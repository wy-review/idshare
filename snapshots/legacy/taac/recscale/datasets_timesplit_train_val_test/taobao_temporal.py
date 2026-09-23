"""TaobaoAd temporal reader for offline processed train/val/test files.

This reader does not join Tianchi raw files at runtime. It expects an offline
preparation step to produce processed files with the same schema as the current
TaobaoAd reader:

USER_SPARSE + AD_SPARSE + cate_his + brand_his + clk

Supported processed file names under dataset.path:
- train.parquet / val.parquet / test.parquet, or
- train.csv / val.csv / test.csv

Set dataset.sequence_enabled=false for pointwise experiments that must not
consume or build sequence features. The default remains sequence-enabled for
backward compatibility.
"""

from __future__ import annotations

import csv
import hashlib
import os

import numpy as np

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

from recscale.datasets import register_dataset
from recscale.datasets.base import BaseDataset
from recscale.datasets.taobao_ad_enhanced import AD_SPARSE, SEQ_COLS, USER_SPARSE


@register_dataset("taobao_ad_temporal")
class TaobaoAdTemporalDataset(BaseDataset):
    CONTENT_FINGERPRINT_ALGORITHM = "length-prefixed-utf8-consumed-columns-v1"
    IDENTITY_SPECIAL_TOKEN_IDS = {
        "padding": 0,
        "oov": 1,
        "missing": 2,
        "reserved_zero": 3,
    }

    _shared_feature_maps = None
    _shared_seq_maps = None
    _shared_cardinalities = None
    _shared_sequence_enabled = None
    _shared_identity_special_token_fields = None

    def __init__(self, config: dict, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"[TaobaoAdTemporal] split must be train/val/test, got {split}")
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        max_rows = dc.get("max_rows", 0)
        self.sequence_enabled = bool(dc.get("sequence_enabled", True))
        self.dual_sequence = dc.get("dual_sequence", True)
        self.sparse_cols = USER_SPARSE + AD_SPARSE
        raw_identity_fields = dc.get("identity_special_token_fields", [])
        if not isinstance(raw_identity_fields, (list, tuple)):
            raise ValueError(
                "[TaobaoAdTemporal] identity_special_token_fields must be a list"
            )
        self.identity_special_token_fields = tuple(str(x) for x in raw_identity_fields)
        unknown_identity_fields = sorted(
            set(self.identity_special_token_fields) - set(self.sparse_cols)
        )
        if unknown_identity_fields:
            raise ValueError(
                "[TaobaoAdTemporal] identity_special_token_fields contains unknown "
                f"columns: {unknown_identity_fields}"
            )
        if len(set(self.identity_special_token_fields)) != len(
            self.identity_special_token_fields
        ):
            raise ValueError(
                "[TaobaoAdTemporal] identity_special_token_fields contains duplicates"
            )

        rows = self._load_rows(data_path, split, max_rows)
        print(f"[TaobaoAdTemporal] {split}: loaded {len(rows):,} processed rows from {data_path}")
        self.content_fingerprint_sha256 = self._verify_split_identity(
            rows=rows,
            split=split,
            config=dc,
        )

        if split == "train":
            self.feature_maps, cardinalities = self._build_feature_maps(rows)
            self.seq_maps = self._build_seq_maps(rows) if self.sequence_enabled else {}
            TaobaoAdTemporalDataset._shared_feature_maps = self.feature_maps
            TaobaoAdTemporalDataset._shared_seq_maps = self.seq_maps
            TaobaoAdTemporalDataset._shared_cardinalities = cardinalities
            TaobaoAdTemporalDataset._shared_sequence_enabled = self.sequence_enabled
            TaobaoAdTemporalDataset._shared_identity_special_token_fields = (
                self.identity_special_token_fields
            )
            dc["cardinalities"] = cardinalities
            dc["num_items"] = len(self.seq_maps.get("cate_his", {})) + 1
            dc["num_items2"] = len(self.seq_maps.get("brand_his", {})) + 1
        else:
            self.feature_maps = TaobaoAdTemporalDataset._shared_feature_maps
            self.seq_maps = TaobaoAdTemporalDataset._shared_seq_maps
            cardinalities = TaobaoAdTemporalDataset._shared_cardinalities
            if self.feature_maps is None or self.seq_maps is None or cardinalities is None:
                raise RuntimeError("[TaobaoAdTemporal] val/test loaded before train split")
            if TaobaoAdTemporalDataset._shared_sequence_enabled != self.sequence_enabled:
                raise RuntimeError(
                    "[TaobaoAdTemporal] train and validation/test must use the same "
                    "sequence_enabled setting"
                )
            if (
                TaobaoAdTemporalDataset._shared_identity_special_token_fields
                != self.identity_special_token_fields
            ):
                raise RuntimeError(
                    "[TaobaoAdTemporal] train and validation/test must use the same "
                    "identity_special_token_fields setting"
                )
            dc["cardinalities"] = cardinalities
            dc["num_items"] = len(self.seq_maps.get("cate_his", {})) + 1
            dc["num_items2"] = len(self.seq_maps.get("brand_his", {})) + 1

        dc["sparse_cols"] = self.sparse_cols
        dc["num_sparse"] = len(self.sparse_cols)
        dc["sequence_enabled"] = self.sequence_enabled
        dc["num_seq_fields"] = (
            (2 if self.dual_sequence else 1) if self.sequence_enabled else 0
        )
        dc["seq_field_names"] = (
            (SEQ_COLS if self.dual_sequence else ["cate_his"])
            if self.sequence_enabled
            else []
        )

        self.labels = np.array([int(float(r.get("clk", 0) or 0)) for r in rows], dtype=np.float32)
        self.sparse = np.array(
            [
                [self._encode_sparse_value(c, r.get(c, "")) for c in self.sparse_cols]
                for r in rows
            ],
            dtype=np.int64,
        )

        if not self.sequence_enabled:
            self.seqs = None
            self.targets = None
        elif self.dual_sequence:
            self.seqs = np.zeros((len(rows), 2, self.maxlen), dtype=np.int64)
            for i, r in enumerate(rows):
                for s, scol in enumerate(SEQ_COLS):
                    self._fill_seq(self.seqs[i, s], r.get(scol, ""), self.seq_maps.get(scol, {}))
            self.targets = np.zeros((len(rows), 2), dtype=np.int64)
            for i, r in enumerate(rows):
                self.targets[i, 0] = self.seq_maps.get("cate_his", {}).get(r.get("cate_id", ""), 0)
                self.targets[i, 1] = self.seq_maps.get("brand_his", {}).get(r.get("brand", ""), 0)
        else:
            self.seqs = np.zeros((len(rows), self.maxlen), dtype=np.int64)
            for i, r in enumerate(rows):
                self._fill_seq(self.seqs[i], r.get("cate_his", ""), self.seq_maps.get("cate_his", {}))
            self.targets = np.array(
                [self.seq_maps.get("cate_his", {}).get(r.get("cate_id", ""), 0) for r in rows],
                dtype=np.int64,
            )

        dc["num_items"] = max(
            dc.get("num_items", 1),
            max(self.seq_maps.get("cate_his", {}).values(), default=0) + 1,
        )
        dc["num_items2"] = max(
            dc.get("num_items2", 1),
            max(self.seq_maps.get("brand_his", {}).values(), default=0) + 1,
        )

        n_pos = int((self.labels > 0.5).sum())
        print(
            f"[TaobaoAdTemporal] {split}: {len(rows):,} samples, "
            f"pos={n_pos:,} ({n_pos / max(len(rows), 1) * 100:.2f}%), "
            f"sequence_enabled={self.sequence_enabled}, "
            f"dual_sequence={self.dual_sequence if self.sequence_enabled else False}"
        )

    def _load_rows(self, data_path: str, split: str, max_rows: int) -> list[dict]:
        parquet_path = os.path.join(data_path, f"{split}.parquet")
        csv_path = os.path.join(data_path, f"{split}.csv")
        if os.path.exists(parquet_path):
            if pq is None:
                raise RuntimeError("pyarrow is required to read parquet processed TaobaoAd files")
            table = pq.read_table(parquet_path)
            rows = table.to_pylist()
        elif os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            raise FileNotFoundError(f"Cannot find {split}.parquet or {split}.csv in {data_path}")
        if max_rows > 0:
            rows = rows[:max_rows]
        return rows

    def _verify_split_identity(
        self,
        *,
        rows: list[dict],
        split: str,
        config: dict,
    ) -> str | None:
        """Fail closed on an optional train/validation content contract."""
        expected_counts = config.get("expected_split_counts")
        expected_fingerprints = config.get("consumed_content_fingerprints")
        if expected_counts is None and expected_fingerprints is None:
            return None
        if split not in {"train", "val"}:
            raise RuntimeError(
                "[TaobaoAdTemporal] split identity contracts only permit train/val"
            )
        if not isinstance(expected_counts, dict) or not isinstance(
            expected_fingerprints, dict
        ):
            raise ValueError(
                "[TaobaoAdTemporal] expected_split_counts and "
                "consumed_content_fingerprints must both be mappings"
            )
        expected = expected_counts.get(split)
        expected_fingerprint = expected_fingerprints.get(split)
        if not isinstance(expected, dict) or not isinstance(
            expected_fingerprint, str
        ):
            raise ValueError(
                f"[TaobaoAdTemporal] missing frozen identity contract for {split}"
            )
        observed_rows = len(rows)
        observed_positives = sum(
            int(float(row.get("clk", 0) or 0)) for row in rows
        )
        if observed_rows != int(expected.get("rows", -1)):
            raise RuntimeError(
                f"[TaobaoAdTemporal] {split} row count mismatch: "
                f"{observed_rows} != {expected.get('rows')}"
            )
        if observed_positives != int(expected.get("positives", -1)):
            raise RuntimeError(
                f"[TaobaoAdTemporal] {split} positive count mismatch: "
                f"{observed_positives} != {expected.get('positives')}"
            )

        digest = hashlib.sha256(self.CONTENT_FINGERPRINT_ALGORITHM.encode("ascii"))
        for row in rows:
            for column in [*self.sparse_cols, "clk"]:
                value = row.get(column, "")
                raw = ("" if value is None else str(value)).encode("utf-8")
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
        observed_fingerprint = digest.hexdigest()
        if observed_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"[TaobaoAdTemporal] {split} consumed-column fingerprint mismatch"
            )
        return observed_fingerprint

    def _fill_seq(self, out: np.ndarray, raw, mapping: dict):
        if raw is None:
            return
        if isinstance(raw, (list, tuple)):
            ids = [str(x) for x in raw][-self.maxlen:]
        else:
            raw = str(raw)
            ids = raw.split("^")[-self.maxlen:] if raw else []
        start = self.maxlen - len(ids)
        for k, v in enumerate(ids):
            out[start + k] = mapping.get(v, 0)

    def _build_feature_maps(self, rows):
        print("[TaobaoAdTemporal] Building train-only feature maps...")
        feature_maps = {}
        cardinalities = []
        for col in self.sparse_cols:
            vals = sorted(v for v in set(str(r.get(col, "")) for r in rows) if v)
            first_id = (
                len(self.IDENTITY_SPECIAL_TOKEN_IDS)
                if col in self.identity_special_token_fields
                else 1
            )
            mapping = {v: i + first_id for i, v in enumerate(vals)}
            feature_maps[col] = mapping
            cardinalities.append(
                max(mapping.values(), default=first_id - 1) + 1
            )
        return feature_maps, cardinalities

    def _encode_sparse_value(self, column: str, raw_value) -> int:
        value = "" if raw_value is None else str(raw_value)
        mapping = self.feature_maps.get(column, {})
        if column not in self.identity_special_token_fields:
            return mapping.get(value, 0)
        if not value:
            return self.IDENTITY_SPECIAL_TOKEN_IDS["missing"]
        return mapping.get(value, self.IDENTITY_SPECIAL_TOKEN_IDS["oov"])

    def _build_seq_maps(self, rows):
        seq_maps = {}
        for scol in SEQ_COLS:
            vals = set()
            for r in rows:
                raw = r.get(scol, "")
                if isinstance(raw, (list, tuple)):
                    vals.update(str(x) for x in raw if x is not None)
                elif raw:
                    vals.update(str(x) for x in str(raw).split("^") if x)
            mapping = {v: i + 1 for i, v in enumerate(sorted(vals))}
            seq_maps[scol] = mapping
        return seq_maps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.sequence_enabled:
            return {
                "sparse": self.sparse[idx],
                "label": self.labels[idx],
            }
        if self.dual_sequence:
            return {
                "sparse": self.sparse[idx],
                "seq": self.seqs[idx, 0],
                "target": self.targets[idx, 0],
                "seq2": self.seqs[idx, 1],
                "target2": self.targets[idx, 1],
                "seqs": self.seqs[idx],
                "targets": self.targets[idx],
                "label": self.labels[idx],
            }
        return {
            "sparse": self.sparse[idx],
            "seq": self.seqs[idx],
            "target": self.targets[idx],
            "label": self.labels[idx],
        }
