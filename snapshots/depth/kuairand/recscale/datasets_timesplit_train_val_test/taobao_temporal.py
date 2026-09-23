"""TaobaoAd temporal reader for offline processed train/val/test files.

This reader does not join Tianchi raw files at runtime. It expects an offline
preparation step to produce processed files with the same schema as the current
TaobaoAd reader:

USER_SPARSE + AD_SPARSE + cate_his + brand_his + clk

Supported processed file names under dataset.path:
- train.parquet / val.parquet / test.parquet, or
- train.csv / val.csv / test.csv
"""

from __future__ import annotations

import csv
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
    _shared_feature_maps = None
    _shared_seq_maps = None
    _shared_cardinalities = None

    def __init__(self, config: dict, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"[TaobaoAdTemporal] split must be train/val/test, got {split}")
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        max_rows = dc.get("max_rows", 0)
        self.dual_sequence = dc.get("dual_sequence", True)
        self.sparse_cols = USER_SPARSE + AD_SPARSE

        rows = self._load_rows(data_path, split, max_rows)
        print(f"[TaobaoAdTemporal] {split}: loaded {len(rows):,} processed rows from {data_path}")

        if split == "train":
            self.feature_maps, self.seq_maps, cardinalities, seq_vocab = self._build_maps(rows)
            TaobaoAdTemporalDataset._shared_feature_maps = self.feature_maps
            TaobaoAdTemporalDataset._shared_seq_maps = self.seq_maps
            TaobaoAdTemporalDataset._shared_cardinalities = cardinalities
            dc["cardinalities"] = cardinalities
            dc["num_items"] = len(self.seq_maps.get("cate_his", {})) + 1
            dc["num_items2"] = len(self.seq_maps.get("brand_his", {})) + 1
        else:
            self.feature_maps = TaobaoAdTemporalDataset._shared_feature_maps
            self.seq_maps = TaobaoAdTemporalDataset._shared_seq_maps
            cardinalities = TaobaoAdTemporalDataset._shared_cardinalities
            if self.feature_maps is None or self.seq_maps is None or cardinalities is None:
                raise RuntimeError("[TaobaoAdTemporal] val/test loaded before train split")
            dc["cardinalities"] = cardinalities
            dc["num_items"] = len(self.seq_maps.get("cate_his", {})) + 1
            dc["num_items2"] = len(self.seq_maps.get("brand_his", {})) + 1

        dc["sparse_cols"] = self.sparse_cols
        dc["num_sparse"] = len(self.sparse_cols)
        dc["num_seq_fields"] = 2 if self.dual_sequence else 1
        dc["seq_field_names"] = SEQ_COLS if self.dual_sequence else ["cate_his"]

        self.labels = np.array([int(float(r.get("clk", 0) or 0)) for r in rows], dtype=np.float32)
        self.sparse = np.array(
            [[self.feature_maps.get(c, {}).get(r.get(c, ""), 0) for c in self.sparse_cols] for r in rows],
            dtype=np.int64,
        )

        if self.dual_sequence:
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

        dc["num_items"] = max(dc.get("num_items", 1), max(self.seq_maps.get("cate_his", {}).values(), default=0) + 1)
        dc["num_items2"] = max(dc.get("num_items2", 1), max(self.seq_maps.get("brand_his", {}).values(), default=0) + 1)

        n_pos = int((self.labels > 0.5).sum())
        print(
            f"[TaobaoAdTemporal] {split}: {len(rows):,} samples, "
            f"pos={n_pos:,} ({n_pos / max(len(rows), 1) * 100:.2f}%), "
            f"dual_sequence={self.dual_sequence}"
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

    def _build_maps(self, rows):
        print("[TaobaoAdTemporal] Building train-only feature maps...")
        feature_maps = {}
        cardinalities = []
        for col in self.sparse_cols:
            vals = sorted(v for v in set(str(r.get(col, "")) for r in rows) if v)
            mapping = {v: i + 1 for i, v in enumerate(vals)}
            feature_maps[col] = mapping
            cardinalities.append(max(mapping.values(), default=0) + 1)
        seq_maps = {}
        seq_vocab_size = 0
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
            seq_vocab_size = max(seq_vocab_size, len(mapping) + 1)
        return feature_maps, seq_maps, cardinalities, seq_vocab_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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
