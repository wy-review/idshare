"""TAAC2025 CTR DataLoader for FuxiCTR.

Returns dict-of-tensors batches compatible with all FuxiCTR rank models.
Performs on-the-fly seq window cutoff + sideinfo lookup; no sample materialization.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data

USER_SPARSE: List[int] = [103, 104, 105, 109]
USER_ARRAY:  List[int] = [106, 107, 108, 110]
ITEM_SPARSE: List[int] = [100, 101, 102, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122]


# ---------- helpers ----------

def _pad_1d(arr, maxlen: int) -> np.ndarray:
    """Left-pad / truncate a 1D sequence of ints to fixed length. Pad value 0."""
    out = np.zeros(maxlen, dtype=np.int64)
    if arr is None or len(arr) == 0:
        return out
    arr = np.asarray(arr, dtype=np.int64)
    if len(arr) >= maxlen:
        out[:] = arr[-maxlen:]
    else:
        out[-len(arr):] = arr
    return out


def _load_feat_dict(feat_dir: Path, key_col: str,
                    sparse_ids: List[int],
                    array_ids: Optional[List[int]] = None) -> Dict[int, dict]:
    """Load a parquet directory into {key: {fid: value or list}}.
    sparse columns -> int; array columns -> python list.
    """
    import pyarrow.dataset as pds  # lazy import
    array_ids = array_ids or []
    cols = [key_col] + [str(f) for f in sparse_ids] + [str(f) for f in array_ids]
    ds = pds.dataset(str(feat_dir), format="parquet")
    tbl = ds.to_table(columns=cols)
    d: Dict[int, dict] = {}
    key_list = tbl.column(key_col).to_pylist()
    sparse_cols = {fid: tbl.column(str(fid)).to_pylist() for fid in sparse_ids}
    array_cols = {fid: tbl.column(str(fid)).to_pylist() for fid in array_ids}
    for i, k in enumerate(key_list):
        row = {}
        for fid in sparse_ids:
            v = sparse_cols[fid][i]
            if v is not None:
                row[fid] = int(v)
        for fid in array_ids:
            v = array_cols[fid][i]
            if v is not None:
                row[fid] = list(v)
        d[k] = row
    return d


def _load_user_seqs(user_seqs_dir: Path, keep_timestamp: bool = False,
                    keep_action_type: bool = False) -> Dict[int, object]:
    """Load user_seqs, optionally keeping timestamp and/or action_type for each item."""
    import pyarrow.dataset as pds  # lazy import
    ds = pds.dataset(str(user_seqs_dir), format="parquet")
    tbl = ds.to_table(columns=["user_id", "seq"])
    uids = tbl.column("user_id").to_pylist()
    seqs = tbl.column("seq").to_pylist()
    out: Dict[int, object] = {}
    for uid, seq in zip(uids, seqs):
        item_ids = np.fromiter(
            (x["item_id"] for x in seq), dtype=np.int64, count=len(seq))
        extra = {}
        if keep_timestamp:
            extra["timestamp"] = np.fromiter(
                (x.get("timestamp", 0) for x in seq), dtype=np.int64, count=len(seq))
        if keep_action_type:
            extra["action_type"] = np.fromiter(
                (x.get("action_type", 0) + 1 for x in seq), dtype=np.int64, count=len(seq))
        if extra:
            out[uid] = {"item_id": item_ids, **extra}
        else:
            out[uid] = item_ids
    return out


# ---------- Dataset ----------

class TaacCTRDataset(torch.utils.data.Dataset):
    def __init__(self, feature_map, manifest_path,
                 maxlen: int = 50, user_array_maxlen: int = 10,
                 preloaded: Optional[dict] = None,
                 max_samples: Optional[int] = None,
                 sequence_side_fields: Optional[List[int]] = None,
                 emit_sequence_timestamp: bool = False,
                 emit_sequence_action_type: bool = False):
        """Args:
            feature_map: FuxiCTR FeatureMap (not used inside but kept for API).
            manifest_path: path to {split}_manifest.json
            preloaded: optional dict with keys 'item_feat', 'user_feat', 'user_seqs',
                       and optionally 'samples' (dict with uid/tid/label/cutoff arrays)
                       to share across train/valid/test datasets or bypass parquet.
            sequence_side_fields: list of ITEM_SPARSE fids to emit as per-step
                       item_seq__feat_<fid> side info sequences. None/[] disables.
            emit_sequence_timestamp: if True, emit item_seq__timestamp aligned with item_seq.
            emit_sequence_action_type: if True, emit item_seq__action_type aligned with
                       item_seq and feat_action_type=0 for target (interaction-level side info).
        """
        self.maxlen = maxlen
        self.user_array_maxlen = user_array_maxlen
        self.sequence_side_fields = list(sequence_side_fields or [])
        self.emit_sequence_timestamp = emit_sequence_timestamp
        self.emit_sequence_action_type = emit_sequence_action_type
        self._raw_root: Optional[Path] = None

        # samples: either from preloaded (unit-test bypass) or from parquet
        if preloaded and "samples" in preloaded:
            s = preloaded["samples"]
            uid    = np.asarray(s["user_id"],        dtype=np.int64)
            tid    = np.asarray(s["target_item_id"], dtype=np.int64)
            label  = np.asarray(s["label"],          dtype=np.float32)
            cutoff = np.asarray(s["seq_cutoff_pos"], dtype=np.int32)
        else:
            import pyarrow.parquet as pq  # lazy import
            with open(manifest_path) as f:
                meta = json.load(f)
            root = Path(meta["raw_data_root"])
            t = pq.read_table(meta["samples_path"]).to_pydict()
            uid    = np.asarray(t["user_id"],        dtype=np.int64)
            tid    = np.asarray(t["target_item_id"], dtype=np.int64)
            label  = np.asarray(t["label"],          dtype=np.float32)
            cutoff = np.asarray(t["seq_cutoff_pos"], dtype=np.int32)

            if meta.get("index_array"):
                idx = np.load(meta["index_array"])
                uid, tid, label, cutoff = uid[idx], tid[idx], label[idx], cutoff[idx]

            self._raw_root = root

        # side info (can be shared via preloaded)
        if preloaded and "item_feat" in preloaded:
            self.item_feat = preloaded["item_feat"]
        else:
            self.item_feat = _load_feat_dict(self._raw_root / "item_feat", "item_id", ITEM_SPARSE)

        if preloaded and "user_feat" in preloaded:
            self.user_feat = preloaded["user_feat"]
        else:
            self.user_feat = _load_feat_dict(self._raw_root / "user_feat", "user_id",
                                              USER_SPARSE, USER_ARRAY)

        if preloaded and "user_seqs" in preloaded:
            self.user_seqs = preloaded["user_seqs"]
        else:
            self.user_seqs = _load_user_seqs(
                self._raw_root / "user_seqs",
                keep_timestamp=emit_sequence_timestamp,
                keep_action_type=emit_sequence_action_type)

        # filter rows whose user has no seq
        keep = np.fromiter((u in self.user_seqs for u in uid), dtype=bool, count=len(uid))
        n_before = len(uid)
        self.uid    = uid[keep]
        self.tid    = tid[keep]
        self.label  = label[keep]
        self.cutoff = cutoff[keep]
        n_after = len(self.uid)
        if n_after < n_before:
            print(f"[TaacCTRDataset] filtered {n_before - n_after}/{n_before} rows missing user_seq")

        # optionally cap to first N samples for smoke testing
        if max_samples is not None and max_samples < len(self.uid):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.uid), size=max_samples, replace=False)
            idx.sort()
            self.uid    = self.uid[idx]
            self.tid    = self.tid[idx]
            self.label  = self.label[idx]
            self.cutoff = self.cutoff[idx]
            print(f"[TaacCTRDataset] max_samples={max_samples}: using {len(self.uid)} samples")

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx: int):
        uid    = int(self.uid[idx])
        tid    = int(self.tid[idx])
        label  = float(self.label[idx])
        cutoff = int(self.cutoff[idx])

        full = self.user_seqs[uid]
        if isinstance(full, dict):
            full_items = full["item_id"]
            full_timestamps = full.get("timestamp")
            full_action_types = full.get("action_type")
        else:
            full_items = full
            full_timestamps = None
            full_action_types = None
        win = full_items[:cutoff][-self.maxlen:] if cutoff > 0 else np.empty(0, dtype=np.int64)
        ts_win = full_timestamps[:cutoff][-self.maxlen:] \
            if full_timestamps is not None and cutoff > 0 else np.empty(0, dtype=np.int64)
        at_win = full_action_types[:cutoff][-self.maxlen:] \
            if full_action_types is not None and cutoff > 0 else np.empty(0, dtype=np.int64)
        seq_pad = np.zeros(self.maxlen, dtype=np.int64)
        ts_pad = np.zeros(self.maxlen, dtype=np.int64)
        at_pad = np.zeros(self.maxlen, dtype=np.int64)
        if len(win):
            seq_pad[-len(win):] = win
            if self.emit_sequence_timestamp:
                ts_pad[-len(ts_win):] = ts_win
            if self.emit_sequence_action_type:
                at_pad[-len(at_win):] = at_win

        out = {
            "target_item_id": np.int64(tid),
            "item_seq":       seq_pad,
            "label":          np.float32(label),
        }
        if self.emit_sequence_timestamp:
            out["item_seq__timestamp"] = ts_pad
        if self.emit_sequence_action_type:
            out["item_seq__action_type"] = at_pad
        uf = self.user_feat.get(uid, {})
        for fid in USER_SPARSE:
            v = uf.get(fid, 0)
            out[f"feat_{fid}"] = np.int64(v if v is not None else 0)
        for fid in USER_ARRAY:
            out[f"feat_{fid}"] = _pad_1d(uf.get(fid), self.user_array_maxlen)
        itf = self.item_feat.get(tid, {})
        for fid in ITEM_SPARSE:
            v = itf.get(fid, 0)
            out[f"feat_{fid}"] = np.int64(v if v is not None else 0)
        # per-step side info for item_seq: look up each history item's item_feat.
        # Padding positions stay at 0 (matches padding_idx=0 in feature_map).
        if self.sequence_side_fields and len(win):
            for fid in self.sequence_side_fields:
                side = np.zeros(self.maxlen, dtype=np.int64)
                vals = [self.item_feat.get(int(iid), {}).get(fid, 0) for iid in win]
                side[-len(win):] = vals
                out[f"item_seq__feat_{fid}"] = side
        else:
            for fid in self.sequence_side_fields:
                out[f"item_seq__feat_{fid}"] = np.zeros(self.maxlen, dtype=np.int64)
        return out


# ---------- DataLoader ----------

class TaacDataLoader(torch.utils.data.DataLoader):
    """Custom DataLoader injected via model_config.yaml `data_loader_class`.
    Compatible with FuxiCTR RankDataLoader expectations: exposes num_samples,
    num_blocks, num_batches, __len__, and yields dict-of-tensors batches.
    """
    def __init__(self, feature_map, data_path, split="train",
                 batch_size: int = 1024, shuffle: bool = False,
                 num_workers: int = 0, **kwargs):
        maxlen = kwargs.get("maxlen", 50)
        user_array_maxlen = kwargs.get("user_array_maxlen", 10)
        preloaded = kwargs.get("_taac_preloaded", None)
        max_samples = kwargs.get("max_samples", None)
        sequence_side_fields = kwargs.get("sequence_side_fields", None)
        emit_sequence_timestamp = kwargs.get("emit_sequence_timestamp", False)
        emit_sequence_action_type = kwargs.get("emit_sequence_action_type", False)

        dataset = TaacCTRDataset(
            feature_map, data_path,
            maxlen=maxlen, user_array_maxlen=user_array_maxlen,
            preloaded=preloaded,
            max_samples=max_samples,
            sequence_side_fields=sequence_side_fields,
            emit_sequence_timestamp=emit_sequence_timestamp,
            emit_sequence_action_type=emit_sequence_action_type,
        )
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=torch.utils.data.dataloader.default_collate,
            drop_last=False,
        )
        self.num_samples = len(dataset)
        self.num_blocks = 1
        self.num_batches = int(math.ceil(self.num_samples / batch_size))

    def __len__(self):
        return self.num_batches
