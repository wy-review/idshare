"""Temporal KuaiRec reader using deterministic day-level splits.

Split rule:
- last date -> test
- penultimate date -> val
- earlier dates -> train

This reader is non-destructive: it only reads the existing KuaiRec files and does
not modify remote data. It registers as `type: kuairec_temporal` when this
package is imported.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

from recscale.datasets import register_dataset
from recscale.datasets.base import BaseDataset, validate_cardinalities
from recscale.datasets.side_features import SideFeatureMixin

from .base import DayStats, infer_splits_from_dates, infer_splits_from_day_stats, split_for_date, timestamp_to_date


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@register_dataset("kuairec_temporal")
class KuaiRecTemporalDataset(SideFeatureMixin, BaseDataset):
    _shared_sparse_specs = None
    _shared_dense_specs = None
    _shared_bucket_boundaries = None
    _shared_user_rows: dict = {}
    _shared_item_rows: dict = {}
    _shared_cat_rows: dict = {}
    _shared_uid2idx: dict = {}
    _shared_vid2idx: dict = {}
    _shared_split_dates = None
    _shared_typeA_edges = None
    _shared_typeB_edges = None
    _shared_item_ctr = None
    _shared_item_exposures = None
    _shared_item_unique_users = None

    def __init__(self, config: dict, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"[KuaiRecTemporal] split must be train/val/test, got {split}")

        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        threshold = dc.get("watch_ratio_threshold", 2.0)
        sequence_filter = dc.get("sequence_filter", "all")
        if sequence_filter not in ("all", "positive", "positive_recent_window"):
            raise ValueError(
                f"[KuaiRecTemporal] Unsupported sequence_filter={sequence_filter!r}; "
                "expected 'all', 'positive', or 'positive_recent_window'"
            )
        max_rows = dc.get("max_rows", 0)
        tz = dc.get("timezone", "Asia/Shanghai")
        min_eval_samples = int(dc.get("min_eval_samples", 0))
        min_eval_users = int(dc.get("min_eval_users", 0))
        if dc.get("use_seq_derived_features", False) or dc.get("use_item_global_stats", False):
            raise NotImplementedError(
                "[KuaiRecTemporal] Type A/B derived features are not implemented yet; "
                "disable use_seq_derived_features/use_item_global_stats or port the original feature block."
            )

        if split == "train":
            self._reset_shared()

        csv_path = os.path.join(data_path, "big_matrix.csv")
        print(f"[KuaiRecTemporal] Loading {csv_path} ...")

        user_interactions = defaultdict(list)
        all_dates = []
        day_stats = defaultdict(DayStats)
        n = 0
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 0 < max_rows <= n:
                    break
                uid = int(row["user_id"])
                vid = int(row["video_id"])
                ts = float(row["timestamp"])
                wr = float(row["watch_ratio"])
                date = timestamp_to_date(ts, unit="s", tz=tz)
                user_interactions[uid].append((ts, date, vid, wr))
                all_dates.append(date)
                day_stats[date].add(uid, vid, int(wr >= threshold))
                n += 1

        if not all_dates:
            raise RuntimeError("[KuaiRecTemporal] No interactions loaded")
        if min_eval_samples > 0 or min_eval_users > 0:
            split_dates = infer_splits_from_day_stats(day_stats, min_samples=min_eval_samples, min_users=min_eval_users)
        else:
            split_dates = infer_splits_from_dates(all_dates)
        KuaiRecTemporalDataset._shared_split_dates = split_dates

        for uid in user_interactions:
            user_interactions[uid].sort(key=lambda x: x[0])

        all_vids = {vid for ints in user_interactions.values() for _, _, vid, _ in ints}
        vid2idx = {v: i + 1 for i, v in enumerate(sorted(all_vids))}
        uid2idx = {u: i + 1 for i, u in enumerate(sorted(user_interactions.keys()))}
        KuaiRecTemporalDataset._shared_uid2idx = uid2idx
        KuaiRecTemporalDataset._shared_vid2idx = vid2idx

        dc["num_items"] = len(vid2idx) + 1
        dc["num_users"] = len(uid2idx) + 1

        samples = []  # (uidx, vidx, seq, label, raw_uid, raw_vid, sample_ts)
        seq_timestamps_list = []
        split_counts = {"train": 0, "val": 0, "test": 0}
        for uid, ints in user_interactions.items():
            uidx = uid2idx[uid]
            history = []
            history_ts = []
            for ts, date, vid, wr in ints:
                vidx = vid2idx[vid]
                label = 1.0 if wr >= threshold else 0.0
                try:
                    sample_split = split_for_date(date, split_dates)
                except KeyError:
                    continue
                split_counts[sample_split] += 1
                if sample_split == split:
                    cut = len(history_ts)
                    while cut > 0 and history_ts[cut - 1] >= ts:
                        cut -= 1
                    if sequence_filter == "positive_recent_window":
                        recent = history[:cut][-self.maxlen:]
                        seq = [h_vidx for h_vidx, h_label in recent if h_label > 0.5]
                        seq_ts = []
                    else:
                        seq = history[:cut][-self.maxlen:]
                        seq_ts = history_ts[:cut][-self.maxlen:]
                    if seq_ts and max(seq_ts) >= ts:
                        raise AssertionError("[KuaiRecTemporal] causal history violation")
                    samples.append((uidx, vidx, seq, label, uid, vid, ts))
                    seq_timestamps_list.append(seq_ts.copy())
                if sequence_filter == "positive_recent_window":
                    history.append((vidx, label))
                    history_ts.append(ts)
                elif sequence_filter == "all" or label > 0.5:
                    history.append(vidx)
                    history_ts.append(ts)

        raw_uids = np.array([s[4] for s in samples], dtype=np.int64)
        raw_vids = np.array([s[5] for s in samples], dtype=np.int64)
        self.labels = np.array([s[3] for s in samples], dtype=np.float32)
        self.target_ids = np.array([s[1] for s in samples], dtype=np.int64)
        self.seqs = np.zeros((len(samples), self.maxlen), dtype=np.int64)
        for i, (_, _, seq, _, _, _, _) in enumerate(samples):
            if seq:
                self.seqs[i, self.maxlen - len(seq):] = seq

        num_buckets = dc.get("num_buckets", 100)
        if split == "train":
            user_rows, item_rows, cat_rows = self._load_side_features(data_path, dc)
            sparse_specs, dense_specs = self._build_feature_specs(
                uid2idx, vid2idx, user_rows, item_rows, cat_rows, dc
            )
            self.sparse, boundaries = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets
            )
            KuaiRecTemporalDataset._shared_sparse_specs = sparse_specs
            KuaiRecTemporalDataset._shared_dense_specs = dense_specs
            KuaiRecTemporalDataset._shared_bucket_boundaries = boundaries
            KuaiRecTemporalDataset._shared_user_rows = user_rows
            KuaiRecTemporalDataset._shared_item_rows = item_rows
            KuaiRecTemporalDataset._shared_cat_rows = cat_rows
        else:
            sparse_specs = KuaiRecTemporalDataset._shared_sparse_specs
            dense_specs = KuaiRecTemporalDataset._shared_dense_specs
            boundaries = KuaiRecTemporalDataset._shared_bucket_boundaries
            user_rows = KuaiRecTemporalDataset._shared_user_rows
            item_rows = KuaiRecTemporalDataset._shared_item_rows
            cat_rows = KuaiRecTemporalDataset._shared_cat_rows
            if sparse_specs is None:
                raise RuntimeError("[KuaiRecTemporal] val/test split loaded before train split")
            self.sparse, _ = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets,
                bucket_boundaries=boundaries
            )
        self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)
        validate_cardinalities(config, self.sparse, dc["sparse_cols"])

        n_pos = int((self.labels > 0.5).sum())
        print(
            f"[KuaiRecTemporal] {split}: {len(samples):,} samples, "
            f"pos={n_pos:,} ({n_pos / max(len(samples), 1) * 100:.2f}%), "
            f"dates train={sorted(split_dates['train'])[0]}..{sorted(split_dates['train'])[-1]} "
            f"val={next(iter(split_dates['val']))} test={next(iter(split_dates['test']))}, "
            f"split_counts={split_counts}"
        )

    @classmethod
    def _reset_shared(cls):
        cls._shared_sparse_specs = None
        cls._shared_dense_specs = None
        cls._shared_bucket_boundaries = None
        cls._shared_user_rows = {}
        cls._shared_item_rows = {}
        cls._shared_cat_rows = {}
        cls._shared_uid2idx = {}
        cls._shared_vid2idx = {}
        cls._shared_split_dates = None
        cls._shared_typeA_edges = None
        cls._shared_typeB_edges = None
        cls._shared_item_ctr = None
        cls._shared_item_exposures = None
        cls._shared_item_unique_users = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sparse": self.sparse[idx],
            "seq": self.seqs[idx],
            "target": self.target_ids[idx],
            "label": self.labels[idx],
        }
