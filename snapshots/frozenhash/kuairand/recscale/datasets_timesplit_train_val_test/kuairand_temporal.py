"""Temporal KuaiRand reader using merged standard logs.

Split rule on standard logs:
- last date -> test
- penultimate date -> val
- earlier dates -> train

Random exposure logs are kept out of train/val/test. An optional
`split="test_unbiased"` loads random logs with histories from prior standard-log
interactions and maps unseen random-log items/users to OOV index 0.
"""

from __future__ import annotations

import bisect
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from recscale.datasets import register_dataset
from recscale.datasets.base import BaseDataset, validate_cardinalities
from recscale.datasets.side_features import SideFeatureMixin

from .base import infer_splits_from_dates, split_for_date, timestamp_to_date


@register_dataset("kuairand_temporal")
class KuaiRandTemporalDataset(SideFeatureMixin, BaseDataset):
    _shared_vid2idx = None
    _shared_uid2idx = None
    _shared_std_timeline = None
    _shared_split_dates = None
    _shared_sparse_specs = None
    _shared_dense_specs = None
    _shared_bucket_boundaries = None
    _shared_user_rows: dict = {}
    _shared_item_rows: dict = {}
    _shared_cat_rows: dict = {}

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _row_any_positive(row, cols):
        return any(int(float(row.get(col, 0) or 0)) > 0 for col in cols)

    @staticmethod
    def _action_type(row):
        if int(float(row.get("is_hate", 0) or 0)) > 0:
            return 4
        if any(int(float(row.get(col, 0) or 0)) > 0 for col in (
            "long_view", "is_like", "is_follow", "is_comment", "is_forward"
        )):
            return 3
        if int(float(row.get("is_click", 0) or 0)) > 0:
            return 2
        return 1

    def __init__(self, config: dict, split: str = "train"):
        if split not in ("train", "val", "test", "test_unbiased"):
            raise ValueError(f"[KuaiRandTemporal] split must be train/val/test/test_unbiased, got {split}")
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        max_rows = dc.get("max_rows", 0)
        tz = dc.get("timezone", "Asia/Shanghai")
        label_col = dc.get("label_col", "is_click")
        label_cols = self._as_list(dc.get("label_cols")) or [label_col]
        positive_history_cols = (
            self._as_list(dc.get("positive_history_cols"))
            or self._as_list(dc.get("history_positive_cols"))
            or label_cols
        )
        negative_label_cols = self._as_list(dc.get("negative_label_cols")) or []
        sequence_filter = dc.get("sequence_filter", "all")
        if sequence_filter not in ("all", "positive"):
            raise ValueError("[KuaiRandTemporal] sequence_filter must be all or positive")
        self.use_action_types = bool(dc.get("use_action_types", False))

        def make_label(row):
            positive = self._row_any_positive(row, label_cols)
            negative = self._row_any_positive(row, negative_label_cols)
            return int(positive and not negative)

        def make_history_positive(row):
            positive = self._row_any_positive(row, positive_history_cols)
            negative = self._row_any_positive(row, negative_label_cols)
            return int(positive and not negative)

        if split == "train" or KuaiRandTemporalDataset._shared_vid2idx is None:
            self._build_shared_state(data_path, dc, max_rows, tz, make_history_positive, sequence_filter)

        vid2idx = KuaiRandTemporalDataset._shared_vid2idx
        uid2idx = KuaiRandTemporalDataset._shared_uid2idx
        std_timeline = KuaiRandTemporalDataset._shared_std_timeline
        split_dates = KuaiRandTemporalDataset._shared_split_dates
        sparse_specs = KuaiRandTemporalDataset._shared_sparse_specs
        dense_specs = KuaiRandTemporalDataset._shared_dense_specs
        user_rows = KuaiRandTemporalDataset._shared_user_rows
        item_rows = KuaiRandTemporalDataset._shared_item_rows
        cat_rows = KuaiRandTemporalDataset._shared_cat_rows
        num_buckets = dc.get("num_buckets", 100)

        if split == "train":
            sparse_specs = KuaiRandTemporalDataset._shared_sparse_specs
            dense_specs = KuaiRandTemporalDataset._shared_dense_specs
        else:
            if sparse_specs is None:
                raise RuntimeError("[KuaiRandTemporal] val/test loaded before train initialized shared state")
            self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)

        if split == "test_unbiased":
            rows = self._load_random_rows(data_path, max_rows, make_label)
        else:
            rows = self._load_standard_rows_for_split(data_path, max_rows, tz, split_dates, split, make_label)

        samples = self._build_samples(rows, uid2idx, vid2idx, std_timeline)
        raw_uids = np.array([s[5] for s in samples], dtype=np.int64)
        raw_vids = np.array([s[6] for s in samples], dtype=np.int64)
        self.target_ids = np.array([s[1] for s in samples], dtype=np.int64)
        self.labels = np.array([s[4] for s in samples], dtype=np.float32)
        self.seqs = np.zeros((len(samples), self.maxlen), dtype=np.int64)
        self.seq_actions = np.zeros((len(samples), self.maxlen), dtype=np.int64)
        for i, (_, _, seq, seq_action, _, _, _) in enumerate(samples):
            if seq:
                start = self.maxlen - len(seq)
                self.seqs[i, start:] = seq
                if self.use_action_types:
                    self.seq_actions[i, start:] = np.array(seq_action, dtype=np.int64)

        if split == "train":
            self.sparse, boundaries = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets
            )
            KuaiRandTemporalDataset._shared_bucket_boundaries = boundaries
            self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)
        else:
            self.sparse, _ = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets,
                bucket_boundaries=KuaiRandTemporalDataset._shared_bucket_boundaries
            )
            self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)

        validate_cardinalities(config, self.sparse, dc["sparse_cols"])
        n_pos = int((self.labels > 0.5).sum())
        print(
            f"[KuaiRandTemporal] {split}: {len(samples):,} samples, "
            f"pos={n_pos:,} ({n_pos / max(len(samples), 1) * 100:.2f}%)"
        )

    def _build_shared_state(self, data_path, dc, max_rows, tz, make_history_positive, sequence_filter):
        std_matches = sorted(glob.glob(os.path.join(data_path, "log_standard*.csv")))
        if not std_matches:
            raise FileNotFoundError(f"Cannot find log_standard*.csv in {data_path}")
        all_vids, all_uids, all_dates = set(), set(), []
        timeline_lists = defaultdict(list)
        n = 0
        for fp in std_matches:
            with open(fp) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 0 < max_rows <= n:
                        break
                    uid = int(row["user_id"])
                    vid = int(row["video_id"])
                    t = int(row["time_ms"])
                    date = timestamp_to_date(t, unit="ms", tz=tz)
                    all_dates.append(date)
                    all_vids.add(vid)
                    all_uids.add(uid)
                    hist_pos = make_history_positive(row)
                    if sequence_filter == "all" or hist_pos > 0:
                        timeline_lists[uid].append((t, vid, self._action_type(row)))
                    n += 1
                if 0 < max_rows <= n:
                    break
        split_dates = infer_splits_from_dates(all_dates)
        vid2idx = {v: i + 1 for i, v in enumerate(sorted(all_vids))}
        uid2idx = {u: i + 1 for i, u in enumerate(sorted(all_uids))}
        std_timeline = {}
        for uid, rows in timeline_lists.items():
            rows.sort(key=lambda x: x[0])
            std_timeline[uid] = (
                np.array([x[0] for x in rows], dtype=np.int64),
                np.array([vid2idx.get(x[1], 0) for x in rows], dtype=np.int64),
                np.array([x[2] for x in rows], dtype=np.int64),
            )
        KuaiRandTemporalDataset._shared_vid2idx = vid2idx
        KuaiRandTemporalDataset._shared_uid2idx = uid2idx
        KuaiRandTemporalDataset._shared_std_timeline = std_timeline
        KuaiRandTemporalDataset._shared_split_dates = split_dates
        dc["num_items"] = len(vid2idx) + 1
        dc["num_users"] = len(uid2idx) + 1

        num_buckets = dc.get("num_buckets", 100)
        user_rows, item_rows, cat_rows = self._load_side_features(data_path, dc)
        sparse_specs, dense_specs = self._build_feature_specs(uid2idx, vid2idx, user_rows, item_rows, cat_rows, dc)
        KuaiRandTemporalDataset._shared_sparse_specs = sparse_specs
        KuaiRandTemporalDataset._shared_dense_specs = dense_specs
        KuaiRandTemporalDataset._shared_user_rows = user_rows
        KuaiRandTemporalDataset._shared_item_rows = item_rows
        KuaiRandTemporalDataset._shared_cat_rows = cat_rows
        self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)
        print(
            f"[KuaiRandTemporal] Shared state: {n:,} standard rows, "
            f"{len(uid2idx):,} users, {len(vid2idx):,} videos, "
            f"train={sorted(split_dates['train'])[0]}..{sorted(split_dates['train'])[-1]} "
            f"val={next(iter(split_dates['val']))} test={next(iter(split_dates['test']))}"
        )

    def _load_standard_rows_for_split(self, data_path, max_rows, tz, split_dates, split, make_label):
        rows = []
        n = 0
        for fp in sorted(glob.glob(os.path.join(data_path, "log_standard*.csv"))):
            with open(fp) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 0 < max_rows <= n:
                        break
                    t = int(row["time_ms"])
                    date = timestamp_to_date(t, unit="ms", tz=tz)
                    if split_for_date(date, split_dates) == split:
                        rows.append((int(row["user_id"]), int(row["video_id"]), t, make_label(row)))
                    n += 1
                if 0 < max_rows <= n:
                    break
        return rows

    def _load_random_rows(self, data_path, max_rows, make_label):
        rows = []
        n = 0
        for fp in sorted(glob.glob(os.path.join(data_path, "log_random*.csv"))):
            with open(fp) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 0 < max_rows <= n:
                        break
                    rows.append((int(row["user_id"]), int(row["video_id"]), int(row["time_ms"]), make_label(row)))
                    n += 1
                if 0 < max_rows <= n:
                    break
        return rows

    def _build_samples(self, rows, uid2idx, vid2idx, std_timeline):
        samples = []
        n_oov = 0
        for uid, vid, t, lbl in rows:
            uidx = uid2idx.get(uid, 0)
            vidx = vid2idx.get(vid, 0)
            if uidx == 0 or vidx == 0:
                n_oov += 1
            timeline = std_timeline.get(uid)
            if timeline is None:
                seq, seq_action = [], []
            else:
                times, vids, actions = timeline
                cut = int(np.searchsorted(times, t, side="left"))
                start = max(0, cut - self.maxlen)
                seq = vids[start:cut].tolist()
                seq_action = actions[start:cut].tolist()
                if cut > 0 and int(times[cut - 1]) >= t:
                    raise AssertionError("[KuaiRandTemporal] causal history violation")
            samples.append((uidx, vidx, seq, seq_action, float(lbl), uid, vid))
        if n_oov:
            print(f"[KuaiRandTemporal] OOV uid/vid rows mapped to 0: {n_oov:,}")
        return samples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "sparse": self.sparse[idx],
            "seq": self.seqs[idx],
            "target": self.target_ids[idx],
            "label": self.labels[idx],
        }
        if self.use_action_types:
            item["seq_action"] = self.seq_actions[idx]
        return item
