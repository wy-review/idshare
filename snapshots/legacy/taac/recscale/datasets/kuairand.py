"""
recscale.datasets.kuairand — KuaiRand 数据集

KuaiRand 在标准推荐 feed 中随机替换 ~0.37% 的位置插入随机视频，
通过对比随机曝光 vs 标准推荐的点击，实现无偏评估。

三种 split：
  train        — log_standard_4_08_to_4_21（标准日志，前期）
  test_biased  — log_standard_4_22_to_5_08 中 is_rand=0 的行（有偏，标准推荐）
  test_unbiased (默认"test") — log_random（随机曝光，无偏）

序列构建：
  train: 当前交互之前的同用户历史（滑窗）
  test_*: 使用 log_standard 全期（4_08 + 4_22）时间点之前的历史
          按 time_ms 截断，防止 leakage

YAML 示例:
```yaml
dataset:
  type: kuairand
  path: /data/kuairand/KuaiRand-1K/data
  maxlen: 50
  max_rows: 0
```
"""

import bisect
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from . import register_dataset
from .base import BaseDataset
from .side_features import SideFeatureMixin


@register_dataset("kuairand")
class KuaiRandDataset(SideFeatureMixin, BaseDataset):
    """
    支持三种 split:
      "train"         — log_standard_4_08_to_4_21
      "test"          — log_random (无偏，默认测试集)
      "test_biased"   — log_standard_4_22_to_5_08 中 is_rand=0 的行（有偏对照）
    """

    # 类级别共享 vocab + 用户历史，train 建立，test 复用
    _shared_vid2idx = None
    _shared_uid2idx = None
    _shared_std_timeline = None
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
        """0=pad, 1=exposed/non-click, 2=click/valid-play, 3=strong engagement, 4=hate."""
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
        """
        split: "train" | "test" | "test_biased"
        """
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        max_rows    = dc.get("max_rows", 0)
        label_col   = dc.get("label_col", "is_click")
        label_cols  = self._as_list(dc.get("label_cols")) or [label_col]
        positive_history_cols = (
            self._as_list(dc.get("positive_history_cols"))
            or self._as_list(dc.get("history_positive_cols"))
            or label_cols
        )
        negative_label_cols = self._as_list(dc.get("negative_label_cols")) or []
        use_action_types = bool(dc.get("use_action_types", False))
        self.use_action_types = use_action_types
        sequence_filter = dc.get("sequence_filter", "all")
        if sequence_filter not in ("all", "positive"):
            raise ValueError(
                f"[KuaiRand] Unsupported sequence_filter={sequence_filter!r}; "
                "expected 'all' or 'positive'"
            )

        def make_label(row):
            positive = self._row_any_positive(row, label_cols)
            negative = self._row_any_positive(row, negative_label_cols)
            return int(positive and not negative)

        def make_history_positive(row):
            positive = self._row_any_positive(row, positive_history_cols)
            negative = self._row_any_positive(row, negative_label_cols)
            return int(positive and not negative)

        assert split in ("train", "test", "test_biased"), \
            f"split must be train/test/test_biased, got {split}"

        print(
            f"[KuaiRand] split={split}, maxlen={self.maxlen}, "
            f"label_cols={label_cols}, positive_history_cols={positive_history_cols}, "
            f"negative_label_cols={negative_label_cols}, sequence_filter={sequence_filter}, "
            f"use_action_types={use_action_types}"
        )

        # ---- 加载所有 standard log（train 样本 + 所有 test 的历史序列）----
        std_pattern = os.path.join(data_path, "log_standard*.csv")
        std_matches = sorted(glob.glob(std_pattern))
        if not std_matches:
            raise FileNotFoundError(f"Cannot find log_standard*.csv in {data_path}")

        # ---- 构建 vocab（train 时建立，test 时复用）----
        if split == "train" or KuaiRandDataset._shared_vid2idx is None:
            compact_timeline = bool(dc.get("compact_timeline", False))
            all_vids, all_uids = set(), set()

            if compact_timeline:
                n_std = 0
                for std_path in std_matches:
                    with open(std_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if 0 < max_rows <= n_std:
                                break
                            all_uids.add(int(row["user_id"]))
                            all_vids.add(int(row["video_id"]))
                            n_std += 1
                    if 0 < max_rows <= n_std:
                        break
                print(f"[KuaiRand] Standard log: {n_std:,} rows, {len(all_uids):,} users")

                # 加入 random log 的 vid/uid
                for rand_path in sorted(glob.glob(os.path.join(data_path, "log_random*.csv"))):
                    with open(rand_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            all_vids.add(int(row["video_id"]))
                            all_uids.add(int(row["user_id"]))

                vid2idx = {v: i+1 for i, v in enumerate(sorted(all_vids))}
                uid2idx = {u: i+1 for i, u in enumerate(sorted(all_uids))}
                KuaiRandDataset._shared_vid2idx = vid2idx
                KuaiRandDataset._shared_uid2idx = uid2idx

                # 构建紧凑全量时序历史（用于 test 序列按时间截断）。
                timeline_lists = defaultdict(list)
                n_timeline = 0
                for std_path in std_matches:
                    with open(std_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if 0 < max_rows <= n_timeline:
                                break
                            hist_pos = make_history_positive(row)
                            if sequence_filter == "all" or hist_pos > 0:
                                vidx = vid2idx.get(int(row["video_id"]), 0)
                                if vidx > 0:
                                    timeline_lists[int(row["user_id"])].append((
                                        int(row["time_ms"]),
                                        vidx,
                                        self._action_type(row),
                                    ))
                            n_timeline += 1
                    if 0 < max_rows <= n_timeline:
                        break

                std_timeline = {}
                for uid, rows in timeline_lists.items():
                    rows.sort(key=lambda x: x[0])
                    std_timeline[uid] = (
                        np.array([x[0] for x in rows], dtype=np.int64),
                        np.array([x[1] for x in rows], dtype=np.int64),
                        np.array([x[2] for x in rows], dtype=np.int64),
                    )
                print(f"[KuaiRand] Compact timeline: {n_timeline:,} rows, {len(std_timeline):,} users")
            else:
                std_interactions = defaultdict(list)  # uid → [(time_ms, vid, label, history_positive, action_type, is_rand)]
                n_std = 0
                for std_path in std_matches:
                    with open(std_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if 0 < max_rows <= n_std:
                                break
                            uid     = int(row["user_id"])
                            vid     = int(row["video_id"])
                            t       = int(row["time_ms"])
                            lbl     = make_label(row)
                            hist_pos = make_history_positive(row)
                            action_type = self._action_type(row)
                            is_rand = int(row.get("is_rand", 0))
                            std_interactions[uid].append((t, vid, lbl, hist_pos, action_type, is_rand))
                            n_std += 1
                    if 0 < max_rows <= n_std:
                        break

                print(f"[KuaiRand] Standard log: {n_std:,} rows, {len(std_interactions):,} users")
                for uid in std_interactions:
                    std_interactions[uid].sort(key=lambda x: x[0])

                for ints in std_interactions.values():
                    for t, vid, _, _, _, _ in ints:
                        all_vids.add(vid)
                all_uids.update(std_interactions.keys())
                # 加入 random log 的 vid/uid
                for rand_path in sorted(glob.glob(os.path.join(data_path, "log_random*.csv"))):
                    with open(rand_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            all_vids.add(int(row["video_id"]))
                            all_uids.add(int(row["user_id"]))

                vid2idx = {v: i+1 for i, v in enumerate(sorted(all_vids))}
                uid2idx = {u: i+1 for i, u in enumerate(sorted(all_uids))}
                KuaiRandDataset._shared_vid2idx = vid2idx
                KuaiRandDataset._shared_uid2idx = uid2idx

                # 构建全量时序历史（用于 test 序列按时间截断）
                std_timeline = {}
                for uid, ints in std_interactions.items():
                    std_timeline[uid] = [
                        (t, vid2idx[vid], action_type)
                        for t, vid, _lbl, hist_pos, action_type, _is_rand in ints
                        if vid2idx.get(vid, 0) > 0
                        and (sequence_filter == "all" or hist_pos > 0)
                    ]
            KuaiRandDataset._shared_std_timeline = std_timeline

            dc["num_items"] = len(vid2idx) + 1
            dc["num_users"] = len(uid2idx) + 1
            print(f"[KuaiRand] Vocab: {len(vid2idx):,} videos, {len(uid2idx):,} users")

            # ---- side features ----
            # KuaiRand 文件名随子集（1K / 27K）不同，不硬编码默认值。
            # 必须在 YAML 中显式配置，例如（KuaiRand-27K 目录下的真实文件名）：
            #   user_feature_file: user_features.csv        # 或 user_features_27k.csv
            #   item_feature_glob: "video_features_*.csv"   # 合并 basic + statistic
            # 若未配置则静默退化为 [user_id, video_id] 两列。
            #
            # video feature 文件的 key 列待真实数据确认：
            #   SideFeatureMixin 默认用 "video_id"；若实际列名不同可配置 item_feature_key。
            num_buckets = dc.get("num_buckets", 100)
            user_rows, item_rows, cat_rows = self._load_side_features(data_path, dc)
            sparse_specs, dense_specs = self._build_feature_specs(
                uid2idx, vid2idx, user_rows, item_rows, cat_rows, dc)
            KuaiRandDataset._shared_sparse_specs = sparse_specs
            KuaiRandDataset._shared_dense_specs  = dense_specs
            KuaiRandDataset._shared_user_rows    = user_rows
            KuaiRandDataset._shared_item_rows    = item_rows
            KuaiRandDataset._shared_cat_rows     = cat_rows
            # bucket boundaries 在 _precompute_sparse_with_buckets 时生成
            KuaiRandDataset._shared_bucket_boundaries = None  # 先置 None，后面填
        else:
            vid2idx      = KuaiRandDataset._shared_vid2idx
            uid2idx      = KuaiRandDataset._shared_uid2idx
            std_timeline = KuaiRandDataset._shared_std_timeline
            sparse_specs = KuaiRandDataset._shared_sparse_specs
            dense_specs  = KuaiRandDataset._shared_dense_specs
            user_rows    = KuaiRandDataset._shared_user_rows
            item_rows    = KuaiRandDataset._shared_item_rows
            cat_rows     = KuaiRandDataset._shared_cat_rows
            num_buckets  = dc.get("num_buckets", 100)
            self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)

        # ---- 构建样本，保留原始 uid/vid 用于查 side feature ----
        if split == "train":
            # 用 log_standard_4_08_to_4_21（第一个文件）构建滑窗样本
            train_file = std_matches[0]
            train_ints = defaultdict(list)
            n_train_rows = 0
            with open(train_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 0 < max_rows <= n_train_rows:
                        break
                    uid = int(row["user_id"])
                    vid = int(row["video_id"])
                    t   = int(row["time_ms"])
                    lbl = make_label(row)
                    hist_pos = make_history_positive(row)
                    action_type = self._action_type(row)
                    train_ints[uid].append((t, vid, lbl, hist_pos, action_type))
                    n_train_rows += 1
            for uid in train_ints:
                train_ints[uid].sort(key=lambda x: x[0])

            samples = []  # (uidx, vidx, seq, label, raw_uid, raw_vid)
            for uid, ints in train_ints.items():
                uidx = uid2idx.get(uid, 0)
                history = []
                history_actions = []
                for t, vid, lbl, hist_pos, action_type in ints:
                    vidx = vid2idx.get(vid, 0)
                    if vidx == 0:
                        continue
                    seq = history[-self.maxlen:]
                    seq_action = history_actions[-self.maxlen:]
                    samples.append((uidx, vidx, seq, seq_action, float(lbl), uid, vid))
                    if sequence_filter == "all" or hist_pos > 0:
                        history.append(vidx)
                        history_actions.append(action_type)
            print(f"[KuaiRand] Train: {len(samples):,} samples")

        else:
            def _build_test_samples(rows):
                """rows: list of (uid, vid, t, lbl)"""
                result = []
                n_skip = 0
                for uid, vid, t, lbl in rows:
                    uidx = uid2idx.get(uid, 0)
                    vidx = vid2idx.get(vid, 0)
                    if uidx == 0 or vidx == 0:
                        n_skip += 1
                        continue
                    timeline = std_timeline.get(uid, [])
                    if isinstance(timeline, tuple):
                        times, vids, actions = timeline
                        cut = int(np.searchsorted(times, t, side="left"))
                        start = max(0, cut - self.maxlen)
                        seq = vids[start:cut].tolist()
                        seq_action = actions[start:cut].tolist()
                    else:
                        times    = [x[0] for x in timeline]
                        cut      = bisect.bisect_left(times, t)
                        window   = timeline[:cut][-self.maxlen:]
                        seq      = [x[1] for x in window]
                        seq_action = [x[2] for x in window]
                    result.append((uidx, vidx, seq, seq_action, float(lbl), uid, vid))
                if n_skip:
                    print(f"[KuaiRand] Skipped {n_skip:,} rows (unknown uid/vid)")
                return result

            if split == "test":
                rows = []
                for rand_path in sorted(glob.glob(os.path.join(data_path, "log_random*.csv"))):
                    with open(rand_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            rows.append((
                                int(row["user_id"]),
                                int(row["video_id"]),
                                int(row["time_ms"]),
                                make_label(row),
                            ))
                samples = _build_test_samples(rows)
                print(f"[KuaiRand] test (unbiased, random log): {len(samples):,} samples")

            else:  # test_biased
                biased_file = std_matches[-1]
                rows = []
                with open(biased_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if int(row.get("is_rand", 0)) == 0:
                            rows.append((
                                int(row["user_id"]),
                                int(row["video_id"]),
                                int(row["time_ms"]),
                                make_label(row),
                            ))
                samples = _build_test_samples(rows)
                print(f"[KuaiRand] test_biased (std log, is_rand=0): {len(samples):,} samples")

        # ---- 转 numpy + 组装完整 sparse（含数值分桶）----
        raw_uids        = np.array([s[5] for s in samples], dtype=np.int64)
        raw_vids        = np.array([s[6] for s in samples], dtype=np.int64)
        self.target_ids = np.array([s[1] for s in samples], dtype=np.int64)
        self.labels     = np.array([s[4] for s in samples], dtype=np.float32)

        self.seqs = np.zeros((len(samples), self.maxlen), dtype=np.int64)
        self.seq_actions = np.zeros((len(samples), self.maxlen), dtype=np.int64)
        for i, (_, _, seq, seq_action, _, _, _) in enumerate(samples):
            if seq:
                start = self.maxlen - len(seq)
                self.seqs[i, start:] = seq
                if use_action_types:
                    self.seq_actions[i, start:] = np.array(seq_action, dtype=np.int64)

        num_buckets = dc.get("num_buckets", 100)
        if split == "train":
            self.sparse, boundaries = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets)
            KuaiRandDataset._shared_bucket_boundaries = boundaries
            self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)
        else:
            self.sparse, _ = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets,
                bucket_boundaries=KuaiRandDataset._shared_bucket_boundaries)
            self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)

        from .base import validate_cardinalities
        validate_cardinalities(config, self.sparse, dc["sparse_cols"])
        n_pos = (self.labels > 0.5).sum()
        print(f"[KuaiRand] {split}: {len(samples):,} samples, "
              f"sparse={self.sparse.shape[1]}, "
              f"pos={n_pos:,} ({n_pos / max(len(samples), 1) * 100:.2f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "sparse": self.sparse[idx],
            "seq":    self.seqs[idx],
            "target": self.target_ids[idx],
            "label":  self.labels[idx],
        }
        if getattr(self, "use_action_types", False):
            item["seq_action"] = self.seq_actions[idx]
        return item
