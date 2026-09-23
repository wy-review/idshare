"""
recscale.datasets.kuairec — KuaiRec 数据集

提供两种 adapter:
1. `type: kuairec`：序列版本，构建 (user, seq, target, label) 样本。
   no-seq 特征与 pointwise 版本完全对齐（user_features.csv + item_daily_features.csv +
   item_categories.csv），文件不存在时静默退化为 [user_id, target_id]。
2. `type: kuairec_pointwise`：非序列 pointwise 版本，输出 sparse/dense 特征，
   适配 RankMixer / UniMixer / MLP / DCN 等点式模型。

Pointwise 版本默认读取:
- big_matrix.csv
- user_features.csv          (可选)
- item_daily_features.csv    (可选)
- item_categories.csv        (可选)

YAML 示例:
```yaml
dataset:
  type: kuairec_pointwise
  path: /data/kuairec/data
  watch_ratio_threshold: 2.0
  max_rows: 0
  train_user_ratio: 0.8
  use_user_features: true
  use_item_features: true
  use_item_category_features: true
```
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import register_dataset
from .base import BaseDataset, validate_cardinalities
from .side_features import SideFeatureMixin


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv_rows(csv_path: str, max_rows: int = 0) -> List[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if 0 < max_rows <= i:
                break
            rows.append(row)
    return rows


def _load_keyed_rows(csv_path: str, key_col: str) -> Dict[int, dict]:
    if not os.path.exists(csv_path):
        return {}
    keyed = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _safe_int(row.get(key_col), 0)
            if key > 0:
                keyed[key] = row
    return keyed


def _infer_feature_groups(
    keyed_rows: Dict[int, dict],
    key_col: str,
    explicit_sparse: Optional[List[str]] = None,
    explicit_dense: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    if not keyed_rows:
        return [], []

    sample_row = next(iter(keyed_rows.values()))
    all_cols = [c for c in sample_row.keys() if c != key_col]

    if explicit_sparse is not None or explicit_dense is not None:
        sparse = [c for c in (explicit_sparse or []) if c in sample_row]
        dense = [c for c in (explicit_dense or []) if c in sample_row and c not in sparse]
        return sparse, dense

    ignore_tokens = (
        "caption", "text", "topic", "tag", "name", "upload_dt", "upload_date",
        "date", "manual_cover", "comment_staytime", "comment", "description",
    )
    sparse_hint_tokens = (
        "_id", "gender", "age", "degree", "level", "type", "brand", "model",
        "platform", "country", "province", "city", "community", "channel",
        "version", "author", "music", "category",
    )
    dense_hint_tokens = (
        "cnt", "num", "ratio", "duration", "days", "price", "score", "rate",
        "play", "like", "follow", "share", "download", "report", "collect",
        "fans", "friend", "register",
    )

    sparse_cols = []
    dense_cols = []

    for col in all_cols:
        col_lower = col.lower()
        if any(tok in col_lower for tok in ignore_tokens):
            continue

        values = [row.get(col, "") for row in keyed_rows.values() if row.get(col, "") not in ("", None)]
        if not values:
            continue

        numeric_values = []
        is_numeric = True
        for v in values:
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                is_numeric = False
                break

        unique_count = len(set(values))

        if col_lower.startswith("onehot_feat") or any(tok in col_lower for tok in sparse_hint_tokens):
            sparse_cols.append(col)
            continue

        if is_numeric:
            all_integer_like = all(abs(v - round(v)) < 1e-8 for v in numeric_values[: min(512, len(numeric_values))])
            if all_integer_like and unique_count <= 128 and not any(tok in col_lower for tok in dense_hint_tokens):
                sparse_cols.append(col)
            else:
                dense_cols.append(col)
        else:
            if unique_count <= 256:
                sparse_cols.append(col)

    return sparse_cols, dense_cols


@register_dataset("kuairec")
class KuaiRecDataset(SideFeatureMixin, BaseDataset):

    _shared_sparse_specs = None
    _shared_dense_specs = None
    _shared_bucket_boundaries = None
    _shared_user_rows: dict = {}
    _shared_item_rows: dict = {}
    _shared_cat_rows: dict = {}
    _shared_uid2idx: dict = {}
    _shared_vid2idx: dict = {}

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        threshold = dc.get("watch_ratio_threshold", 2.0)
        sequence_filter = dc.get("sequence_filter", "all")
        if sequence_filter not in ("all", "positive", "positive_recent_window"):
            raise ValueError(
                f"[KuaiRec] Unsupported sequence_filter={sequence_filter!r}; "
                "expected 'all', 'positive', or 'positive_recent_window'"
            )
        max_rows = dc.get("max_rows", 0)
        seed = config.get("seed", 42)

        # Reset class-level shared variables when starting a fresh train split
        # to avoid contamination from previous jobs on the same worker process
        if split == "train":
            KuaiRecDataset._shared_sparse_specs = None
            KuaiRecDataset._shared_dense_specs = None
            KuaiRecDataset._shared_bucket_boundaries = None
            KuaiRecDataset._shared_user_rows = {}
            KuaiRecDataset._shared_item_rows = {}
            KuaiRecDataset._shared_cat_rows = {}
            KuaiRecDataset._shared_uid2idx = {}
            KuaiRecDataset._shared_vid2idx = {}

        csv_path = os.path.join(data_path, "big_matrix.csv")
        print(f"[KuaiRec] Loading {csv_path} ...")

        user_interactions = defaultdict(list)
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
                user_interactions[uid].append((ts, vid, wr))
                n += 1

        print(f"[KuaiRec] {n:,} interactions, {len(user_interactions):,} users")

        for uid in user_interactions:
            user_interactions[uid].sort(key=lambda x: x[0])

        all_vids = set()
        for ints in user_interactions.values():
            for _, vid, _ in ints:
                all_vids.add(vid)
        vid2idx = {v: i + 1 for i, v in enumerate(sorted(all_vids))}
        uid2idx = {u: i + 1 for i, u in enumerate(sorted(user_interactions.keys()))}

        dc["num_items"] = len(vid2idx) + 1
        dc["num_users"] = len(uid2idx) + 1

        # ---- 构建所有样本（保留原始 uid/vid 用于查 side feature）--------
        samples = []          # (uidx, vidx, seq_idxs, label, raw_uid, raw_vid)
        # 额外保存序列时间戳，用于 Type A-seq 时间统计
        seq_timestamps_list = []  # 每条样本对应的序列时间戳
        for uid, ints in user_interactions.items():
            uidx = uid2idx[uid]
            history = []       # vidx list
            history_ts = []    # timestamp list (parallel to history)
            for ts, vid, wr in ints:
                vidx = vid2idx[vid]
                label = 1.0 if wr >= threshold else 0.0
                if sequence_filter == "positive_recent_window":
                    recent = history[-self.maxlen:]
                    seq = [h_vidx for h_vidx, h_label in recent if h_label > 0.5]
                    seq_ts = []  # 暂不支持 positive_recent_window 的时间统计
                else:
                    seq = history[-self.maxlen:]
                    seq_ts = history_ts[-self.maxlen:]
                samples.append((uidx, vidx, seq, label, uid, vid))
                seq_timestamps_list.append(seq_ts.copy())
                if sequence_filter == "positive_recent_window":
                    history.append((vidx, label))
                    history_ts.append(ts)
                elif sequence_filter == "all" or label > 0.5:
                    history.append(vidx)
                    history_ts.append(ts)

        # ---- 按用户切分 train/test -----------------------------------------
        all_uids = sorted(user_interactions.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(all_uids)
        n_train = int(len(all_uids) * 0.8)
        keep_uids = set(all_uids[:n_train] if split == "train" else all_uids[n_train:])
        keep_uidxs = {uid2idx[u] for u in keep_uids}
        keep_mask = [s[0] in keep_uidxs for s in samples]
        samples = [s for s, keep in zip(samples, keep_mask) if keep]
        seq_timestamps_list = [t for t, keep in zip(seq_timestamps_list, keep_mask) if keep]

        # ---- 提取数组 -------------------------------------------------------
        raw_uids = np.array([s[4] for s in samples], dtype=np.int64)
        raw_vids = np.array([s[5] for s in samples], dtype=np.int64)
        self.labels = np.array([s[3] for s in samples], dtype=np.float32)

        self.seqs = np.zeros((len(samples), self.maxlen), dtype=np.int64)
        for i, (_, _, seq, _, _, _) in enumerate(samples):
            if seq:
                start = self.maxlen - len(seq)
                self.seqs[i, start:] = seq

        # ---- 加载 side features 并构建完整 sparse 矩阵 ----------------------
        num_buckets = dc.get("num_buckets", 100)
        if split == "train":
            user_rows, item_rows, cat_rows = self._load_side_features(data_path, dc)
            sparse_specs, dense_specs = self._build_feature_specs(
                uid2idx, vid2idx, user_rows, item_rows, cat_rows, dc)
            self.sparse, boundaries = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets)
            # 保存到类级别，供 test split 复用
            KuaiRecDataset._shared_sparse_specs = sparse_specs
            KuaiRecDataset._shared_dense_specs = dense_specs
            KuaiRecDataset._shared_bucket_boundaries = boundaries
            KuaiRecDataset._shared_user_rows = user_rows
            KuaiRecDataset._shared_item_rows = item_rows
            KuaiRecDataset._shared_cat_rows = cat_rows
            KuaiRecDataset._shared_uid2idx = uid2idx
            KuaiRecDataset._shared_vid2idx = vid2idx
        else:
            sparse_specs = KuaiRecDataset._shared_sparse_specs
            dense_specs  = KuaiRecDataset._shared_dense_specs
            boundaries   = KuaiRecDataset._shared_bucket_boundaries
            user_rows    = KuaiRecDataset._shared_user_rows
            item_rows    = KuaiRecDataset._shared_item_rows
            cat_rows     = KuaiRecDataset._shared_cat_rows
            uid2idx      = KuaiRecDataset._shared_uid2idx
            vid2idx      = KuaiRecDataset._shared_vid2idx
            if sparse_specs is None:
                raise RuntimeError("[KuaiRec] test split loaded before train split")
            self.sparse, _ = self._precompute_sparse_with_buckets(
                raw_uids, raw_vids, uid2idx, vid2idx,
                user_rows, item_rows, cat_rows,
                sparse_specs, dense_specs, num_buckets=num_buckets,
                bucket_boundaries=boundaries)
        self._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets)

        # target_ids：用于 seq_emb 查表（保持 vidx）
        self.target_ids = np.array([s[1] for s in samples], dtype=np.int64)

        # ---- Type A/B 探索性特征 -------------------------------------------
        # Type A: 从序列可统计的特征（序列长度、多样性、target是否重复）
        # Type B: 跨样本全局统计（item 全局 CTR、曝光量）
        use_seq_derived = dc.get("use_seq_derived_features", False)
        # 可选：只启用指定的 Type A 子特征（逗号分隔或单个名称）
        seq_derived_only = dc.get("seq_derived_features_only", None)
        use_item_global = dc.get("use_item_global_stats", False)

        if use_seq_derived or use_item_global:
            from .side_features import compute_bucket_edges, apply_bucket
            extra_cols = []
            extra_names = []
            extra_cards = []
            n_bkt = num_buckets

            if use_seq_derived:
                # 决定要计算哪些 Type A 特征
                if seq_derived_only:
                    if isinstance(seq_derived_only, str):
                        selected_feats = set(seq_derived_only.split(","))
                    else:
                        selected_feats = set(seq_derived_only)
                else:
                    # 默认全部
                    selected_feats = {
                        "seq_length", "seq_unique_items",
                        "target_in_history", "target_history_count",
                        # 新增 A-seq 统计
                        "seq_fill_ratio", "repeat_ratio", "top1_item_ratio", "seq_entropy",
                        "seq_time_span", "avg_time_interval",
                        # 多窗口类目分布（需要 cat_rows）
                        "recent_3_cat_concentration", "recent_10_cat_concentration",
                        "recent_30_cat_concentration",
                    }

                N = len(self.seqs)

                # ---- 基础统计 ----
                seq_lengths = np.array([(s != 0).sum() for s in self.seqs], dtype=np.float32)
                seq_unique = np.array([len(set(s[s != 0].tolist())) for s in self.seqs], dtype=np.float32)
                target_in_hist = np.array(
                    [1.0 if self.target_ids[i] in set(self.seqs[i][self.seqs[i] != 0].tolist()) else 0.0
                     for i in range(N)], dtype=np.float32)
                target_hist_count = np.array(
                    [(self.seqs[i] == self.target_ids[i]).sum()
                     for i in range(N)], dtype=np.float32)

                # ---- 频次/集中度统计 ----
                seq_fill_ratio = seq_lengths / max(self.maxlen, 1)
                repeat_ratio = np.where(seq_lengths > 0, 1.0 - seq_unique / seq_lengths, 0.0)

                top1_item_ratio = np.zeros(N, dtype=np.float32)
                seq_entropy = np.zeros(N, dtype=np.float32)
                for i in range(N):
                    valid = self.seqs[i][self.seqs[i] != 0]
                    if len(valid) > 0:
                        _, counts = np.unique(valid, return_counts=True)
                        top1_item_ratio[i] = counts.max() / len(valid)
                        probs = counts / counts.sum()
                        seq_entropy[i] = -(probs * np.log(probs + 1e-10)).sum()

                # ---- 时间统计 ----
                seq_time_span = np.zeros(N, dtype=np.float32)
                avg_time_interval = np.zeros(N, dtype=np.float32)
                for i in range(N):
                    ts_list = seq_timestamps_list[i]
                    if len(ts_list) >= 2:
                        seq_time_span[i] = ts_list[-1] - ts_list[0]
                        intervals = [ts_list[j+1] - ts_list[j] for j in range(len(ts_list)-1)]
                        avg_time_interval[i] = sum(intervals) / len(intervals)
                    elif len(ts_list) == 1:
                        seq_time_span[i] = 0.0

                # ---- 多窗口类目分布 ----
                # 构建 vidx -> category 映射（使用 cat_rows + vid2idx）
                vidx_to_cat = {}
                if cat_rows:
                    item_key = dc.get("item_feature_key", "video_id")
                    for raw_vid, row in cat_rows.items():
                        vidx = vid2idx.get(raw_vid, 0)
                        if vidx > 0:
                            # 取第一个类目字段作为 category
                            for col in row:
                                if col != item_key:
                                    vidx_to_cat[vidx] = row[col]
                                    break

                def compute_cat_concentration(seq_arr, window_size):
                    """计算序列最后 window_size 个 item 中最高频类目占比"""
                    result = np.zeros(len(seq_arr), dtype=np.float32)
                    if not vidx_to_cat:
                        return result
                    for i in range(len(seq_arr)):
                        valid = seq_arr[i][seq_arr[i] != 0]
                        recent = valid[-window_size:] if len(valid) > 0 else np.array([])
                        if len(recent) == 0:
                            continue
                        cats = [vidx_to_cat.get(int(v), "unk") for v in recent]
                        if cats:
                            from collections import Counter
                            c = Counter(cats)
                            result[i] = c.most_common(1)[0][1] / len(cats)
                    return result

                recent_3_cat_conc = compute_cat_concentration(self.seqs, 3)
                recent_10_cat_conc = compute_cat_concentration(self.seqs, 10)
                recent_30_cat_conc = compute_cat_concentration(self.seqs, 30)

                # ---- 分桶并添加到 extra_cols ----
                all_feats = {
                    "seq_length": seq_lengths,
                    "seq_unique_items": seq_unique,
                    "seq_fill_ratio": seq_fill_ratio,
                    "repeat_ratio": repeat_ratio,
                    "top1_item_ratio": top1_item_ratio,
                    "seq_entropy": seq_entropy,
                    "seq_time_span": seq_time_span,
                    "avg_time_interval": avg_time_interval,
                    "recent_3_cat_concentration": recent_3_cat_conc,
                    "recent_10_cat_concentration": recent_10_cat_conc,
                    "recent_30_cat_concentration": recent_30_cat_conc,
                }
                # 二值/特殊特征
                binary_feats = {
                    "target_in_history": target_in_hist,
                    "target_history_count": target_hist_count,
                }

                if split == "train":
                    typeA_edges = {}
                    for fname, arr in all_feats.items():
                        valid_vals = arr[arr > 0] if arr.max() > 0 else arr
                        typeA_edges[fname] = compute_bucket_edges(valid_vals if len(valid_vals) > 0 else arr, n_bkt)
                    typeA_edges["target_history_count"] = compute_bucket_edges(target_hist_count, n_bkt)
                    KuaiRecDataset._shared_typeA_edges = typeA_edges
                else:
                    typeA_edges = KuaiRecDataset._shared_typeA_edges

                for fname in sorted(selected_feats):
                    if fname == "target_in_history":
                        extra_cols.append((target_in_hist + 1).astype(np.int64))
                        extra_names.append("typeA__target_in_history")
                        extra_cards.append(3)
                    elif fname == "target_history_count":
                        extra_cols.append(apply_bucket(target_hist_count, typeA_edges["target_history_count"], n_bkt))
                        extra_names.append("typeA__target_history_count")
                        extra_cards.append(n_bkt + 1)
                    elif fname in all_feats:
                        extra_cols.append(apply_bucket(all_feats[fname], typeA_edges[fname], n_bkt))
                        extra_names.append(f"typeA__{fname}")
                        extra_cards.append(n_bkt + 1)

            if use_item_global:
                # Type B 跨样本统计：从全训练集统计 item 的全局 CTR 和曝光量
                if split == "train":
                    # 统计每个 vidx 的曝光次数和正样本数
                    n_items = dc["num_items"]
                    item_exposures = np.zeros(n_items, dtype=np.float32)
                    item_positives = np.zeros(n_items, dtype=np.float32)
                    item_users = [set() for _ in range(n_items)]
                    for uidx, vidx, _, label, uid, vid in [(s[0], s[1], s[2], s[3], s[4], s[5]) for s in samples]:
                        item_exposures[vidx] += 1
                        if label > 0.5:
                            item_positives[vidx] += 1
                        item_users[vidx].add(uidx)
                    item_ctr = np.where(item_exposures > 0, item_positives / item_exposures, 0.0)
                    item_unique_user_counts = np.array([len(s) for s in item_users], dtype=np.float32)

                    edges_ctr = compute_bucket_edges(item_ctr[item_ctr > 0], n_bkt)
                    edges_exp = compute_bucket_edges(item_exposures[item_exposures > 0], n_bkt)
                    edges_uuser = compute_bucket_edges(item_unique_user_counts[item_unique_user_counts > 0], n_bkt)
                    KuaiRecDataset._shared_typeB_edges = (edges_ctr, edges_exp, edges_uuser)
                    KuaiRecDataset._shared_item_ctr = item_ctr
                    KuaiRecDataset._shared_item_exposures = item_exposures
                    KuaiRecDataset._shared_item_unique_users = item_unique_user_counts
                else:
                    edges_ctr, edges_exp, edges_uuser = KuaiRecDataset._shared_typeB_edges
                    item_ctr = KuaiRecDataset._shared_item_ctr
                    item_exposures = KuaiRecDataset._shared_item_exposures
                    item_unique_user_counts = KuaiRecDataset._shared_item_unique_users

                # 按 target item 查表
                target_ctr = item_ctr[self.target_ids]
                target_exp = item_exposures[self.target_ids]
                target_uusers = item_unique_user_counts[self.target_ids]

                extra_cols.append(apply_bucket(target_ctr, edges_ctr, n_bkt))
                extra_names.append("typeB__item_global_ctr")
                extra_cards.append(n_bkt + 1)

                extra_cols.append(apply_bucket(target_exp, edges_exp, n_bkt))
                extra_names.append("typeB__item_exposure_count")
                extra_cards.append(n_bkt + 1)

                extra_cols.append(apply_bucket(target_uusers, edges_uuser, n_bkt))
                extra_names.append("typeB__item_unique_users")
                extra_cards.append(n_bkt + 1)

            # 追加到 sparse 矩阵
            if extra_cols:
                extra_matrix = np.stack(extra_cols, axis=1)  # (N, num_extra)
                self.sparse = np.concatenate([self.sparse, extra_matrix], axis=1)
                # 重新设置 dc metadata 以确保与 sparse 矩阵一致
                # _update_dc_feature_meta 可能包含 dense_specs 的分桶项，
                # 这里直接从 sparse 矩阵列数重建
                base_cols = dc.get("sparse_cols", [])
                base_cards = dc.get("cardinalities", [])
                # 如果 base 已经多于 sparse - extra，截断到正确长度
                n_base = self.sparse.shape[1] - len(extra_cols)
                dc["sparse_cols"] = base_cols[:n_base] + extra_names
                dc["cardinalities"] = base_cards[:n_base] + extra_cards
                print(f"[KuaiRec] Added {len(extra_names)} Type A/B features: {extra_names}, "
                      f"total sparse={self.sparse.shape[1]}")

        validate_cardinalities(config, self.sparse, dc["sparse_cols"])
        n_pos = (self.labels > 0.5).sum()
        print(f"[KuaiRec] {split}: {len(samples):,} samples, "
              f"sparse={self.sparse.shape[1]}, "
              f"pos={n_pos:,} ({n_pos / max(len(samples), 1) * 100:.2f}%), "
              f"sequence_filter={sequence_filter}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sparse": self.sparse[idx],
            "seq":    self.seqs[idx],
            "target": self.target_ids[idx],
            "label":  self.labels[idx],
        }


@register_dataset("kuairec_pointwise")
class KuaiRecPointwiseDataset(BaseDataset):
    """KuaiRec pointwise adapter for non-sequence models."""

    _shared_uid2idx = None
    _shared_vid2idx = None
    _shared_train_users = None
    _shared_test_users = None
    _shared_sparse_specs = None
    _shared_dense_specs = None
    _shared_bucket_boundaries = None

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        max_rows = dc.get("max_rows", 0)
        threshold = dc.get("watch_ratio_threshold", 2.0)
        train_user_ratio = dc.get("train_user_ratio", 0.8)
        seed = config.get("seed", 42)
        bucketize = dc.get("bucketize_dense", False)
        self.bucketize_dense = bool(bucketize)
        self.bucket_method = bucketize if isinstance(bucketize, str) else dc.get("bucket_method", "log")
        self.num_buckets = dc.get("num_buckets", 100)

        interactions_path = os.path.join(data_path, dc.get("interaction_file", "big_matrix.csv"))
        print(f"[KuaiRecPointwise] Loading {interactions_path} ...")
        interactions = _read_csv_rows(interactions_path, max_rows=max_rows)
        print(f"[KuaiRecPointwise] Loaded {len(interactions):,} interactions")

        all_users = sorted({_safe_int(r.get("user_id"), 0) for r in interactions if _safe_int(r.get("user_id"), 0) > 0})
        all_videos = sorted({_safe_int(r.get("video_id"), 0) for r in interactions if _safe_int(r.get("video_id"), 0) > 0})

        if split == "train":
            uid2idx = {u: i + 1 for i, u in enumerate(all_users)}
            vid2idx = {v: i + 1 for i, v in enumerate(all_videos)}

            rng = np.random.RandomState(seed)
            shuffled_users = list(all_users)
            rng.shuffle(shuffled_users)
            n_train = int(len(shuffled_users) * train_user_ratio)
            train_users = set(shuffled_users[:n_train])
            test_users = set(shuffled_users[n_train:])

            sparse_specs, dense_specs = self._build_feature_specs(data_path, dc, uid2idx, vid2idx)

            KuaiRecPointwiseDataset._shared_uid2idx = uid2idx
            KuaiRecPointwiseDataset._shared_vid2idx = vid2idx
            KuaiRecPointwiseDataset._shared_train_users = train_users
            KuaiRecPointwiseDataset._shared_test_users = test_users
            KuaiRecPointwiseDataset._shared_sparse_specs = sparse_specs
            KuaiRecPointwiseDataset._shared_dense_specs = dense_specs
        else:
            uid2idx = KuaiRecPointwiseDataset._shared_uid2idx
            vid2idx = KuaiRecPointwiseDataset._shared_vid2idx
            train_users = KuaiRecPointwiseDataset._shared_train_users
            test_users = KuaiRecPointwiseDataset._shared_test_users
            sparse_specs = KuaiRecPointwiseDataset._shared_sparse_specs
            dense_specs = KuaiRecPointwiseDataset._shared_dense_specs
            if uid2idx is None or vid2idx is None or sparse_specs is None or dense_specs is None:
                raise RuntimeError(
                    "[KuaiRecPointwise] Test split loaded before train split. "
                    "Must load train split first to establish vocab and feature specs."
                )

        keep_users = train_users if split == "train" else test_users
        user_feature_rows, item_feature_rows, item_category_rows = self._load_side_features(data_path, dc)

        filtered = []
        for row in interactions:
            uid = _safe_int(row.get("user_id"), 0)
            vid = _safe_int(row.get("video_id"), 0)
            if uid in keep_users and uid > 0 and vid > 0:
                filtered.append(row)

        dc["num_users"] = len(uid2idx) + 1
        dc["num_items"] = len(vid2idx) + 1

        base_sparse_cols = [spec[0] for spec in sparse_specs]
        base_dense_cols = [spec[0] for spec in dense_specs]
        base_cardinalities = [spec[3] for spec in sparse_specs]
        dc["sparse_cols"] = base_sparse_cols
        dc["dense_cols"] = base_dense_cols
        dc["cardinalities"] = base_cardinalities

        num_sparse = len(sparse_specs)
        num_dense = len(dense_specs)
        self.sparse = np.zeros((len(filtered), num_sparse), dtype=np.int64)
        raw_dense = np.zeros((len(filtered), num_dense), dtype=np.float32) if num_dense > 0 else None
        dense_as_sparse = None
        self.dense = None
        self.labels = np.zeros(len(filtered), dtype=np.float32)

        for i, row in enumerate(filtered):
            uid = _safe_int(row.get("user_id"), 0)
            vid = _safe_int(row.get("video_id"), 0)
            self.labels[i] = 1.0 if _safe_float(row.get("watch_ratio"), 0.0) >= threshold else 0.0

            sources = {
                "interaction": row,
                "user": user_feature_rows.get(uid, {}),
                "item": item_feature_rows.get(vid, {}),
                "item_category": item_category_rows.get(vid, {}),
            }

            for j, (_, source_name, col_name, cardinality, mapping) in enumerate(sparse_specs):
                source_row = sources.get(source_name, {})
                if source_name == "interaction" and col_name == "user_id":
                    self.sparse[i, j] = uid2idx.get(uid, 0)
                elif source_name == "interaction" and col_name == "video_id":
                    self.sparse[i, j] = vid2idx.get(vid, 0)
                else:
                    raw_value = source_row.get(col_name, "")
                    self.sparse[i, j] = mapping.get(str(raw_value), 0)

            if raw_dense is not None:
                for j, (_, source_name, col_name) in enumerate(dense_specs):
                    source_row = sources.get(source_name, {})
                    raw_dense[i, j] = _safe_float(source_row.get(col_name), 0.0)

        if raw_dense is not None:
            if self.bucketize_dense:
                if split == "train":
                    dense_as_sparse, boundaries = self._bucketize_train(raw_dense)
                    KuaiRecPointwiseDataset._shared_bucket_boundaries = boundaries
                    print(
                        f"[KuaiRecPointwise] Bucketized {num_dense} dense cols -> sparse "
                        f"({self.num_buckets} buckets, method={self.bucket_method})"
                    )
                else:
                    boundaries = KuaiRecPointwiseDataset._shared_bucket_boundaries
                    if boundaries is None:
                        raise RuntimeError(
                            "[KuaiRecPointwise] Test split loaded before train split bucket boundaries. "
                            "Must load train split first when bucketize_dense is enabled."
                        )
                    dense_as_sparse = self._bucketize_test(raw_dense, boundaries)
                    print(f"[KuaiRecPointwise] Bucketized {num_dense} dense cols (split={split})")
            else:
                self.dense = np.log1p(np.abs(raw_dense)) * np.sign(raw_dense)

        if dense_as_sparse is not None:
            self.sparse = np.concatenate([self.sparse, dense_as_sparse], axis=1)
            bucket_sparse_cols = [f"{name}_bucket" for name in base_dense_cols]
            bucket_card = self.num_buckets + 2
            dc["_original_sparse_cols"] = list(base_sparse_cols)
            dc["_original_dense_cols"] = list(base_dense_cols)
            dc["sparse_cols"] = base_sparse_cols + bucket_sparse_cols
            dc["dense_cols"] = []
            dc["cardinalities"] = base_cardinalities + [bucket_card] * len(bucket_sparse_cols)
            self.dense = None

        validate_cardinalities(config, self.sparse, dc["sparse_cols"])

        final_num_sparse = self.sparse.shape[1] if self.sparse.ndim == 2 else num_sparse
        final_num_dense = 0 if self.dense is None else num_dense
        pos = int((self.labels > 0.5).sum())
        print(f"[KuaiRecPointwise] {split}: {len(filtered):,} samples, "
              f"sparse={final_num_sparse}, dense={final_num_dense}, "
              f"pos={pos:,} ({pos / max(len(filtered), 1) * 100:.2f}%)")

    def _build_feature_specs(self, data_path: str, dc: dict, uid2idx: dict, vid2idx: dict):
        user_feature_rows, item_feature_rows, item_category_rows = self._load_side_features(data_path, dc)

        user_sparse_cols, user_dense_cols = _infer_feature_groups(
            user_feature_rows,
            key_col="user_id",
            explicit_sparse=dc.get("user_sparse_cols"),
            explicit_dense=dc.get("user_dense_cols"),
        )
        item_sparse_cols, item_dense_cols = _infer_feature_groups(
            item_feature_rows,
            key_col="video_id",
            explicit_sparse=dc.get("item_sparse_cols"),
            explicit_dense=dc.get("item_dense_cols"),
        )
        item_cat_sparse_cols, item_cat_dense_cols = _infer_feature_groups(
            item_category_rows,
            key_col="video_id",
            explicit_sparse=dc.get("item_category_sparse_cols"),
            explicit_dense=dc.get("item_category_dense_cols"),
        )

        sparse_specs = []
        if dc.get("use_interaction_ids", True):
            if dc.get("use_user_id", True):
                sparse_specs.append(
                    ("user_id", "interaction", "user_id", len(uid2idx) + 1, {}),
                )
            sparse_specs.append(
                ("video_id", "interaction", "video_id", len(vid2idx) + 1, {}),
            )
        dense_specs = []

        for prefix, source_name, keyed_rows, cols in [
            ("user", "user", user_feature_rows, user_sparse_cols),
            ("item", "item", item_feature_rows, item_sparse_cols),
            ("item_category", "item_category", item_category_rows, item_cat_sparse_cols),
        ]:
            for col in cols:
                values = sorted({str(row.get(col, "")) for row in keyed_rows.values() if row.get(col, "") not in ("", None)})
                mapping = {v: i + 1 for i, v in enumerate(values)}
                sparse_specs.append((f"{prefix}__{col}", source_name, col, len(mapping) + 1, mapping))

        for prefix, source_name, cols in [
            ("user", "user", user_dense_cols),
            ("item", "item", item_dense_cols),
            ("item_category", "item_category", item_cat_dense_cols),
        ]:
            for col in cols:
                dense_specs.append((f"{prefix}__{col}", source_name, col))

        print(
            "[KuaiRecPointwise] Feature spec: "
            f"user_sparse={len(user_sparse_cols)}, user_dense={len(user_dense_cols)}, "
            f"item_sparse={len(item_sparse_cols)}, item_dense={len(item_dense_cols)}, "
            f"item_cat_sparse={len(item_cat_sparse_cols)}, item_cat_dense={len(item_cat_dense_cols)}"
        )
        return sparse_specs, dense_specs

    def _bucketize_train(self, raw_dense: np.ndarray):
        n_cols = raw_dense.shape[1]
        boundaries = {}
        result = np.zeros_like(raw_dense, dtype=np.int64)

        for i in range(n_cols):
            col = raw_dense[:, i]

            if self.bucket_method == "log":
                transformed = np.log1p(np.abs(col)) * np.sign(col)
                vmin, vmax = transformed.min(), transformed.max()
                if vmax - vmin < 1e-8:
                    edges = np.array([vmin])
                else:
                    edges = np.linspace(vmin, vmax, self.num_buckets + 1)[1:-1]
                indices = np.searchsorted(edges, transformed) + 1
            elif self.bucket_method == "quantile":
                percentiles = np.linspace(0, 100, self.num_buckets + 1)[1:-1]
                edges = np.percentile(col, percentiles)
                edges = np.unique(edges)
                indices = np.searchsorted(edges, col) + 1
            elif self.bucket_method == "uniform":
                vmin, vmax = col.min(), col.max()
                if vmax - vmin < 1e-8:
                    edges = np.array([vmin])
                else:
                    edges = np.linspace(vmin, vmax, self.num_buckets + 1)[1:-1]
                indices = np.searchsorted(edges, col) + 1
            else:
                raise ValueError(f"Unknown bucket method: {self.bucket_method}")

            indices = np.clip(indices, 1, self.num_buckets)
            result[:, i] = indices
            boundaries[i] = {"method": self.bucket_method, "edges": edges.tolist()}

        return result, boundaries

    def _bucketize_test(self, raw_dense: np.ndarray, boundaries: dict):
        n_cols = raw_dense.shape[1]
        result = np.zeros_like(raw_dense, dtype=np.int64)

        for i in range(n_cols):
            col = raw_dense[:, i]
            info = boundaries.get(i, {})
            edges = np.array(info.get("edges", []))
            method = info.get("method", "log")

            if method == "log":
                transformed = np.log1p(np.abs(col)) * np.sign(col)
                indices = np.searchsorted(edges, transformed) + 1
            else:
                indices = np.searchsorted(edges, col) + 1

            indices = np.clip(indices, 1, self.num_buckets)
            result[:, i] = indices

        return result

    def _load_side_features(self, data_path: str, dc: dict):
        user_feature_rows = {}
        item_feature_rows = {}
        item_category_rows = {}

        if dc.get("use_user_features", True):
            user_file = os.path.join(data_path, dc.get("user_feature_file", "user_features.csv"))
            user_feature_rows = _load_keyed_rows(user_file, "user_id")
            if user_feature_rows:
                print(f"[KuaiRecPointwise] Loaded user features: {len(user_feature_rows):,} rows")

        if dc.get("use_item_features", True):
            item_file = os.path.join(data_path, dc.get("item_feature_file", "item_daily_features.csv"))
            item_feature_rows = _load_keyed_rows(item_file, "video_id")
            if item_feature_rows:
                print(f"[KuaiRecPointwise] Loaded item features: {len(item_feature_rows):,} rows")

        if dc.get("use_item_category_features", True):
            item_cat_file = os.path.join(data_path, dc.get("item_category_file", "item_categories.csv"))
            item_category_rows = _load_keyed_rows(item_cat_file, "video_id")
            if item_category_rows:
                print(f"[KuaiRecPointwise] Loaded item category features: {len(item_category_rows):,} rows")

        return user_feature_rows, item_feature_rows, item_category_rows

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            "sparse": self.sparse[idx],
            "label": self.labels[idx],
        }
        if self.dense is not None:
            sample["dense"] = self.dense[idx]
        return sample
