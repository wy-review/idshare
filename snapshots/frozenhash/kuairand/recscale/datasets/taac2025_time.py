"""
recscale.datasets.taac2025_time — TAAC2025 按时间切分 (10m-time-split)

train/ 和 test/ 各有 samples.parquet，共享 user_seqs/ 目录。
每条样本通过 seq_cutoff_pos 动态滑窗截断序列。

No-seq 特征：
  - user_id + target_item_id (基础 2 字段)
  - 若数据目录包含 user_feat/ 和 item_feat/ parquet 目录，自动加载 side features：
      user sparse features: feat IDs 103, 104, 105, 109
      item sparse features: feat IDs 100, 117, 118, 101, 102, 119, 120, 114, 112, 121, 115, 122, 116
    这些特征已经是整数索引，vocab size 从 indexer['f'][feat_id] 读取。
  - 文件不存在时静默退化为 [user_id, target_id] 2 字段（向后兼容）

YAML 示例:
```yaml
dataset:
  type: taac2025_time
  path: /data/taac2025/10m-time-split
  maxlen: 99
```
"""

import os
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pds

from . import register_dataset
from .base import BaseDataset, validate_cardinalities


def _load_taac_feat_parquet(parquet_dir: str, id_col: str, feat_ids: list) -> dict:
    """
    从 parquet 目录加载 TAAC side features，返回 {int_id: {feat_id: int_value}}。

    - sparse features (103/104/105/109 / item sparse): 值是单个整数，直接 int()
    - array features (106/107/108/110): 值是 list of ints，取第一个非零元素作为代表值

    特征值已经是索引化整数（来自 indexer.pkl），不需要 vocab mapping。
    目录不存在时静默返回空 dict。
    """
    if not os.path.exists(parquet_dir):
        return {}
    try:
        cols = [id_col] + feat_ids
        dataset = pds.dataset(parquet_dir, format="parquet")
        table = dataset.to_table(columns=cols)
        data = table.to_pydict()
        ids = data[id_col]
        out = {}
        for i in range(len(ids)):
            raw_id = ids[i]
            if raw_id is None:
                continue
            key = int(raw_id)
            feat = {}
            for fid in feat_ids:
                v = data[fid][i]
                if v is None:
                    continue
                # array feature: list → 取第一个非零值
                if isinstance(v, (list, tuple)):
                    chosen = 0
                    for elem in v:
                        if elem is not None and int(elem) != 0:
                            chosen = int(elem)
                            break
                    feat[fid] = chosen
                else:
                    feat[fid] = int(v)
            out[key] = feat
        return out
    except Exception as e:
        print(f"[TAAC2025-Time] Warning: failed to load {parquet_dir}: {e}")
        return {}


@register_dataset("taac2025_time")
class TAAC2025TimeDataset(BaseDataset):

    # 类级别共享
    _shared_user_seqs = None
    _shared_needed_uids = None
    _all_seqs_cache = None
    # B9/B11 fix: cache key 标记 (data_path, mode)。
    # 同进程切换 use_action_types 或 data_path 时必须 reset sequence cache,
    # 否则旧路径写入的序列(可能是 np.ndarray-only 或来自不同 TAAC 路径)
    # 会让新 dataset 拿到错误的数据,seq_action 静默归零或 uid 串数据集。
    _shared_seq_cache_key = None  # tuple (data_path, "item_only"|"with_action")
    # side feature 共享（train 建立，test 复用）
    _shared_user_feat = None   # {int uid: {feat_id: int_val}}
    _shared_item_feat = None   # {int iid: {feat_id: int_val}}
    _shared_sparse_cols = None  # list of col names
    _shared_cardinalities = None  # list of cardinalities

    # TAAC 固定特征 ID（来自 ctr_dataset.py）
    _ITEM_SPARSE_IDS = [
        '100', '117', '118', '101', '102', '119', '120',
        '114', '112', '121', '115', '122', '116',
    ]
    _USER_SPARSE_IDS = ['103', '104', '105', '109']
    # array features: 实际存储为 list of ints
    # 第一版处理：取第一个非零元素作为单一 sparse field
    _USER_ARRAY_IDS = ['106', '107', '108', '110']

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 99)
        max_rows = dc.get("max_rows", 0)
        # If True: load action_type from raw parquet (skip pkl fast path).
        # action_type ∈ {0=exposure, 1=click, 2=conversion} from TAAC schema;
        # we offset to {1,2,3} so 0 can be padding (matches KuaiRand convention).
        self.use_action_types = bool(dc.get("use_action_types", False))

        # 加载 indexer
        indexer_path = os.path.join(data_path, "indexer.pkl")
        with open(indexer_path, "rb") as f:
            indexer = pickle.load(f)
        num_users = len(indexer["u"]) + 1
        num_items = len(indexer["i"]) + 1
        dc["num_items"] = num_items
        dc["num_users"] = num_users

        # ----------------------------------------------------------------
        # side features (train 时加载并建立元信息，test 时复用)
        # ----------------------------------------------------------------
        if split == "train":
            user_feat_dir = os.path.join(data_path, "user_feat")
            item_feat_dir = os.path.join(data_path, "item_feat")

            if dc.get("use_user_features", True):
                user_feat = _load_taac_feat_parquet(
                    user_feat_dir, "user_id",
                    self._USER_SPARSE_IDS + self._USER_ARRAY_IDS)  # 包含 array features
            else:
                user_feat = {}

            if dc.get("use_item_features", True):
                item_feat = _load_taac_feat_parquet(
                    item_feat_dir, "item_id", self._ITEM_SPARSE_IDS)
            else:
                item_feat = {}

            if user_feat:
                all_user_ids = self._USER_SPARSE_IDS + self._USER_ARRAY_IDS
                print(f"[TAAC2025-Time] Loaded user side features: "
                      f"{len(user_feat):,} users, fields={all_user_ids}")
            if item_feat:
                print(f"[TAAC2025-Time] Loaded item side features: "
                      f"{len(item_feat):,} items, fields={self._ITEM_SPARSE_IDS}")

            # 构建 sparse_cols 和 cardinalities
            # TAAC 特征已经是索引化整数，vocab size 从 indexer['f'] 读取
            sparse_cols = ["user_id", "target_item_id"]
            cardinalities = [num_users, num_items]

            feat_vocab = indexer.get("f", {})
            if user_feat:
                for fid in self._USER_SPARSE_IDS + self._USER_ARRAY_IDS:
                    raw_vocab = feat_vocab.get(fid)
                    if raw_vocab is None:
                        continue
                    vsize = len(raw_vocab) if not isinstance(raw_vocab, int) else int(raw_vocab)
                    if vsize > 0:
                        sparse_cols.append(f"user_feat_{fid}")
                        cardinalities.append(vsize + 1)  # +1 for missing/0

            if item_feat:
                for fid in self._ITEM_SPARSE_IDS:
                    raw_vocab = feat_vocab.get(fid)
                    if raw_vocab is None:
                        continue
                    vsize = len(raw_vocab) if not isinstance(raw_vocab, int) else int(raw_vocab)
                    if vsize > 0:
                        sparse_cols.append(f"item_feat_{fid}")
                        cardinalities.append(vsize + 1)

            # 保存到类级别共享变量
            TAAC2025TimeDataset._shared_user_feat = user_feat
            TAAC2025TimeDataset._shared_item_feat = item_feat
            TAAC2025TimeDataset._shared_sparse_cols = sparse_cols
            TAAC2025TimeDataset._shared_cardinalities = cardinalities

        else:
            # test split: 复用 train 建立的 specs
            user_feat = TAAC2025TimeDataset._shared_user_feat or {}
            item_feat = TAAC2025TimeDataset._shared_item_feat or {}
            sparse_cols = TAAC2025TimeDataset._shared_sparse_cols or ["user_id", "target_item_id"]
            cardinalities = TAAC2025TimeDataset._shared_cardinalities or [num_users, num_items]

        dc["sparse_cols"] = sparse_cols
        dc["cardinalities"] = cardinalities
        self._num_sparse = len(sparse_cols)
        self._has_side_features = len(sparse_cols) > 2
        # 全部 user features（sparse + array）
        self._user_all_ids = (self._USER_SPARSE_IDS + self._USER_ARRAY_IDS) if self._has_side_features else []
        self._item_sparse_ids = self._ITEM_SPARSE_IDS if self._has_side_features else []
        self._user_feat = user_feat
        self._item_feat = item_feat

        # ----------------------------------------------------------------
        # 加载 samples
        # ----------------------------------------------------------------
        samples_path = os.path.join(data_path, split, "samples.parquet")
        print(f"[TAAC2025-Time] Loading {samples_path} ...")
        if max_rows > 0:
            pf = pq.ParquetFile(samples_path)
            chunks = []
            rows_left = max_rows
            for batch in pf.iter_batches(batch_size=min(262144, rows_left)):
                chunks.append(pa.Table.from_batches([batch]))
                rows_left -= batch.num_rows
                if rows_left <= 0:
                    break
            tbl = pa.concat_tables(chunks) if chunks else pf.read().slice(0, 0)
        else:
            tbl = pq.read_table(samples_path)
        data = tbl.to_pydict()

        self.sample_user_ids    = np.array(data["user_id"],        dtype=np.int64)
        self.sample_target_items = np.array(data["target_item_id"], dtype=np.int64)
        self.sample_labels      = np.array(data["label"],          dtype=np.int32)
        self.sample_cutoff_pos  = np.array(data["seq_cutoff_pos"], dtype=np.int32)

        if 0 < max_rows < len(self.sample_user_ids):
            self.sample_user_ids     = self.sample_user_ids[:max_rows]
            self.sample_target_items = self.sample_target_items[:max_rows]
            self.sample_labels       = self.sample_labels[:max_rows]
            self.sample_cutoff_pos   = self.sample_cutoff_pos[:max_rows]

        # ----------------------------------------------------------------
        # 预计算 sparse 矩阵（含 side features）
        # ----------------------------------------------------------------
        N = len(self.sample_user_ids)
        self.sparse = np.zeros((N, self._num_sparse), dtype=np.int64)
        # col 0: user_id, col 1: target_item_id (直接用整数 ID 作为索引)
        self.sparse[:, 0] = self.sample_user_ids
        self.sparse[:, 1] = self.sample_target_items

        if self._has_side_features:
            # 向量化填充 user features（sparse + array）
            col_offset = 2
            for fid in self._user_all_ids:
                col_name = f"user_feat_{fid}"
                if col_name not in sparse_cols:
                    continue
                if self._user_feat:
                    vals = np.array(
                        [self._user_feat.get(int(uid), {}).get(fid, 0)
                         for uid in self.sample_user_ids],
                        dtype=np.int64)
                    self.sparse[:, col_offset] = vals
                col_offset += 1
            # 向量化填充 item features
            for fid in self._item_sparse_ids:
                col_name = f"item_feat_{fid}"
                if col_name not in sparse_cols:
                    continue
                if self._item_feat:
                    vals = np.array(
                        [self._item_feat.get(int(iid), {}).get(fid, 0)
                         for iid in self.sample_target_items],
                        dtype=np.int64)
                    self.sparse[:, col_offset] = vals
                col_offset += 1

        validate_cardinalities(config, self.sparse, sparse_cols)

        # ----------------------------------------------------------------
        # 加载 user_seqs
        # ----------------------------------------------------------------
        needed_uids = set(self.sample_user_ids.tolist())
        print(f"[TAAC2025-Time] {split}: need {len(needed_uids):,} unique users")

        # B9/B11 fix: 同进程切换 use_action_types 或 data_path 时 reset sequence cache。
        # cache_key=(data_path, mode) 比单纯 mode 更稳:也覆盖 notebook / smoke test 里
        # 切换不同 TAAC 路径但保持 use_action_types 不变的场景。
        cache_mode = "with_action" if self.use_action_types else "item_only"
        cache_key = (data_path, cache_mode)
        if (TAAC2025TimeDataset._shared_seq_cache_key is not None
                and TAAC2025TimeDataset._shared_seq_cache_key != cache_key):
            print(f"[TAAC2025-Time] cache key change detected "
                  f"({TAAC2025TimeDataset._shared_seq_cache_key} → {cache_key}); "
                  f"clearing sequence cache to avoid silent degradation.")
            TAAC2025TimeDataset._shared_user_seqs = None
            TAAC2025TimeDataset._shared_needed_uids = None
            TAAC2025TimeDataset._all_seqs_cache = None
        TAAC2025TimeDataset._shared_seq_cache_key = cache_key

        if TAAC2025TimeDataset._shared_user_seqs is None:
            TAAC2025TimeDataset._shared_user_seqs = {}
            TAAC2025TimeDataset._shared_needed_uids = set()

        new_uids = needed_uids - TAAC2025TimeDataset._shared_needed_uids
        loaded = 0
        if new_uids:
            pkl_path = os.path.join(data_path, "user_seqs_numpy", "user_seqs.pkl")
            if (not self.use_action_types) and os.path.exists(pkl_path) and not TAAC2025TimeDataset._shared_user_seqs:
                import time as _t
                _t0 = _t.time()
                print(f"[TAAC2025-Time] Loading preprocessed sequences from {pkl_path} ...")
                with open(pkl_path, "rb") as f:
                    all_seqs = pickle.load(f)
                for uid in new_uids:
                    if uid in all_seqs:
                        TAAC2025TimeDataset._shared_user_seqs[uid] = all_seqs[uid]
                TAAC2025TimeDataset._all_seqs_cache = all_seqs
                loaded = sum(1 for uid in new_uids if uid in all_seqs)
                print(f"[TAAC2025-Time] Loaded {loaded:,} user sequences in {_t.time()-_t0:.1f}s")
            elif (not self.use_action_types) and TAAC2025TimeDataset._all_seqs_cache:
                all_seqs = TAAC2025TimeDataset._all_seqs_cache
                for uid in new_uids:
                    if uid in all_seqs:
                        TAAC2025TimeDataset._shared_user_seqs[uid] = all_seqs[uid]
                        loaded += 1
                print(f"[TAAC2025-Time] Loaded {loaded:,} from cache")
            else:
                seqs_path = os.path.join(data_path, "user_seqs")
                if self.use_action_types:
                    # B12 fix: 不能把 9.3M user_seqs 以 dict-form 存内存
                    # (Python dict overhead ~200B/item × 9.3M × ~50 = ~90GB → OOM)。
                    # 改为 compact numpy tuple (item_ids, action_types):
                    #   item_ids:    np.int64 array
                    #   action_types: np.int8 array (offset +1, 0=padding, 1=exp, 2=click, 3=conv)
                    # 内存量级:9.3M × 50 × 9 bytes ≈ 4.2 GB(可接受)。
                    print(f"[TAAC2025-Time] Loading raw parquet (with action_type) {seqs_path} ...")
                    import time as _t
                    _t0 = _t.time()
                    seq_dataset = pds.dataset(seqs_path, format="parquet")
                    seq_table = seq_dataset.to_table(columns=["user_id", "seq"])
                    seq_data = seq_table.to_pydict()
                    for uid, seq in zip(seq_data["user_id"], seq_data["seq"]):
                        if uid not in new_uids:
                            continue
                        if not seq:
                            TAAC2025TimeDataset._shared_user_seqs[uid] = (
                                np.zeros(0, dtype=np.int64),
                                np.zeros(0, dtype=np.int8),
                            )
                            loaded += 1
                            continue
                        # seq is list of dicts: [{"item_id": int, "action_type": int, "timestamp": int}, ...]
                        n = len(seq)
                        items_arr = np.empty(n, dtype=np.int64)
                        actions_arr = np.empty(n, dtype=np.int8)
                        for k, it in enumerate(seq):
                            items_arr[k] = int(it.get("item_id", 0))
                            raw_at = it.get("action_type", 0)
                            raw_at = 0 if raw_at is None else int(raw_at)
                            if raw_at < 0 or raw_at > 2:
                                raise ValueError(
                                    f"Unexpected TAAC action_type={raw_at} at uid={uid}, "
                                    f"expected 0(exp)/1(click)/2(conv)."
                                )
                            actions_arr[k] = raw_at + 1  # offset so 0=padding
                        TAAC2025TimeDataset._shared_user_seqs[uid] = (items_arr, actions_arr)
                        loaded += 1
                    print(f"[TAAC2025-Time] Loaded {loaded:,} user sequences (with action_type) "
                          f"as compact numpy in {_t.time()-_t0:.1f}s")
                else:
                    print(f"[TAAC2025-Time] Loading from parquet {seqs_path} (slow) ...")
                    seq_dataset = pds.dataset(seqs_path, format="parquet")
                    seq_table = seq_dataset.to_table(columns=["user_id", "seq"])
                    seq_data = seq_table.to_pydict()
                    for uid, seq in zip(seq_data["user_id"], seq_data["seq"]):
                        if uid in new_uids:
                            TAAC2025TimeDataset._shared_user_seqs[uid] = seq if seq else []
                            loaded += 1
                    print(f"[TAAC2025-Time] Loaded {loaded:,} user sequences from parquet")
            TAAC2025TimeDataset._shared_needed_uids.update(new_uids)

        self.user_seqs = TAAC2025TimeDataset._shared_user_seqs

        # 过滤没有序列的样本
        valid = np.array([uid in self.user_seqs for uid in self.sample_user_ids])
        if not valid.all():
            n_before = len(self.sample_user_ids)
            self.sample_user_ids     = self.sample_user_ids[valid]
            self.sample_target_items = self.sample_target_items[valid]
            self.sample_labels       = self.sample_labels[valid]
            self.sample_cutoff_pos   = self.sample_cutoff_pos[valid]
            self.sparse              = self.sparse[valid]
            print(f"[TAAC2025-Time] Filtered {n_before - len(self.sample_user_ids)} "
                  f"samples without seq")

        n_pos   = (self.sample_labels == 1).sum()
        n_total = len(self.sample_user_ids)
        print(f"[TAAC2025-Time] {split}: {n_total:,} samples, "
              f"sparse={self.sparse.shape[1]}, "
              f"pos={n_pos:,} ({n_pos / max(n_total, 1) * 100:.2f}%)")

        # ---- Type A/B 探索性特征 -------------------------------------------
        use_seq_derived = dc.get("use_seq_derived_features", False)
        use_item_global = dc.get("use_item_global_stats", False)
        seq_derived_only = dc.get("seq_derived_features_only", None)

        if use_seq_derived or use_item_global:
            from .side_features import compute_bucket_edges, apply_bucket
            extra_cols = []
            extra_names_new = []
            extra_cards_new = []
            n_bkt = dc.get("num_buckets", 100)

            if use_seq_derived:
                if seq_derived_only:
                    if isinstance(seq_derived_only, str):
                        selected_feats = set(seq_derived_only.split(","))
                    else:
                        selected_feats = set(seq_derived_only)
                else:
                    selected_feats = {"seq_length", "seq_unique_items", "target_in_history", "target_history_count"}

                # 预计算每个 sample 的序列统计量（向量化加速版）
                # 先批量构建所有样本的序列矩阵
                print(f"[TAAC2025-Time] Pre-computing Type A features for {n_total:,} samples...")
                import time as _time
                _t0 = _time.time()

                seq_lengths = np.zeros(n_total, dtype=np.float32)
                seq_unique_arr = np.zeros(n_total, dtype=np.float32)
                target_in_hist = np.zeros(n_total, dtype=np.float32)
                target_hist_count_arr = np.zeros(n_total, dtype=np.float32)

                # 按 user 分组处理（同一用户共享 full_seq，避免重复查找）
                from collections import defaultdict
                user_sample_indices = defaultdict(list)
                for i in range(n_total):
                    user_sample_indices[int(self.sample_user_ids[i])].append(i)

                for uid, indices in user_sample_indices.items():
                    full_seq = self.user_seqs.get(uid, [])
                    if isinstance(full_seq, np.ndarray):
                        seq_arr = full_seq
                    else:
                        seq_arr = np.array(
                            [int(x) if not isinstance(x, dict) else x.get("item_id", 0) for x in full_seq],
                            dtype=np.int64) if full_seq else np.array([], dtype=np.int64)

                    for i in indices:
                        cutoff = int(self.sample_cutoff_pos[i])
                        target = int(self.sample_target_items[i])
                        windowed = seq_arr[:cutoff][-self.maxlen:]
                        valid = windowed[windowed != 0]
                        seq_lengths[i] = len(valid)
                        seq_unique_arr[i] = len(np.unique(valid)) if len(valid) > 0 else 0
                        target_in_hist[i] = 1.0 if np.any(valid == target) else 0.0
                        target_hist_count_arr[i] = np.sum(valid == target)

                print(f"[TAAC2025-Time] Type A pre-compute done in {_time.time()-_t0:.1f}s")

                if split == "train":
                    edges_len = compute_bucket_edges(seq_lengths, n_bkt)
                    edges_uniq = compute_bucket_edges(seq_unique_arr, n_bkt)
                    edges_cnt = compute_bucket_edges(target_hist_count_arr, n_bkt)
                    TAAC2025TimeDataset._shared_typeA_edges = (edges_len, edges_uniq, edges_cnt)
                else:
                    edges_len, edges_uniq, edges_cnt = TAAC2025TimeDataset._shared_typeA_edges

                if "seq_length" in selected_feats:
                    extra_cols.append(apply_bucket(seq_lengths, edges_len, n_bkt))
                    extra_names_new.append("typeA__seq_length")
                    extra_cards_new.append(n_bkt + 1)
                if "seq_unique_items" in selected_feats:
                    extra_cols.append(apply_bucket(seq_unique_arr, edges_uniq, n_bkt))
                    extra_names_new.append("typeA__seq_unique_items")
                    extra_cards_new.append(n_bkt + 1)
                if "target_in_history" in selected_feats:
                    extra_cols.append((target_in_hist + 1).astype(np.int64))
                    extra_names_new.append("typeA__target_in_history")
                    extra_cards_new.append(3)
                if "target_history_count" in selected_feats:
                    extra_cols.append(apply_bucket(target_hist_count_arr, edges_cnt, n_bkt))
                    extra_names_new.append("typeA__target_history_count")
                    extra_cards_new.append(n_bkt + 1)

            if use_item_global:
                if split == "train":
                    # item 全局统计
                    num_items_total = dc["num_items"]
                    item_exp = np.zeros(num_items_total, dtype=np.float32)
                    item_pos = np.zeros(num_items_total, dtype=np.float32)
                    item_users_set = [set() for _ in range(num_items_total)]
                    for i in range(n_total):
                        tid = int(self.sample_target_items[i])
                        uid = int(self.sample_user_ids[i])
                        if tid < num_items_total:
                            item_exp[tid] += 1
                            if self.sample_labels[i] > 0:
                                item_pos[tid] += 1
                            item_users_set[tid].add(uid)
                    item_ctr = np.where(item_exp > 0, item_pos / item_exp, 0.0)
                    item_uusers = np.array([len(s) for s in item_users_set], dtype=np.float32)
                    edges_ctr = compute_bucket_edges(item_ctr[item_ctr > 0], n_bkt)
                    edges_exp = compute_bucket_edges(item_exp[item_exp > 0], n_bkt)
                    edges_uu = compute_bucket_edges(item_uusers[item_uusers > 0], n_bkt)
                    TAAC2025TimeDataset._shared_typeB_edges = (edges_ctr, edges_exp, edges_uu)
                    TAAC2025TimeDataset._shared_item_ctr = item_ctr
                    TAAC2025TimeDataset._shared_item_exp = item_exp
                    TAAC2025TimeDataset._shared_item_uusers = item_uusers
                else:
                    edges_ctr, edges_exp, edges_uu = TAAC2025TimeDataset._shared_typeB_edges
                    item_ctr = TAAC2025TimeDataset._shared_item_ctr
                    item_exp = TAAC2025TimeDataset._shared_item_exp
                    item_uusers = TAAC2025TimeDataset._shared_item_uusers

                t_ids = self.sample_target_items.clip(0, len(item_ctr) - 1)
                extra_cols.append(apply_bucket(item_ctr[t_ids], edges_ctr, n_bkt))
                extra_names_new.append("typeB__item_global_ctr")
                extra_cards_new.append(n_bkt + 1)
                extra_cols.append(apply_bucket(item_exp[t_ids], edges_exp, n_bkt))
                extra_names_new.append("typeB__item_exposure_count")
                extra_cards_new.append(n_bkt + 1)
                extra_cols.append(apply_bucket(item_uusers[t_ids], edges_uu, n_bkt))
                extra_names_new.append("typeB__item_unique_users")
                extra_cards_new.append(n_bkt + 1)

            if extra_cols:
                extra_matrix = np.stack(extra_cols, axis=1)
                self.sparse = np.concatenate([self.sparse, extra_matrix], axis=1)
                # 更新 config metadata（每次都追加，因为 _update_dc_feature_meta 或 上面的初始化每次覆写）
                dc["sparse_cols"] = dc["sparse_cols"] + extra_names_new
                dc["cardinalities"] = dc["cardinalities"] + extra_cards_new
                print(f"[TAAC2025-Time] Added {len(extra_names_new)} Type A/B features: {extra_names_new}")

    def __len__(self):
        return len(self.sample_user_ids)

    def __getitem__(self, idx):
        uid    = int(self.sample_user_ids[idx])
        target = int(self.sample_target_items[idx])
        label  = float(self.sample_labels[idx])
        cutoff = int(self.sample_cutoff_pos[idx])

        full_seq = self.user_seqs.get(uid, [])
        seq_action = None  # Only populated when use_action_types=True
        # B12 fix: when use_action_types=True, full_seq is tuple (items_np, actions_np)
        if isinstance(full_seq, tuple) and len(full_seq) == 2:
            items_arr, actions_arr = full_seq
            windowed_items = items_arr[:cutoff][-self.maxlen:]
            windowed_actions = actions_arr[:cutoff][-self.maxlen:]
            seq = np.zeros(self.maxlen, dtype=np.int64)
            seq_action = np.zeros(self.maxlen, dtype=np.int64)
            if len(windowed_items) > 0:
                start = self.maxlen - len(windowed_items)
                seq[start:] = windowed_items
                seq_action[start:] = windowed_actions
        elif isinstance(full_seq, np.ndarray):
            windowed = full_seq[:cutoff][-self.maxlen:]
            seq = np.zeros(self.maxlen, dtype=np.int64)
            if len(windowed) > 0:
                start = self.maxlen - len(windowed)
                seq[start:] = windowed
            if self.use_action_types:
                # Edge case: cache 是 PKL 写的 np.ndarray(只有 item_id)。
                # B9 fix 已在 __init__ 时 reset cache 避免此路径,这里 zeros 仅作兜底。
                seq_action = np.zeros(self.maxlen, dtype=np.int64)
        else:
            windowed = full_seq[:cutoff][-self.maxlen:]
            seq = np.zeros(self.maxlen, dtype=np.int64)
            if self.use_action_types:
                seq_action = np.zeros(self.maxlen, dtype=np.int64)
            if windowed:
                start = self.maxlen - len(windowed)
                for k, item in enumerate(windowed):
                    if isinstance(item, dict):
                        seq[start + k] = item.get("item_id", 0)
                        if self.use_action_types:
                            # Raw action_type ∈ {0=exp, 1=click, 2=conv}; offset +1 so 0=padding.
                            # B10 fix: fail-fast 校验,避免 None / 异常值导致
                            # 训练中 Embedding(num_seq_action_types) index OOB(CUDA 上难定位)。
                            raw_at = item.get("action_type", 0)
                            raw_at = 0 if raw_at is None else int(raw_at)
                            if raw_at < 0 or raw_at > 2:
                                raise ValueError(
                                    f"Unexpected TAAC action_type={raw_at} at uid={uid}, "
                                    f"expected 0(exp)/1(click)/2(conv)."
                                )
                            seq_action[start + k] = raw_at + 1
                    else:
                        seq[start + k] = int(item)

        out = {
            "sparse": self.sparse[idx],
            "seq":    seq,
            "target": np.int64(target),
            "label":  np.float32(label),
        }
        if self.use_action_types and seq_action is not None:
            out["seq_action"] = seq_action
        return out
