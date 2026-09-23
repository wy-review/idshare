"""
recscale.datasets.side_features — 通用 side feature 加载工具

提供：
  - load_keyed_rows(path, key_col)           从 CSV 按主键构建查找表
  - load_keyed_rows_glob(pattern, key_col)   Glob 多文件合并
  - infer_feature_groups(rows, key_col, ...) 自动推断 sparse / dense 特征列
  - SideFeatureMixin                         供 sequential dataset 类继承

设计原则：
  - 数值型特征（dense_cols）一律通过 log 分桶转为类别型 sparse，不传入 batch["dense"]
  - 最终所有 non-seq 特征都以 sparse index 形式输入模型
  - 文件不存在时静默退化为 [user_id, video_id]（向后兼容）

调用顺序（sequential dataset __init__ 里）：
  1. user_rows, item_rows, cat_rows = self._load_side_features(data_path, dc)
  2. sparse_specs, dense_specs = self._build_feature_specs(
         uid2idx, vid2idx, user_rows, item_rows, cat_rows, dc)
  3. 若 split == "train":
       self.sparse, boundaries = self._precompute_sparse_with_buckets(
           raw_uids, raw_vids, uid2idx, vid2idx,
           user_rows, item_rows, cat_rows, sparse_specs, dense_specs)
       # 保存 boundaries 到类级别共享变量
     else:
       self.sparse = self._precompute_sparse_with_buckets(
           raw_uids, raw_vids, uid2idx, vid2idx,
           user_rows, item_rows, cat_rows, sparse_specs, dense_specs,
           bucket_boundaries=<shared_boundaries>)
  4. self._update_dc_feature_meta(dc, sparse_specs, dense_specs, dc.get("num_buckets", 100))
"""

import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 底层工具函数
# ---------------------------------------------------------------------------

def _sf_safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sf_safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_keyed_rows(csv_path: str, key_col: str) -> Dict[int, dict]:
    """从 CSV 文件按整数主键构建查找表。文件不存在时返回空 dict。"""
    if not os.path.exists(csv_path):
        return {}
    keyed: Dict[int, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _sf_safe_int(row.get(key_col), 0)
            if key > 0:
                keyed[key] = row
    return keyed


def load_keyed_rows_glob(pattern: str, key_col: str) -> Dict[int, dict]:
    """Glob 多个 CSV 文件并合并成一个按主键索引的查找表。"""
    merged: Dict[int, dict] = {}
    for path in sorted(glob.glob(pattern)):
        merged.update(load_keyed_rows(path, key_col))
    return merged


def infer_feature_groups(
    keyed_rows: Dict[int, dict],
    key_col: str,
    explicit_sparse: Optional[List[str]] = None,
    explicit_dense: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    自动将 CSV 列分类为 sparse（类别）或 dense（连续）特征。
    注：dense 列后续会通过分桶转为 sparse，不会以 float 形式传入模型。
    """
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

    sparse_cols: List[str] = []
    dense_cols: List[str] = []

    for col in all_cols:
        col_lower = col.lower()
        if any(tok in col_lower for tok in ignore_tokens):
            continue

        values = [
            row.get(col, "")
            for row in keyed_rows.values()
            if row.get(col, "") not in ("", None)
        ]
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
            all_int_like = all(
                abs(v - round(v)) < 1e-8
                for v in numeric_values[: min(512, len(numeric_values))]
            )
            if (
                all_int_like
                and unique_count <= 128
                and not any(tok in col_lower for tok in dense_hint_tokens)
            ):
                sparse_cols.append(col)
            else:
                # 数值型（含连续型整数）→ 走分桶路径
                dense_cols.append(col)
        else:
            if unique_count <= 256:
                sparse_cols.append(col)

    return sparse_cols, dense_cols


# ---------------------------------------------------------------------------
# 分桶工具
# ---------------------------------------------------------------------------

def compute_bucket_edges(values: np.ndarray, num_buckets: int = 100) -> np.ndarray:
    """
    对一列数值用 log1p 变换后计算 num_buckets 个等宽桶边界。
    返回 edges 数组（长度 = num_buckets - 1）。
    """
    transformed = np.log1p(np.abs(values)) * np.sign(values)
    vmin, vmax = transformed.min(), transformed.max()
    if vmax - vmin < 1e-8:
        return np.array([vmin])
    return np.linspace(vmin, vmax, num_buckets + 1)[1:-1]


def apply_bucket(values: np.ndarray, edges: np.ndarray, num_buckets: int = 100) -> np.ndarray:
    """将数值列按 edges 分桶，返回 1-indexed 桶号（shape 与 values 相同）。"""
    transformed = np.log1p(np.abs(values)) * np.sign(values)
    indices = np.searchsorted(edges, transformed) + 1   # 1-indexed
    return np.clip(indices, 1, num_buckets).astype(np.int64)


# ---------------------------------------------------------------------------
# SideFeatureMixin
# ---------------------------------------------------------------------------

class SideFeatureMixin:
    """
    为 sequential dataset 类提供完整的 no-seq side feature 加载能力。

    Config 选项（均可选，有合理默认值）：
      use_user_features:          True/False (default True)
      user_feature_file:          "user_features.csv"（单文件）
      user_feature_glob:          glob 模式，优先于 user_feature_file，可匹配多文件并 merge
      use_item_features:          True/False (default True)
      item_feature_file:          "item_daily_features.csv"（单文件）
      item_feature_glob:          glob 模式，优先于 item_feature_file，可匹配多文件并 merge
                                  （适用于 basic + statistic 两类文件需要合并的场景）
      item_feature_key:           item feature CSV/parquet 的主键列名，默认 "video_id"
                                  （若文件用 "item_id" 等其他列名时在此配置）
      use_item_category_features: True/False (default True)
      item_category_file:         "item_categories.csv"
      num_buckets:                100（数值型特征分桶数）
      # 显式列指定（不设置则自动推断）
      user_sparse_cols / user_dense_cols
      item_sparse_cols / item_dense_cols
      item_category_sparse_cols

    子类调用流程（在 __init__ 中）：
      user_rows, item_rows, cat_rows = self._load_side_features(data_path, dc)
      sparse_specs, dense_specs = self._build_feature_specs(
          uid2idx, vid2idx, user_rows, item_rows, cat_rows, dc)

      if split == "train":
          self.sparse, boundaries = self._precompute_sparse_with_buckets(
              raw_uids, raw_vids, uid2idx, vid2idx,
              user_rows, item_rows, cat_rows,
              sparse_specs, dense_specs, num_buckets=dc.get("num_buckets", 100))
          MyClass._shared_bucket_boundaries = boundaries
      else:
          self.sparse, _ = self._precompute_sparse_with_buckets(
              raw_uids, raw_vids, uid2idx, vid2idx,
              user_rows, item_rows, cat_rows,
              sparse_specs, dense_specs,
              num_buckets=dc.get("num_buckets", 100),
              bucket_boundaries=MyClass._shared_bucket_boundaries)

      self._update_dc_feature_meta(dc, sparse_specs, dense_specs, dc.get("num_buckets", 100))
    """

    # ------------------------------------------------------------------ load

    def _load_side_features(
        self, data_path: str, dc: dict
    ) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, dict]]:
        """加载 user / item / item_category 侧特征查找表。文件不存在时静默返回空 dict。"""
        tag = self.__class__.__name__.replace("Dataset", "")

        user_rows: Dict[int, dict] = {}
        item_rows: Dict[int, dict] = {}
        cat_rows: Dict[int, dict] = {}

        if dc.get("use_user_features", True):
            glob_pat = dc.get("user_feature_glob")
            if glob_pat:
                user_rows = load_keyed_rows_glob(os.path.join(data_path, glob_pat), "user_id")
            else:
                user_rows = load_keyed_rows(
                    os.path.join(data_path, dc.get("user_feature_file", "user_features.csv")),
                    "user_id")
            if user_rows:
                print(f"[{tag}] Loaded user side features: {len(user_rows):,} users")

        if dc.get("use_item_features", True):
            glob_pat  = dc.get("item_feature_glob")
            # item feature 的主键列名可以配置（默认 "video_id"）
            item_key  = dc.get("item_feature_key", "video_id")
            if glob_pat:
                item_rows = load_keyed_rows_glob(os.path.join(data_path, glob_pat), item_key)
            else:
                item_rows = load_keyed_rows(
                    os.path.join(data_path, dc.get("item_feature_file", "item_daily_features.csv")),
                    item_key)
            if item_rows:
                print(f"[{tag}] Loaded item side features: {len(item_rows):,} items "
                      f"(key={item_key})")

        if dc.get("use_item_category_features", True):
            item_key = dc.get("item_feature_key", "video_id")
            cat_rows = load_keyed_rows(
                os.path.join(data_path, dc.get("item_category_file", "item_categories.csv")),
                item_key)
            if cat_rows:
                print(f"[{tag}] Loaded item category features: {len(cat_rows):,} items")

        return user_rows, item_rows, cat_rows

    # ------------------------------------------------------------------ specs

    def _build_feature_specs(
        self,
        uid2idx: dict,
        vid2idx: dict,
        user_rows: Dict[int, dict],
        item_rows: Dict[int, dict],
        cat_rows: Dict[int, dict],
        dc: dict,
    ) -> Tuple[List[tuple], List[tuple]]:
        """
        构建特征 specs：
          sparse_specs: [(name, source, col, cardinality, mapping_dict), ...]
            前两项固定为 user_id / video_id
          dense_specs:  [(name, source, col), ...]
            数值型特征，后续通过分桶转 sparse

        Returns (sparse_specs, dense_specs)
        """
        sparse_specs = [
            ("user_id",  "interaction", "user_id",  len(uid2idx) + 1, {}),
            ("video_id", "interaction", "video_id", len(vid2idx) + 1, {}),
        ]
        dense_specs: List[tuple] = []

        # item feature 的主键列名可配置（默认 "video_id"，KuaiRand 等可能用 "item_id"）
        item_key = dc.get("item_feature_key", "video_id")

        for prefix, source, rows, key_col, sp_key, de_key in [
            ("user",          "user",          user_rows, "user_id",  "user_sparse_cols",          "user_dense_cols"),
            ("item",          "item",          item_rows, item_key,   "item_sparse_cols",          "item_dense_cols"),
            ("item_category", "item_category", cat_rows,  item_key,   "item_category_sparse_cols", None),
        ]:
            sp_cols, de_cols = infer_feature_groups(
                rows, key_col=key_col,
                explicit_sparse=dc.get(sp_key),
                explicit_dense=dc.get(de_key) if de_key else None,
            )
            # 类别型 → 直接建 vocab mapping
            for col in sp_cols:
                values = sorted({
                    str(r.get(col, ""))
                    for r in rows.values()
                    if r.get(col, "") not in ("", None)
                })
                mapping = {v: i + 1 for i, v in enumerate(values)}
                sparse_specs.append((f"{prefix}__{col}", source, col, len(mapping) + 1, mapping))
            # 数值型 → 加入 dense_specs，后续分桶
            for col in de_cols:
                dense_specs.append((f"{prefix}__{col}", source, col))

        tag = self.__class__.__name__.replace("Dataset", "")
        n_side_sp = len(sparse_specs) - 2
        n_side_de = len(dense_specs)
        if n_side_sp + n_side_de > 0:
            print(f"[{tag}] side features: {n_side_sp} categorical + "
                  f"{n_side_de} numerical (will be bucketized) "
                  f"= {n_side_sp + n_side_de} extra fields")
        return sparse_specs, dense_specs

    # ------------------------------------------------------------------ assemble

    def _precompute_sparse_with_buckets(
        self,
        raw_uids: np.ndarray,
        raw_vids: np.ndarray,
        uid2idx: dict,
        vid2idx: dict,
        user_rows: Dict[int, dict],
        item_rows: Dict[int, dict],
        cat_rows: Dict[int, dict],
        sparse_specs: List[tuple],
        dense_specs: List[tuple],
        num_buckets: int = 100,
        bucket_boundaries: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        预计算每个样本的完整 sparse 向量（含分桶后的数值特征）。

        Returns:
          result:      np.ndarray shape [N, num_sparse + num_dense_bucketized]
          boundaries:  dict {col_idx: edges_array}，train 时计算，test 时传入复用
        """
        N = len(raw_uids)
        is_train = bucket_boundaries is None

        # --- 1. 类别型 sparse ---
        n_sp = len(sparse_specs)
        sp_matrix = np.zeros((N, n_sp), dtype=np.int64)
        for i in range(N):
            raw_uid = int(raw_uids[i])
            raw_vid = int(raw_vids[i])
            sources = {
                "interaction": {"user_id": raw_uid, "video_id": raw_vid},
                "user":          user_rows.get(raw_uid, {}),
                "item":          item_rows.get(raw_vid, {}),
                "item_category": cat_rows.get(raw_vid, {}),
            }
            for j, (_, source, col, _, mapping) in enumerate(sparse_specs):
                src = sources[source]
                if source == "interaction":
                    sp_matrix[i, j] = uid2idx.get(raw_uid, 0) if col == "user_id" else vid2idx.get(raw_vid, 0)
                else:
                    sp_matrix[i, j] = mapping.get(str(src.get(col, "")), 0)

        if not dense_specs:
            return sp_matrix, {}

        # --- 2. 数值型 dense → 分桶 → sparse ---
        n_de = len(dense_specs)
        de_raw = np.zeros((N, n_de), dtype=np.float32)
        for i in range(N):
            raw_uid = int(raw_uids[i])
            raw_vid = int(raw_vids[i])
            sources = {
                "user":          user_rows.get(raw_uid, {}),
                "item":          item_rows.get(raw_vid, {}),
                "item_category": cat_rows.get(raw_vid, {}),
            }
            for j, (_, source, col) in enumerate(dense_specs):
                de_raw[i, j] = _sf_safe_float(sources.get(source, {}).get(col, ""), 0.0)

        boundaries_out: dict = {}
        de_bucketed = np.zeros((N, n_de), dtype=np.int64)
        for j in range(n_de):
            col_vals = de_raw[:, j]
            if is_train:
                edges = compute_bucket_edges(col_vals, num_buckets)
                boundaries_out[j] = edges
            else:
                edges = bucket_boundaries.get(j, np.array([]))
            de_bucketed[:, j] = apply_bucket(col_vals, edges, num_buckets)

        result = np.concatenate([sp_matrix, de_bucketed], axis=1)
        return result, boundaries_out

    # ------------------------------------------------------------------ meta

    def _update_dc_feature_meta(
        self,
        dc: dict,
        sparse_specs: List[tuple],
        dense_specs: List[tuple],
        num_buckets: int = 100,
    ) -> None:
        """
        更新 dc["cardinalities"] 和 dc["sparse_cols"]。
        数值型特征分桶后 cardinality = num_buckets + 1（含 bucket 0 = missing）。
        """
        cardinalities = [spec[3] for spec in sparse_specs]
        col_names     = [spec[0] for spec in sparse_specs]
        for name, _, _ in dense_specs:
            col_names.append(f"{name}_bucket")
            cardinalities.append(num_buckets + 1)
        dc["cardinalities"] = cardinalities
        dc["sparse_cols"]   = col_names
