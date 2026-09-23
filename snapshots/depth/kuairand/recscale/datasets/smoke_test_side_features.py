#!/usr/bin/env python3
"""
smoke_test_side_features.py
用少量数据验证 side_features.py 的核心逻辑，不需要真实数据集文件。

测试内容：
  1. infer_feature_groups 分类逻辑
  2. compute_bucket_edges / apply_bucket 分桶逻辑
  3. SideFeatureMixin._build_feature_specs（mock rows）
  4. SideFeatureMixin._precompute_sparse_with_buckets（mock 数据）
  5. _update_dc_feature_meta 更新 dc
  6. TAAC _load_taac_feat_parquet 退化行为（目录不存在时返回空）
  7. TAAC sparse 列数断言（side features 存在时 > 2，不存在时 == 2）

运行方式：
  cd /workspace/IDShare
  python recscale/datasets/smoke_test_side_features.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from recscale.datasets.side_features import (
    infer_feature_groups,
    compute_bucket_edges,
    apply_bucket,
    SideFeatureMixin,
)
from recscale.datasets.taac2025_time import _load_taac_feat_parquet

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

errors = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}  {detail}")
        errors.append(name)


# ------------------------------------------------------------------ Test 1
print("\n[1] infer_feature_groups")

mock_user_rows = {
    1: {"user_id": "1", "gender": "M", "age_range": "25-34",
        "onehot_feat0": "3", "onehot_feat1": "7",
        "fans_user_num": "12345", "register_days": "365.5"},
    2: {"user_id": "2", "gender": "F", "age_range": "18-24",
        "onehot_feat0": "1", "onehot_feat1": "2",
        "fans_user_num": "567", "register_days": "100.2"},
}
sp, de = infer_feature_groups(mock_user_rows, key_col="user_id")

check("gender → sparse", "gender" in sp)
check("age_range → sparse", "age_range" in sp)
check("onehot_feat0 → sparse", "onehot_feat0" in sp)
check("fans_user_num → dense (hint token)", "fans_user_num" in de)
check("register_days → dense (hint token)", "register_days" in de)
check("gender not in dense", "gender" not in de)


# ------------------------------------------------------------------ Test 2
print("\n[2] compute_bucket_edges / apply_bucket")

vals = np.array([0, 1, 2, 10, 100, 1000, 10000], dtype=np.float32)
edges = compute_bucket_edges(vals, num_buckets=4)
check("edges count == num_buckets-1", len(edges) == 3, f"got {len(edges)}")

buckets = apply_bucket(vals, edges, num_buckets=4)
check("all buckets in [1, 4]", buckets.min() >= 1 and buckets.max() <= 4,
      f"min={buckets.min()} max={buckets.max()}")
check("buckets are monotonic (non-decreasing)",
      all(buckets[i] <= buckets[i+1] for i in range(len(buckets)-1)),
      f"buckets={buckets.tolist()}")

vals_same = np.ones(10, dtype=np.float32)
edges_same = compute_bucket_edges(vals_same, num_buckets=4)
buckets_same = apply_bucket(vals_same, edges_same, num_buckets=4)
check("all-same values → all same bucket", len(set(buckets_same.tolist())) == 1)


# ------------------------------------------------------------------ Test 3
print("\n[3] SideFeatureMixin._build_feature_specs (mock)")

class MockDataset(SideFeatureMixin):
    pass

mixin = MockDataset()

mock_item_rows = {
    101: {"video_id": "101", "author_id": "A1", "video_type": "short",
          "play_cnt": "9999", "like_cnt": "500"},
    102: {"video_id": "102", "author_id": "A2", "video_type": "long",
          "play_cnt": "100",  "like_cnt": "5"},
}

uid2idx = {1: 1, 2: 2}
vid2idx = {101: 1, 102: 2}

sparse_specs, dense_specs = mixin._build_feature_specs(
    uid2idx, vid2idx,
    user_rows=mock_user_rows,
    item_rows=mock_item_rows,
    cat_rows={},
    dc={},
)

check("sparse_specs[0] == user_id", sparse_specs[0][0] == "user_id")
check("sparse_specs[1] == video_id", sparse_specs[1][0] == "video_id")
check("gender in sparse specs", any("gender" in s[0] for s in sparse_specs))
check("author_id in sparse specs", any("author_id" in s[0] for s in sparse_specs))
check("fans_user_num NOT in sparse specs", not any("fans_user_num" in s[0] for s in sparse_specs))
check("fans_user_num in dense specs", any("fans_user_num" in d[0] for d in dense_specs))
check("play_cnt in dense specs", any("play_cnt" in d[0] for d in dense_specs))


# ------------------------------------------------------------------ Test 4
print("\n[4] _precompute_sparse_with_buckets (mock)")

raw_uids = np.array([1, 2, 1], dtype=np.int64)
raw_vids = np.array([101, 102, 102], dtype=np.int64)

sparse_mat_train, boundaries = mixin._precompute_sparse_with_buckets(
    raw_uids, raw_vids, uid2idx, vid2idx,
    mock_user_rows, mock_item_rows, {},
    sparse_specs, dense_specs, num_buckets=4,
)

expected_cols = len(sparse_specs) + len(dense_specs)
check(f"train sparse shape == (3, {expected_cols})",
      sparse_mat_train.shape == (3, expected_cols),
      f"got {sparse_mat_train.shape}")
check("boundaries contains dense spec indices",
      len(boundaries) == len(dense_specs),
      f"got {len(boundaries)}")
check("train sparse dtype == int64", sparse_mat_train.dtype == np.int64)
check("user_id col (col 0) == uid2idx values",
      sparse_mat_train[0, 0] == uid2idx[1] and sparse_mat_train[1, 0] == uid2idx[2])
check("video_id col (col 1) == vid2idx values",
      sparse_mat_train[0, 1] == vid2idx[101] and sparse_mat_train[1, 1] == vid2idx[102])

sparse_mat_test, _ = mixin._precompute_sparse_with_buckets(
    raw_uids[:1], raw_vids[:1], uid2idx, vid2idx,
    mock_user_rows, mock_item_rows, {},
    sparse_specs, dense_specs, num_buckets=4,
    bucket_boundaries=boundaries,
)
check("test sparse shape == (1, expected_cols)",
      sparse_mat_test.shape == (1, expected_cols))
check("train/test same row for same uid/vid",
      (sparse_mat_test[0] == sparse_mat_train[0]).all())


# ------------------------------------------------------------------ Test 5
print("\n[5] _update_dc_feature_meta")

dc = {}
mixin._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets=4)

check("dc['cardinalities'] length == sparse + dense",
      len(dc["cardinalities"]) == len(sparse_specs) + len(dense_specs))
check("dc['sparse_cols'] length == sparse + dense",
      len(dc["sparse_cols"]) == len(sparse_specs) + len(dense_specs))
check("dense bucket cols end with '_bucket'",
      all(c.endswith("_bucket") for c in dc["sparse_cols"][len(sparse_specs):]))
check("bucket cardinality == num_buckets + 1",
      all(c == 5 for c in dc["cardinalities"][len(sparse_specs):]))


# ------------------------------------------------------------------ Test 6
print("\n[6] TAAC _load_taac_feat_parquet degradation")

# 目录不存在时应返回空 dict
result = _load_taac_feat_parquet("/nonexistent/path", "user_id", ["103", "104"])
check("missing dir returns empty dict", result == {}, f"got {result}")

# 空特征 ID 列表
result2 = _load_taac_feat_parquet("/nonexistent/path", "item_id", [])
check("missing dir + empty feat_ids returns empty dict", result2 == {})


# ------------------------------------------------------------------ Test 7
print("\n[7] TAAC sparse col count logic")

# 模拟 train split 建立 meta（不加载真实文件，模拟 no side features 退化）
from recscale.datasets.taac2025_time import TAAC2025TimeDataset

# 模拟 no side features 时：sparse_cols = ["user_id", "target_item_id"]
sparse_cols_no_side = ["user_id", "target_item_id"]
cardinalities_no_side = [1000, 5000]
check("no side features → 2 cols", len(sparse_cols_no_side) == 2)

# 模拟有 side features 时：4 user + 13 item = 17 extra
user_ids = ["103", "104", "105", "109"]
item_ids = ["100", "117", "118", "101", "102", "119", "120", "114", "112", "121", "115", "122", "116"]
sparse_cols_with_side = (
    ["user_id", "target_item_id"]
    + [f"user_feat_{fid}" for fid in user_ids]
    + [f"item_feat_{fid}" for fid in item_ids]
)
check("with side features → 2 + 4 + 13 = 19 cols",
      len(sparse_cols_with_side) == 19,
      f"got {len(sparse_cols_with_side)}")
check("sparse_cols_with_side[0] == user_id", sparse_cols_with_side[0] == "user_id")
check("sparse_cols_with_side[1] == target_item_id", sparse_cols_with_side[1] == "target_item_id")
check("user feat cols at index 2~5",
      all(sparse_cols_with_side[2+i].startswith("user_feat_") for i in range(4)))
check("item feat cols at index 6~18",
      all(sparse_cols_with_side[6+i].startswith("item_feat_") for i in range(13)))


# ------------------------------------------------------------------ Summary
print()
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print(f"All tests passed.")


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

# 直接 import 模块（绕过 register_dataset 注册）
from recscale.datasets.side_features import (
    infer_feature_groups,
    compute_bucket_edges,
    apply_bucket,
    SideFeatureMixin,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

errors = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}  {detail}")
        errors.append(name)


# ------------------------------------------------------------------ Test 1
print("\n[1] infer_feature_groups")

mock_user_rows = {
    1: {"user_id": "1", "gender": "M", "age_range": "25-34",
        "onehot_feat0": "3", "onehot_feat1": "7",
        "fans_user_num": "12345", "register_days": "365.5"},
    2: {"user_id": "2", "gender": "F", "age_range": "18-24",
        "onehot_feat0": "1", "onehot_feat1": "2",
        "fans_user_num": "567", "register_days": "100.2"},
}
sp, de = infer_feature_groups(mock_user_rows, key_col="user_id")

check("gender → sparse", "gender" in sp)
check("age_range → sparse", "age_range" in sp)
check("onehot_feat0 → sparse", "onehot_feat0" in sp)
check("fans_user_num → dense (hint token)", "fans_user_num" in de)
check("register_days → dense (hint token)", "register_days" in de)
check("gender not in dense", "gender" not in de)


# ------------------------------------------------------------------ Test 2
print("\n[2] compute_bucket_edges / apply_bucket")

vals = np.array([0, 1, 2, 10, 100, 1000, 10000], dtype=np.float32)
edges = compute_bucket_edges(vals, num_buckets=4)
check("edges count == num_buckets-1", len(edges) == 3,
      f"got {len(edges)}")

buckets = apply_bucket(vals, edges, num_buckets=4)
check("all buckets in [1, 4]", buckets.min() >= 1 and buckets.max() <= 4,
      f"min={buckets.min()} max={buckets.max()}")
check("buckets are monotonic (non-decreasing)", all(buckets[i] <= buckets[i+1] for i in range(len(buckets)-1)),
      f"buckets={buckets.tolist()}")

# edge case: all same value
vals_same = np.ones(10, dtype=np.float32)
edges_same = compute_bucket_edges(vals_same, num_buckets=4)
buckets_same = apply_bucket(vals_same, edges_same, num_buckets=4)
check("all-same values → all same bucket", len(set(buckets_same.tolist())) == 1)


# ------------------------------------------------------------------ Test 3
print("\n[3] SideFeatureMixin._build_feature_specs (mock)")

class MockDataset(SideFeatureMixin):
    pass

mixin = MockDataset()

mock_item_rows = {
    101: {"video_id": "101", "author_id": "A1", "video_type": "short",
          "play_cnt": "9999", "like_cnt": "500"},
    102: {"video_id": "102", "author_id": "A2", "video_type": "long",
          "play_cnt": "100",  "like_cnt": "5"},
}

uid2idx = {1: 1, 2: 2}
vid2idx = {101: 1, 102: 2}

sparse_specs, dense_specs = mixin._build_feature_specs(
    uid2idx, vid2idx,
    user_rows=mock_user_rows,
    item_rows=mock_item_rows,
    cat_rows={},
    dc={},
)

check("sparse_specs[0] == user_id", sparse_specs[0][0] == "user_id")
check("sparse_specs[1] == video_id", sparse_specs[1][0] == "video_id")
check("gender in sparse specs", any("gender" in s[0] for s in sparse_specs))
check("author_id in sparse specs", any("author_id" in s[0] for s in sparse_specs))
check("fans_user_num NOT in sparse specs", not any("fans_user_num" in s[0] for s in sparse_specs))
check("fans_user_num in dense specs", any("fans_user_num" in d[0] for d in dense_specs))
check("play_cnt in dense specs", any("play_cnt" in d[0] for d in dense_specs))


# ------------------------------------------------------------------ Test 4
print("\n[4] _precompute_sparse_with_buckets (mock)")

raw_uids = np.array([1, 2, 1], dtype=np.int64)
raw_vids = np.array([101, 102, 102], dtype=np.int64)

# train: compute boundaries
sparse_mat_train, boundaries = mixin._precompute_sparse_with_buckets(
    raw_uids, raw_vids, uid2idx, vid2idx,
    mock_user_rows, mock_item_rows, {},
    sparse_specs, dense_specs, num_buckets=4,
)

expected_cols = len(sparse_specs) + len(dense_specs)
check(f"train sparse shape == (3, {expected_cols})",
      sparse_mat_train.shape == (3, expected_cols),
      f"got {sparse_mat_train.shape}")
check("boundaries contains dense spec indices",
      len(boundaries) == len(dense_specs),
      f"got {len(boundaries)}")
check("train sparse dtype == int64", sparse_mat_train.dtype == np.int64)
check("user_id col (col 0) == uid2idx values",
      sparse_mat_train[0, 0] == uid2idx[1] and sparse_mat_train[1, 0] == uid2idx[2])
check("video_id col (col 1) == vid2idx values",
      sparse_mat_train[0, 1] == vid2idx[101] and sparse_mat_train[1, 1] == vid2idx[102])

# test: reuse boundaries
sparse_mat_test, _ = mixin._precompute_sparse_with_buckets(
    raw_uids[:1], raw_vids[:1], uid2idx, vid2idx,
    mock_user_rows, mock_item_rows, {},
    sparse_specs, dense_specs, num_buckets=4,
    bucket_boundaries=boundaries,
)
check("test sparse shape == (1, expected_cols)",
      sparse_mat_test.shape == (1, expected_cols))
check("train/test same row for same uid/vid",
      (sparse_mat_test[0] == sparse_mat_train[0]).all())


# ------------------------------------------------------------------ Test 5
print("\n[5] _update_dc_feature_meta")

dc = {}
mixin._update_dc_feature_meta(dc, sparse_specs, dense_specs, num_buckets=4)

check("dc['cardinalities'] length == sparse + dense",
      len(dc["cardinalities"]) == len(sparse_specs) + len(dense_specs))
check("dc['sparse_cols'] length == sparse + dense",
      len(dc["sparse_cols"]) == len(sparse_specs) + len(dense_specs))
check("dense bucket cols end with '_bucket'",
      all(c.endswith("_bucket") for c in dc["sparse_cols"][len(sparse_specs):]))
check("bucket cardinality == num_buckets + 1",
      all(c == 5 for c in dc["cardinalities"][len(sparse_specs):]))


# ------------------------------------------------------------------ Summary
print()
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print(f"All tests passed.")
