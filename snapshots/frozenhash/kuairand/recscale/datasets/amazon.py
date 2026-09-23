"""
recscale.datasets.amazon — Amazon Reviews 2023 (Electronics)

从评分记录构建用户行为序列。
Label: rating >= 4 → 正样本, rating <= 2 → 负样本, rating == 3 → 丢弃。

CSV 格式（prepare_data.py 含 metadata 版）：
  user_id, item_id, store_id, main_cat_id,
  price_bucket, avg_rating_bucket, rating_num_bucket,
  label, history_items

CSV 格式（旧版，无 metadata）：
  user_id, item_id, category_id, label, history_items

两种格式均支持，通过列名自动检测。

YAML 示例:
```yaml
dataset:
  type: amazon
  path: /data/amazon_reviews/electronics_ctr
  maxlen: 50
```
"""

import csv
import os

import numpy as np

from . import register_dataset
from .base import BaseDataset

# 新版 CSV（含 metadata）的列名集合
_META_COLS = {"store_id", "main_cat_id", "price_bucket", "avg_rating_bucket", "rating_num_bucket"}


@register_dataset("amazon")
class AmazonDataset(BaseDataset):
    """Amazon Reviews 序列模型 adapter。
    no-seq 特征：
      - 新版 CSV：user_id + item_id + store_id + main_cat_id +
                  price_bucket + avg_rating_bucket + rating_num_bucket (7 列)
      - 旧版 CSV：user_id + item_id (+ category_id 若非全零) (2-3 列)
    """

    _shared_uid2idx = None
    _shared_iid2idx = None
    _shared_sparse_cols = None
    _shared_cardinalities = None
    # 新版 meta 特征的 vocab（store / main_cat 是类别型，已在 prepare_data.py 中映射为整数）
    _shared_n_stores = None
    _shared_n_cats   = None
    _shared_num_buckets = 20

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen  = dc.get("maxlen", 50)
        max_rows     = dc.get("max_rows", 0)

        csv_name = "train.csv" if split == "train" else "test.csv"
        csv_path = os.path.join(data_path, csv_name)
        print(f"[Amazon] Loading {csv_path} ...")

        rows_raw = []
        fieldnames = None
        n = 0
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                if 0 < max_rows <= n:
                    break
                rows_raw.append(row)
                n += 1

        has_meta = bool(_META_COLS & set(fieldnames))
        print(f"[Amazon] {n:,} rows loaded, has_meta={has_meta}, cols={fieldnames}")

        # ---- 提取字段 ----------------------------------------
        users  = [r["user_id"]  for r in rows_raw]
        items  = [r["item_id"]  for r in rows_raw]
        labels = [float(r["label"]) for r in rows_raw]
        seqs   = [
            r.get("history_items", "").split()
            if r.get("history_items", "").strip() else []
            for r in rows_raw
        ]

        user_set = set(users)
        item_set = set(items)
        for seq in seqs:
            item_set.update(seq)

        if has_meta:
            # 新版：store_id / main_cat_id / *_bucket 已是整数，直接读
            store_ids     = [int(r.get("store_id", 0) or 0)          for r in rows_raw]
            main_cat_ids  = [int(r.get("main_cat_id", 0) or 0)       for r in rows_raw]
            price_b       = [int(r.get("price_bucket", 0) or 0)      for r in rows_raw]
            avg_rat_b     = [int(r.get("avg_rating_bucket", 0) or 0) for r in rows_raw]
            rat_num_b     = [int(r.get("rating_num_bucket", 0) or 0) for r in rows_raw]
        else:
            # 旧版：只有 category_id（全为 0 时不加入 sparse）
            cat_vals = [r.get("category_id", "0") for r in rows_raw]
            cat_set  = {c for c in cat_vals if c and c != "0"}

        # ---- vocab -------------------------------------------
        if split == "train":
            uid2idx = {u: i+1 for i, u in enumerate(sorted(user_set))}
            iid2idx = {v: i+1 for i, v in enumerate(sorted(item_set))}
            AmazonDataset._shared_uid2idx = uid2idx
            AmazonDataset._shared_iid2idx = iid2idx

            dc["num_users"] = len(uid2idx) + 1
            dc["num_items"] = len(iid2idx) + 1

            if has_meta:
                # store_id / main_cat_id 已经是整数 ID（来自 prepare_data vocab），
                # cardinality 从 vocab.pkl 读取（若存在）或用最大值+2 估算
                import pickle
                vocab_path = os.path.join(data_path, "vocab.pkl")
                if os.path.exists(vocab_path):
                    with open(vocab_path, "rb") as vf:
                        voc = pickle.load(vf)
                    n_stores      = voc.get("n_stores", max(store_ids, default=0)+2)
                    n_cats        = voc.get("n_cats",   max(main_cat_ids, default=0)+2)
                    num_buckets   = voc.get("num_buckets", 20)
                else:
                    n_stores    = max(store_ids, default=0) + 2
                    n_cats      = max(main_cat_ids, default=0) + 2
                    num_buckets = 20
                AmazonDataset._shared_n_stores      = n_stores
                AmazonDataset._shared_n_cats        = n_cats
                AmazonDataset._shared_num_buckets   = num_buckets

                sparse_cols   = ["user_id", "item_id",
                                 "store_id", "main_cat_id",
                                 "price_bucket", "avg_rating_bucket", "rating_num_bucket"]
                cardinalities = [dc["num_users"], dc["num_items"],
                                 n_stores, n_cats,
                                 num_buckets+1, num_buckets+1, num_buckets+1]
                print(f"[Amazon] Meta vocab: n_stores={n_stores}, n_cats={n_cats}, "
                      f"num_buckets={num_buckets}")
            else:
                cid2idx = {c: i+1 for i, c in enumerate(sorted(cat_set))} if cat_set else {}
                AmazonDataset._shared_cid2idx = cid2idx
                sparse_cols   = ["user_id", "item_id"]
                cardinalities = [dc["num_users"], dc["num_items"]]
                if cid2idx:
                    sparse_cols.append("category_id")
                    cardinalities.append(len(cid2idx)+1)

            AmazonDataset._shared_sparse_cols   = sparse_cols
            AmazonDataset._shared_cardinalities = cardinalities
            print(f"[Amazon] Train vocab: {len(uid2idx):,} users, {len(iid2idx):,} items, "
                  f"sparse_cols={sparse_cols}")

        else:
            if AmazonDataset._shared_uid2idx is None:
                raise RuntimeError(
                    "[Amazon] Test split loaded before train. Load train first.")
            uid2idx       = AmazonDataset._shared_uid2idx
            iid2idx       = AmazonDataset._shared_iid2idx
            sparse_cols   = AmazonDataset._shared_sparse_cols
            cardinalities = AmazonDataset._shared_cardinalities
            dc["num_users"] = len(uid2idx) + 1
            dc["num_items"] = len(iid2idx) + 1
            if not has_meta:
                cid2idx = getattr(AmazonDataset, "_shared_cid2idx", {}) or {}
            print(f"[Amazon] Test using shared vocab from train")

        dc["sparse_cols"]   = sparse_cols
        dc["cardinalities"] = cardinalities

        # ---- 构建 numpy arrays -----------------------------------
        self.user_ids = np.array([uid2idx.get(u, 0) for u in users], dtype=np.int64)
        self.item_ids = np.array([iid2idx.get(v, 0) for v in items], dtype=np.int64)
        self.labels   = np.array(labels, dtype=np.float32)

        self.seqs = np.zeros((n, self.maxlen), dtype=np.int64)
        for i, seq in enumerate(seqs):
            tr = seq[-self.maxlen:]
            for k, v in enumerate(tr):
                self.seqs[i, self.maxlen - len(tr) + k] = iid2idx.get(v, 0)

        if has_meta:
            self.sparse = np.column_stack([
                self.user_ids,
                self.item_ids,
                np.array(store_ids,    dtype=np.int64),
                np.array(main_cat_ids, dtype=np.int64),
                np.array(price_b,      dtype=np.int64),
                np.array(avg_rat_b,    dtype=np.int64),
                np.array(rat_num_b,    dtype=np.int64),
            ]).astype(np.int64)
        else:
            cat_ids = np.array([cid2idx.get(c, 0) for c in cat_vals], dtype=np.int64)
            cols = [self.user_ids, self.item_ids]
            if cid2idx:
                cols.append(cat_ids)
            self.sparse = np.column_stack(cols).astype(np.int64)

        from .base import validate_cardinalities
        validate_cardinalities(config, self.sparse, sparse_cols)
        n_pos = (self.labels > 0.5).sum()
        print(f"[Amazon] {split}: {n:,} samples, sparse={self.sparse.shape[1]}, "
              f"pos={n_pos:,} ({n_pos/n*100:.2f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sparse": self.sparse[idx],
            "seq":    self.seqs[idx],
            "target": self.item_ids[idx],
            "label":  self.labels[idx],
        }
