"""
recscale.datasets.taac2025_user — TAAC2025 按用户切分 (10m-user-split / 1m-user-split)

直接读 sharded parquet，每行包含完整特征 + 序列。

YAML 示例:
```yaml
dataset:
  type: taac2025_user
  path: /data/taac2025/10m-user-split
  maxlen: 100
  mm_emb_ids: ['81']
  sparse_ids: ['100', '117', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116']
  user_sparse_ids: ['103', '104', '105', '109']
```
"""

import os
import pickle

import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as pds

from . import register_dataset
from .base import BaseDataset


@register_dataset("taac2025_user")
class TAAC2025UserDataset(BaseDataset):

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 100)
        max_rows = dc.get("max_rows", 0)

        # 加载 indexer
        indexer_path = os.path.join(data_path, "indexer.pkl")
        with open(indexer_path, "rb") as f:
            indexer = pickle.load(f)
        self.num_items = len(indexer["i"])
        self.num_users = len(indexer["u"])
        dc["num_items"] = self.num_items + 1
        dc["num_users"] = self.num_users + 1
        
        # FIX for Bug #7: Add cardinalities
        dc["cardinalities"] = [dc["num_users"], dc["num_items"]]

        # 读取 samples
        samples_path = os.path.join(data_path, split, "samples.parquet")
        print(f"[TAAC2025-User] Loading {samples_path} ...")
        tbl = pq.read_table(samples_path)
        data = tbl.to_pydict()

        n_total = len(data["user_id"])
        if 0 < max_rows < n_total:
            for k in data:
                data[k] = data[k][:max_rows]
            n_total = max_rows

        self.user_ids = np.array(data["user_id"], dtype=np.int64)
        self.target_items = np.array(data["target_item_id"], dtype=np.int64)
        self.labels = np.array(data["label"], dtype=np.float32)

        # 序列 (如果 samples 里有 seq 列)
        if "seq" in data:
            self.seqs = np.zeros((n_total, self.maxlen), dtype=np.int64)
            for i, seq in enumerate(data["seq"]):
                if seq:
                    truncated = seq[-self.maxlen:]
                    self.seqs[i, self.maxlen - len(truncated):] = truncated
        else:
            self.seqs = np.zeros((n_total, self.maxlen), dtype=np.int64)

        n_pos = (self.labels > 0.5).sum()
        print(f"[TAAC2025-User] {split}: {n_total:,} samples, "
              f"pos={n_pos:,} ({n_pos / n_total * 100:.2f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sparse": np.array([self.user_ids[idx], self.target_items[idx]], dtype=np.int64),
            "seq": self.seqs[idx],
            "target": self.target_items[idx],
            "label": self.labels[idx],
        }
