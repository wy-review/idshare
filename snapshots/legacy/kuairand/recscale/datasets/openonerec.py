"""
recscale.datasets.openonerec — OpenOneRec 快手推荐数据集

数据格式 (processed_10k/ 或 processed/):
  - train/samples.parquet: (user_id, target_item_rid, label)
  - test/samples.parquet: 同上
  - train/user_seqs.pkl: dict[user_id] → np.array(item_ids, int32)
  - train/user_masks.pkl: dict[user_id] → np.array([5, seq_len], int8)
  - train/user_profiles.pkl: dict[user_id] → (gender_id, age_id)
  - sid_array.npy: [item_num+1, 3] Semantic ID 三层

YAML 示例:
```yaml
dataset:
  type: openonerec
  path: /data/openonerec/processed
  maxlen: 512
```
"""

import os
import pickle

import numpy as np
import pyarrow.parquet as pq

from . import register_dataset
from .base import BaseDataset

NUM_BEHAVIORS = 5  # longview, like, follow, forward, not_interested


@register_dataset("openonerec")
class OpenOneRecDataset(BaseDataset):
    # FIX for Bug #9: Track max item ID across splits
    _shared_max_item_id = None

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 512)
        max_rows = dc.get("max_rows", 0)

        split_dir = os.path.join(data_path, split)

        # 加载 samples
        samples_path = os.path.join(split_dir, "samples.parquet")
        print(f"[OpenOneRec] Loading {samples_path} ...")
        samples = pq.read_table(samples_path).to_pydict()

        self.user_ids = np.array(samples["user_id"])
        self.target_items = np.array(samples["target_item_rid"])
        self.labels = np.array(samples["label"], dtype=np.float32)

        if 0 < max_rows < len(self.labels):
            self.user_ids = self.user_ids[:max_rows]
            self.target_items = self.target_items[:max_rows]
            self.labels = self.labels[:max_rows]

        # 加载 user_seqs: dict[user_id] → np.array(item_ids)
        seqs_path = os.path.join(split_dir, "user_seqs.pkl")
        print(f"[OpenOneRec] Loading {seqs_path} ...")
        with open(seqs_path, "rb") as f:
            self.user_seqs = pickle.load(f)

        # 加载 user_masks: dict[user_id] → np.array([5, seq_len])
        masks_path = os.path.join(split_dir, "user_masks.pkl")
        if os.path.exists(masks_path):
            with open(masks_path, "rb") as f:
                self.user_masks = pickle.load(f)
        else:
            self.user_masks = {}

        # 加载 user_profiles: dict[user_id] → (gender_id, age_id)
        profiles_path = os.path.join(split_dir, "user_profiles.pkl")
        if os.path.exists(profiles_path):
            with open(profiles_path, "rb") as f:
                self.user_profiles = pickle.load(f)
        else:
            self.user_profiles = {}

        # Semantic ID array: [item_num+1, 3]
        sid_path = os.path.join(data_path, "sid_array.npy")
        if os.path.exists(sid_path):
            self.sid_array = np.load(sid_path)
        else:
            self.sid_array = None

        # FIX for Bug #9: Handle out-of-range test items
        if split == "train":
            train_max = int(self.target_items.max()) if len(self.target_items) > 0 else 0
            OpenOneRecDataset._shared_max_item_id = train_max
            
            # num_items for model (from sid_array if available, else from data)
            if self.sid_array is not None:
                dc["num_items"] = self.sid_array.shape[0]
            else:
                dc["num_items"] = train_max + 1
        else:
            # Test split: check if items exceed training range
            test_max = int(self.target_items.max()) if len(self.target_items) > 0 else 0
            train_max = OpenOneRecDataset._shared_max_item_id or 0
            
            if test_max > train_max:
                print(f"[OpenOneRec] Warning: Test has items beyond training range. "
                      f"Training max: {train_max}, Test max: {test_max}. "
                      f"Items {train_max+1}..{test_max} will get zeros features.")
                # Expand num_items to cover test range
                if self.sid_array is not None:
                    # sid_array should already cover all items, but use max for safety
                    dc["num_items"] = max(self.sid_array.shape[0], test_max + 1)
                else:
                    dc["num_items"] = test_max + 1
            else:
                # Test items within training range
                if self.sid_array is not None:
                    dc["num_items"] = self.sid_array.shape[0]
                else:
                    dc["num_items"] = train_max + 1

        # cardinalities: [gender(3), age(8), item]
        dc.setdefault("cardinalities", [3, 8, dc["num_items"]])

        n_pos = (self.labels > 0.5).sum()
        n_total = len(self.labels)
        print(f"[OpenOneRec] {split}: {n_total:,} samples, "
              f"pos={n_pos:,} ({n_pos / max(n_total, 1) * 100:.2f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        target_rid = self.target_items[idx]

        # Sequence
        seq = self.user_seqs.get(uid, np.array([], dtype=np.int32))
        seq_len = len(seq)
        if seq_len > self.maxlen:
            seq = seq[-self.maxlen:]
            seq_len = self.maxlen

        # Pad (right-align)
        padded_seq = np.zeros(self.maxlen, dtype=np.int64)
        if seq_len > 0:
            padded_seq[-seq_len:] = seq

        # User profile
        gender_id, age_id = self.user_profiles.get(uid, (0, 0))

        # Semantic ID of target (FIX for Bug #9: bounds check)
        if self.sid_array is not None and target_rid < len(self.sid_array):
            target_sid = self.sid_array[target_rid]  # [3]
        else:
            # Out of range or no sid_array: return zeros
            target_sid = np.zeros(3, dtype=np.int64)

        return {
            "sparse": np.array([gender_id, age_id, target_rid], dtype=np.int64),
            "seq": padded_seq,
            "target": np.int64(target_rid),
            "label": self.labels[idx],
        }
