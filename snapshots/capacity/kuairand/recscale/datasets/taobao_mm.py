"""
recscale.datasets.taobao_mm — Taobao-MM 多模态推荐数据集

Sharded Parquet 格式, 官方按时间切分。
特征: 6 user sparse + 4 item sparse + 128d SCL emb + 行为序列 (mean≈985, max=1000)

对齐旧版 datasets/taobao_mm/ 的关键设计:
1. hash_mod=1000000 (和旧版一致，避免碰撞)
2. 加载 SCL 多模态 embedding (raw/scl_embedding_int8_p90.parquet)
3. 所有 sparse 使用统一 hash 空间 (target/seq/sparse 共享)

YAML 示例:
```yaml
dataset:
  type: taobao_mm
  path: /data/taobao_mm
  maxlen: 50
  max_rows: 5000000
  hash_mod: 1000000     # 和旧版对齐
```
"""

import os
import glob

import numpy as np
import pyarrow.parquet as pq

from . import register_dataset
from .base import BaseDataset

USER_SPARSE = ["130_1", "130_2", "130_3", "130_4", "130_5"]
ITEM_SPARSE = ["206", "213", "214"]
SEQ_COL = "150_2_180"
LABEL_COL = "label_0"
MM_EMB_ID_COL = "205"
MM_EMB_VEC_COL = "205_c"


@register_dataset("taobao_mm")
class TaobaoMMDataset(BaseDataset):

    # 类级别共享: train 加载的 SCL embedding 供 test 复用
    _shared_mm_emb: dict = None

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        max_rows = dc.get("max_rows", 0)
        self.hash_mod = dc.get("hash_mod", 1000000)  # 对齐旧版默认 1M

        # Sharded parquet
        shard_dir = os.path.join(data_path, split)
        shards = sorted(glob.glob(os.path.join(shard_dir, f"{split}-shard-*.parquet")))
        if not shards:
            shards = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
            shards = [s for s in shards if not s.endswith("metadata.json")]
        if not shards:
            raise FileNotFoundError(
                f"[TaobaoMM] No shards in {shard_dir}")

        print(f"[TaobaoMM] Found {len(shards)} shards in {shard_dir}")

        all_cols = USER_SPARSE + ITEM_SPARSE + [SEQ_COL, LABEL_COL, "205"]  # 205 for target + SCL

        labels_list = []
        sparse_list = []
        seqs_list = []
        item_ids_raw = []  # 原始 item_id，用于查 SCL embedding
        n_total = 0

        for shard_path in shards:
            tbl = pq.read_table(shard_path, columns=all_cols)
            data = tbl.to_pydict()
            n_rows = len(data[LABEL_COL])

            for i in range(n_rows):
                if 0 < max_rows <= n_total:
                    break

                # Label: [1,0]=non-click, [0,1]=click → take index 1
                label_vec = data[LABEL_COL][i]
                label = float(label_vec[1]) if len(label_vec) > 1 else 0.0
                labels_list.append(label)

                # Sparse (原始值，后面统一 hash)
                sparse = [data[c][i] for c in USER_SPARSE + ITEM_SPARSE]
                sparse_list.append(sparse)

                # 保存原始 item_id (col 205) 用于查 SCL embedding
                item_ids_raw.append(data["205"][i])

                # Sequence: truncate to last maxlen
                seq = data[SEQ_COL][i]
                if seq is None:
                    seq = []
                seq = seq[-self.maxlen:]
                seqs_list.append(seq)

                n_total += 1

            if 0 < max_rows <= n_total:
                break

        print(f"[TaobaoMM] Loaded {n_total:,} samples from {split}")

        # Convert to numpy
        self.labels = np.array(labels_list, dtype=np.float32)

        # Sparse: 统一 hash 取模 (abs(val) % hash_mod + 1)
        raw_sparse = np.array(sparse_list, dtype=np.int64)
        self.sparse = np.abs(raw_sparse) % self.hash_mod + 1

        # 设置 sparse_cols 和 cardinalities
        dc["sparse_cols"] = USER_SPARSE + ITEM_SPARSE
        dc.setdefault("cardinalities", [self.hash_mod + 2] * len(USER_SPARSE + ITEM_SPARSE))
        dc["num_items"] = self.hash_mod + 2  # for DIN seq/target embedding
        dc["mm_dim"] = self.mm_dim if hasattr(self, 'mm_dim') else 128  # for model mm_proj

        # Sequences: pad，右对齐，0 为 padding
        self.seqs = np.zeros((n_total, self.maxlen), dtype=np.int64)
        for i, seq in enumerate(seqs_list):
            if seq:
                padded = seq[-self.maxlen:]
                start = self.maxlen - len(padded)
                for k, v in enumerate(padded):
                    self.seqs[i, start + k] = abs(v) % self.hash_mod + 1

        # Target: item_id (col 205) 的 hash，和序列在同一空间
        raw_item_ids = np.array([abs(iid) % self.hash_mod + 1 for iid in item_ids_raw], dtype=np.int64)
        self.targets = raw_item_ids

        # ============================================================
        # SCL 多模态 Embedding (对齐旧版: raw/scl_embedding_int8_p90.parquet)
        # ============================================================
        if split == "train":
            self._load_mm_emb(data_path)
            TaobaoMMDataset._shared_mm_emb = self.mm_emb_dict
        else:
            if TaobaoMMDataset._shared_mm_emb is not None:
                self.mm_emb_dict = TaobaoMMDataset._shared_mm_emb
                self.mm_dim = len(next(iter(self.mm_emb_dict.values()))) if self.mm_emb_dict else 128
                print(f"[TaobaoMM] Using shared SCL embeddings: {len(self.mm_emb_dict):,} entries")
            else:
                self._load_mm_emb(data_path)

        # 预取每个样本的 mm_emb (避免 __getitem__ 里查 dict)
        self.mm_embs = np.zeros((n_total, self.mm_dim), dtype=np.float32)
        n_found = 0
        for i, iid in enumerate(item_ids_raw):
            emb = self.mm_emb_dict.get(iid)
            if emb is not None:
                self.mm_embs[i] = emb
                n_found += 1

        n_pos = (self.labels > 0.5).sum()
        print(f"[TaobaoMM] {split}: {n_total:,} samples, "
              f"pos={n_pos:,} ({n_pos / max(n_total, 1) * 100:.2f}%), "
              f"mm_emb coverage={n_found}/{n_total} ({n_found/max(n_total,1)*100:.1f}%)")

    def _load_mm_emb(self, data_path):
        """加载 SCL 多模态 embedding (int8 128d)"""
        emb_path = os.path.join(data_path, "raw", "scl_embedding_int8_p90.parquet")
        self.mm_emb_dict = {}
        self.mm_dim = 128

        if os.path.exists(emb_path):
            print(f"[TaobaoMM] Loading SCL embeddings from {emb_path} ...")
            emb_tbl = pq.read_table(emb_path, columns=[MM_EMB_ID_COL, MM_EMB_VEC_COL])
            emb_data = emb_tbl.to_pydict()
            for iid, emb in zip(emb_data[MM_EMB_ID_COL], emb_data[MM_EMB_VEC_COL]):
                if emb is not None:
                    self.mm_emb_dict[iid] = np.array(emb, dtype=np.float32)
            if self.mm_emb_dict:
                self.mm_dim = len(next(iter(self.mm_emb_dict.values())))
            print(f"[TaobaoMM] Loaded {len(self.mm_emb_dict):,} SCL embeddings, dim={self.mm_dim}")
        else:
            print(f"[TaobaoMM] SCL embedding not found at {emb_path}, using zeros")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sparse": self.sparse[idx],
            "seq": self.seqs[idx],
            "target": self.targets[idx],
            "mm_emb": self.mm_embs[idx],
            "label": self.labels[idx],
        }
