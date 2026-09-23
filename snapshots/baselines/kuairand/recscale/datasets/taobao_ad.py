"""
recscale.datasets.taobao_ad — Taobao Ad CTR 数据集

支持两个版本:
1. reczoo TaobaoAd_x1.zip: 已 join, 序列已截断到 50, CSV 格式
2. 天池原始 4 文件: 需要自行 join, 序列完整

YAML 示例:
```yaml
dataset:
  type: taobao_ad
  path: /data/taobao_ad_ctr/TaobaoAd_x1.zip
  version: reczoo           # reczoo | raw
  maxlen: 50
```
"""

import csv
import io
import zipfile

import numpy as np

from . import register_dataset
from .base import BaseDataset

USER_SPARSE = ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code',
               'age_level', 'pvalue_level', 'shopping_level', 'occupation',
               'new_user_class_level']
AD_SPARSE = ['adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag']
SEQ_COLS = ['cate_his', 'brand_his']


@register_dataset("taobao_ad")
class TaobaoAdDataset(BaseDataset):

    # 类级别共享 feature maps
    _shared_feature_maps = None
    _shared_seq_maps = None
    _shared_cardinalities = None

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        max_rows = dc.get("max_rows", 0)
        self.sparse_cols = USER_SPARSE + AD_SPARSE

        # reczoo 版: zip 内嵌 train.csv / test.csv
        csv_name = "train.csv" if split == "train" else "test.csv"
        print(f"[TaobaoAd] Loading {split}: {csv_name} from {data_path}")

        rows = []
        with zipfile.ZipFile(data_path) as z:
            with z.open(csv_name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for i, row in enumerate(reader):
                    if 0 < max_rows <= i:
                        break
                    rows.append(row)

        print(f"[TaobaoAd] Loaded {len(rows):,} rows")

        # Feature maps
        if split == "train":
            self.feature_maps, self.seq_maps, cardinalities, seq_vocab = self._build_maps(rows)
            TaobaoAdDataset._shared_feature_maps = self.feature_maps
            TaobaoAdDataset._shared_seq_maps = self.seq_maps
            TaobaoAdDataset._shared_cardinalities = cardinalities
            dc["cardinalities"] = cardinalities  # only sparse, not seq
            dc["num_items"] = seq_vocab  # seq vocab for DIN
            # FIX for Bug #8: Store sparse_cols in config for models
            dc["sparse_cols"] = self.sparse_cols
            dc["num_sparse"] = len(self.sparse_cols)
        else:
            # FIX for Bug #2: Expand vocabulary for test items not in training
            self.feature_maps = TaobaoAdDataset._shared_feature_maps or {}
            self.seq_maps = self._expand_seq_maps(TaobaoAdDataset._shared_seq_maps or {}, rows)
            
            # Use shared cardinalities
            cardinalities = TaobaoAdDataset._shared_cardinalities or [len(m) + 1 for m in self.feature_maps.values()]
            dc["cardinalities"] = cardinalities
            
            # Update num_items to account for expanded vocabulary
            max_seq_vocab = max([len(m) + 1 for m in self.seq_maps.values()]) if self.seq_maps else 1
            dc["num_items"] = max(dc.get("num_items", 1), max_seq_vocab)
            
            # FIX for Bug #8: Store sparse_cols in config
            dc["sparse_cols"] = self.sparse_cols
            dc["num_sparse"] = len(self.sparse_cols)

        # Labels
        self.labels = np.array([int(r["clk"]) for r in rows], dtype=np.float32)

        # Sparse features
        self.sparse = np.array(
            [[self.feature_maps.get(c, {}).get(r.get(c, ""), 0) for c in self.sparse_cols]
             for r in rows],
            dtype=np.int64,
        )

        # Sequence features (cate_his + brand_his, ^ 分隔)
        # FIX for Bug #13: Use separate ID ranges to avoid collision
        self.seqs = np.zeros((len(rows), len(SEQ_COLS), self.maxlen), dtype=np.int64)
        for i, r in enumerate(rows):
            for s, scol in enumerate(SEQ_COLS):
                raw = r.get(scol, "")
                if raw:
                    ids = raw.split("^")[-self.maxlen:]
                    for k, v in enumerate(ids):
                        seq_id = self.seq_maps.get(scol, {}).get(v, 0)
                        self.seqs[i, s, self.maxlen - len(ids) + k] = seq_id

        # Target 1: cate_id → cate_his ID 空间 (DIN 原论文)
        self.targets = np.zeros(len(rows), dtype=np.int64)
        for i, r in enumerate(rows):
            cate_val = r.get("cate_id", "")
            self.targets[i] = self.seq_maps.get("cate_his", {}).get(cate_val, 0)
        dc["num_items"] = max(dc.get("num_items", 1),
                              max(self.seq_maps.get("cate_his", {}).values(), default=0) + 1)

        # Target 2: brand → brand_his ID 空间 (dual-sequence DIN)
        self.targets2 = np.zeros(len(rows), dtype=np.int64)
        for i, r in enumerate(rows):
            brand_val = r.get("brand", "")
            self.targets2[i] = self.seq_maps.get("brand_his", {}).get(brand_val, 0)
        dc["num_items2"] = max(dc.get("num_items2", 1),
                               max(self.seq_maps.get("brand_his", {}).values(), default=0) + 1)

        n_pos = (self.labels > 0.5).sum()
        print(f"[TaobaoAd] {split}: {len(rows):,} samples, "
              f"pos={n_pos:,} ({n_pos / len(rows) * 100:.2f}%)")

    def _expand_seq_maps(self, shared_seq_maps, rows):
        """FIX for Bug #2: Expand vocabulary for test items not in training"""
        seq_maps = {}
        for scol in SEQ_COLS:
            # Start with training vocab
            seq_maps[scol] = (shared_seq_maps.get(scol, {})).copy()
            max_id = max(seq_maps[scol].values()) if seq_maps[scol] else 0
            
            # Add any new items from test set
            for r in rows:
                raw = r.get(scol, "")
                if raw:
                    for v in raw.split("^"):
                        if v and v not in seq_maps[scol]:
                            max_id += 1
                            seq_maps[scol][v] = max_id
                            if max_id % 1000 == 0:
                                print(f"[TaobaoAd] Expanded {scol}: new ID {max_id}")
        
        return seq_maps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # DIN 原论文: cate_his + brand_his 两路独立 attention
        cate_seq = self.seqs[idx, 0]   # (maxlen,) cate_his
        brand_seq = self.seqs[idx, 1]  # (maxlen,) brand_his
        return {
            "sparse": self.sparse[idx],
            "seq": cate_seq,
            "seq2": brand_seq,
            "target": self.targets[idx],
            "target2": self.targets2[idx],
            "label": self.labels[idx],
        }

    def _build_maps(self, rows):
        print("[TaobaoAd] Building feature maps...")
        feature_maps = {}
        cardinalities = []

        for col in self.sparse_cols:
            vals = sorted(set(r.get(col, "") for r in rows))
            mapping = {v: i + 1 for i, v in enumerate(vals) if v}
            feature_maps[col] = mapping
            cardinalities.append(len(mapping) + 1)

        seq_maps = {}
        seq_vocab_size = 0
        for scol in SEQ_COLS:
            vals = set()
            for r in rows:
                raw = r.get(scol, "")
                if raw:
                    for v in raw.split("^"):
                        vals.add(v)
            mapping = {v: i + 1 for i, v in enumerate(sorted(vals))}
            seq_maps[scol] = mapping
            seq_vocab_size = max(seq_vocab_size, len(mapping) + 1)

        return feature_maps, seq_maps, cardinalities, seq_vocab_size
