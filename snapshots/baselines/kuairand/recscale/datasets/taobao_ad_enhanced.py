"""
recscale.datasets.taobao_ad_enhanced — Enhanced Taobao Ad CTR dataset

Improvements:
1. Support dual-sequence processing (cate_his + brand_his)
2. Proper target alignment (cate_id for cate_his, brand for brand_his)
3. Backward compatibility with single-sequence mode
4. Proper ID mapping to avoid collisions
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


@register_dataset("taobao_ad_enhanced")
class TaobaoAdEnhancedDataset(BaseDataset):
    """
    Enhanced TaobaoAd dataset supporting dual-sequence DIN attention.
    
    Each row returns:
    - sparse: [B, 16] (9 user + 7 ad features)
    - seqs: [B, 2, maxlen] (cate_his + brand_his sequences)
    - targets: [B, 2] (cate_id target for cate_his, brand target for brand_his)
    - label: [B]
    """

    # Class-level shared feature maps
    _shared_feature_maps = None
    _shared_seq_maps = None
    _shared_cardinalities = None

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        data_path = dc["path"]
        self.maxlen = dc.get("maxlen", 50)
        max_rows = dc.get("max_rows", 0)
        self.sparse_cols = USER_SPARSE + AD_SPARSE
        
        # Dual-sequence mode flag
        self.dual_sequence = dc.get("dual_sequence", True)

        csv_name = "train.csv" if split == "train" else "test.csv"
        print(f"[TaobaoAdEnhanced] Loading {split}: {csv_name} from {data_path}")

        rows = []
        with zipfile.ZipFile(data_path) as z:
            with z.open(csv_name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for i, row in enumerate(reader):
                    if 0 < max_rows <= i:
                        break
                    rows.append(row)

        print(f"[TaobaoAdEnhanced] Loaded {len(rows):,} rows")

        # Feature maps
        if split == "train":
            self.feature_maps, self.seq_maps, cardinalities, seq_vocab = self._build_maps(rows)
            TaobaoAdEnhancedDataset._shared_feature_maps = self.feature_maps
            TaobaoAdEnhancedDataset._shared_seq_maps = self.seq_maps
            TaobaoAdEnhancedDataset._shared_cardinalities = cardinalities
            dc["cardinalities"] = cardinalities
            dc["num_items"] = seq_vocab
            dc["sparse_cols"] = self.sparse_cols
            dc["num_sparse"] = len(self.sparse_cols)
            dc["num_seq_fields"] = 2 if self.dual_sequence else 1
            dc["seq_field_names"] = SEQ_COLS if self.dual_sequence else ["cate_his"]
        else:
            self.feature_maps = TaobaoAdEnhancedDataset._shared_feature_maps or {}
            self.seq_maps = self._expand_seq_maps(TaobaoAdEnhancedDataset._shared_seq_maps or {}, rows)
            
            cardinalities = TaobaoAdEnhancedDataset._shared_cardinalities or [len(m) + 1 for m in self.feature_maps.values()]
            dc["cardinalities"] = cardinalities
            
            max_seq_vocab = max([len(m) + 1 for m in self.seq_maps.values()]) if self.seq_maps else 1
            dc["num_items"] = max(dc.get("num_items", 1), max_seq_vocab)
            
            dc["sparse_cols"] = self.sparse_cols
            dc["num_sparse"] = len(self.sparse_cols)
            dc["num_seq_fields"] = 2 if self.dual_sequence else 1
            dc["seq_field_names"] = SEQ_COLS if self.dual_sequence else ["cate_his"]

        # Labels
        self.labels = np.array([int(r["clk"]) for r in rows], dtype=np.float32)

        # Sparse features
        self.sparse = np.array(
            [[self.feature_maps.get(c, {}).get(r.get(c, ""), 0) for c in self.sparse_cols]
             for r in rows],
            dtype=np.int64,
        )

        # Sequence features
        if self.dual_sequence:
            # Store both sequences: cate_his and brand_his
            self.seqs = np.zeros((len(rows), 2, self.maxlen), dtype=np.int64)
            for i, r in enumerate(rows):
                for s, scol in enumerate(SEQ_COLS):
                    raw = r.get(scol, "")
                    if raw:
                        ids = raw.split("^")[-self.maxlen:]
                        for k, v in enumerate(ids):
                            seq_id = self.seq_maps.get(scol, {}).get(v, 0)
                            self.seqs[i, s, self.maxlen - len(ids) + k] = seq_id
        else:
            # Single sequence mode (backward compatible)
            self.seqs = np.zeros((len(rows), self.maxlen), dtype=np.int64)
            for i, r in enumerate(rows):
                raw = r.get("cate_his", "")
                if raw:
                    ids = raw.split("^")[-self.maxlen:]
                    for k, v in enumerate(ids):
                        seq_id = self.seq_maps.get("cate_his", {}).get(v, 0)
                        self.seqs[i, self.maxlen - len(ids) + k] = seq_id

        # Targets: map sparse feature values to sequence ID spaces
        if self.dual_sequence:
            self.targets = np.zeros((len(rows), 2), dtype=np.int64)
            for i, r in enumerate(rows):
                # Target 1: cate_id → cate_his ID space
                cate_val = r.get("cate_id", "")
                self.targets[i, 0] = self.seq_maps.get("cate_his", {}).get(cate_val, 0)
                
                # Target 2: brand → brand_his ID space
                brand_val = r.get("brand", "")
                self.targets[i, 1] = self.seq_maps.get("brand_his", {}).get(brand_val, 0)
        else:
            # Single target mode
            self.targets = np.zeros(len(rows), dtype=np.int64)
            for i, r in enumerate(rows):
                cate_val = r.get("cate_id", "")
                self.targets[i] = self.seq_maps.get("cate_his", {}).get(cate_val, 0)

        # Update num_items for all sequence fields
        for scol in SEQ_COLS:
            max_val = max(self.seq_maps.get(scol, {}).values(), default=0)
            dc["num_items"] = max(dc.get("num_items", 1), max_val + 1)

        n_pos = (self.labels > 0.5).sum()
        print(f"[TaobaoAdEnhanced] {split}: {len(rows):,} samples, "
              f"pos={n_pos:,} ({n_pos / len(rows) * 100:.2f}%), "
              f"dual_sequence={self.dual_sequence}")

    def _expand_seq_maps(self, shared_seq_maps, rows):
        """Expand vocabulary for test items not in training."""
        seq_maps = {}
        for scol in SEQ_COLS:
            seq_maps[scol] = (shared_seq_maps.get(scol, {})).copy()
            max_id = max(seq_maps[scol].values()) if seq_maps[scol] else 0
            
            for r in rows:
                raw = r.get(scol, "")
                if raw:
                    for v in raw.split("^"):
                        if v and v not in seq_maps[scol]:
                            max_id += 1
                            seq_maps[scol][v] = max_id
                            if max_id % 1000 == 0:
                                print(f"[TaobaoAdEnhanced] Expanded {scol}: new ID {max_id}")
        
        return seq_maps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.dual_sequence:
            return {
                "sparse": self.sparse[idx],
                "seqs": self.seqs[idx],  # (2, maxlen)
                "targets": self.targets[idx],  # (2,)
                "label": self.labels[idx],
            }
        else:
            return {
                "sparse": self.sparse[idx],
                "seq": self.seqs[idx],  # (maxlen,)
                "target": self.targets[idx],  # scalar
                "label": self.labels[idx],
            }

    def _build_maps(self, rows):
        """Build feature and sequence vocabulary maps."""
        print("[TaobaoAdEnhanced] Building feature maps...")
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
