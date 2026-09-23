"""
recscale.datasets.base — BaseDataset ABC + 通用 collate_fn
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    数据集抽象基类。子类只需实现:
    - __len__()
    - __getitem__(idx) -> dict

    输出 dict 的 key 对齐 RecModel.forward(batch) 约定。
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        返回单条样本的 dict, 例如:
        {
            "sparse": np.array([...], dtype=np.int64),  # (S,)
            "dense": np.array([...], dtype=np.float32),  # (D,) 可选
            "label": np.float32,
        }
        """
        ...

    @property
    def num_samples(self) -> int:
        return len(self)

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        """
        通用 collate: 将 list of dict 合并为 batch dict。
        自动处理:
        - numpy array → tensor (stack)
        - scalar → tensor
        - 变长序列 → pad to max length (RIGHT-aligned to preserve recent history)
        - 多维数组 (e.g., seqs [2, maxlen]) → proper handling
        
        FIX for Bug #5: Changed from LEFT-align to RIGHT-align for sequences.
        This ensures attention models see actual data first, not padding.
        
        FIX for Enhanced DIN: Handle multi-dimensional arrays for dual-sequence
        """
        batch = {}
        keys = samples[0].keys()

        for key in keys:
            vals = [s[key] for s in samples]

            if isinstance(vals[0], np.ndarray):
                # 检查是否变长 (不同 shape)
                shapes = set(v.shape for v in vals)
                if len(shapes) == 1:
                    # 等长, 直接 stack
                    batch[key] = torch.from_numpy(np.stack(vals))
                else:
                    # 变长, pad to max on last dimension
                    # 对于 1D 数组 [L], pad 第一维
                    # 对于 2D 数组 [M, L], pad 最后一维
                    # 对于 3D 数组 [M, N, L], pad 最后一维
                    
                    ndim = vals[0].ndim
                    if ndim == 1:
                        # 1D case: [L] with variable L
                        max_len = max(v.shape[0] for v in vals)
                        padded = np.zeros((len(vals), max_len), dtype=vals[0].dtype)
                        mask = np.zeros((len(vals), max_len), dtype=np.bool_)
                        
                        use_right_align = "seq" in key or "history" in key
                        
                        for i, v in enumerate(vals):
                            if use_right_align:
                                start_idx = max_len - len(v)
                                padded[i, start_idx:] = v
                                mask[i, start_idx:] = True
                            else:
                                padded[i, :len(v)] = v
                                mask[i, :len(v)] = True
                        
                        batch[key] = torch.from_numpy(padded)
                        batch[key + "_mask"] = torch.from_numpy(mask)
                        
                    elif ndim == 2:
                        # 2D case: [L, D] with variable L on first dimension
                        # or [N, L] where N fixed, L variable
                        # Check if first dimension is fixed (for sequences)
                        first_dims = [v.shape[0] for v in vals]
                        if len(set(first_dims)) == 1:
                            # First dim fixed (e.g., num_seq_fields=2 for dual-sequence)
                            # Pad the second dimension (sequence length)
                            N = vals[0].shape[0]
                            max_len = max(v.shape[1] for v in vals)
                            
                            padded = np.zeros((len(vals), N, max_len), dtype=vals[0].dtype)
                            mask = np.zeros((len(vals), N, max_len), dtype=np.bool_)
                            
                            use_right_align = "seq" in key or "history" in key
                            
                            for i, v in enumerate(vals):
                                if use_right_align:
                                    start_idx = max_len - v.shape[1]
                                    padded[i, :, start_idx:] = v
                                    mask[i, :, start_idx:] = True
                                else:
                                    padded[i, :, :v.shape[1]] = v
                                    mask[i, :, :v.shape[1]] = True
                            
                            batch[key] = torch.from_numpy(padded)
                            batch[key + "_mask"] = torch.from_numpy(mask)
                        else:
                            # First dim variable: pad first dimension
                            max_len = max(v.shape[0] for v in vals)
                            feature_dim = vals[0].shape[1]
                            
                            padded = np.zeros((len(vals), max_len, feature_dim), dtype=vals[0].dtype)
                            mask = np.zeros((len(vals), max_len), dtype=np.bool_)
                            
                            use_right_align = "seq" in key or "history" in key
                            
                            for i, v in enumerate(vals):
                                if use_right_align:
                                    start_idx = max_len - v.shape[0]
                                    padded[i, start_idx:, :] = v
                                    mask[i, start_idx:] = True
                                else:
                                    padded[i, :v.shape[0], :] = v
                                    mask[i, :v.shape[0]] = True
                            
                            batch[key] = torch.from_numpy(padded)
                            batch[key + "_mask"] = torch.from_numpy(mask)

            elif isinstance(vals[0], (int, float, np.integer, np.floating)):
                dtype = torch.float32 if key == "label" else torch.long
                batch[key] = torch.tensor(vals, dtype=dtype)

            else:
                # 其他类型直接保留 list
                batch[key] = vals

        return batch


def validate_cardinalities(config: dict, sparse_array: np.ndarray = None, sparse_cols: list = None):
    """
    Validate that dataset properly configured cardinalities (FIX for Bug #14).
    
    This function ensures all datasets:
    1. Define cardinalities in config
    2. Have matching number of cardinalities vs sparse features
    3. Have no out-of-bounds indices that would crash embedding layers
    
    Args:
        config: Dataset config dict with 'dataset' key
        sparse_array: Optional numpy array of sparse features (B, num_sparse)
        sparse_cols: Optional list of sparse column names
    
    Raises:
        AssertionError: If cardinalities improperly configured
        
    Returns:
        bool: True if validation passes
    """
    dc = config.get("dataset", {})
    cardinalities = dc.get("cardinalities", None)
    
    if cardinalities is None:
        raise AssertionError(
            f"Dataset config missing 'cardinalities' key. "
            f"Must set: dc['cardinalities'] = [card1, card2, ...] "
            f"where each cardinality is num_embeddings (including padding)"
        )
    
    if not isinstance(cardinalities, (list, tuple)):
        raise AssertionError(
            f"cardinalities must be list/tuple, got {type(cardinalities)}"
        )
    
    if len(cardinalities) == 0:
        raise AssertionError("cardinalities cannot be empty")
    
    # If sparse features provided, validate cardinalities vs data
    if sparse_array is not None:
        num_sparse = sparse_array.shape[1] if len(sparse_array.shape) > 1 else 1
        if len(cardinalities) != num_sparse:
            raise AssertionError(
                f"cardinalities length {len(cardinalities)} != "
                f"num_sparse features {num_sparse}"
            )
        
        # Check for out-of-bounds indices
        for col_idx, card in enumerate(cardinalities):
            if len(sparse_array) > 0:
                max_idx = int(sparse_array[:, col_idx].max())
                if max_idx >= card:
                    raise AssertionError(
                        f"Column {col_idx}: max index {max_idx} >= cardinality {card}. "
                        f"This will crash the embedding layer!"
                    )
    
    return True
