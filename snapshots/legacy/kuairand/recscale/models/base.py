"""
recscale.models.base — RecModel 抽象基类 + 共享 building blocks

参考 recommendation_demo/python/models/base.py 的设计:
- DenseArch: Bottom MLP (BatchNorm + ReLU + Dropout)
- SparseArch: Per-feature embedding tables
- RecModel: forward(batch: dict) → logits
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# =============================================================================
# Building Blocks
# =============================================================================

class DenseArch(nn.Module):
    """
    Dense 特征的 Bottom MLP。

    输入 (B, num_dense) → 输出 (B, output_dim)
    支持 num_dense=0 的情况 (如 Avazu 无数值特征)
    """

    def __init__(self, num_dense: int, layer_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.num_dense = num_dense
        if num_dense == 0:
            self.output_dim = 0
            self.mlp = None
            return

        layers = []
        in_dim = num_dense
        for out_dim in layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = layer_dims[-1] if layer_dims else num_dense

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        if self.mlp is None:
            return dense.new_zeros(dense.size(0), 0)
        return self.mlp(dense)


class SparseArch(nn.Module):
    """
    Sparse 特征的 Embedding 层。

    每个 sparse 特征一张独立的 embedding table。
    默认 Init: uniform(-1/sqrt(emb_dim), 1/sqrt(emb_dim))，
    也可显式选择 normal 或 zero，用于稀疏 ID 鲁棒性实验。

    输入 (B, num_sparse) → 输出 (B, num_sparse * emb_dim)
    
    Note: cardinalities parameter should be the actual number of embeddings needed
          (already includes padding_idx=0). This avoids double-counting +1.
    """

    def __init__(
        self,
        num_sparse: int,
        embedding_dim: int,
        cardinalities: list[int],
        use_layer_norm: bool = False,
        embedding_init: str = "uniform",
        embedding_init_std: float = 0.01,
    ):
        super().__init__()
        assert len(cardinalities) == num_sparse
        if embedding_init not in ("uniform", "normal", "zero"):
            raise ValueError(
                "embedding_init must be one of 'uniform', 'normal', or 'zero', "
                f"got {embedding_init!r}"
            )
        if embedding_init == "normal" and embedding_init_std <= 0:
            raise ValueError(
                f"embedding_init_std must be positive for normal init, got {embedding_init_std}"
            )
        self.num_sparse = num_sparse
        self.embedding_dim = embedding_dim
        self.output_dim = num_sparse * embedding_dim
        self.embedding_init = embedding_init
        self.embedding_init_std = float(embedding_init_std)

        init_range = 1.0 / math.sqrt(embedding_dim)
        # FIX for Bug #11: Use cardinality directly, don't add +1
        # cardinalities already includes +1 for padding (indices 0..card-1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim, padding_idx=0)  # 0 = padding/unknown
            for card in cardinalities
        ])
        for emb in self.embeddings:
            if embedding_init == "uniform":
                nn.init.uniform_(emb.weight, -init_range, init_range)
            elif embedding_init == "normal":
                nn.init.normal_(emb.weight, mean=0.0, std=self.embedding_init_std)
            else:
                nn.init.zeros_(emb.weight)
            emb.weight.data[0].zero_()  # padding vector = 0

        self.layer_norm = nn.LayerNorm(embedding_dim) if use_layer_norm else None

    def forward(self, sparse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse: (B, num_sparse) int64
        Returns:
            (B, num_sparse * emb_dim)
        """
        embs = []
        for i, emb in enumerate(self.embeddings):
            e = emb(sparse[:, i])  # (B, emb_dim)
            if self.layer_norm is not None:
                e = self.layer_norm(e)
            embs.append(e)
        return torch.cat(embs, dim=1)  # (B, num_sparse * emb_dim)

    def forward_per_feature(self, sparse: torch.Tensor) -> torch.Tensor:
        """
        返回 per-feature embeddings (B, num_sparse, emb_dim)，
        用于 FM/DCN 等需要 per-feature 交互的模型。
        """
        embs = []
        for i, emb in enumerate(self.embeddings):
            e = emb(sparse[:, i])  # (B, emb_dim)
            if self.layer_norm is not None:
                e = self.layer_norm(e)
            embs.append(e.unsqueeze(1))
        return torch.cat(embs, dim=1)  # (B, num_sparse, emb_dim)


class FeatureFieldPooling(nn.Module):
    """
    将 sparse feature tokens 压缩成更少的 field tokens。

    当前实现使用固定随机分组 + sum pooling：
    - 每个原始 sparse 特征只属于一个 field
    - 所有 field 共享同一个 token dim
    - 缺失/unknown 特征由于 embedding[0]=0，不会在 sum pooling 中引入额外噪声
    """

    def __init__(self, num_features: int, num_fields: int, seed: int = 42):
        super().__init__()
        if not (0 < num_fields <= num_features):
            raise ValueError(
                f"num_fields must be in [1, {num_features}], got {num_fields}"
            )
        self.num_features = num_features
        self.num_fields = num_fields

        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(num_features, generator=generator).tolist()
        groups = [[] for _ in range(num_fields)]
        for idx, feat_idx in enumerate(perm):
            groups[idx % num_fields].append(feat_idx)

        max_group_size = max(len(g) for g in groups)
        group_indices = torch.zeros(num_fields, max_group_size, dtype=torch.long)
        group_mask = torch.zeros(num_fields, max_group_size, dtype=torch.bool)
        for field_idx, feature_indices in enumerate(groups):
            group_indices[field_idx, :len(feature_indices)] = torch.tensor(feature_indices, dtype=torch.long)
            group_mask[field_idx, :len(feature_indices)] = True

        self.register_buffer("group_indices", group_indices)
        self.register_buffer("group_mask", group_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_features, emb_dim) -> (B, num_fields, emb_dim)"""
        grouped = x[:, self.group_indices, :]
        mask = self.group_mask.view(1, self.num_fields, -1, 1).to(dtype=x.dtype)
        return (grouped * mask).sum(dim=2)


class CrossNetwork(nn.Module):
    """
    DCN-V2 Cross Network: x_{l+1} = x_0 * (W_l @ x_l + b_l) + x_l
    """

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.weight = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, input_dim))
            for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
        for w in self.weight:
            nn.init.xavier_normal_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for i in range(self.num_layers):
            xw = torch.matmul(x, self.weight[i])  # (B, D)
            x = x0 * (xw + self.bias[i]) + x
        return x


class FMInteraction(nn.Module):
    """
    FM 二阶交互: 0.5 * sum((sum v_i*x_i)^2 - sum(v_i^2 * x_i^2))
    输入 (B, num_fields, emb_dim) → 输出 (B, 1)
    """

    def __init__(self, num_fields: int = 0, emb_dim: int = 0):
        super().__init__()
        # BF16 安全: scale factor 防止溢出
        self.scale = 1.0 / math.sqrt(max(num_fields * emb_dim, 1))

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """embs: (B, F, E)"""
        sum_sq = embs.sum(dim=1).pow(2).sum(dim=1, keepdim=True)  # (B, 1)
        sq_sum = embs.pow(2).sum(dim=1).sum(dim=1, keepdim=True)  # (B, 1)
        return 0.5 * (sum_sq - sq_sum) * self.scale


# =============================================================================
# RecModel ABC
# =============================================================================

class RecModel(nn.Module, ABC):
    """
    推荐模型抽象基类。

    所有模型子类只需实现:
    1. __init__(config): 定义网络结构
    2. forward(batch: dict) -> logits (B,)
    """

    model_name: str = "base"

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict of tensors, 由 Dataset + collate_fn 生成。
                约定 key:
                - "sparse": (B, S) int64
                - "dense": (B, D) float32 (可选)
                - "seq": (B, L) int64 (可选)
                - "seq_mask": (B, L) bool (可选)
                - "target": (B,) int64 (可选)
                - "mm_emb": (B, E) float32 (可选)
                - "label": (B,) float32
        Returns:
            logits: (B,) float tensor (raw, before sigmoid)
        """
        ...

    def get_num_params(self) -> Dict[str, float]:
        """参数统计 (单位: 百万)"""
        total = sum(p.numel() for p in self.parameters())
        emb = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "embedding" in name.lower() or "sparse" in name.lower() or "item_emb" in name.lower()
        )
        return {
            "total_M": total / 1e6,
            "embedding_M": emb / 1e6,
            "non_embedding_M": (total - emb) / 1e6,
        }
