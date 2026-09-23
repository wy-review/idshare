"""
recscale.models.wukong — Wukong (ICML 2024)

"Towards a Scaling Law for Large-Scale Recommendation"
Paper: https://arxiv.org/abs/2403.02545
Reference: FuxiCTR model_zoo/WuKong/src/WuKong.py

核心思想: 堆叠 FMB (Factorization Machine Block) + LCB (Linear Compression Block)，
每层捕获 1~2^i 阶交互，通过 binary exponentiation 实现指数阶交互增长。

Architecture:
  sparse → SparseArch(per-feature) → (B, F, E)
  dense → DenseArch → project → (B, 1, E)  (optional)
  concat → (B, N, E)
      ↓
  Stack of WuKongLayer × L
  ├── FMB: X@X^T@Y → LN → MLP → (B, fmb_features, E)
  └── LCB: W_L @ X  →  (B, lcb_features, E)
  concat + residual + LayerNorm
      ↓
  flatten → MLP → logit
"""

import math

import torch
import torch.nn as nn

from . import register_model
from .base import RecModel, DenseArch, SparseArch


class FactorizationMachineBlock(nn.Module):
    """
    FMB: Optimized FM + LayerNorm + MLP

    FM(X) = X @ X^T (n×n interaction matrix)
    Optimized: X @ (X^T @ Y) where Y ∈ R^{n×k}, k << n

    Output: reshape(MLP(LN(flatten(FM(X))))) → (B, output_features, emb_dim)
    """

    def __init__(self, input_features: int, output_features: int, embedding_dim: int,
                 rank_k: int = 8, mlp_dims: list = None, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_features = output_features
        self.rank_k = rank_k

        # Low-rank projection for optimized FM
        if rank_k is not None and rank_k > 0:
            self.proj_Y = nn.Parameter(torch.randn(input_features, rank_k))
            nn.init.xavier_normal_(self.proj_Y)
            fm_out_dim = input_features * rank_k
        else:
            self.proj_Y = None
            fm_out_dim = input_features * input_features

        self.layer_norm = nn.LayerNorm(fm_out_dim)

        # MLP: fm_out_dim → output_features * embedding_dim
        mlp_dims = mlp_dims or [64, 64]
        layers = []
        in_dim = fm_out_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim
        layers.append(nn.Linear(in_dim, output_features * embedding_dim))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, E) — N feature embeddings of dim E
        Returns:
            (B, output_features, E)
        """
        # Optimized FM
        if self.proj_Y is not None:
            # X @ (X^T @ Y): (B,N,E) @ ((B,E,N) @ (N,k)) = (B,N,E) @ (B,E,k) = (B,N,k)
            projected = torch.matmul(x.transpose(1, 2), self.proj_Y)  # (B, E, k)
            fm_matrix = torch.bmm(x, projected)  # (B, N, k)
        else:
            fm_matrix = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)

        flat = fm_matrix.flatten(start_dim=1)  # (B, N*k or N*N)
        normed = self.layer_norm(flat)
        out = self.mlp(normed)  # (B, output_features * E)
        return out.view(-1, self.output_features, self.embedding_dim)


class LinearCompressionBlock(nn.Module):
    """
    LCB: Linear recombination without increasing interaction orders.
    W_L ∈ R^{output_features × input_features}

    X (B, N, E) → transpose → (B, E, N) → Linear → (B, E, out) → transpose → (B, out, E)
    """

    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, E) → (B, output_features, E)"""
        out = self.linear(x.transpose(1, 2))  # (B, E, out)
        return out.transpose(1, 2)  # (B, out, E)


class WuKongLayer(nn.Module):
    """
    Single Wukong interaction layer:
    X_{i+1} = LN(concat(FMB(X_i), LCB(X_i)) + residual(X_i))

    Layer i captures interactions from order 1 to 2^i.
    """

    def __init__(self, input_features: int, lcb_features: int, fmb_features: int,
                 embedding_dim: int, fmb_rank_k: int = 8,
                 fmb_mlp_dims: list = None, dropout: float = 0.0):
        super().__init__()
        self.output_features = lcb_features + fmb_features

        self.fmb = FactorizationMachineBlock(
            input_features, fmb_features, embedding_dim,
            rank_k=fmb_rank_k, mlp_dims=fmb_mlp_dims, dropout=dropout
        )
        self.lcb = LinearCompressionBlock(input_features, lcb_features)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Residual projection if dimensions don't match
        if input_features != self.output_features:
            self.residual_proj = nn.Linear(input_features, self.output_features, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, E) → (B, lcb+fmb, E)"""
        fmb_out = self.fmb(x)   # (B, fmb_features, E)
        lcb_out = self.lcb(x)   # (B, lcb_features, E)
        concat_out = torch.cat([fmb_out, lcb_out], dim=1)  # (B, fmb+lcb, E)

        # Residual connection
        if self.residual_proj is not None:
            res = self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
        else:
            res = x

        out = self.layer_norm(concat_out + res)
        return out


@register_model
class WukongModel(RecModel):
    """
    Wukong: Stacked Factorization Machine layers for scalable CTR prediction.

    Supports batch keys: "sparse", "dense" (optional), "label"
    """
    model_name = "wukong"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_wukong_layers", 3)
        lcb_features = mc.get("lcb_features", 40)
        fmb_features = mc.get("fmb_features", 40)
        fmb_rank_k = mc.get("fmb_rank_k", 8)
        fmb_mlp_dims = mc.get("fmb_mlp_dims", [32, 32])
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        # Embedding layers
        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        # Dense → project to (1, emb_dim) token
        self.has_dense = num_dense > 0
        if self.has_dense:
            self.dense_proj = nn.Sequential(
                nn.Linear(num_dense, emb_dim),
                nn.ReLU(),
            )
            num_fields = num_sparse + 1  # sparse fields + 1 dense token
        else:
            self.dense_proj = None
            num_fields = num_sparse

        # Wukong stacked layers
        output_features = lcb_features + fmb_features
        layers = []
        for i in range(num_layers):
            in_features = num_fields if i == 0 else output_features
            layers.append(WuKongLayer(
                input_features=in_features,
                lcb_features=lcb_features,
                fmb_features=fmb_features,
                embedding_dim=emb_dim,
                fmb_rank_k=fmb_rank_k,
                fmb_mlp_dims=fmb_mlp_dims,
                dropout=dropout,
            ))
        self.wukong_stack = nn.Sequential(*layers)

        # Top MLP: flatten → prediction
        top_layers = []
        in_dim = output_features * emb_dim
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        # Sparse embeddings: per-feature (B, S, E)
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])  # (B, S, E)

        tokens = [sparse_per]

        # Dense → single token
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            dense_token = self.dense_proj(batch["dense"]).unsqueeze(1)  # (B, 1, E)
            tokens.append(dense_token)

        x = torch.cat(tokens, dim=1)  # (B, N, E)

        # Wukong layers
        x = self.wukong_stack(x)  # (B, lcb+fmb, E)

        # Flatten → MLP → logit
        x = x.flatten(start_dim=1)  # (B, (lcb+fmb)*E)
        return self.top_mlp(x).squeeze(-1)
