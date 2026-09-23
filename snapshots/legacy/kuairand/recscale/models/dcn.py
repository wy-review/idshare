"""
recscale.models.dcn — DCN-V2 (Deep & Cross Network)
"""

import torch
import torch.nn as nn

from . import register_model
from .base import RecModel, DenseArch, SparseArch, CrossNetwork


@register_model
class DCNModel(RecModel):
    """
    DCN-V2: Parallel Cross + Deep paths.

    concat(dense_emb, sparse_emb) → [CrossNetwork || DeepMLP] → prediction head
    """
    model_name = "dcn"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_cross_layers = mc.get("num_cross_layers", 3)
        mlp_dims = mc.get("mlp_dims", [256, 128, 64])
        dropout = mc.get("dropout", 0.0)

        num_dense = len(dc.get("dense_cols") or [])
        num_sparse = len(dc.get("sparse_cols") or [])
        cardinalities = dc.get("cardinalities", [10000] * num_sparse)

        # Embedding layers
        dense_out_dim = emb_dim if num_dense > 0 else 0
        self.dense_arch = DenseArch(num_dense, [dense_out_dim] if num_dense > 0 else [], dropout)
        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        input_dim = self.dense_arch.output_dim + self.sparse_arch.output_dim

        # Cross path
        self.cross_net = CrossNetwork(input_dim, num_cross_layers)

        # Deep path
        layers = []
        in_dim = input_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.deep_net = nn.Sequential(*layers)

        # Prediction head
        self.pred_head = nn.Linear(input_dim + mlp_dims[-1], 1)

    def forward(self, batch: dict) -> torch.Tensor:
        parts = []
        if "dense" in batch and batch["dense"] is not None:
            parts.append(self.dense_arch(batch["dense"]))
        if "sparse" in batch:
            parts.append(self.sparse_arch(batch["sparse"]))

        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        cross_out = self.cross_net(x)       # (B, input_dim)
        deep_out = self.deep_net(x)         # (B, mlp_dims[-1])

        combined = torch.cat([cross_out, deep_out], dim=1)
        return self.pred_head(combined).squeeze(-1)
