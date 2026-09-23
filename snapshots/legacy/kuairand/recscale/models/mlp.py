"""
recscale.models.mlp — MLP Baseline
"""

import torch
import torch.nn as nn

from . import register_model
from .base import RecModel, DenseArch, SparseArch


@register_model
class MLPModel(RecModel):
    """
    纯 MLP baseline: dense → DenseArch, sparse → SparseArch, concat → MLP → logit
    """
    model_name = "mlp"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        mlp_dims = mc.get("mlp_dims", [256, 128, 64])
        dropout = mc.get("dropout", 0.0)

        num_dense = len(dc.get("dense_cols") or [])
        num_sparse = len(dc.get("sparse_cols") or [])
        cardinalities = dc.get("cardinalities", [10000] * num_sparse)

        # DenseArch
        dense_out_dim = emb_dim if num_dense > 0 else 0
        self.dense_arch = DenseArch(num_dense, [dense_out_dim] if num_dense > 0 else [], dropout)

        # SparseArch
        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        # Top MLP
        input_dim = self.dense_arch.output_dim + self.sparse_arch.output_dim
        layers = []
        in_dim = input_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        parts = []

        if "dense" in batch and batch["dense"] is not None:
            parts.append(self.dense_arch(batch["dense"]))

        if "sparse" in batch:
            parts.append(self.sparse_arch(batch["sparse"]))

        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.top_mlp(x).squeeze(-1)
