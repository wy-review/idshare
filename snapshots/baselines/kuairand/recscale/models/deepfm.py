"""
recscale.models.deepfm — DeepFM (Wide + FM + Deep)
"""

import torch
import torch.nn as nn

from . import register_model
from .base import RecModel, DenseArch, SparseArch, FMInteraction


@register_model
class DeepFMModel(RecModel):
    """
    DeepFM: 三路并行
    - Wide: 1st order (per-feature linear)
    - FM: 2nd order (pairwise interaction)
    - Deep: high order (MLP)

    logits = wide_out + fm_out + deep_out
    """
    model_name = "deepfm"

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

        # Shared embedding
        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        self.dense_arch = DenseArch(num_dense, [emb_dim] if num_dense > 0 else [], dropout)
        self.has_dense = num_dense > 0

        # Wide: per-feature 1-dim projection
        self.wide_proj = nn.Linear(num_sparse * emb_dim + self.dense_arch.output_dim, 1)

        # FM interaction: include dense token as a field if present
        num_fm_fields = num_sparse + (1 if self.has_dense else 0)
        self.fm = FMInteraction(num_fm_fields, emb_dim)

        # Deep MLP
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
        self.deep_net = nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        sparse_flat = self.sparse_arch(batch["sparse"])  # (B, S*E)
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])  # (B, S, E)

        parts = [sparse_flat]
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            dense_emb = self.dense_arch(batch["dense"])  # (B, E)
            parts.append(dense_emb)
            # FM: include dense as extra field
            fm_input = torch.cat([sparse_per, dense_emb.unsqueeze(1)], dim=1)  # (B, S+1, E)
        else:
            fm_input = sparse_per

        x_flat = torch.cat(parts, dim=1)

        # Three paths
        wide_out = self.wide_proj(x_flat)       # (B, 1)
        fm_out = self.fm(fm_input)              # (B, 1)
        deep_out = self.deep_net(x_flat)        # (B, 1)

        logits = (wide_out + fm_out + deep_out).squeeze(-1)
        return logits
