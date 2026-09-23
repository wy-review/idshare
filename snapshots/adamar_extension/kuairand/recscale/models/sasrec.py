"""
recscale.models.sasrec — SASRec (Self-Attentive Sequential Recommendation) CTR variant

Adapted from SASRec (Kang & McAuley, ICDM 2018) for pointwise CTR prediction:
instead of next-item prediction with sampled softmax, we use the last (or pooled)
self-attended hidden state together with the target item embedding to predict
binary click label.

Architecture:
  seq → item_emb + pos_emb → [self-attn blocks × N] → seq_enc (B, L, D)
  target → item_emb → target_emb (B, D)
  sparse → SparseArch → sparse_emb (B, S·E)
  repr = concat(seq_enc_last, target_emb, sparse_emb[,dense]) → MLP → logit

Supports batch keys: "sparse", "seq", "target", optional "dense".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, SparseArch


class SASRecBlock(nn.Module):
    """Pre-LN self-attention block with causal mask (classic SASRec)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-LN
        h = self.norm1(x)
        h_attn, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + h_attn
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


@register_model
class SASRecModel(RecModel):
    """
    SASRec for CTR.

    Config keys:
      embedding_dim:  sparse / item embedding size (=hidden_dim for simplicity)
      hidden_dim:     self-attention width
      num_layers:     # self-attention blocks
      num_heads:      attention heads
      ffn_dim:        FFN hidden size (default 2*hidden_dim)
      mlp_dims:       top MLP hidden sizes
      dropout
      max_seq_len:    needed for positional embedding
      pool_strategy:  "last" | "mean" | "target_attn" (default "last")
    """
    model_name = "sasrec"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        hidden_dim = mc.get("hidden_dim", emb_dim)
        num_layers = mc.get("num_layers", 2)
        num_heads = mc.get("num_heads", 2)
        ffn_dim = mc.get("ffn_dim", hidden_dim * 2)
        dropout = mc.get("dropout", 0.1)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        self.max_seq_len = mc.get("max_seq_len", dc.get("maxlen", 1024))
        self.pool_strategy = mc.get("pool_strategy", "last")

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None
        self.has_dense = num_dense > 0

        # item embedding — shared between seq and target
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.item_emb.weight.data[0].zero_()

        # positional embedding (learnable)
        self.pos_emb = nn.Embedding(self.max_seq_len, hidden_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.emb_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # top MLP
        in_dim = num_sparse * emb_dim + (num_dense if self.has_dense else 0)
        in_dim += hidden_dim  # seq pooled
        in_dim += hidden_dim  # target emb
        layers = []
        for d in mlp_dims:
            layers += [nn.Linear(in_dim, d), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = d
        layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*layers)

    def _causal_mask(self, L, device):
        # Upper-triangular mask: True means "cannot attend"
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]  # (B, L)
        B, L = seq.shape

        # item + positional
        x = self.item_emb(seq)  # (B, L, D)
        pos = torch.arange(L, device=seq.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.input_norm(self.emb_dropout(x))

        # masks
        pad_mask = (seq == 0)  # True at padding
        causal = self._causal_mask(L, seq.device)

        # NaN-safe: for rows where entire seq is padded, PyTorch MultiheadAttention
        # returns NaN (softmax over all -inf). We mark those rows as fully non-pad for
        # attention purposes, then zero out their attention output afterwards.
        all_pad = pad_mask.all(dim=1)  # (B,)
        safe_pad_mask = pad_mask.clone()
        if all_pad.any():
            # For fully-pad rows, pretend position 0 is not pad (will zero out later)
            safe_pad_mask[all_pad, 0] = False

        for blk in self.blocks:
            x = blk(x, attn_mask=causal, key_padding_mask=safe_pad_mask)

        # Zero out fully-pad rows (they contribute nothing)
        if all_pad.any():
            x = x.masked_fill(all_pad.view(-1, 1, 1), 0.0)

        # Extra safety: NaN / Inf guard (covers any lingering numerical issue)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.final_norm(x)

        # pool
        if self.pool_strategy == "last":
            # last non-pad position; fallback to zeros for all-pad rows
            # Find last non-pad position per batch row
            valid = (~pad_mask).float()  # (B, L)
            # Use arange × valid, take max index per row; for all-pad rows default to 0
            idx = (valid * torch.arange(L, device=seq.device).float().unsqueeze(0)).argmax(dim=1)
            # argmax ties at index 0 when all zeros; that's fine since we'll zero those rows
            seq_repr = x[torch.arange(B, device=seq.device), idx]
            # Zero out rows with all-pad sequences
            seq_repr = seq_repr * (~all_pad).float().unsqueeze(-1)
        elif self.pool_strategy == "mean":
            valid = (~pad_mask).float().unsqueeze(-1)
            seq_repr = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            seq_repr = x[:, -1, :]

        # target
        target = batch.get("target")
        if target is not None:
            target_emb = self.item_emb(target)
        else:
            target_emb = torch.zeros(
                B, self.item_emb.embedding_dim, device=seq.device
            )

        # sparse / dense
        parts = [seq_repr, target_emb]
        if self.sparse_arch is not None and "sparse" in batch:
            parts.insert(0, self.sparse_arch(batch["sparse"]))
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            parts.insert(1, batch["dense"])
        final = torch.cat(parts, dim=1)

        return self.top_mlp(final).squeeze(-1)
