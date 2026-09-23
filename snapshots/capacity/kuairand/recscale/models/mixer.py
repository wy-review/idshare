"""
recscale.models.mixer — Basic Mixer (Token Mixing)

Position-aware feature interaction without attention.

Architecture:
  sparse → SparseArch(per-feature) → (B, T, D)
  dense → project → (B, 1, D)  (optional)
      ↓
  Stack of MixerBlock × L
  ├── Token Mixing: Linear layer mixing tokens (position-aware)
  └── Channel Mixing: Per-token FFN (feature-wise)
  + Pre-LN + Residual
      ↓
  flatten → MLP → logit

Key difference from Attention:
  - No internal product of query/key (no softmax)
  - Simple parameter matrix for token interactions
  - Efficient: O(T²D) vs O(T²) for softmax in attention
"""

import torch
import torch.nn as nn

from . import register_model
from .base import RecModel, DenseArch, SparseArch


class TokenMixingLayer(nn.Module):
    """
    Token Mixing: Mix information across features/tokens
    
    Simple approach: Use a learned linear transformation across tokens.
    Input: (B, T, D) → Linear(T, T) → (B, T, D)
    """

    def __init__(self, num_tokens: int, token_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        # Shared mixing layer across all features
        self.mixing = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        Permute to (B, D, T) for mixing, then back
        """
        # (B, T, D) → (B, D, T)
        x = x.transpose(1, 2)
        # (B, D, T) → Mix across T dimension → (B, D, T)
        x = self.mixing(x)
        # (B, D, T) → (B, T, D)
        x = x.transpose(1, 2)
        return x


class ChannelMixingLayer(nn.Module):
    """
    Channel Mixing: FFN per token
    
    Apply the same MLP to each token independently.
    """

    def __init__(self, token_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        return self.ffn(x)


class MixerBlock(nn.Module):
    """
    Single Mixer block:
    x = x + ChannelMixing(LN(TokenMixing(LN(x))))
    
    Or more commonly (pre-LN):
    x = x + TokenMixing(LN(x))    # token mixing
    x = x + ChannelMixing(LN(x))  # channel mixing
    """

    def __init__(self, num_tokens: int, token_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(token_dim)
        self.token_mixing = TokenMixingLayer(num_tokens, token_dim, dropout)
        
        self.ln2 = nn.LayerNorm(token_dim)
        self.channel_mixing = ChannelMixingLayer(token_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing
        x = x + self.token_mixing(self.ln1(x))
        # Channel mixing
        x = x + self.channel_mixing(self.ln2(x))
        return x


@register_model
class MixerModel(RecModel):
    """
    Mixer: Position-aware token mixing for CTR prediction.
    
    Simpler and more interpretable than RankMixer:
    - No multi-head complexity
    - Direct token-to-token interaction matrix
    - Efficient and suitable for scaling experiments
    
    Config keys:
        embedding_dim: embedding dimension
        num_mixer_layers: number of mixer blocks
        mixer_hidden_dim: hidden dimension in channel mixing FFN
        mlp_dims: top MLP layer dimensions
        dropout: dropout rate
    """
    model_name = "mixer"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_mixer_layers", 2)
        mixer_hidden_dim = mc.get("mixer_hidden_dim", 256)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        # Embedding
        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        self.has_dense = num_dense > 0
        if self.has_dense:
            self.dense_proj = nn.Sequential(
                nn.Linear(num_dense, emb_dim),
                nn.ReLU(),
            )
            num_tokens = num_sparse + 1
        else:
            self.dense_proj = None
            num_tokens = num_sparse

        # Mixer blocks
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(num_tokens, emb_dim, mixer_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Top MLP
        top_layers = []
        in_dim = num_tokens * emb_dim
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        # Per-feature embeddings
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])  # (B, S, E)
        tokens = [sparse_per]

        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            dense_token = self.dense_proj(batch["dense"]).unsqueeze(1)  # (B, 1, E)
            tokens.append(dense_token)

        x = torch.cat(tokens, dim=1)  # (B, T, E)

        # Mixer blocks
        x = self.mixer_blocks(x)  # (B, T, E)

        # Flatten → MLP → logit
        x = x.flatten(start_dim=1)  # (B, T*E)
        return self.top_mlp(x).squeeze(-1)
