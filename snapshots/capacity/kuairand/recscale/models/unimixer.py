"""
recscale.models.unimixer — UniMixer

UniMixer keeps the RankMixer-style multi-head token-mixing backbone, but replaces
fixed interaction rules with a learnable token-token interaction matrix. Each
head owns a normalized interaction matrix over tokens, so feature combinations
are explicitly parameterized instead of being hard-coded as FM/attention/mixer
style branches.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from . import register_model
from .base import FeatureFieldPooling, RecModel, SparseArch


class LearnableInteractionTokenMixing(nn.Module):
    """
    Multi-head token mixing with a learnable interaction matrix.

    1. Split each token embedding into H heads.
    2. For every head, learn a token-token interaction matrix A_h ∈ R^(T×T).
    3. Apply A_h to mix token slices across the token dimension.
    4. Refine the mixed representation with a shared per-head FFN.

    `num_rank` optionally factorizes the interaction logits to reduce parameters.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        ffn_dim: int = 256,
        num_rank: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        assert token_dim % num_heads == 0, (
            f"token_dim {token_dim} must be divisible by num_heads {num_heads}"
        )

        self.rank = num_rank if num_rank is not None and 0 < num_rank < num_tokens else None
        if self.rank is None:
            self.interaction_logits = nn.Parameter(
                torch.empty(num_heads, num_tokens, num_tokens)
            )
            self.interaction_left = None
            self.interaction_right = None
        else:
            self.interaction_logits = None
            self.interaction_left = nn.Parameter(
                torch.empty(num_heads, num_tokens, self.rank)
            )
            self.interaction_right = nn.Parameter(
                torch.empty(num_heads, num_tokens, self.rank)
            )

        self.self_bias = nn.Parameter(torch.zeros(num_heads, num_tokens))

        mixing_dim = num_tokens * self.head_dim
        self.mixing_ffn = nn.Sequential(
            nn.Linear(mixing_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, mixing_dim),
            nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.interaction_logits is not None:
            nn.init.normal_(self.interaction_logits, mean=0.0, std=0.02)
        else:
            nn.init.xavier_uniform_(self.interaction_left)
            nn.init.xavier_uniform_(self.interaction_right)
        nn.init.constant_(self.self_bias, 1.0)

    def _interaction_logits_tensor(self) -> torch.Tensor:
        if self.interaction_logits is not None:
            logits = self.interaction_logits / math.sqrt(self.num_tokens)
        else:
            logits = torch.matmul(
                self.interaction_left,
                self.interaction_right.transpose(-1, -2),
            ) / math.sqrt(self.rank)
        return logits + torch.diag_embed(self.self_bias)

    def get_interaction_matrix(self) -> torch.Tensor:
        return torch.softmax(self._interaction_logits_tensor(), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        batch_size, num_tokens, token_dim = x.shape
        assert num_tokens == self.num_tokens, (
            f"Expected {self.num_tokens} tokens, got {num_tokens}"
        )
        assert token_dim == self.token_dim, (
            f"Expected token dim {self.token_dim}, got {token_dim}"
        )

        x_heads = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        x_heads = x_heads.permute(0, 2, 1, 3)  # (B, H, T, Dh)

        interaction = self.get_interaction_matrix()  # (H, T, T)
        x_mixed = torch.einsum("hts,bhsd->bhtd", interaction, x_heads)

        x_flat = x_mixed.reshape(batch_size * self.num_heads, num_tokens * self.head_dim)
        x_flat = self.mixing_ffn(x_flat)
        x_mixed = x_flat.view(batch_size, self.num_heads, num_tokens, self.head_dim)

        return x_mixed.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, token_dim)


class PerTokenFFN(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int, ffn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, token_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_tokens)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {x.size(1)}")
        outputs = [ffn(x[:, i, :]).unsqueeze(1) for i, ffn in enumerate(self.ffns)]
        return torch.cat(outputs, dim=1)


class UniMixerBlock(nn.Module):
    """
    Single UniMixer block:
    x = x + LearnableInteractionTokenMixing(LN(x))
    x = x + PerTokenFFN(LN(x))
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        ffn_dim: int = 256,
        num_rank: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(token_dim)
        self.token_mixing = LearnableInteractionTokenMixing(
            num_tokens=num_tokens,
            token_dim=token_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_rank=num_rank,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(token_dim)
        self.channel_mixing = PerTokenFFN(num_tokens, token_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(self.ln1(x))
        x = x + self.channel_mixing(self.ln2(x))
        return x


@register_model
class UniMixerModel(RecModel):
    """
    UniMixer: RankMixer-like mixer backbone with learnable token interaction.

    Config keys:
        embedding_dim: token embedding dimension
        num_mixer_layers: number of UniMixer blocks
        num_heads: number of token-mixing heads
        ffn_dim: hidden dimension for token/channel FFNs
        num_rank: optional low-rank factorization rank for interaction logits
        mlp_dims: top MLP layer dimensions
        dropout: dropout rate
    """

    model_name = "unimixer"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_mixer_layers", 2)
        ffn_dim = mc.get("ffn_dim", mc.get("mixer_hidden_dim", 256))
        num_rank = mc.get("num_rank", mc.get("interaction_rank", None))
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        num_feature_fields = mc.get("num_feature_fields")
        if num_feature_fields is not None and 0 < num_feature_fields < num_sparse:
            self.feature_field_pool = FeatureFieldPooling(
                num_features=num_sparse,
                num_fields=num_feature_fields,
                seed=config.get("seed", 42),
            )
            sparse_token_count = num_feature_fields
        else:
            self.feature_field_pool = None
            sparse_token_count = num_sparse

        self.has_dense = num_dense > 0
        if self.has_dense:
            self.dense_proj = nn.Sequential(
                nn.Linear(num_dense, emb_dim),
                nn.ReLU(),
            )
            num_tokens = sparse_token_count + 1
        else:
            self.dense_proj = None
            num_tokens = sparse_token_count

        num_heads = mc.get("num_heads", None)
        if num_heads is None:
            for candidate in range(min(num_tokens, emb_dim), 0, -1):
                if emb_dim % candidate == 0:
                    num_heads = candidate
                    break
        if num_heads is None:
            raise ValueError("Could not infer a valid num_heads value")

        self.mixer_blocks = nn.Sequential(
            *[
                UniMixerBlock(
                    num_tokens=num_tokens,
                    token_dim=emb_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    num_rank=num_rank,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

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
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])
        if self.feature_field_pool is not None:
            sparse_per = self.feature_field_pool(sparse_per)
        tokens = [sparse_per]

        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            dense_token = self.dense_proj(batch["dense"]).unsqueeze(1)
            tokens.append(dense_token)

        x = torch.cat(tokens, dim=1)
        x = self.mixer_blocks(x)
        x = x.flatten(start_dim=1)
        return self.top_mlp(x).squeeze(-1)
