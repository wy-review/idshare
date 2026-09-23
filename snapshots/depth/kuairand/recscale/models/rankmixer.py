"""
recscale.models.rankmixer — RankMixer

"Scaling Up Ranking Models in Industrial Recommenders"
Paper: https://arxiv.org/abs/2507.15551

核心思想: 用 Multi-Head Token Mixing 替代 Transformer self-attention，
避免异构特征空间的内积相似度计算问题，同时大幅提升 MFU (4.5% → 45%)。

Architecture:
  sparse → SparseArch(per-feature) → (B, T, D)
  dense → project → (B, 1, D)  (optional)
      ↓
  Stack of RankMixerBlock × L
  ├── Multi-Head Token Mixing: shuffle heads across tokens → FFN
  └── Per-Token FFN (channel mixing)
  + Pre-LN + Residual
      ↓
  flatten → MLP → logit

Token Mixing 关键区别于 Attention:
  - Attention: Q@K^T 计算相似度 → softmax → V, 需要内积有意义
  - TokenMix: 参数无关地重排 head → FFN 学习交互, 无需内积假设
"""

import torch
import torch.nn as nn

from . import register_model
from .base import FeatureFieldPooling, RecModel, DenseArch, SparseArch


class TokenMixingLayer(nn.Module):
    """
    Multi-Head Token Mixing (核心组件)

    1. 每个 token 切成 H 个 head: (B, T, D) → (B, T, H, D//H)
    2. 按 head 维度重组: (B, H, T, D//H)
    3. 对每个 head: 将所有 token 的片段拼在一起做 FFN
       → (B, H, T*(D//H)) → FFN → (B, H, T*(D//H))
    4. 还原: (B, T, D)

    设 H=T (论文推荐), 则 head_dim = D//T
    """

    def __init__(self, num_tokens: int, token_dim: int, num_heads: int = None,
                 ffn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads or num_tokens  # H=T by default
        self.head_dim = token_dim // self.num_heads
        assert token_dim % self.num_heads == 0, \
            f"token_dim {token_dim} must be divisible by num_heads {self.num_heads}"

        # Per-head FFN: input = T * head_dim (all tokens' h-th head concatenated)
        mixing_dim = num_tokens * self.head_dim
        self.mixing_ffn = nn.Sequential(
            nn.Linear(mixing_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, mixing_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) → (B, T, D)
        """
        B, T, D = x.shape
        H = self.num_heads

        # Split into heads: (B, T, H, head_dim)
        x_heads = x.view(B, T, H, self.head_dim)

        # Rearrange: group by head → (B, H, T, head_dim)
        x_heads = x_heads.permute(0, 2, 1, 3)

        # Concat all tokens per head: (B, H, T*head_dim)
        x_concat = x_heads.reshape(B, H, T * self.head_dim)

        # Apply shared FFN per head (broadcasts over H)
        # Process all heads together: (B*H, T*head_dim)
        x_flat = x_concat.reshape(B * H, T * self.head_dim)
        x_mixed = self.mixing_ffn(x_flat)
        x_mixed = x_mixed.reshape(B, H, T * self.head_dim)

        # Restore: (B, H, T, head_dim) → (B, T, H, head_dim) → (B, T, D)
        x_mixed = x_mixed.view(B, H, T, self.head_dim)
        x_mixed = x_mixed.permute(0, 2, 1, 3).reshape(B, T, D)

        return x_mixed


class PerTokenFFN(nn.Module):
    """
    Per-Token FFN (channel mixing): 每个 token 位置有独立的 FFN 参数。

    区别于共享 FFN（nn.Linear 对所有 token 广播同一套权重）：
    - 共享 FFN：T 个 token 用同一个 Linear(D, ffn_dim)，权重共享
    - Per-Token FFN：T 个 token 各自有独立的 Linear，参数量 × T

    实现：用 (T, D, ffn_dim) 的参数矩阵 + einsum 实现逐 token 矩阵乘
    （数学等价于 MR 的 ModuleList 版，但 einsum 实现快 ~1.4x，无 Python 循环）
    """

    def __init__(self, num_tokens: int, token_dim: int,
                 ffn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.ffn_dim = ffn_dim

        # 每个 token 独立的 up-projection 和 down-projection
        # w1: (T, D, ffn_dim)  w2: (T, ffn_dim, D)
        self.w1 = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, ffn_dim))
        self.w2 = nn.Parameter(torch.empty(num_tokens, ffn_dim, token_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, token_dim))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init
        nn.init.kaiming_uniform_(self.w1, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w2, a=0, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) → (B, T, D)

        每个 token t 独立计算：
          h = GELU(x[:, t, :] @ w1[t] + b1[t])    (B, ffn_dim)
          out = h @ w2[t] + b2[t]                   (B, D)
        """
        # up: (B, T, D) × (T, D, ffn_dim) → (B, T, ffn_dim)
        h = torch.einsum("btd,tdf->btf", x, self.w1) + self.b1  # (B, T, ffn_dim)
        h = self.act(h)
        h = self.dropout(h)
        # down: (B, T, ffn_dim) × (T, ffn_dim, D) → (B, T, D)
        out = torch.einsum("btf,tfd->btd", h, self.w2) + self.b2  # (B, T, D)
        out = self.dropout(out)
        return out


class RankMixerBlock(nn.Module):
    """
    Single RankMixer block:
    x = x + TokenMixing(LN(x))    # token mixing (cross-token)
    x = x + PerTokenFFN(LN(x))    # channel mixing (per-token independent FFN)
    """

    def __init__(self, num_tokens: int, token_dim: int, num_heads: int = None,
                 ffn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(token_dim)
        self.token_mixing = TokenMixingLayer(
            num_tokens, token_dim, num_heads, ffn_dim, dropout
        )
        self.ln2 = nn.LayerNorm(token_dim)
        # Per-token FFN：每个 token 位置有独立参数（einsum 实现）
        self.channel_mixing = PerTokenFFN(num_tokens, token_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(self.ln1(x))
        x = x + self.channel_mixing(self.ln2(x))
        return x


@register_model
class RankMixerModel(RecModel):
    """
    RankMixer: Token Mixing for efficient CTR prediction.

    Supports batch keys: "sparse", "dense" (optional), "label"
    """
    model_name = "rankmixer"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_mixer_layers", 3)
        ffn_dim = mc.get("ffn_dim", 256)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        # Embedding
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

        # Ensure emb_dim is divisible by num_tokens for head splitting
        # If not, use a smaller num_heads
        num_heads = mc.get("num_heads", None)
        if num_heads is None:
            # Find largest divisor of emb_dim that is <= num_tokens
            for h in range(min(num_tokens, emb_dim), 0, -1):
                if emb_dim % h == 0:
                    num_heads = h
                    break

        # RankMixer blocks
        self.mixer_blocks = nn.Sequential(*[
            RankMixerBlock(num_tokens, emb_dim, num_heads, ffn_dim, dropout)
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
        if self.feature_field_pool is not None:
            sparse_per = self.feature_field_pool(sparse_per)
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
