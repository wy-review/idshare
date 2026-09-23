"""
recscale.models.rankmixer_v2 — RankMixer V2（论文正确实现）

严格遵循论文 "Scaling Up Ranking Models in Industrial Recommenders"
(https://arxiv.org/abs/2507.15551) 的设计：

核心约束：H = T（头数 = token 数），且 D 能被 T 整除（head_dim = D/T）

Token Mixing 是纯参数无关的重排操作（permute），无 FFN 参数。
非线性变换完全由 Per-Token FFN（channel mixing）提供。

架构：
  sparse(S fields) → SparseArch → (B, S, D)
                 → FeatureFieldPooling → (B, T, D)   T | D, T <= S
       ↓
  Stack of RankMixerV2Block × L
  ├── Token Mixing: 纯 permute，参数无关   X.view(B,T,T,dh).permute(0,2,1,3).reshape(B,T,D)
  └── Per-Token FFN: 每个 token 独立参数（einsum 实现）
  + Pre-LN + Residual
       ↓
  flatten / mean pooling → MLP → logit
"""

import torch
import torch.nn as nn

from . import register_model
from .base import FeatureFieldPooling, RecModel, SparseArch


class TokenMixingV2(nn.Module):
    """
    Multi-Head Token Mixing（论文正确版本）

    H = T（强制），head_dim = D / T。
    纯参数无关的重排：把 (B, T, D) 视为 (B, T_token, H=T, dh)，
    交换 T_token 和 H 两个维度，得到新的 (B, T, D)。

    等价于：
      x.view(B, T, T, dh).permute(0, 2, 1, 3).reshape(B, T, D)

    没有任何参数。交互能力完全来自后面的 Per-Token FFN。
    """

    def __init__(self, num_tokens: int, token_dim: int):
        super().__init__()
        assert token_dim % num_tokens == 0, (
            f"D={token_dim} must be divisible by T={num_tokens} (H=T constraint)"
        )
        self.T = num_tokens
        self.head_dim = token_dim // num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)，纯 permute，无参数"""
        B, T, D = x.shape
        dh = self.head_dim
        # view as (B, T_token, H=T, dh), swap T_token and H dimensions
        return x.view(B, T, T, dh).permute(0, 2, 1, 3).reshape(B, T, D)


class PerTokenFFN(nn.Module):
    """
    Per-Token FFN（channel mixing）：每个 token 独立参数。
    用 (T, D, ffn_dim) 参数矩阵 + einsum 实现，无 Python 循环。
    """

    def __init__(self, num_tokens: int, token_dim: int,
                 ffn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, ffn_dim))
        self.w2 = nn.Parameter(torch.empty(num_tokens, ffn_dim, token_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, token_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.w1, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w2, a=0, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("btd,tdf->btf", x, self.w1) + self.b1
        h = self.act(h)
        h = self.dropout(h)
        out = torch.einsum("btf,tfd->btd", h, self.w2) + self.b2
        return self.dropout(out)


class RankMixerV2Block(nn.Module):
    """
    Single RankMixer V2 block (Pre-LN + Residual):
      x = x + TokenMixingV2(LN(x))   # 纯参数无关重排
      x = x + PerTokenFFN(LN(x))     # per-token 独立 FFN
    """

    def __init__(self, num_tokens: int, token_dim: int,
                 ffn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(token_dim)
        self.token_mixing = TokenMixingV2(num_tokens, token_dim)
        self.ln2 = nn.LayerNorm(token_dim)
        self.channel_mixing = PerTokenFFN(num_tokens, token_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(self.ln1(x))
        x = x + self.channel_mixing(self.ln2(x))
        return x


def _find_valid_num_tokens(num_sparse: int, emb_dim: int) -> int:
    """找满足 D%T==0 的最大 T（T <= num_sparse）"""
    for t in range(min(num_sparse, emb_dim), 0, -1):
        if emb_dim % t == 0:
            return t
    return 1


@register_model
class RankMixerV2Model(RecModel):
    """
    RankMixer V2：严格遵循论文，H=T，Token Mixing 纯参数无关。

    Config keys:
        embedding_dim:      D（必须能被 T 整除）
        num_mixer_layers:   层数 L
        num_feature_fields: 手动指定 T（必须满足 D%T==0）；不指定则自动选
        ffn_dim:            Per-Token FFN 中间层维度
        mlp_dims:           top MLP 层维度
        dropout:            dropout rate
        pool_output:        True=mean pooling（论文），False=flatten
    """
    model_name = "rankmixer_v2"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim     = mc["embedding_dim"]
        num_layers  = mc.get("num_mixer_layers", 2)
        ffn_dim     = mc.get("ffn_dim", 256)
        mlp_dims    = mc.get("mlp_dims", [128, 64])
        dropout     = mc.get("dropout", 0.0)
        pool_output = mc.get("pool_output", False)

        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        # 确定合法的 T：先由用户指定，否则自动推断
        num_feature_fields = mc.get("num_feature_fields", None)
        if num_feature_fields is not None:
            assert emb_dim % num_feature_fields == 0, (
                f"num_feature_fields={num_feature_fields} must divide emb_dim={emb_dim}"
            )
            T = num_feature_fields
        else:
            T = _find_valid_num_tokens(num_sparse, emb_dim)

        if T < num_sparse:
            self.feature_field_pool = FeatureFieldPooling(
                num_features=num_sparse,
                num_fields=T,
                seed=config.get("seed", 42),
            )
        else:
            self.feature_field_pool = None

        head_dim = emb_dim // T
        print(f"[RankMixerV2] T={T}, D={emb_dim}, head_dim={head_dim}, "
              f"L={num_layers}, ffn_dim={ffn_dim} | "
              f"TokenMixing=parameter-free permute | "
              f"field_pool: {num_sparse}→{T}")

        self.mixer_blocks = nn.ModuleList([
            RankMixerV2Block(T, emb_dim, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.pool_output = pool_output
        in_dim = emb_dim if pool_output else (T * emb_dim)
        top_layers = []
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.sparse_arch.forward_per_feature(batch["sparse"])  # (B, S, D)
        if self.feature_field_pool is not None:
            x = self.feature_field_pool(x)  # (B, T, D)
        for block in self.mixer_blocks:
            x = block(x)
        if self.pool_output:
            x = x.mean(dim=1)
        else:
            x = x.flatten(start_dim=1)
        return self.top_mlp(x).squeeze(-1)
