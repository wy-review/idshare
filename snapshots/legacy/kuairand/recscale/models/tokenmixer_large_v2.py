"""
recscale.models.tokenmixer_large_v2 — TokenMixer-Large V2

基于 RankMixer V2 骨干（H=T, 纯 permute Token Mixing），加入 TokenMixer-Large
论文 (https://arxiv.org/abs/2602.06563) 的核心改进：

1. Mixing-and-Reverting: 重排 → pSwiGLU → OTR(原始输入残差)
2. Per-token pSwiGLU: SwiGLU 门控替代 GELU FFN，einsum 实现
3. Inter-layer Residuals: 每隔 N 层从前面拿 skip，最后一层可排除
4. Learnable residual scale: mix/ffn/inter 三个 scale 可学习
5. Down-matrix 小初始化: pSwiGLU 的 down_proj 用 std=0.01 初始化

架构：
  sparse(S fields) → SparseArch → (B, S, D)
                 → FeatureFieldPooling → (B, T, D)
       ↓
  Stack of MixingAndRevertingBlock × L
  ├── Token Mixing (纯 permute, 无参数)
  ├── RMSNorm + learnable scale + OTR
  ├── Per-Token pSwiGLU (einsum, down 小初始化)
  └── Optional inter-layer residual
       ↓
  mean pooling / flatten → MLP → logit
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import FeatureFieldPooling, RecModel, SparseArch
from .din import DINAttention
from .rankmixer_v2 import TokenMixingV2, _find_valid_num_tokens


# =============================================================================
# Per-Token pSwiGLU（einsum 实现 + down 小初始化）
# =============================================================================

class PerTokenPSwiGLU(nn.Module):
    """
    Per-token pSwiGLU: 每个 token 独立的 gate/up/down 参数。

    pSwiGLU(x_t) = down_t @ (silu(gate_t @ x_t) * (up_t @ x_t))

    用 (T, D, F) 参数矩阵 + einsum 实现，避免 Python 循环。
    down_proj 使用小初始化（std=0.01），让早期训练近似恒等映射。
    """

    def __init__(self, num_tokens: int, token_dim: int,
                 ffn_dim: int = 256, dropout: float = 0.0,
                 down_init_std: float = 0.01):
        super().__init__()
        self.num_tokens = num_tokens

        # gate, up: (T, D, F)
        self.w_gate = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, ffn_dim))
        self.w_up = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b_up = nn.Parameter(torch.zeros(num_tokens, ffn_dim))

        # down: (T, F, D)
        self.w_down = nn.Parameter(torch.empty(num_tokens, ffn_dim, token_dim))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, token_dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 标准初始化 gate/up
        nn.init.kaiming_uniform_(self.w_gate, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_up, a=0, mode="fan_in", nonlinearity="relu")

        # Down-matrix 小初始化（论文要求）
        nn.init.normal_(self.w_down, std=down_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        gate = torch.einsum("btd,tdf->btf", x, self.w_gate) + self.b_gate
        gate = F.silu(gate)
        up = torch.einsum("btd,tdf->btf", x, self.w_up) + self.b_up
        hidden = gate * up  # (B, T, F)
        out = torch.einsum("btf,tfd->btd", hidden, self.w_down) + self.b_down
        return self.dropout(out)


# =============================================================================
# Mixing-and-Reverting Block
# =============================================================================

class MixingAndRevertingBlock(nn.Module):
    """
    TokenMixer-Large 的核心 block：Mixing-and-Reverting + pSwiGLU。

    流程：
      original = x
      mixed = TokenMixing(RMSNorm(x))            # 纯重排
      reverted = original + mix_scale * mixed     # OTR: 和本层原始输入做残差
      transformed = pSwiGLU(RMSNorm(reverted))    # Per-token pSwiGLU
      out = reverted + ffn_scale * transformed    # 标准残差
      if skip: out = out + inter_scale * skip     # inter-layer residual

    Learnable scale 控制每个残差路径的权重。
    非 learnable 时用 register_buffer 保证 device 跟随。
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        init_mix_scale: float = 1.0,
        init_ffn_scale: float = 1.0,
        init_inter_scale: float = 0.1,
        learnable_scale: bool = True,
        down_init_std: float = 0.01,
    ):
        super().__init__()
        # RMSNorm (PyTorch 2.4+)
        self.norm1 = nn.RMSNorm(token_dim)
        self.token_mixing = TokenMixingV2(num_tokens, token_dim)

        self.norm2 = nn.RMSNorm(token_dim)
        self.channel_mixing = PerTokenPSwiGLU(
            num_tokens, token_dim, ffn_dim, dropout, down_init_std
        )

        # Learnable residual scales
        if learnable_scale:
            self.mix_scale = nn.Parameter(torch.tensor(float(init_mix_scale)))
            self.ffn_scale = nn.Parameter(torch.tensor(float(init_ffn_scale)))
            self.inter_scale = nn.Parameter(torch.tensor(float(init_inter_scale)))
        else:
            self.register_buffer("mix_scale", torch.tensor(float(init_mix_scale)))
            self.register_buffer("ffn_scale", torch.tensor(float(init_ffn_scale)))
            self.register_buffer("inter_scale", torch.tensor(float(init_inter_scale)))

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        original = x
        mixed = self.token_mixing(self.norm1(x))
        reverted = original + self.mix_scale * mixed
        transformed = self.channel_mixing(self.norm2(reverted))
        out = reverted + self.ffn_scale * transformed
        if skip is not None:
            out = out + self.inter_scale * skip
        return out


# =============================================================================
# Backbone（共享 tokenization + block stack 逻辑）
# =============================================================================

class _TokenMixerLargeV2Backbone(RecModel):
    def __init__(self, config: dict, extra_tokens: int = 0):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_mixer_layers", 3)
        ffn_dim = mc.get("ffn_dim", 256)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)
        pool_output = mc.get("pool_output", False)

        self.inter_layer_residual = mc.get("inter_layer_residual", True)
        self.residual_skip_stride = mc.get("residual_skip_stride", 2)
        self.exclude_last_inter_residual = mc.get("exclude_last_inter_layer_residual", True)
        self.emb_dim = emb_dim
        self.pool_output = pool_output

        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        # 确定合法的 T（extra_tokens 额外占位，如 seq token）
        num_feature_fields = mc.get("num_feature_fields", None)
        base_sparse = num_sparse
        if num_feature_fields is not None:
            assert emb_dim % (num_feature_fields + extra_tokens) == 0, (
                f"num_feature_fields + extra_tokens = {num_feature_fields + extra_tokens} "
                f"must divide emb_dim={emb_dim}"
            )
            T_sparse = num_feature_fields
        else:
            T_sparse = _find_valid_num_tokens(num_sparse, emb_dim)
            # 如果有 extra_tokens，要确保总 T 也整除 D
            total_T = T_sparse + extra_tokens
            while total_T > 0 and emb_dim % total_T != 0:
                T_sparse -= 1
                total_T = T_sparse + extra_tokens
            if T_sparse <= 0:
                T_sparse = 1

        if T_sparse < num_sparse:
            self.feature_field_pool = FeatureFieldPooling(
                num_features=num_sparse, num_fields=T_sparse,
                seed=config.get("seed", 42),
            )
        else:
            self.feature_field_pool = None

        self.num_tokens = T_sparse + extra_tokens
        self.T_sparse = T_sparse

        print(f"[TokenMixerLargeV2] T={self.num_tokens} (sparse={T_sparse}+extra={extra_tokens}), "
              f"D={emb_dim}, head_dim={emb_dim // self.num_tokens}, "
              f"L={num_layers}, ffn={ffn_dim} "
              f"(field_pool: {num_sparse}->{T_sparse})")

        self.mixer_blocks = nn.ModuleList([
            MixingAndRevertingBlock(
                num_tokens=self.num_tokens,
                token_dim=emb_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                init_mix_scale=mc.get("init_mix_scale", 1.0),
                init_ffn_scale=mc.get("init_ffn_scale", 1.0),
                init_inter_scale=mc.get("init_inter_scale", 0.1),
                learnable_scale=mc.get("learnable_scale", True),
                down_init_std=mc.get("down_init_std", 0.01),
            )
            for _ in range(num_layers)
        ])

        top_layers = []
        in_dim = emb_dim if pool_output else self.num_tokens * emb_dim
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def _sparse_tokens(self, batch: dict) -> torch.Tensor:
        x = self.sparse_arch.forward_per_feature(batch["sparse"])
        if self.feature_field_pool is not None:
            x = self.feature_field_pool(x)
        return x

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = []
        last_idx = len(self.mixer_blocks) - 1
        for layer_idx, block in enumerate(self.mixer_blocks):
            skip = None
            use_inter = self.inter_layer_residual and not (
                self.exclude_last_inter_residual and layer_idx == last_idx
            )
            if use_inter and layer_idx >= self.residual_skip_stride:
                skip = hidden_states[layer_idx - self.residual_skip_stride]
            x = block(x, skip=skip)
            hidden_states.append(x)
        return x

    def _output(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_output:
            x = x.mean(dim=1)
        else:
            x = x.flatten(start_dim=1)
        return self.top_mlp(x).squeeze(-1)


# =============================================================================
# Pointwise Model
# =============================================================================

@register_model
class TokenMixerLargeV2Model(_TokenMixerLargeV2Backbone):
    """
    TokenMixer-Large V2（pointwise）。

    Config keys:
        embedding_dim, num_mixer_layers, num_feature_fields, ffn_dim, mlp_dims,
        dropout, pool_output, inter_layer_residual, residual_skip_stride,
        exclude_last_inter_layer_residual, init_mix_scale, init_ffn_scale,
        init_inter_scale, learnable_scale, down_init_std
    """
    model_name = "tokenmixer_large_v2"

    def __init__(self, config: dict):
        super().__init__(config, extra_tokens=0)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._sparse_tokens(batch)
        x = self._run_blocks(x)
        return self._output(x)


# =============================================================================
# Sequence-aware Model
# =============================================================================

@register_model
class TokenMixerLargeV2SeqModel(_TokenMixerLargeV2Backbone):
    """
    TokenMixer-Large V2（sequence-aware）。

    用 DIN attention 从行为序列生成一个 interest token，追加到 sparse tokens 后面。

    Config keys: 同 pointwise + attention_dim
    """
    model_name = "tokenmixer_large_v2_seq"

    def __init__(self, config: dict):
        super().__init__(config, extra_tokens=1)
        mc = config["model"]
        dc = config["dataset"]

        num_items = dc.get("num_items")
        if num_items is None:
            cardinalities = dc.get("cardinalities", [])
            num_items = max(cardinalities) if cardinalities else 100000

        self.item_emb = nn.Embedding(num_items + 1, self.emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()
        self.attention = DINAttention(self.emb_dim, mc.get("attention_dim", 64))

    def _sequence_token(self, batch: dict, B: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        seq = batch.get("seq")
        target = batch.get("target")
        if seq is None or target is None:
            return torch.zeros(B, 1, self.emb_dim, device=device, dtype=dtype)
        seq_emb = self.item_emb(seq.long())
        target_emb = self.item_emb(target.long())
        mask = seq != 0
        interest = self.attention(target_emb, seq_emb, mask)
        return interest.unsqueeze(1)

    def forward(self, batch: dict) -> torch.Tensor:
        sparse_tokens = self._sparse_tokens(batch)
        seq_token = self._sequence_token(
            batch, sparse_tokens.size(0), sparse_tokens.device, sparse_tokens.dtype
        )
        x = torch.cat([sparse_tokens, seq_token], dim=1)  # (B, T, D)
        x = self._run_blocks(x)
        return self._output(x)
