"""
recscale.models.tokenmixer_large_v3 — TokenMixer-Large V3

Paper-faithful TokenMixer-Large block based on arxiv:2602.06563 Figure 1:

  RMSNorm
    -> Mix
    -> Sparse-PerToken SwiGLU / dense PerToken pSwiGLU
    -> Revert
    -> residual to the block input
    -> RMSNorm
    -> Sparse-PerToken SwiGLU / dense PerToken pSwiGLU
    -> residual
    -> optional inter-layer residual

The important difference from tokenmixer_large_v2.py is that "Revert" is an
actual inverse token-mixing operation. With RankMixer/TokenMixerV2, token mixing
is a swap of token/head axes and is self-inverse when H = T, so applying the
same TokenMixingV2 module a second time maps the mixed representation back to
the original token coordinate system.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import FeatureFieldPooling, RecModel, SparseArch
from .din import DINAttention
from .init_utils import init_per_token_kaiming_uniform_, init_per_token_xavier_uniform_
from .rankmixer_v2 import TokenMixingV2, _find_valid_num_tokens


class _RMSNorm(nn.Module):
    """Small RMSNorm fallback for environments where torch.nn.RMSNorm is absent."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class PerTokenPSwiGLU(nn.Module):
    """Dense per-token pSwiGLU: each token position owns an independent SwiGLU."""

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        down_init_std: float = 0.01,
        per_token_init_v2: bool = False,
    ):
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, ffn_dim))
        self.w_up = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b_up = nn.Parameter(torch.zeros(num_tokens, ffn_dim))
        self.w_down = nn.Parameter(torch.empty(num_tokens, ffn_dim, token_dim))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, token_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if per_token_init_v2:
            init_per_token_kaiming_uniform_(self.w_gate, a=0, fan_in=token_dim)
            init_per_token_kaiming_uniform_(self.w_up, a=0, fan_in=token_dim)
        else:
            nn.init.kaiming_uniform_(self.w_gate, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_uniform_(self.w_up, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.w_down, std=down_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.einsum("btd,tdf->btf", x, self.w_gate) + self.b_gate
        gate = F.silu(gate)
        up = torch.einsum("btd,tdf->btf", x, self.w_up) + self.b_up
        hidden = gate * up
        out = torch.einsum("btf,tfd->btd", hidden, self.w_down) + self.b_down
        return self.dropout(out)


class SparsePerTokenSwiGLU(nn.Module):
    """
    Sparse per-token SwiGLU MoE.

    Each token position has E routed SwiGLU experts and optional shared experts.
    We compute all experts for simplicity, then apply top-k router weights. This
    preserves the paper's sparse routing semantics while keeping the implementation
    straightforward and deterministic for research experiments.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        ffn_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        num_shared: int = 1,
        dropout: float = 0.0,
        down_init_std: float = 0.01,
        router_scale: float = 1.0,
        per_token_init_v2: bool = False,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if ffn_dim % num_experts != 0:
            raise ValueError(f"ffn_dim={ffn_dim} must be divisible by num_experts={num_experts}")
        if top_k < num_shared:
            raise ValueError("top_k must be >= num_shared")

        self.num_experts = num_experts
        self.num_shared = num_shared
        self.top_k_routed = min(top_k - num_shared, num_experts)
        self.router_scale = float(router_scale)
        self.expert_dim = ffn_dim // num_experts
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        T, D, E, F_pe = num_tokens, token_dim, num_experts, self.expert_dim

        self.w_gate_r = nn.Parameter(torch.empty(T, E, D, F_pe))
        self.b_gate_r = nn.Parameter(torch.zeros(T, E, F_pe))
        self.w_up_r = nn.Parameter(torch.empty(T, E, D, F_pe))
        self.b_up_r = nn.Parameter(torch.zeros(T, E, F_pe))
        self.w_down_r = nn.Parameter(torch.empty(T, E, F_pe, D))
        self.b_down_r = nn.Parameter(torch.zeros(T, E, D))

        if per_token_init_v2:
            init_per_token_kaiming_uniform_(self.w_gate_r, a=0, fan_in=token_dim)
            init_per_token_kaiming_uniform_(self.w_up_r, a=0, fan_in=token_dim)
        else:
            nn.init.kaiming_uniform_(self.w_gate_r, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_uniform_(self.w_up_r, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.w_down_r, std=down_init_std)

        if num_shared > 0:
            self.w_gate_s = nn.Parameter(torch.empty(T, num_shared, D, F_pe))
            self.b_gate_s = nn.Parameter(torch.zeros(T, num_shared, F_pe))
            self.w_up_s = nn.Parameter(torch.empty(T, num_shared, D, F_pe))
            self.b_up_s = nn.Parameter(torch.zeros(T, num_shared, F_pe))
            self.w_down_s = nn.Parameter(torch.empty(T, num_shared, F_pe, D))
            self.b_down_s = nn.Parameter(torch.zeros(T, num_shared, D))
            if per_token_init_v2:
                init_per_token_kaiming_uniform_(self.w_gate_s, a=0, fan_in=token_dim)
                init_per_token_kaiming_uniform_(self.w_up_s, a=0, fan_in=token_dim)
            else:
                nn.init.kaiming_uniform_(self.w_gate_s, a=0, mode="fan_in", nonlinearity="relu")
                nn.init.kaiming_uniform_(self.w_up_s, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.normal_(self.w_down_s, std=down_init_std)

        if self.top_k_routed > 0:
            self.router_weight = nn.Parameter(torch.empty(T, D, E))
            self.router_bias = nn.Parameter(torch.zeros(T, E))
            if per_token_init_v2:
                init_per_token_xavier_uniform_(self.router_weight, fan_in=token_dim, fan_out=num_experts)
            else:
                nn.init.xavier_uniform_(self.router_weight)

    @staticmethod
    def _run_experts(
        x: torch.Tensor,
        w_gate: torch.Tensor,
        b_gate: torch.Tensor,
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_down: torch.Tensor,
        b_down: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.einsum("btd,tedf->btef", x, w_gate) + b_gate
        gate = F.silu(gate)
        up = torch.einsum("btd,tedf->btef", x, w_up) + b_up
        hidden = gate * up
        return torch.einsum("btef,tefd->bted", hidden, w_down) + b_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)

        if self.num_shared > 0:
            shared = self._run_experts(
                x,
                self.w_gate_s,
                self.b_gate_s,
                self.w_up_s,
                self.b_up_s,
                self.w_down_s,
                self.b_down_s,
            )
            out = out + shared.sum(dim=2)

        if self.top_k_routed > 0:
            expert_out = self._run_experts(
                x,
                self.w_gate_r,
                self.b_gate_r,
                self.w_up_r,
                self.b_up_r,
                self.w_down_r,
                self.b_down_r,
            )
            router_logits = torch.einsum("btd,tde->bte", x, self.router_weight) + self.router_bias
            topk_vals, topk_idx = router_logits.topk(self.top_k_routed, dim=-1)
            topk_gates = F.softmax(topk_vals, dim=-1) * self.router_scale
            gates = torch.zeros_like(router_logits)
            gates.scatter_(-1, topk_idx, topk_gates)
            out = out + torch.einsum("bte,bted->btd", gates, expert_out)

        return self.dropout(out)


def _make_channel_mixer(
    mixer_type: str,
    num_tokens: int,
    token_dim: int,
    ffn_dim: int,
    dropout: float,
    down_init_std: float,
    num_experts: int,
    top_k: int,
    num_shared: int,
    router_scale: float,
    per_token_init_v2: bool = False,
) -> nn.Module:
    if mixer_type == "dense":
        return PerTokenPSwiGLU(
            num_tokens, token_dim, ffn_dim, dropout, down_init_std, per_token_init_v2,
        )
    if mixer_type == "moe":
        return SparsePerTokenSwiGLU(
            num_tokens,
            token_dim,
            ffn_dim,
            num_experts,
            top_k,
            num_shared,
            dropout,
            down_init_std,
            router_scale,
            per_token_init_v2,
        )
    raise ValueError(f"Unknown channel_mixer_type={mixer_type!r}; expected 'dense' or 'moe'")


class TokenMixerLargeV3Block(nn.Module):
    """Paper-faithful TokenMixer-Large block with real mix-and-revert."""

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        channel_mixer_type: str = "moe",
        num_experts: int = 4,
        top_k: int = 2,
        num_shared: int = 1,
        init_mix_scale: float = 1.0,
        init_ffn_scale: float = 1.0,
        init_inter_scale: float = 0.1,
        learnable_scale: bool = True,
        down_init_std: float = 0.01,
        router_scale: float = 1.0,
        per_token_init_v2: bool = False,
    ):
        super().__init__()
        self.norm_mix = _RMSNorm(token_dim)
        self.token_mixing = TokenMixingV2(num_tokens, token_dim)
        _mixer_kw = dict(
            mixer_type=channel_mixer_type,
            num_tokens=num_tokens,
            token_dim=token_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            down_init_std=down_init_std,
            num_experts=num_experts,
            top_k=top_k,
            num_shared=num_shared,
            router_scale=router_scale,
            per_token_init_v2=per_token_init_v2,
        )
        self.mixed_channel_mixing = _make_channel_mixer(**_mixer_kw)

        self.norm_ffn = _RMSNorm(token_dim)
        self.channel_mixing = _make_channel_mixer(**_mixer_kw)

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

        # Mix -> S-P SwiGLU -> Revert. TokenMixingV2 is self-inverse under H=T.
        mixed = self.token_mixing(self.norm_mix(x))
        mixed = self.mixed_channel_mixing(mixed)
        reverted = self.token_mixing(mixed)
        out = original + self.mix_scale * reverted

        transformed = self.channel_mixing(self.norm_ffn(out))
        out = out + self.ffn_scale * transformed

        if skip is not None:
            out = out + self.inter_scale * skip
        return out


class _AuxPredictor(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tokens * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1)).squeeze(-1)


class _TokenMixerLargeV3Backbone(RecModel):
    def __init__(self, config: dict, extra_tokens: int = 0):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_mixer_layers", 3)
        ffn_dim = mc.get("ffn_dim", 256)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)
        self.pool_output = mc.get("pool_output", False)
        self.emb_dim = emb_dim

        self.inter_layer_residual = mc.get("inter_layer_residual", True)
        self.residual_skip_stride = mc.get("residual_skip_stride", 2)
        self.exclude_last_inter_residual = mc.get("exclude_last_inter_layer_residual", True)

        self.aux_loss_weight = float(mc.get("aux_loss_weight", 0.0))
        self.aux_pred_hidden_dim = int(mc.get("aux_pred_hidden_dim", 64))
        self.aux_exclude_last_layer = bool(mc.get("aux_exclude_last_layer", True))

        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        num_feature_fields = mc.get("num_feature_fields", None)
        if num_feature_fields is not None:
            if emb_dim % (num_feature_fields + extra_tokens) != 0:
                raise ValueError(
                    f"num_feature_fields + extra_tokens = {num_feature_fields + extra_tokens} "
                    f"must divide embedding_dim={emb_dim}"
                )
            t_sparse = num_feature_fields
        else:
            t_sparse = _find_valid_num_tokens(num_sparse, emb_dim)
            total_t = t_sparse + extra_tokens
            while total_t > 0 and emb_dim % total_t != 0:
                t_sparse -= 1
                total_t = t_sparse + extra_tokens
            if t_sparse <= 0:
                t_sparse = 1

        if t_sparse < num_sparse:
            self.feature_field_pool = FeatureFieldPooling(
                num_features=num_sparse,
                num_fields=t_sparse,
                seed=config.get("seed", 42),
            )
        else:
            self.feature_field_pool = None

        self.num_tokens = t_sparse + extra_tokens
        self.t_sparse = t_sparse
        self.channel_mixer_type = mc.get("channel_mixer_type", "moe")
        self.per_token_init_v2 = bool(mc.get("per_token_init_v2", False))

        print(
            f"[TokenMixerLargeV3] T={self.num_tokens} "
            f"(sparse={t_sparse}+extra={extra_tokens}), D={emb_dim}, "
            f"head_dim={emb_dim // self.num_tokens}, L={num_layers}, "
            f"ffn={ffn_dim}, channel_mixer={self.channel_mixer_type}, "
            f"field_pool: {num_sparse}->{t_sparse}"
        )

        self.mixer_blocks = nn.ModuleList([
            TokenMixerLargeV3Block(
                num_tokens=self.num_tokens,
                token_dim=emb_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                channel_mixer_type=self.channel_mixer_type,
                num_experts=mc.get("num_experts", 4),
                top_k=mc.get("top_k", 2),
                num_shared=mc.get("num_shared", 1),
                init_mix_scale=mc.get("init_mix_scale", 1.0),
                init_ffn_scale=mc.get("init_ffn_scale", 1.0),
                init_inter_scale=mc.get("init_inter_scale", 0.1),
                learnable_scale=mc.get("learnable_scale", True),
                down_init_std=mc.get("down_init_std", 0.01),
                router_scale=mc.get("router_scale", 1.0),
                per_token_init_v2=self.per_token_init_v2,
            )
            for _ in range(num_layers)
        ])

        self.num_aux = 0
        if self.aux_loss_weight > 0.0:
            self.num_aux = num_layers - 1 if self.aux_exclude_last_layer else num_layers
            if self.num_aux > 0:
                self.aux_predictors = nn.ModuleList([
                    _AuxPredictor(self.num_tokens, emb_dim, self.aux_pred_hidden_dim)
                    for _ in range(self.num_aux)
                ])

        in_dim = emb_dim if self.pool_output else self.num_tokens * emb_dim
        top_layers = []
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

    def _run_blocks(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
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

        aux_loss = None
        if (
            self.training
            and label is not None
            and self.aux_loss_weight > 0.0
            and self.num_aux > 0
        ):
            aux_loss = x.new_zeros(())
            for i, predictor in enumerate(self.aux_predictors):
                aux_logit = predictor(hidden_states[i])
                aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                    aux_logit,
                    label.float(),
                )
            aux_loss = self.aux_loss_weight * aux_loss / self.num_aux

        return x, aux_loss

    def _output(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_output:
            x = x.mean(dim=1)
        else:
            x = x.flatten(start_dim=1)
        return self.top_mlp(x).squeeze(-1)


@register_model
class TokenMixerLargeV3Model(_TokenMixerLargeV3Backbone):
    """TokenMixer-Large V3 pointwise CTR model."""

    model_name = "tokenmixer_large_v3"

    def __init__(self, config: dict):
        super().__init__(config, extra_tokens=0)

    def forward(self, batch: dict):
        x = self._sparse_tokens(batch)
        x, aux_loss = self._run_blocks(x, label=batch.get("label"))
        logits = self._output(x)
        if aux_loss is not None:
            return logits, aux_loss
        return logits


@register_model
class TokenMixerLargeV3SeqModel(_TokenMixerLargeV3Backbone):
    """Sequence-aware V3: DIN attention produces one extra sequence token."""

    model_name = "tokenmixer_large_v3_seq"

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

    def _sequence_token(
        self,
        batch: dict,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        seq = batch.get("seq")
        target = batch.get("target")
        if seq is None or target is None:
            return torch.zeros(batch_size, 1, self.emb_dim, device=device, dtype=dtype)

        seq_emb = self.item_emb(seq.long())
        target_emb = self.item_emb(target.long())
        mask = seq != 0
        interest = self.attention(target_emb, seq_emb, mask)
        return interest.unsqueeze(1)

    def forward(self, batch: dict):
        sparse_tokens = self._sparse_tokens(batch)
        seq_token = self._sequence_token(
            batch,
            sparse_tokens.size(0),
            sparse_tokens.device,
            sparse_tokens.dtype,
        )
        x = torch.cat([sparse_tokens, seq_token], dim=1)
        x, aux_loss = self._run_blocks(x, label=batch.get("label"))
        logits = self._output(x)
        if aux_loss is not None:
            return logits, aux_loss
        return logits
