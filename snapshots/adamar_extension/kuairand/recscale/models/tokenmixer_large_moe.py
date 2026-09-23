"""
recscale.models.tokenmixer_large_moe — Per-Token Sparse MoE for TokenMixer-Large

Reference: arxiv:2602.06563 §3.4 "Sparse-Pertoken MoE"

Stage S3 (correctness check):  E experts per token, k=E (all active, == dense)
Stage S4 (sparse scaling):     E experts per token, k=2 (1 shared + 1 routed)

Key design decisions (see explore/tokenmixer-large/DESIGN.md):
  - Each of T tokens has E independent "sub-experts" (SwiGLU FFN), each with
    hidden dim F_per_expert = ffn_dim // num_experts
  - Router: per-token Linear(D, E_routed) → softmax (train) / topk (infer)
  - Shared expert: always-active dense SwiGLU with F_per_expert hidden (Eq.20)
  - Gate value scaling α (Eq.21, optional)
  - Forward returns (logit,) or (logit, None) compatible with trainer.py

Discipline: tokenmixer_large_v2.py is NOT modified. New code lives here until
validated, then will be folded into tokenmixer_large_v3.py.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import FeatureFieldPooling, RecModel, SparseArch
from .tokenmixer_large_v2 import (
    MixingAndRevertingBlock,
    TokenMixingV2,
    _TokenMixerLargeV2Backbone,
    _find_valid_num_tokens,
)


# =============================================================================
# Per-Token Sparse MoE (replaces PerTokenPSwiGLU in MixingAndRevertingBlock)
# =============================================================================

class PerTokenMoE(nn.Module):
    """
    Per-Token Sparse MoE: E experts per token position, activating top-k.

    Implements arxiv:2602.06563 Eq. 19-21.

    Architecture:
      - E_routed routed experts (each: per-token SwiGLU, hidden F_per_expert)
      - E_shared shared experts (always active, not gated)
      - k_routed = top_k - num_shared routed experts activated per token
      - Router: per-token Linear(D, E_routed) → softmax (train) / topk (infer)
      - Gate scaling α optionally scales routed expert outputs

    Parameter accounting:
      Per token position t, per expert j:
        W_gate^{t,j}: (D, F_pe),  W_up^{t,j}: (D, F_pe),  W_down^{t,j}: (F_pe, D)
      Where F_pe = ffn_dim // num_experts  (so total capacity = ffn_dim, same as dense)

      Router weight (routed experts): (T, D, E_routed)
      Shared expert weight:           (T, D, F_pe) × 3  (if num_shared > 0)

    Total parameters ≈ dense (PerTokenPSwiGLU with same ffn_dim) × 1.0 when
    E_routed = num_experts, slightly more due to router and shared expert.
    True scaling comes from *expanding* ffn_dim by E while keeping k=2.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,           # D
        ffn_dim: int = 256,        # total capacity F = E * F_pe
        num_experts: int = 4,      # E_routed (routed experts)
        top_k: int = 2,            # activated per token (includes shared)
        num_shared: int = 1,       # E_shared (always active, no routing)
        dropout: float = 0.0,
        down_init_std: float = 0.01,
        router_scale: float = 1.0,  # α gate scaling (Eq.21)
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_experts = num_experts       # routed experts
        self.num_shared = num_shared
        self.top_k_total = top_k             # total activated = shared + routed
        self.top_k_routed = top_k - num_shared
        self.router_scale = router_scale
        assert self.top_k_routed >= 0, "top_k must be >= num_shared"
        assert ffn_dim % num_experts == 0, \
            f"ffn_dim {ffn_dim} must be divisible by num_experts {num_experts}"
        self.F_pe = ffn_dim // num_experts   # per-expert hidden dim

        # --- Routed expert parameters: (T, E, D, F_pe) ---
        E, T, D, F = num_experts, num_tokens, token_dim, self.F_pe
        self.w_gate_r = nn.Parameter(torch.empty(T, E, D, F))
        self.b_gate_r = nn.Parameter(torch.zeros(T, E, F))
        self.w_up_r = nn.Parameter(torch.empty(T, E, D, F))
        self.b_up_r = nn.Parameter(torch.zeros(T, E, F))
        self.w_down_r = nn.Parameter(torch.empty(T, E, F, D))
        self.b_down_r = nn.Parameter(torch.zeros(T, E, D))

        nn.init.kaiming_uniform_(self.w_gate_r, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_up_r, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.w_down_r, std=down_init_std)

        # --- Shared expert parameters: (T, D, F_pe) ---
        if num_shared > 0:
            # One shared expert (same width F_pe as a single routed expert)
            self.w_gate_s = nn.Parameter(torch.empty(T, D, F))
            self.b_gate_s = nn.Parameter(torch.zeros(T, F))
            self.w_up_s = nn.Parameter(torch.empty(T, D, F))
            self.b_up_s = nn.Parameter(torch.zeros(T, F))
            self.w_down_s = nn.Parameter(torch.empty(T, F, D))
            self.b_down_s = nn.Parameter(torch.zeros(T, D))
            nn.init.kaiming_uniform_(self.w_gate_s, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_uniform_(self.w_up_s, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.normal_(self.w_down_s, std=down_init_std)

        # --- Router: (T, D, E_routed) ---
        if self.top_k_routed > 0:
            self.router_weight = nn.Parameter(torch.empty(T, D, E))
            nn.init.xavier_uniform_(self.router_weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _run_expert(
        self,
        x: torch.Tensor,          # (B, T, D)
        w_gate: torch.Tensor,     # (T, [E,] D, F)
        b_gate: torch.Tensor,
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        has_expert_dim: bool = False,
    ) -> torch.Tensor:
        """Run a SwiGLU expert. Returns (B, T, [E,] D)."""
        if has_expert_dim:
            # (B,T,D) × (T,E,D,F) → (B,T,E,F)
            gate = torch.einsum("btd,tedi->btei", x, w_gate) + b_gate
            gate = F.silu(gate)
            up = torch.einsum("btd,tedi->btei", x, w_up) + b_up
            h = gate * up  # (B, T, E, F)
            out = torch.einsum("btei,teid->bted", h, w_down) + b_down  # (B,T,E,D)
        else:
            # (B,T,D) × (T,D,F) → (B,T,F)
            gate = torch.einsum("btd,tdf->btf", x, w_gate) + b_gate
            gate = F.silu(gate)
            up = torch.einsum("btd,tdf->btf", x, w_up) + b_up
            h = gate * up  # (B, T, F)
            out = torch.einsum("btf,tfd->btd", h, w_down) + b_down  # (B,T,D)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape

        # --- Shared expert (always active) ---
        if self.num_shared > 0:
            shared_out = self._run_expert(
                x,
                self.w_gate_s, self.b_gate_s,
                self.w_up_s, self.b_up_s,
                self.w_down_s, self.b_down_s,
                has_expert_dim=False,
            )  # (B, T, D)
        else:
            shared_out = torch.zeros_like(x)

        # --- Routed experts ---
        if self.top_k_routed == 0:
            return self.dropout(shared_out)

        # Compute all E expert outputs: (B, T, E, D)
        all_expert_out = self._run_expert(
            x,
            self.w_gate_r, self.b_gate_r,
            self.w_up_r, self.b_up_r,
            self.w_down_r, self.b_down_r,
            has_expert_dim=True,
        )

        # Router logits: (B, T, E)
        router_logits = torch.einsum("btd,tde->bte", x, self.router_weight)

        if self.training:
            # Softmax (dense) gating during training
            gates = F.softmax(router_logits, dim=-1)  # (B, T, E)
        else:
            # Sparse top-k gating at inference
            topk_vals, topk_idx = router_logits.topk(self.top_k_routed, dim=-1)
            gates = torch.zeros_like(router_logits)
            gates.scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))

        # Apply gate scaling α
        if self.router_scale != 1.0:
            gates = gates * self.router_scale

        # Weighted sum of expert outputs: (B, T, D)
        routed_out = torch.einsum("bte,bted->btd", gates, all_expert_out)

        out = shared_out + routed_out
        return self.dropout(out)


# =============================================================================
# MixingAndRevertingBlock with MoE channel mixing (replaces PerTokenPSwiGLU)
# =============================================================================

class MixingAndRevertingMoEBlock(nn.Module):
    """
    MixingAndRevertingBlock with PerTokenMoE instead of PerTokenPSwiGLU.
    Otherwise identical to MixingAndRevertingBlock.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        ffn_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 4,           # S3: k=E (all active); S4: k=2 (1 shared+1 routed)
        num_shared: int = 1,
        dropout: float = 0.0,
        init_mix_scale: float = 1.0,
        init_ffn_scale: float = 1.0,
        init_inter_scale: float = 0.1,
        learnable_scale: bool = True,
        down_init_std: float = 0.01,
        router_scale: float = 1.0,
    ):
        super().__init__()
        from torch import nn as _nn
        import torch.nn as nn

        self.norm1 = nn.RMSNorm(token_dim)
        self.token_mixing = TokenMixingV2(num_tokens, token_dim)

        self.norm2 = nn.RMSNorm(token_dim)
        self.channel_mixing = PerTokenMoE(
            num_tokens, token_dim, ffn_dim,
            num_experts, top_k, num_shared,
            dropout, down_init_std, router_scale,
        )

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
# TokenMixerLargeMoE Backbone
# =============================================================================

class _TokenMixerLargeMoEBackbone(RecModel):
    """
    Same as _TokenMixerLargeV2Backbone but replaces PerTokenPSwiGLU with PerTokenMoE.
    """

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

        # MoE-specific
        self.num_experts = mc.get("num_experts", 4)
        self.top_k = mc.get("top_k", 4)           # 4 = all-active for S3; 2 for S4
        self.num_shared = mc.get("num_shared", 0)  # 1 for S4
        self.router_scale = float(mc.get("router_scale", 1.0))

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

        num_feature_fields = mc.get("num_feature_fields", None)
        base_sparse = num_sparse
        if num_feature_fields is not None:
            assert emb_dim % (num_feature_fields + extra_tokens) == 0
            T_sparse = num_feature_fields
        else:
            T_sparse = _find_valid_num_tokens(num_sparse, emb_dim)
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

        effective_k = min(self.top_k, self.num_experts + self.num_shared)
        print(
            f"[TokenMixerLargeMoE] T={self.num_tokens}, D={emb_dim}, L={num_layers}, "
            f"ffn={ffn_dim}, E_routed={self.num_experts}, E_shared={self.num_shared}, "
            f"k={self.top_k} (→ k_routed={self.top_k - self.num_shared}), "
            f"F_pe={ffn_dim // self.num_experts}"
        )

        self.mixer_blocks = nn.ModuleList([
            MixingAndRevertingMoEBlock(
                num_tokens=self.num_tokens,
                token_dim=emb_dim,
                ffn_dim=ffn_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                num_shared=self.num_shared,
                dropout=dropout,
                init_mix_scale=mc.get("init_mix_scale", 1.0),
                init_ffn_scale=mc.get("init_ffn_scale", 1.0),
                init_inter_scale=mc.get("init_inter_scale", 0.1),
                learnable_scale=mc.get("learnable_scale", True),
                down_init_std=mc.get("down_init_std", 0.01),
                router_scale=self.router_scale,
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


@register_model
class TokenMixerLargeMoEModel(_TokenMixerLargeMoEBackbone):
    """
    TokenMixer-Large with Per-Token Sparse MoE channel mixing (pointwise).

    Stage S3 config: num_experts=4, top_k=4, num_shared=0
      → all 4 experts active (dense-equivalent), correctness check
    Stage S4 config: num_experts=4, top_k=2, num_shared=1
      → 1 shared + 1 routed active (50% sparsity, paper 1:2 ratio)

    Config keys (in addition to base tmlv2 keys):
      num_experts:   E_routed (default 4)
      top_k:         total activated per token (shared + routed, default 4)
      num_shared:    always-active experts (default 0; use 1 for S4)
      router_scale:  α gate scaling (default 1.0)
    """
    model_name = "tokenmixer_large_moe"

    def __init__(self, config: dict):
        super().__init__(config, extra_tokens=0)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._sparse_tokens(batch)
        x = self._run_blocks(x)
        return self._output(x)
