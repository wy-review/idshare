"""
recscale.models.onetrans_v2 — OneTrans V2（论文正确实现）

严格遵循论文 "OneTrans: One Transformer for All" (arXiv:2510.26104) 的设计，
并修复了原始实现中的以下问题：

P0 修复：
  1. 因果注意力 Mask（可选，默认开启）:
     - S-tokens: causal mask（只能看前面的 S + 所有 NS）
     - NS-tokens: 可以看所有 S + 前面的 NS
     - 通过超参数 `use_causal_mask` 控制（默认 True）
     - 另有 `strict_s_causal`（默认 False），开启后 S-tokens 不看任何 NS-tokens
       （论文严格派，用于消融实验）
  2. Pyramid Pruning（方案 C）:
     - 每层 forward 后裁剪 S-tokens，下一层 Q/K/V 都用裁剪后的序列
     - 右对齐保留最近行为（原实现已正确）
     - 同时正确裁剪 causal mask

P1 改进（可选，通过超参数控制）:
  3. NS-Tokens 完全独立参数（可选，默认关闭）:
     - `ns_param_mode: "shared_bias" | "independent"`
     - 论文用完全独立参数，但内存消耗大

架构：
  sparse/dense → NS-tokens (B, L_NS, D)
  seq → item_emb → S-tokens (B, L_S, D)
  concat → [S-tokens || NS-tokens]
      ↓
  Stack of MixedTransformerBlockV2 × N
  ├── RMSNorm + MixedMHA V2 (S: shared, NS: per-token, optional causal mask)
  └── RMSNorm + MixedFFN V2 (S: shared, NS: per-token)
  + Residual
  + Pyramid Pruning (optional, 逐层裁减 S-tokens)
      ↓
  pool NS-tokens → MLP → logit
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, DenseArch, SparseArch


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MixedMultiHeadAttentionV2(nn.Module):
    """
    Mixed Parameterization Multi-Head Attention (V2).

    S-tokens 共享一套 Q/K/V 投影参数;
    NS-tokens 每个 token 有独立的 Q/K/V 投影参数（或 shared + bias）。

    支持因果注意力 Mask（论文推荐，但可通过开关关闭）:
      - S-tokens: causal mask（只能看前面的 S-tokens + 所有 NS-tokens）
      - NS-tokens: 可以看所有 S-tokens + 前面的 NS-tokens
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_ns_tokens: int,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        ns_param_mode: str = "shared_bias",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_ns_tokens = num_ns_tokens
        self.use_causal_mask = use_causal_mask
        self.ns_param_mode = ns_param_mode

        # S-tokens: shared Q/K/V projections
        self.s_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)

        # NS-tokens: parameter handling
        if ns_param_mode == "independent":
            # 完全独立参数：每个 NS-token 有独立的 Q/K/V 投影（einsum 实现）
            self.ns_qkv_weight = nn.Parameter(
                torch.empty(num_ns_tokens, hidden_dim, 3 * hidden_dim)
            )
            self.ns_qkv_bias = nn.Parameter(
                torch.zeros(num_ns_tokens, 3 * hidden_dim)
            )
            nn.init.kaiming_uniform_(self.ns_qkv_weight, a=math.sqrt(5))
        else:
            # shared + per-token bias（默认，内存友好）
            self.ns_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
            if num_ns_tokens > 0:
                self.ns_bias = nn.Parameter(
                    torch.zeros(num_ns_tokens, 3 * hidden_dim)
                )
                nn.init.normal_(self.ns_bias, std=0.02)
            else:
                self.ns_bias = None

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        num_s_tokens: int,
        causal_mask: Optional[torch.Tensor] = None,
        ns_s_bias: Optional[torch.Tensor] = None,
        s_s_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D) where L = L_S + L_NS
        num_s_tokens: how many tokens at the beginning are S-tokens
        causal_mask: (L, L) bool tensor, True 表示可见（可选）
        ns_s_bias: (B, L_NS, L_S) — TCPB target-conditioned bias added ONLY to
                   the NS×S sub-block of `attn` (B, H, L, L), broadcast over
                   heads. Added BEFORE the causal mask so that any position
                   that should be masked out still ends at -inf.
        s_s_bias: (L_S, L_S) — static relative position bias added to the S×S
                  sub-block of `attn`, broadcast over batch and heads.
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        L_S = num_s_tokens
        L_NS = L - L_S

        # Compute Q/K/V for S-tokens (shared)
        s_tokens = x[:, :L_S]
        s_qkv = self.s_qkv(s_tokens)  # (B, L_S, 3D)

        # Compute Q/K/V for NS-tokens
        ns_tokens = x[:, L_S:]
        if self.ns_param_mode == "independent" and L_NS > 0:
            # einsum: (B, L_NS, D) × (L_NS, D, 3D) → (B, L_NS, 3D)
            # 注意：L_NS 可能小于 num_ns_tokens，取前 L_NS 个
            W = self.ns_qkv_weight[:L_NS]  # (L_NS, D, 3D)
            b = self.ns_qkv_bias[:L_NS]  # (L_NS, 3D)
            ns_qkv = torch.einsum("bld,ldf->blf", ns_tokens, W) + b
        else:
            ns_qkv = self.ns_qkv(ns_tokens)  # (B, L_NS, 3D)
            if L_NS > 0 and self.ns_bias is not None:
                bias = self.ns_bias[:L_NS]
                ns_qkv = ns_qkv + bias.unsqueeze(0)

        # Concatenate back
        qkv = torch.cat([s_qkv, ns_qkv], dim=1)  # (B, L, 3D)

        # Reshape to multi-head
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # S×S relative position bias, broadcast over batch and heads.
        # Shape: (L_S, L_S). Applied BEFORE causal-mask.
        if s_s_bias is not None and L_S > 0:
            attn[:, :, :L_S, :L_S] = attn[:, :, :L_S, :L_S] + s_s_bias.unsqueeze(0).unsqueeze(0)

        # TCPB / SlotPos bias on the NS×S sub-block, broadcast over heads.
        # Shape: (B, L_NS, L_S) for TCPB, or (1, L_NS, L_S) for static SlotPos,
        # or their sum. All are broadcast-compatible with attn[:, :, L_S:, :L_S].
        # Applied BEFORE causal-mask so masked entries still get -inf.
        if ns_s_bias is not None and L_NS > 0 and L_S > 0:
            # NOTE: in-place add is fine here because `attn` is a fresh tensor.
            attn[:, :, L_S:, :L_S] = attn[:, :, L_S:, :L_S] + ns_s_bias.unsqueeze(1)

        # Apply causal mask if enabled
        if self.use_causal_mask and causal_mask is not None:
            attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)

        return self.out_proj(out)

    @staticmethod
    def build_causal_mask(
        L_S: int,
        L_NS: int,
        device: torch.device,
        dtype: torch.dtype,
        strict_s_causal: bool = False,
    ) -> torch.Tensor:
        """
        构建非对称因果 mask，形状 (L_S + L_NS, L_S + L_NS).

        默认规则（S 可以看 NS，宽松）:
          - S-tokens (0:L_S): causal mask（只能看前面的 S + 所有 NS）
          - NS-tokens (L_S:): 可以看所有 S + 前面的 NS

        strict_s_causal=True 时（论文严格派）:
          - S-tokens 只看前面的 S-tokens，不看任何 NS-tokens
          - NS-tokens 仍然可以看所有 S + 前面的 NS

        Returns:
          mask: (L, L) bool tensor, True 表示可见
        """
        L = L_S + L_NS
        mask = torch.ones(L, L, device=device, dtype=torch.bool)

        # S-tokens 部分: causal mask（上三角置 False）
        if L_S > 0:
            s_mask = torch.tril(torch.ones(L_S, L_S, device=device, dtype=torch.bool))
            mask[:L_S, :L_S] = s_mask
            # S-tokens 对 NS-tokens 的可见性
            if strict_s_causal and L_NS > 0:
                # 论文严格派：S 不能看 NS
                mask[:L_S, L_S:] = False

        # NS-tokens 部分: causal mask（上三角置 False）
        if L_NS > 0:
            ns_mask = torch.tril(torch.ones(L_NS, L_NS, device=device, dtype=torch.bool))
            mask[L_S:, L_S:] = ns_mask
            # NS-tokens 看所有 S-tokens: 全 True（已经满足）

        return mask


class MixedFFNV2(nn.Module):
    """
    Mixed Parameterization FFN (V2).

    S-tokens 共享 FFN, NS-tokens 用独立 bias 或完全独立参数。
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_ns_tokens: int,
        dropout: float = 0.0,
        ns_param_mode: str = "shared_bias",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_ns_tokens = num_ns_tokens
        self.ns_param_mode = ns_param_mode

        if ns_param_mode == "independent":
            # 完全独立参数
            self.ns_fc1 = nn.Parameter(torch.empty(num_ns_tokens, hidden_dim, ffn_dim))
            self.ns_fc2 = nn.Parameter(torch.empty(num_ns_tokens, ffn_dim, hidden_dim))
            self.ns_bias1 = nn.Parameter(torch.zeros(num_ns_tokens, ffn_dim))
            self.ns_bias2 = nn.Parameter(torch.zeros(num_ns_tokens, hidden_dim))
            nn.init.kaiming_uniform_(self.ns_fc1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.ns_fc2, a=math.sqrt(5))
            # S-tokens 用共享 FFN
            self.shared_fc1 = nn.Linear(hidden_dim, ffn_dim)
            self.shared_fc2 = nn.Linear(ffn_dim, hidden_dim)
        else:
            # shared FFN + per-token bias（默认）
            self.shared_fc1 = nn.Linear(hidden_dim, ffn_dim)
            self.shared_fc2 = nn.Linear(ffn_dim, hidden_dim)
            if num_ns_tokens > 0:
                self.ns_bias1 = nn.Parameter(torch.zeros(num_ns_tokens, ffn_dim))
                self.ns_bias2 = nn.Parameter(torch.zeros(num_ns_tokens, hidden_dim))
                nn.init.normal_(self.ns_bias1, std=0.02)
                nn.init.normal_(self.ns_bias2, std=0.02)
            else:
                self.ns_bias1 = None
                self.ns_bias2 = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, num_s_tokens: int) -> torch.Tensor:
        B, L, D = x.shape
        L_S = num_s_tokens
        L_NS = L - L_S

        if self.ns_param_mode == "independent" and L_NS > 0:
            # 分别处理
            s_tokens = x[:, :L_S]
            ns_tokens = x[:, L_S:]

            # S-tokens: 共享 FFN
            s_h = F.gelu(self.shared_fc1(s_tokens))
            s_h = self.dropout(s_h)
            s_out = self.shared_fc2(s_h)
            s_out = self.dropout(s_out)

            # NS-tokens: 独立 FFN (einsum，取前 L_NS 个参数)
            W1 = self.ns_fc1[:L_NS]
            W2 = self.ns_fc2[:L_NS]
            b1 = self.ns_bias1[:L_NS]
            b2 = self.ns_bias2[:L_NS]
            ns_h = F.gelu(torch.einsum("bsd,sdf->bsf", ns_tokens, W1) + b1)
            ns_h = self.dropout(ns_h)
            ns_out = torch.einsum("bsf,sfd->bsd", ns_h, W2) + b2
            ns_out = self.dropout(ns_out)

            return torch.cat([s_out, ns_out], dim=1)
        else:
            # 共享 FFN + per-token bias
            h = F.gelu(self.shared_fc1(x))

            if L_NS > 0 and self.ns_bias1 is not None:
                bias1 = self.ns_bias1[:L_NS].unsqueeze(0)
                h = torch.cat([h[:, :L_S], h[:, L_S:] + bias1], dim=1)

            h = self.dropout(h)
            out = self.shared_fc2(h)

            if L_NS > 0 and self.ns_bias2 is not None:
                bias2 = self.ns_bias2[:L_NS].unsqueeze(0)
                out = torch.cat([out[:, :L_S], out[:, L_S:] + bias2], dim=1)

            return self.dropout(out)


class MixedTransformerBlockV2(nn.Module):
    """
    Pre-norm Transformer block with Mixed Parameterization (V2).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_ns_tokens: int,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        ns_param_mode: str = "shared_bias",
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = MixedMultiHeadAttentionV2(
            hidden_dim, num_heads, num_ns_tokens, dropout,
            use_causal_mask, ns_param_mode,
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = MixedFFNV2(
            hidden_dim, ffn_dim, num_ns_tokens, dropout, ns_param_mode,
        )
        self.use_causal_mask = use_causal_mask

    def forward(
        self,
        x: torch.Tensor,
        num_s_tokens: int,
        causal_mask: Optional[torch.Tensor] = None,
        ns_s_bias: Optional[torch.Tensor] = None,
        s_s_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), num_s_tokens, causal_mask, ns_s_bias=ns_s_bias, s_s_bias=s_s_bias)
        x = x + self.ffn(self.norm2(x), num_s_tokens)
        return x


class TargetAttentionOT(nn.Module):
    """DIN-style target attention shortcut for OneTrans."""

    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, E = keys.shape
        q = query.unsqueeze(1).expand(-1, L, -1)
        attn_input = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.attn_mlp(attn_input).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.bmm(weights.unsqueeze(1), keys).squeeze(1)


@register_model
class OneTransV2Model(RecModel):
    """
    OneTrans V2: One Transformer for All features (论文正确实现).

    新超参数:
      - use_causal_mask (bool): 是否使用因果注意力 mask，默认 True
      - strict_s_causal (bool): 严格论文派，S-tokens 不看 NS-tokens，默认 False
      - ns_param_mode (str): "shared_bias" | "independent"，默认 "shared_bias"

    Supports batch keys: "sparse", "dense" (opt), "seq" (opt), "target" (opt), "label"
    """
    model_name = "onetrans_v2"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc.get("embedding_dim", 16)
        hidden_dim = mc.get("hidden_dim", 128)
        num_layers = mc.get("num_layers", 4)
        num_heads = mc.get("num_heads", 4)
        ffn_dim = mc.get("ffn_dim", hidden_dim * 4)
        dropout = mc.get("dropout", 0.1)
        mlp_dims = mc.get("mlp_dims", [128, 64])

        self.pyramid_prune_ratio = mc.get("pyramid_prune_ratio", 0)
        self.use_causal_mask = mc.get("use_causal_mask", True)
        self.strict_s_causal = mc.get("strict_s_causal", False)
        self.ns_param_mode = mc.get("ns_param_mode", "shared_bias")

        # ----- TCQD-OT knobs (target-conditioned modulation, all default OFF) -----
        # use_target_gating  (TCQG-OT): two per-channel gates that scale
        #   s_tokens / ns groups by g_s(c), g_ns(c) ∈ ℝ^(B, hidden_dim) before cat.
        # use_target_time_bias (TCPB-OT): per-layer low-rank bias added to the
        #   NS×S block of self-attention, B_tc ∈ ℝ^(B, L_NS, L_S).
        self.use_target_gating = mc.get("use_target_gating", False)
        self.use_target_time_bias = mc.get("use_target_time_bias", False)
        self.use_position_bias = mc.get("use_position_bias", False)
        self.pos_bias_max_seq_len = int(mc.get("pos_bias_max_seq_len", 1024))
        self.target_time_rank = mc.get("target_time_rank", 8)
        self.target_condition_source = mc.get("target_condition_source", "item")
        self.gate_activation_mode = mc.get("gate_activation_mode", "2sigmoid")
        self.tcpb_share_layers = mc.get("tcpb_share_layers", False)
        self.use_din_shortcut = mc.get("use_din_shortcut", True)  # DIN target_attn shortcut
        self.tcpb_max_seq_len = mc.get("tcpb_max_seq_len", 1024)
        assert self.target_condition_source in ("item", "item_seq"), (
            f"target_condition_source must be 'item' or 'item_seq', "
            f"got {self.target_condition_source}"
        )
        assert self.gate_activation_mode in (
            "2sigmoid", "sigmoid", "residual_tanh"
        ), self.gate_activation_mode

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        self.num_ns_tokens = num_sparse + (1 if num_dense > 0 else 0) + 1
        self.has_dense = num_dense > 0
        self.has_seq = "seq" in (dc.get("batch_keys") or []) or dc.get("num_items", 0) > 0

        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        self.sparse_proj = nn.Linear(emb_dim, hidden_dim) if emb_dim != hidden_dim else nn.Identity()

        if self.has_dense:
            self.dense_proj = nn.Sequential(
                nn.Linear(num_dense, hidden_dim),
                nn.ReLU(),
            )

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.item_emb.weight.data[0].zero_()

        self.target_attn = TargetAttentionOT(hidden_dim, hidden_dim) if self.use_din_shortcut else None

        self.blocks = nn.ModuleList([
            MixedTransformerBlockV2(
                hidden_dim, num_heads, ffn_dim,
                self.num_ns_tokens, dropout,
                self.use_causal_mask, self.ns_param_mode,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)

        self.num_layers = num_layers

        top_layers = []
        in_dim = self.num_ns_tokens * hidden_dim
        if self.use_din_shortcut:
            in_dim += hidden_dim + hidden_dim  # interest + target_emb
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

        # ----- TCQD-OT modules (built only when knobs are True) -----
        # Static Slot-Specific Position Bias (SlotPos): per-NS-token,
        # per-S-position learnable bias on the NS×S attention sub-block.
        if self.use_position_bias:
            self.slot_pos_bias = nn.Parameter(
                torch.zeros(self.num_ns_tokens, self.pos_bias_max_seq_len)
            )

        # S×S relative position bias: Toeplitz-style shared across heads/layers.
        # Adds a learnable relative-position curve to the S×S sub-block of
        # self-attention, giving S-tokens position awareness when attending to
        # each other. Zero-init → identity at init.
        self.use_seq_pos_bias = mc.get("use_seq_pos_bias", False)
        if self.use_seq_pos_bias:
            max_len = self.pos_bias_max_seq_len
            self.seq_pos_bias_table = nn.Parameter(
                torch.zeros(2 * max_len - 1)
            )

        # condition vector: target_emb (item) or [target_emb; s̄] (item_seq).
        cond_dim = hidden_dim
        if self.target_condition_source == "item_seq":
            cond_dim = hidden_dim + hidden_dim
        self._cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        if self.use_target_gating:
            # Two per-channel gates, shared across positions within group.
            self.tcqg_proj_s = nn.Linear(cond_dim, hidden_dim)
            self.tcqg_proj_ns = nn.Linear(cond_dim, hidden_dim)
            for proj in (self.tcqg_proj_s, self.tcqg_proj_ns):
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

        if self.use_target_time_bias:
            R = self.target_time_rank
            num_modules = 1 if self.tcpb_share_layers else num_layers
            # M(c) ∈ ℝ^(B, L_NS, R) — per NS-token mixture
            self.tcpb_proj = nn.ModuleList([
                nn.Linear(cond_dim, self.num_ns_tokens * R)
                for _ in range(num_modules)
            ])
            # P ∈ ℝ^(L_S, R) — recency-aligned position table
            self.tcpb_pos_emb = nn.ModuleList([
                nn.Embedding(self.tcpb_max_seq_len, R) for _ in range(num_modules)
            ])
            for proj in self.tcpb_proj:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            for emb in self.tcpb_pos_emb:
                nn.init.normal_(emb.weight, std=0.02)

        # TCQD status line for the log
        tcqd_bits = []
        if self.use_position_bias:
            tcqd_bits.append(f"SlotPos({self.num_ns_tokens}x{self.pos_bias_max_seq_len})")
        if self.use_seq_pos_bias:
            tcqd_bits.append(f"SeqPosBias(2*{self.pos_bias_max_seq_len}-1)")
        if self.use_target_gating:
            tcqd_bits.append(f"TCQG({self.gate_activation_mode})")
        if self.use_target_time_bias:
            share_str = "shared" if self.tcpb_share_layers else "per-layer"
            tcqd_bits.append(f"TCPB(R={self.target_time_rank},{share_str})")
        if tcqd_bits:
            print(
                f"[OneTransV2] modules=[{', '.join(tcqd_bits)}], "
                f"cond={self.target_condition_source}({self._cond_dim}d)"
            )

    def _gate_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Activation for TCQG gates — mirrors HyFormer convention."""
        if self.gate_activation_mode == "2sigmoid":
            return 2.0 * torch.sigmoid(logits)
        if self.gate_activation_mode == "residual_tanh":
            return 1.0 + torch.tanh(logits)
        return torch.sigmoid(logits)

    def forward(self, batch: dict) -> torch.Tensor:
        # NS-tokens
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])
        ns_tokens = [self.sparse_proj(sparse_per)]

        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            ns_tokens.append(self.dense_proj(batch["dense"]).unsqueeze(1))

        target = batch.get("target")
        if target is not None:
            target_emb = self.item_emb(target)
            target_token = target_emb.unsqueeze(1)
            ns_tokens.append(target_token)
        else:
            B = sparse_per.size(0)
            target_emb = torch.zeros(
                B, self.item_emb.embedding_dim, device=sparse_per.device
            )
            ns_tokens.append(target_emb.unsqueeze(1))

        ns = torch.cat(ns_tokens, dim=1)
        L_NS = ns.size(1)

        # S-tokens
        seq = batch.get("seq")
        seq_mask = None
        seq_mean = None
        if seq is not None and seq.sum() > 0:
            s_tokens = self.item_emb(seq)
            seq_mask = (seq != 0)
            s_tokens = s_tokens * seq_mask.unsqueeze(-1).float()
            L_S = s_tokens.size(1)
            # Masked mean of S-tokens for the "item_seq" maturity branch.
            m = seq_mask.float().unsqueeze(-1)               # (B, L_S, 1)
            denom = m.sum(dim=1).clamp_min(1.0)               # (B, 1)
            seq_mean = (s_tokens * m).sum(dim=1) / denom      # (B, hidden_dim)
            interest = self.target_attn(target_emb, s_tokens, seq_mask) if self.use_din_shortcut else None
        else:
            L_S = 0
            s_tokens = None
            interest = torch.zeros_like(target_emb) if self.use_din_shortcut else None

        # ---------- TCQD: build condition vector + apply TCQG gates ----------
        cond = None
        if self.use_target_gating or self.use_target_time_bias:
            if self.target_condition_source == "item":
                cond = target_emb                              # (B, hidden_dim)
            else:  # "item_seq"
                if seq_mean is None:
                    seq_mean = torch.zeros(
                        target_emb.size(0), self.hidden_dim,
                        device=target_emb.device, dtype=target_emb.dtype,
                    )
                cond = torch.cat([target_emb, seq_mean], dim=1)

        if self.use_target_gating:
            # Per-channel gates, broadcast across positions in each group.
            g_ns = self._gate_activation(self.tcqg_proj_ns(cond)).unsqueeze(1)
            ns = ns * g_ns                                     # (B, L_NS, D)
            if s_tokens is not None:
                g_s = self._gate_activation(self.tcqg_proj_s(cond)).unsqueeze(1)
                s_tokens = s_tokens * g_s                      # (B, L_S, D)

        # Concatenate after gating.
        if s_tokens is not None:
            x = torch.cat([s_tokens, ns], dim=1)
        else:
            x = ns

        # Transformer blocks with Pyramid Pruning (方案 C: 直接裁剪 x)
        current_L_S = L_S
        for layer_idx, block in enumerate(self.blocks):
            # 预计算当前层的 causal mask
            causal_mask = None
            if self.use_causal_mask:
                causal_mask = MixedMultiHeadAttentionV2.build_causal_mask(
                    current_L_S, L_NS, x.device, x.dtype,
                    strict_s_causal=self.strict_s_causal,
                )

            # ---------- Static SlotPos: NS×S position bias ----------
            ns_s_bias = None
            if self.use_position_bias and current_L_S > 0 and L_NS > 0:
                ns_s_bias = self.slot_pos_bias[:L_NS, :current_L_S].unsqueeze(0)  # (1, L_NS, L_S)

            # ---------- S×S relative position bias ----------
            s_s_bias = None
            if self.use_seq_pos_bias and current_L_S > 0:
                positions = torch.arange(current_L_S, device=x.device)
                rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L_S, L_S)
                offset = self.pos_bias_max_seq_len - 1
                indices = (rel_pos + offset).clamp(0, 2 * self.pos_bias_max_seq_len - 2)
                s_s_bias = self.seq_pos_bias_table[indices]  # (L_S, L_S)

            # ---------- TCPB: per-layer NS×S bias (additive on top of SlotPos) ----------
            if self.use_target_time_bias and current_L_S > 0 and L_NS > 0:
                idx = 0 if self.tcpb_share_layers else layer_idx
                R = self.target_time_rank
                B_ = cond.size(0)
                M = self.tcpb_proj[idx](cond).view(B_, L_NS, R)        # (B, L_NS, R)
                pos_idx = torch.arange(current_L_S, device=x.device).clamp_max(
                    self.tcpb_pos_emb[idx].num_embeddings - 1
                )
                P = self.tcpb_pos_emb[idx](pos_idx)                     # (L_S, R)
                tcpb_bias = torch.einsum("bnr,lr->bnl", M, P)           # (B, L_NS, L_S)
                if ns_s_bias is not None:
                    ns_s_bias = ns_s_bias + tcpb_bias
                else:
                    ns_s_bias = tcpb_bias

            x = block(
                x,
                num_s_tokens=current_L_S,
                causal_mask=causal_mask,
                ns_s_bias=ns_s_bias,
                s_s_bias=s_s_bias,
            )

            # Pyramid Pruning: 裁减 S-tokens（右对齐，保留最近行为）
            if self.pyramid_prune_ratio > 0 and current_L_S > 1:
                target_L_S = max(1, int(current_L_S * (1 - self.pyramid_prune_ratio)))
                if target_L_S < current_L_S:
                    keep_start = current_L_S - target_L_S
                    x_s = x[:, keep_start:current_L_S]
                    x_ns = x[:, current_L_S:]
                    x = torch.cat([x_s, x_ns], dim=1)
                    current_L_S = target_L_S

        x = self.final_norm(x)

        # Pool NS-tokens for prediction
        ns_out = x[:, current_L_S:]
        ns_flat = ns_out.flatten(start_dim=1)

        if self.use_din_shortcut:
            final = torch.cat([ns_flat, interest, target_emb], dim=1)
        else:
            final = ns_flat
        return self.top_mlp(final).squeeze(-1)
