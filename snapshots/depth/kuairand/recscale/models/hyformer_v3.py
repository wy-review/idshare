"""
recscale.models.hyformer_v3 — HyFormer V3(论文严格复现 + LENS 优化)

设计文档:`recscale/docs/hyformer_v3_design.md`(已与 reviewer 讨论 2 轮 + 用户拍板)

相对 V2 的核心变化:
  1. SeqEncoder per-layer(论文 Eq.5/6/7 都带下标 l)
  2. LONGER 输出保持 L_H 长度,不反向映射回 L_S(Eq.6)
  3. Query Boosting RankMixer-style:零参数 reshape + PerToken-FFN(Eq.10–15)
     - 复用 `rankmixer_v2.TokenMixingV2` + `PerTokenFFN`
     - 不复用 `RankMixerV2Block`(避免内部 LN/residual 与外层 Pre-LN 冲突)
     - 强制 `hidden_dim % T == 0`,T = sum(num_queries_per_sequence) + num_ns_tokens
     - 报错时给出可行的 `num_sparse_buckets` 候选
  4. NS token bucketing 复用 `base.FeatureFieldPooling`(static hash + sum pool)
  5. Multi-sequence 严格按论文 §3.7 / 图 2:
     - 每条序列独立的 query 集
     - 每条序列独立的 cross-attn(per-layer)
     - Boosting 阶段 concat 所有序列的 queries + NS tokens
  6. `attn_num_heads` 与 boosting 的 H=T 解耦
  7. QueryGen 默认 'independent'(论文 Eq.3)
  8. DIN shortcut 默认 False(论文未含此模块)
  9. 删除 `use_v1_residual` 历史兼容分支
 10. 删除 LONGER 反向映射代码(`longer_back_*`)

保留(LENS 论文优化点):
  - QueryPos(`use_position_bias`)
  - TCQG(`use_target_gating`,`2σ` + 零初始化)
  - TCPB(`use_target_time_bias`,low-rank `r=8`)
  - condition source: `item` / `item_seq`(LENS 论文);`item_side` / `item_side_seq`(扩展)

保留(扩展消融开关,默认关):
  - `tcqg_per_layer`:每层重新 gate
  - `diversity_weight`:query 正交化辅助 loss
  - `share_pos_bias_across_layers`:QueryPos 跨层共享

架构:
  for each sequence s:
    seq_repr_raw_s = item_emb_s(seq_s)                            # B8: mean-pool 用 raw
    seq_repr_s     = seq_repr_raw_s [+ pos_emb] [+ action_emb]    # SeqEncoder cross-attn 用增强版
    seq_mask_s     = (seq_s != 0)
    seq_mean_s     = MeanPool(seq_repr_raw_s, seq_mask_s)         # B8 fix: 只用 raw item_emb,
                                                                   # 与 target_emb 在同一空间
    target_emb_s = item_emb_s(target_s)

  Global Info = Concat(F_1, ..., F_M, target_emb_1, ..., target_emb_S, seq_mean_1, ..., seq_mean_S)

  for each sequence s:
    queries_s = QueryGen_s(Global Info)                    # N_s 套独立 FFN(Eq.3)
    if use_target_gating: queries_s ⊙= 2σ(W_t · c)         # LENS TCQG

  ns_tokens = [sparse_buckets, dense_token, target_1, ..., target_S]  # bucketing 后 (B, M, D)
  T = sum(N_s) + M;assert hidden_dim % T == 0

  for layer l in range(num_layers):
    for each sequence s:
      H_l_s, kv_mask_l_s = SeqEncoder_l_s(seq_repr_s, seq_mask_s)   # per-layer 独立编码
      queries_s = queries_s + CrossAttn_l_s(LN(queries_s), H_l_s, kv_mask_l_s, c)
      if tcqg_per_layer: queries_s ⊙= 2σ(W_t^l_s · c)

    # Boosting(在所有 sequence 之间共享同一个 Boosting 模块)
    X = Concat(*[LN_q^l_s(queries_s) for s in S], LN_ns^l(ns_tokens))   # (B, T, D)
    out = PerTokenFFN(TokenMixing(X))                                    # (B, T, D)
    queries_s = queries_s + out[s_offset:s_offset+N_s] for each s
    ns_tokens = ns_tokens + out[ns_offset:]

  final_norm(queries_s) for each s → flatten/concat → MLP → logit
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, DenseArch, SparseArch, FeatureFieldPooling
from .rankmixer_v2 import TokenMixingV2, PerTokenFFN


# =====================================================================
# Helpers
# =====================================================================


def _enumerate_valid_sparse_buckets(
    hidden_dim: int,
    query_tokens: int,
    fixed_ns_tokens: int,
    num_sparse: int,
) -> List[int]:
    """
    枚举使 `hidden_dim % T == 0` 的所有可行 `num_sparse_buckets`。

    T = query_tokens + num_sparse_buckets + fixed_ns_tokens
    fixed_ns_tokens = (1 if has_dense else 0) + num_targets
    """
    candidates = []
    for T in range(1, hidden_dim + 1):
        if hidden_dim % T != 0:
            continue
        buckets = T - query_tokens - fixed_ns_tokens
        if 1 <= buckets <= num_sparse:
            candidates.append(buckets)
    return candidates


def _normalize_num_queries(num_queries_arg, num_sequences: int, fallback: int) -> List[int]:
    """`num_queries_per_sequence` 配置项转 list[int]:
        None → [fallback] * num_sequences
        int  → [int] * num_sequences
        list → 检查长度后原样返回
    """
    if num_queries_arg is None:
        return [fallback] * num_sequences
    if isinstance(num_queries_arg, int):
        return [num_queries_arg] * num_sequences
    if isinstance(num_queries_arg, (list, tuple)):
        if len(num_queries_arg) != num_sequences:
            raise ValueError(
                f"num_queries_per_sequence length {len(num_queries_arg)} "
                f"!= num_sequences {num_sequences}"
            )
        return [int(x) for x in num_queries_arg]
    raise TypeError(
        f"num_queries_per_sequence must be None/int/list, got {type(num_queries_arg)}"
    )


# =====================================================================
# Query Generation V3(论文 Eq.3:N 套独立 2 层 MLP)
# =====================================================================


class QueryGenerationV3(nn.Module):
    """
    Q = [FFN_1(GI), FFN_2(GI), ..., FFN_N(GI)] ∈ R^{N×D}     (论文 Eq.3)

    默认 `mode='independent'`(N 套独立参数);`mode='shared'` 是 V2 工程优化,保留作为消融。

    可选 LENS TCQG:`use_target_gating=True` 时,
        Q_0 = Q_raw ⊙ g(c),其中 g(c) = 2σ(W_t · c) 默认零初始化(Q_0=Q_raw at step 0)
    """

    def __init__(
        self,
        input_dim: int,
        num_queries: int,
        hidden_dim: int,
        mode: str = "independent",
        use_target_gating: bool = False,
        target_emb_dim: Optional[int] = None,
        gate_activation_mode: str = "2sigmoid",
        gate_network_mode: str = "linear",
        target_gate_hidden_dim: Optional[int] = None,
        target_gate_scale: float = 1.0,
    ):
        super().__init__()
        if mode not in ("independent", "shared"):
            raise ValueError(f"mode must be 'independent' or 'shared', got {mode!r}")
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.use_target_gating = use_target_gating
        assert gate_activation_mode in ("2sigmoid", "sigmoid", "residual_tanh"), gate_activation_mode
        assert gate_network_mode in ("linear", "mlp"), gate_network_mode
        # B7 fix: target_gate_scale 仅在 residual_tanh 模式下生效;其他模式下传非 1.0 容易误用
        if target_gate_scale != 1.0 and gate_activation_mode != "residual_tanh":
            raise ValueError(
                f"target_gate_scale={target_gate_scale} only takes effect when "
                f"gate_activation_mode='residual_tanh' (got {gate_activation_mode!r}). "
                f"Use 'residual_tanh' or keep target_gate_scale=1.0."
            )
        self.gate_activation_mode = gate_activation_mode
        self.gate_network_mode = gate_network_mode
        self.target_gate_scale = float(target_gate_scale)

        if mode == "independent":
            # 论文 Eq.3:N 套独立 2 层 MLP
            self.projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_queries)
            ])
        else:
            # 工程优化版本(消融用)
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * num_queries),
                nn.GELU(),
                nn.Linear(hidden_dim * num_queries, hidden_dim * num_queries),
            )

        if use_target_gating:
            if target_emb_dim is None:
                target_emb_dim = hidden_dim
            if gate_network_mode == "mlp":
                if target_gate_hidden_dim is None:
                    target_gate_hidden_dim = max(hidden_dim, target_emb_dim)
                self.target_gate = nn.Sequential(
                    nn.LayerNorm(target_emb_dim),
                    nn.Linear(target_emb_dim, target_gate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(target_gate_hidden_dim, num_queries * hidden_dim),
                )
                nn.init.zeros_(self.target_gate[-1].weight)
                nn.init.zeros_(self.target_gate[-1].bias)
            else:
                self.target_gate = nn.Linear(target_emb_dim, num_queries * hidden_dim)
                nn.init.zeros_(self.target_gate.weight)
                nn.init.zeros_(self.target_gate.bias)

    def forward(self, features: torch.Tensor, target_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        features: (B, input_dim) — Global Info 拼接结果
        target_emb: (B, target_emb_dim) — 仅当 use_target_gating=True 时必需
        Returns: (B, N, D)
        """
        if self.mode == "independent":
            out = torch.stack([proj(features) for proj in self.projs], dim=1)
        else:
            out = self.proj(features).view(-1, self.num_queries, self.hidden_dim)

        if self.use_target_gating:
            if target_emb is None:
                raise ValueError("target_emb required when use_target_gating=True")
            gate_logits = self.target_gate(target_emb).view(-1, self.num_queries, self.hidden_dim)
            if self.gate_activation_mode == "2sigmoid":
                gate = 2.0 * torch.sigmoid(gate_logits)
            elif self.gate_activation_mode == "residual_tanh":
                gate = 1.0 + self.target_gate_scale * torch.tanh(gate_logits)
            else:  # "sigmoid"
                gate = torch.sigmoid(gate_logits)
            out = out * gate

        return out


# =====================================================================
# Query Decoding(cross-attn,带 LENS QueryPos / 全局 pos bias / TCPB)
# =====================================================================


class QueryDecoding(nn.Module):
    """
    Multi-head cross-attention from queries to per-layer K/V.

    可选 bias:
      - `use_position_bias=True`:LENS QueryPos,per-query per-position 可学 bias
      - `use_global_position_bias=True`:全局 recency curve(对照基线)
      - `use_target_time_bias=True`:LENS TCPB,target 调制的 low-rank bias
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_position_bias: bool = False,
        use_global_position_bias: bool = False,
        use_target_time_bias: bool = False,
        num_queries: int = 8,
        max_seq_len: int = 1024,
        target_time_rank: int = 8,
        target_time_input_dim: Optional[int] = None,
        shared_position_bias: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by attn_num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.use_position_bias = use_position_bias
        self.use_global_position_bias = use_global_position_bias
        self.use_target_time_bias = use_target_time_bias
        if target_time_input_dim is None:
            target_time_input_dim = hidden_dim

        if use_position_bias:
            if shared_position_bias is not None:
                # B5:跨层共享 QueryPos 参数(消融用)。
                # 用 object.__setattr__ 绕过 nn.Module 的 __setattr__,**故意**不把这个
                # Parameter 注册到当前 QueryDecoding 的 _parameters 中。
                # 原因:外层 HyFormerV3Model 已经在 register_parameter("shared_position_bias_seqX")
                # 注册过一次。如果改成 `self.position_bias = shared_position_bias`,
                # 该 Parameter 会出现在 model.parameters() 中 num_layers 次,导致 optimizer
                # 重复更新同一个 Parameter(累积梯度)。
                object.__setattr__(self, "position_bias", shared_position_bias)
            else:
                self.position_bias = nn.Parameter(torch.zeros(num_queries, max_seq_len))
        if use_global_position_bias:
            self.global_position_bias = nn.Parameter(torch.zeros(max_seq_len))
        if use_target_time_bias:
            self.target_time_proj = nn.Linear(target_time_input_dim, num_queries * target_time_rank)
            self.target_time_pos_emb = nn.Embedding(max_seq_len, target_time_rank)
            nn.init.zeros_(self.target_time_proj.weight)
            nn.init.zeros_(self.target_time_proj.bias)
            nn.init.normal_(self.target_time_pos_emb.weight, std=0.02)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        target_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        queries: (B, N_q, D)
        kv:      (B, L, D)
        kv_mask: (B, L) bool, True=valid;LONGER 输出时传 None(全 valid)
        target_emb: (B, target_time_input_dim) for TCPB
        Returns: (B, N_q, D)
        """
        B, N_q, D = queries.shape
        L = kv.size(1)
        H = self.num_heads

        q = self.q_proj(queries).view(B, N_q, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L, H, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N_q, L)

        if self.use_position_bias:
            # B1 defensive check: torch slicing silently truncates if L > size(1),
            # which then breaks broadcast at attn + pos_bias. Fail fast with a clear msg.
            max_len = self.position_bias.size(1)
            if L > max_len:
                raise RuntimeError(
                    f"QueryDecoding: KV length L={L} exceeds position_bias capacity "
                    f"max_seq_len={max_len}. Increase pos_bias_max_len or "
                    f"check dataset.maxlen vs model max_seq_len_per_seq config."
                )
            pos_bias = self.position_bias[:, :L]  # (N_q, L)
            attn = attn + pos_bias.unsqueeze(0).unsqueeze(0)
        if self.use_global_position_bias:
            max_len_g = self.global_position_bias.size(0)
            if L > max_len_g:
                raise RuntimeError(
                    f"QueryDecoding: KV length L={L} exceeds global_position_bias capacity "
                    f"max_seq_len={max_len_g}."
                )
            global_pos_bias = self.global_position_bias[:L]
            attn = attn + global_pos_bias.view(1, 1, 1, L)
        if self.use_target_time_bias:
            if target_emb is None:
                raise ValueError("target_emb is required when use_target_time_bias=True")
            R = self.target_time_pos_emb.embedding_dim
            target_mix = self.target_time_proj(target_emb).view(B, N_q, R)
            pos_idx = torch.arange(L, device=kv.device).clamp_max(self.target_time_pos_emb.num_embeddings - 1)
            pos_basis = self.target_time_pos_emb(pos_idx)  # (L, R)
            dyn_bias = torch.einsum("bqr,lr->bql", target_mix, pos_basis)
            attn = attn + dyn_bias.unsqueeze(1)

        if kv_mask is not None:
            attn = attn.masked_fill(~kv_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # B8:对全 padding 样本(整行 -inf 后 softmax → NaN)做静默零增量处理。
        # 含义:这些样本的 query 不从 history 读取任何信息,cross-attn 输出 = 0(residual 不变)。
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)


# =====================================================================
# SeqEncoder V3(per-layer;LONGER 输出 L_H 长度,无反向映射)
# =====================================================================


class SeqEncoderV3(nn.Module):
    """
    Per-layer sequence encoder. 论文 Eq.5/6/7 的不同实例。

    forward 返回 `(H_l, kv_mask_l)`:
      - `None` / `transformer` / `swiglu`:H_l 与输入同形状 (B, L, D),kv_mask_l = 输入 mask
      - `longer`:H_l 形状 (B, L_H, D),kv_mask_l = None(全 valid)— 论文 Eq.6,
        **不再做反向映射回 L** —
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        encoder_type: Optional[str] = None,
        longer_num_tokens: int = 16,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
        self.longer_num_tokens = longer_num_tokens

        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        elif encoder_type == "longer":
            # 论文 Eq.6:H_l = CrossAttn(S_short, S, S),输出 (B, L_H, D)
            self.longer_query = nn.Parameter(torch.empty(longer_num_tokens, hidden_dim))
            nn.init.normal_(self.longer_query, std=0.02)
            self.longer_q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.longer_k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.longer_v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.longer_out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.longer_norm = nn.LayerNorm(hidden_dim)
            self.longer_dropout = nn.Dropout(dropout)
        elif encoder_type == "swiglu":
            self.w1 = nn.Linear(hidden_dim, ffn_dim)
            self.w2 = nn.Linear(hidden_dim, ffn_dim)
            self.w3 = nn.Linear(ffn_dim, hidden_dim)
        elif encoder_type is None:
            pass
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")

    def _longer_attn(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """LONGER cross-attn:Q_short attends to S(全长度)。"""
        B, L, D = x.shape
        H = self.num_heads
        hd = self.head_dim
        scale = 1.0 / math.sqrt(hd)

        q_short = self.longer_query.unsqueeze(0).expand(B, -1, -1)  # (B, L_H, D)
        Lq = q_short.size(1)

        q_ = self.longer_q_proj(q_short).view(B, Lq, H, hd).transpose(1, 2)
        k_ = self.longer_k_proj(x).view(B, L, H, hd).transpose(1, 2)
        v_ = self.longer_v_proj(x).view(B, L, H, hd).transpose(1, 2)

        attn = torch.matmul(q_, k_.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.longer_dropout(attn)

        out = torch.matmul(attn, v_).transpose(1, 2).reshape(B, Lq, D)
        return self.longer_out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, L, D)
        mask: (B, L) bool, True=valid
        Returns: (H_l, kv_mask_l)
          - None/transformer/swiglu:H_l shape == x shape,kv_mask_l == mask
          - longer:H_l shape (B, L_H, D),kv_mask_l = None(全 valid)
        """
        if self.encoder_type == "transformer":
            if mask is not None:
                # B3 fix: per-sample all-padding rows produce all -inf in self-attn,
                # softmax → NaN. For those samples temporarily flip mask to all-valid;
                # the encoder output for those samples is still driven by zero
                # embeddings so it carries no information, but it doesn't NaN.
                row_has_valid = mask.any(dim=1, keepdim=True)        # (B, 1) True if any valid
                safe_mask = mask | (~row_has_valid)                  # all-padding rows → all True
                out = self.encoder(x, src_key_padding_mask=~safe_mask)
                # Final NaN guard (if upstream still produces non-finite values)
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                out = self.encoder(x)
            return out, mask
        elif self.encoder_type == "longer":
            s_short = self._longer_attn(x, mask)         # (B, L_H, D)
            s_short = self.longer_norm(s_short)
            return s_short, None                          # 全 valid
        elif self.encoder_type == "swiglu":
            out = self.w3(F.silu(self.w1(x)) * self.w2(x))
            return out, mask
        else:  # None
            return x, mask


# =====================================================================
# Query Boosting V3(论文 Eq.10–15:RankMixer-style)
# =====================================================================


class QueryBoostingV3(nn.Module):
    """
    Query Boosting V3:严格按论文 Eq.10–15。

    内部:
      Eq.13 — `TokenMixingV2`:零参数 reshape/permute(从 rankmixer_v2 复用)
      Eq.14 — `PerTokenFFN`:T 套独立 FFN(从 rankmixer_v2 复用)

    **不复用** `RankMixerV2Block`,因为它内部有 LN + 双 residual,会与外层 Pre-LN block
    的 residual 叠加,语义错。

    `forward()` 返回的是 *increment*(等价于论文 Eq.14 的 \\tilde Q)。
    论文 Eq.15 的 residual `Q_boost = Q + tilde Q` 由外层 `HyFormerBlockV3` 统一处理。
    """

    def __init__(
        self,
        total_tokens: int,
        hidden_dim: int,
        ffn_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % total_tokens != 0:
            raise ValueError(
                f"QueryBoostingV3: hidden_dim={hidden_dim} must be divisible by "
                f"total_tokens={total_tokens}"
            )
        self.total_tokens = total_tokens
        self.hidden_dim = hidden_dim
        # 复用底层模块,不复用 RankMixerV2Block
        self.token_mixing = TokenMixingV2(total_tokens, hidden_dim)
        self.per_token_ffn = PerTokenFFN(total_tokens, hidden_dim, ffn_dim, dropout)

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        """
        x_normed: (B, T, D),已经在外层做完 LN 并 concat 好(顺序:queries_1, ..., queries_S, ns_tokens)
        Returns: (B, T, D),纯 increment;外层加 residual
        """
        x_hat = self.token_mixing(x_normed)        # 零参数 reshape/permute (Eq.13)
        return self.per_token_ffn(x_hat)            # PerToken-FFN (Eq.14);外层加 Eq.15 residual


# =====================================================================
# DIN-style Target Attention(可选 shortcut,默认关)
# =====================================================================


class TargetAttention(nn.Module):
    """DIN-style target attention shortcut(默认关闭,与论文一致)。"""

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


# =====================================================================
# HyFormer V3 Block(per-layer SeqEncoder + multi-seq cross-attn + 共享 Boosting)
# =====================================================================


class HyFormerBlockV3(nn.Module):
    """
    单层 HyFormer V3 block:

    forward 输入:
      queries_list: List[(B, N_s, D)] — 每条 sequence 的 queries
      ns_tokens:    (B, M, D)
      seq_repr_list:List[(B, L_s, D)] — 每条 sequence 的原始 embedding(pre-encoder)
      seq_mask_list:List[(B, L_s)]    — 每条 sequence 的 mask
      target_condition: (B, d_c) — TCQG/TCPB 共享的条件向量

    流程:
      for s in S:
        H_l_s, kv_mask_l_s = seq_encoders[s](seq_repr_list[s], seq_mask_list[s])
        queries_s = queries_s + cross_attns[s](LN(queries_s), H_l_s, kv_mask_l_s, c)
        if tcqg_per_layer: queries_s ⊙= 2σ(W_t · c)

      X = Concat(*[LN_q[s](queries_s) for s], LN_ns(ns_tokens))   # (B, T, D)
      out = boosting(X)                                             # 纯 increment
      queries_s = queries_s + out[s_offset:s_offset+N_s] for each s
      ns_tokens = ns_tokens + out[ns_offset:]
    """

    def __init__(
        self,
        num_sequences: int,
        num_queries_per_seq: List[int],
        num_ns_tokens: int,
        hidden_dim: int,
        attn_num_heads: int,
        ffn_dim: int = 128,
        dropout: float = 0.0,
        # SeqEncoder 配置
        seq_encoder_type: Optional[str] = None,
        seq_encoder_ffn_dim: int = 256,
        longer_num_tokens: int = 16,
        # Cross-attn bias 配置
        use_position_bias: bool = False,
        use_global_position_bias: bool = False,
        use_target_time_bias: bool = False,
        max_seq_len_per_seq: Optional[List[int]] = None,
        target_time_rank: int = 8,
        target_time_input_dim: Optional[int] = None,
        shared_position_bias_per_seq: Optional[List[Optional[nn.Parameter]]] = None,
        # 扩展开关
        use_tcqg_per_layer: bool = False,
        tcqg_per_layer_norm: bool = False,
        tcqg_activation_mode: str = "2sigmoid",
        tcqg_target_gate_scale: float = 1.0,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries_per_seq = num_queries_per_seq
        self.num_ns_tokens = num_ns_tokens
        self.hidden_dim = hidden_dim
        self.use_tcqg_per_layer = use_tcqg_per_layer
        self.tcqg_per_layer_norm = tcqg_per_layer_norm
        self.tcqg_activation_mode = tcqg_activation_mode
        self.tcqg_target_gate_scale = float(tcqg_target_gate_scale)

        if max_seq_len_per_seq is None:
            max_seq_len_per_seq = [1024] * num_sequences
        if shared_position_bias_per_seq is None:
            shared_position_bias_per_seq = [None] * num_sequences

        # Per-sequence per-layer SeqEncoder
        self.seq_encoders = nn.ModuleList([
            SeqEncoderV3(
                hidden_dim=hidden_dim,
                num_heads=attn_num_heads,
                ffn_dim=seq_encoder_ffn_dim,
                dropout=dropout,
                encoder_type=seq_encoder_type,
                longer_num_tokens=longer_num_tokens,
            )
            for _ in range(num_sequences)
        ])

        # Per-sequence Pre-LN before cross-attn
        self.norm1_list = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_sequences)
        ])

        # Per-sequence cross-attn
        # 注意:cross-attn 的 max_seq_len 应该是 KV 真实长度。
        # LONGER 时 KV 长度是 longer_num_tokens(全 valid),而不是原始 max_seq_len。
        self.cross_attns = nn.ModuleList()
        for s in range(num_sequences):
            attn_kv_max_len = (
                longer_num_tokens
                if seq_encoder_type == "longer"
                else max_seq_len_per_seq[s]
            )
            self.cross_attns.append(
                QueryDecoding(
                    hidden_dim=hidden_dim,
                    num_heads=attn_num_heads,
                    dropout=dropout,
                    use_position_bias=use_position_bias,
                    use_global_position_bias=use_global_position_bias,
                    use_target_time_bias=use_target_time_bias,
                    num_queries=num_queries_per_seq[s],
                    max_seq_len=attn_kv_max_len,
                    target_time_rank=target_time_rank,
                    target_time_input_dim=target_time_input_dim,
                    shared_position_bias=shared_position_bias_per_seq[s],
                )
            )

        # Per-sequence per-layer TCQG(扩展开关,默认关)
        if use_tcqg_per_layer:
            if target_time_input_dim is None:
                target_time_input_dim = hidden_dim
            self.per_layer_tcqg = nn.ModuleList([
                nn.Linear(target_time_input_dim, num_queries_per_seq[s] * hidden_dim)
                for s in range(num_sequences)
            ])
            for proj in self.per_layer_tcqg:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            if tcqg_per_layer_norm:
                self.per_layer_tcqg_ln = nn.ModuleList([
                    nn.LayerNorm(hidden_dim) for _ in range(num_sequences)
                ])

        # Pre-LN before Boosting:per-sequence queries + 1 个 ns LN
        self.norm2_q_list = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_sequences)
        ])
        if num_ns_tokens > 0:
            self.norm2_ns = nn.LayerNorm(hidden_dim)

        total_tokens = sum(num_queries_per_seq) + num_ns_tokens
        self.total_tokens = total_tokens
        self.boosting = QueryBoostingV3(
            total_tokens=total_tokens,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def _tcqg_gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.tcqg_activation_mode == "2sigmoid":
            return 2.0 * torch.sigmoid(logits)
        elif self.tcqg_activation_mode == "residual_tanh":
            return 1.0 + self.tcqg_target_gate_scale * torch.tanh(logits)
        else:  # "sigmoid"
            return torch.sigmoid(logits)

    def forward(
        self,
        queries_list: List[torch.Tensor],          # [(B, N_s, D)] × S
        ns_tokens: torch.Tensor,                    # (B, M, D)
        seq_repr_list: List[torch.Tensor],          # [(B, L_s, D)] × S(原始 embedding)
        seq_mask_list: List[Optional[torch.Tensor]],
        target_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        S = self.num_sequences
        assert len(queries_list) == S, f"expected {S} queries, got {len(queries_list)}"
        assert len(seq_repr_list) == S
        assert len(seq_mask_list) == S

        # ---- Per-sequence Query Decoding ----
        for s in range(S):
            # 每层独立的 H_l_s
            H_l_s, kv_mask_l_s = self.seq_encoders[s](seq_repr_list[s], seq_mask_list[s])

            q_normed = self.norm1_list[s](queries_list[s])
            decoded = self.cross_attns[s](
                q_normed, H_l_s, kv_mask_l_s, target_emb=target_condition
            )
            queries_list[s] = queries_list[s] + decoded

            # 扩展:per-layer TCQG re-gate
            if self.use_tcqg_per_layer:
                gl = self.per_layer_tcqg[s](target_condition)
                gl = gl.view(-1, self.num_queries_per_seq[s], queries_list[s].size(-1))
                queries_list[s] = queries_list[s] * self._tcqg_gate(gl)
                if self.tcqg_per_layer_norm:
                    queries_list[s] = self.per_layer_tcqg_ln[s](queries_list[s])

        # ---- Query Boosting(所有 sequence + NS tokens 一起做 token mixing)----
        # X 顺序:queries_1, queries_2, ..., queries_S, ns_tokens
        normed_parts = []
        for s in range(S):
            normed_parts.append(self.norm2_q_list[s](queries_list[s]))
        if self.num_ns_tokens > 0:
            normed_parts.append(self.norm2_ns(ns_tokens))
        X = torch.cat(normed_parts, dim=1)   # (B, T, D)

        inc = self.boosting(X)                # (B, T, D),纯 increment

        # split + 外层 residual(对应论文 Eq.15)
        offset = 0
        for s in range(S):
            n_q_s = self.num_queries_per_seq[s]
            queries_list[s] = queries_list[s] + inc[:, offset:offset + n_q_s]
            offset += n_q_s
        if self.num_ns_tokens > 0:
            ns_tokens = ns_tokens + inc[:, offset:]

        return queries_list, ns_tokens


# =====================================================================
# HyFormer V3 Top-level Model
# =====================================================================


@register_model
class HyFormerV3Model(RecModel):
    """
    HyFormer V3:严格复现 HyFormer 论文 + LENS 优化点(详见文件头注释)。

    主要 config(`config["model"]`):
      hidden_dim:        D — token 维度
      num_layers:        HyFormer block 层数
      num_queries:       默认每条序列的 query 数(被 num_queries_per_sequence 覆盖)
      num_queries_per_sequence:  int 或 list[int],未指定则 = [num_queries] * num_sequences
      num_heads / attn_num_heads: cross-attn / SeqEncoder 内部多头数(Boosting 不读)
      ffn_dim:           Boosting PerTokenFFN 中间维度
      mlp_dims:          顶层 MLP

      seq_encoder_type:  None / "transformer" / "longer" / "swiglu"
      seq_encoder_ffn_dim:  SeqEncoder 内 FFN 维度
      longer_num_tokens: LONGER 的 L_H

      num_sparse_buckets:NS bucketing 数量;不指定则 = num_sparse(每个 sparse_col 一 token)
                         T = sum(N_s) + num_sparse_buckets + (1 if dense else 0) + S
                         必须满足 hidden_dim % T == 0,否则报错并提示候选

      q_gen_mode:        "independent"(默认,论文 Eq.3) / "shared"(消融)

      use_din_shortcut:  默认 False(论文未含)
      use_mean_pool:     最终输出是否拼接 MeanPool(seq)(默认 True)
      use_meanpool_in_qgen: QueryGen Global Info 是否含 MeanPool(seq)(默认 True,论文 Eq.4)

    LENS 开关:
      use_position_bias:   QueryPos
      use_global_position_bias / use_abs_seq_pos_emb: 对照基线
      use_target_gating:   TCQG
      use_target_time_bias / target_time_rank:        TCPB
      target_condition_source: "item"/"item_seq"(LENS) | "item_side"/"item_side_seq"(扩展)

    扩展消融开关(默认关):
      tcqg_per_layer / tcqg_per_layer_norm
      diversity_weight
      share_pos_bias_across_layers
    """

    model_name = "hyformer_v3"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        # ---- 基本超参 ----
        emb_dim = mc.get("embedding_dim", 16)
        hidden_dim = mc.get("hidden_dim", 128)
        num_layers = mc.get("num_layers", 3)
        num_heads = mc.get("num_heads", 4)
        # Q7:cross-attn / SeqEncoder 用 attn_num_heads,默认沿用 num_heads
        attn_num_heads = mc.get("attn_num_heads", num_heads)
        num_queries_default = mc.get("num_queries", 8)
        ffn_dim = mc.get("ffn_dim", hidden_dim * 2)
        seq_encoder_ffn_dim = mc.get("seq_encoder_ffn_dim", ffn_dim)
        dropout = mc.get("dropout", 0.1)
        mlp_dims = mc.get("mlp_dims", [128, 64])

        # ---- Sequence 配置 ----
        self.has_seq2 = dc.get("num_items2", 0) > 0
        num_sequences = 2 if self.has_seq2 else 1
        self.num_sequences = num_sequences
        # Q2:num_queries_per_sequence,默认 = [num_queries] * num_sequences
        num_queries_per_seq = _normalize_num_queries(
            mc.get("num_queries_per_sequence"),
            num_sequences,
            fallback=num_queries_default,
        )
        self.num_queries_per_seq = num_queries_per_seq

        # ---- 序列编码器配置 ----
        self.seq_encoder_type = mc.get("seq_encoder_type", None)
        self.longer_num_tokens = int(mc.get("longer_num_tokens", 16))

        # ---- 数据集字段 ----
        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)
        num_items2 = dc.get("num_items2", 0)

        # 序列长度上限(per-sequence)
        max_seq_len_default = int(dc.get("maxlen", 1024))
        max_seq_len_seq2 = int(dc.get("maxlen2", max_seq_len_default))
        max_seq_len_per_seq = (
            [max_seq_len_default, max_seq_len_seq2] if self.has_seq2 else [max_seq_len_default]
        )

        # ---- NS Token Bucketing ----
        # Q3/Q4:默认不 pool(num_sparse_buckets = num_sparse);失败时显式报错
        num_sparse_buckets = int(mc.get("num_sparse_buckets", num_sparse))
        if num_sparse > 0 and num_sparse_buckets < 1:
            raise ValueError(
                f"num_sparse_buckets={num_sparse_buckets} must be >= 1 when num_sparse={num_sparse}"
            )
        if num_sparse_buckets > num_sparse:
            raise ValueError(
                f"num_sparse_buckets={num_sparse_buckets} cannot exceed num_sparse={num_sparse}"
            )
        self.num_sparse_buckets = num_sparse_buckets

        # 计算 ns_token 总数:sparse_buckets + (1 if dense) + num_sequences (target tokens)
        fixed_ns_tokens = (1 if num_dense > 0 else 0) + num_sequences
        num_ns_tokens = num_sparse_buckets + fixed_ns_tokens
        self.num_ns_tokens = num_ns_tokens

        # ---- T 整除约束(RankMixer-style boosting)----
        query_tokens = sum(num_queries_per_seq)
        T = query_tokens + num_ns_tokens
        if hidden_dim % T != 0:
            candidates = _enumerate_valid_sparse_buckets(
                hidden_dim, query_tokens, fixed_ns_tokens, num_sparse
            )
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by total tokens T={T}\n"
                f"  query_tokens={query_tokens} (= sum({num_queries_per_seq}))\n"
                f"  num_sparse_buckets={num_sparse_buckets}\n"
                f"  fixed_ns_tokens={fixed_ns_tokens} "
                f"(= {(1 if num_dense > 0 else 0)} dense + {num_sequences} target)\n"
                f"  Try num_sparse_buckets in {candidates} "
                f"or adjust hidden_dim/num_queries_per_sequence."
            )
        self.total_tokens = T

        # ---- LENS 开关 ----
        self.use_position_bias = mc.get("use_position_bias", False)
        self.use_global_position_bias = mc.get("use_global_position_bias", False)
        self.use_abs_seq_pos_emb = mc.get("use_abs_seq_pos_emb", False)
        self.use_target_time_bias = mc.get("use_target_time_bias", False)
        self.target_time_rank = int(mc.get("target_time_rank", 8))
        self.use_target_gating = mc.get("use_target_gating", False)
        self._gate_activation_mode = mc.get("gate_activation_mode", "2sigmoid")
        self._target_gate_scale = float(mc.get("target_gate_scale", 1.0))

        self.target_condition_source = mc.get("target_condition_source", "item")
        if self.target_condition_source not in ("item", "item_seq", "item_side", "item_side_seq"):
            raise ValueError(f"Unknown target_condition_source: {self.target_condition_source!r}")
        self.target_condition_detach = bool(mc.get("target_condition_detach", False))
        self.target_condition_embedding = mc.get("target_condition_embedding", "shared")
        if self.target_condition_embedding not in ("shared", "separate"):
            raise ValueError("target_condition_embedding must be 'shared' or 'separate'")
        self.target_condition_embedding_frozen = bool(mc.get("target_condition_embedding_frozen", False))
        self.target_condition_align_weight = float(mc.get("target_condition_align_weight", 0.0))

        # ---- 扩展消融开关(默认关)----
        # Q5:删 use_v1_residual(V3 完全没有这个概念)
        self.tcqg_per_layer = bool(mc.get("tcqg_per_layer", False))
        self.tcqg_per_layer_norm = bool(mc.get("tcqg_per_layer_norm", False))
        self.diversity_weight = float(mc.get("diversity_weight", 0.0))
        self.share_pos_bias_across_layers = bool(mc.get("share_pos_bias_across_layers", False))

        # ---- 其他保留开关 ----
        # Q1:QueryGen 默认 independent
        self.q_gen_mode = mc.get("q_gen_mode", "independent")
        self.use_mean_pool = bool(mc.get("use_mean_pool", True))
        self.use_meanpool_in_qgen = bool(mc.get("use_meanpool_in_qgen", True))
        # Q4(P4):DIN shortcut 默认 False(论文未含)
        self.use_din_shortcut = bool(mc.get("use_din_shortcut", False))

        self.use_seq_action_type_embedding = bool(mc.get("use_seq_action_type_embedding", False))
        self.seq_action_fusion = mc.get("seq_action_fusion", "add")
        num_seq_action_types = int(mc.get("num_seq_action_types", 5))

        # ---- Embedding 模块 ----
        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None
        self.has_dense = num_dense > 0

        # 把 sparse 从 emb_dim 投到 hidden_dim
        self.sparse_proj = (
            nn.Linear(emb_dim, hidden_dim) if emb_dim != hidden_dim else nn.Identity()
        )
        if self.has_dense:
            self.dense_proj = nn.Sequential(
                nn.Linear(num_dense, hidden_dim),
                nn.ReLU(),
            )

        # NS bucketing(可选,在 emb_dim 空间内 sum-pool)
        if num_sparse > 0 and num_sparse_buckets < num_sparse:
            self.sparse_field_pool = FeatureFieldPooling(
                num_features=num_sparse,
                num_fields=num_sparse_buckets,
                seed=int(mc.get("sparse_bucket_seed", 42)),
            )
        else:
            self.sparse_field_pool = None

        # 主序列 item embedding
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.item_emb.weight.data[0].zero_()
        if self.use_abs_seq_pos_emb:
            self.seq_pos_emb = nn.Embedding(max_seq_len_default, hidden_dim)
            nn.init.normal_(self.seq_pos_emb.weight, std=0.02)
        if self.use_seq_action_type_embedding:
            self.seq_action_emb = nn.Embedding(num_seq_action_types, hidden_dim, padding_idx=0)
            nn.init.normal_(self.seq_action_emb.weight, std=0.02)
            self.seq_action_emb.weight.data[0].zero_()
            if self.seq_action_fusion == "concat_mlp":
                self.seq_action_fuse = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            elif self.seq_action_fusion != "add":
                raise ValueError(f"Unsupported seq_action_fusion={self.seq_action_fusion!r}")

        # 第二序列(可选)
        if self.has_seq2:
            self.item_emb2 = nn.Embedding(num_items2 + 1, hidden_dim, padding_idx=0)
            nn.init.normal_(self.item_emb2.weight, std=0.02)
            self.item_emb2.weight.data[0].zero_()
            if self.use_abs_seq_pos_emb:
                self.seq_pos_emb2 = nn.Embedding(max_seq_len_seq2, hidden_dim)
                nn.init.normal_(self.seq_pos_emb2.weight, std=0.02)

        # target_condition 用的独立 emb table(可选,LENS 扩展)
        if self.target_condition_embedding == "separate":
            rng_state = torch.get_rng_state()
            self.target_condition_item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
            torch.set_rng_state(rng_state)
            self.target_condition_item_emb.weight.data.copy_(self.item_emb.weight.data)
            if self.target_condition_embedding_frozen:
                self.target_condition_item_emb.weight.requires_grad_(False)
            if self.has_seq2:
                rng_state = torch.get_rng_state()
                self.target_condition_item_emb2 = nn.Embedding(num_items2 + 1, hidden_dim, padding_idx=0)
                torch.set_rng_state(rng_state)
                self.target_condition_item_emb2.weight.data.copy_(self.item_emb2.weight.data)
                if self.target_condition_embedding_frozen:
                    self.target_condition_item_emb2.weight.requires_grad_(False)
            else:
                self.target_condition_item_emb2 = None
        else:
            self.target_condition_item_emb = None
            self.target_condition_item_emb2 = None

        # ---- 算 target_condition 维度(用于 TCQG/TCPB 的 W_t)----
        qgen_side_dim = num_sparse * emb_dim + (num_dense if self.has_dense else 0)
        target_condition_dim = hidden_dim
        if self.target_condition_source != "item":
            if self.has_seq2:
                target_condition_dim += hidden_dim
            if self.target_condition_source in ("item_side", "item_side_seq"):
                target_condition_dim += qgen_side_dim
            if self.target_condition_source in ("item_seq", "item_side_seq"):
                target_condition_dim += hidden_dim
                if self.has_seq2:
                    target_condition_dim += hidden_dim
        self.target_condition_dim = target_condition_dim

        # ---- QueryGen Global Info 维度(论文 Eq.4)----
        # GI = sparse_flat + dense_raw + target_emb(per seq) + meanpool(seq, per seq)
        feat_dim = qgen_side_dim
        feat_dim += hidden_dim * num_sequences      # target_emb_1, ..., target_emb_S
        if self.use_meanpool_in_qgen:
            feat_dim += hidden_dim * num_sequences  # seq_mean_1, ..., seq_mean_S
        self.feat_dim = feat_dim

        # ---- 每条序列独立的 QueryGen ----
        self.query_gens = nn.ModuleList([
            QueryGenerationV3(
                input_dim=feat_dim,
                num_queries=num_queries_per_seq[s],
                hidden_dim=hidden_dim,
                mode=self.q_gen_mode,
                use_target_gating=self.use_target_gating,
                target_emb_dim=target_condition_dim,
                gate_activation_mode=self._gate_activation_mode,
                gate_network_mode=mc.get("target_gate_network", "linear"),
                target_gate_hidden_dim=mc.get("target_gate_hidden_dim"),
                target_gate_scale=self._target_gate_scale,
            )
            for s in range(num_sequences)
        ])

        # ---- 跨层共享的 QueryPos(消融,默认 False)----
        shared_pb_per_seq: List[Optional[nn.Parameter]] = [None] * num_sequences
        if self.use_position_bias and self.share_pos_bias_across_layers:
            for s in range(num_sequences):
                attn_kv_max_len = (
                    self.longer_num_tokens
                    if self.seq_encoder_type == "longer"
                    else max_seq_len_per_seq[s]
                )
                p = nn.Parameter(torch.zeros(num_queries_per_seq[s], attn_kv_max_len))
                self.register_parameter(f"shared_position_bias_seq{s}", p)
                shared_pb_per_seq[s] = p

        # ---- HyFormer V3 Block 堆叠 ----
        self.blocks = nn.ModuleList([
            HyFormerBlockV3(
                num_sequences=num_sequences,
                num_queries_per_seq=num_queries_per_seq,
                num_ns_tokens=num_ns_tokens,
                hidden_dim=hidden_dim,
                attn_num_heads=attn_num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                seq_encoder_type=self.seq_encoder_type,
                seq_encoder_ffn_dim=seq_encoder_ffn_dim,
                longer_num_tokens=self.longer_num_tokens,
                use_position_bias=self.use_position_bias,
                use_global_position_bias=self.use_global_position_bias,
                use_target_time_bias=self.use_target_time_bias,
                max_seq_len_per_seq=max_seq_len_per_seq,
                target_time_rank=self.target_time_rank,
                target_time_input_dim=target_condition_dim,
                shared_position_bias_per_seq=shared_pb_per_seq,
                use_tcqg_per_layer=self.use_target_gating and self.tcqg_per_layer,
                tcqg_per_layer_norm=self.tcqg_per_layer_norm,
                tcqg_activation_mode=self._gate_activation_mode,
                tcqg_target_gate_scale=self._target_gate_scale,
            )
            for _ in range(num_layers)
        ])

        # ---- Final norm(per-sequence,因为各 sequence 的 queries 分布可能不同)----
        self.final_norm_list = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_sequences)
        ])

        # ---- 可选 DIN shortcut(默认关)----
        if self.use_din_shortcut:
            self.target_attn_list = nn.ModuleList([
                TargetAttention(hidden_dim, hidden_dim) for _ in range(num_sequences)
            ])
        else:
            self.target_attn_list = None

        # ---- Top MLP ----
        in_dim = sum(num_queries_per_seq) * hidden_dim
        if self.use_din_shortcut:
            in_dim += (hidden_dim + hidden_dim) * num_sequences  # interest_s + target_emb_s
        if self.use_mean_pool:
            in_dim += hidden_dim * num_sequences

        top_layers = []
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    # ============================================================
    # forward helpers
    # ============================================================

    def _encode_sequence(
        self,
        seq: Optional[torch.Tensor],
        item_emb_module: nn.Embedding,
        seq_pos_emb_module: Optional[nn.Embedding],
        action_batch: Optional[torch.Tensor],
        device: torch.device,
        B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码单条序列的 raw embedding(pre-encoder)+ mask + meanpool。
        Returns: (seq_repr, seq_mask, seq_mean)
        """
        if seq is None or seq.sum() == 0:
            seq_repr = torch.zeros(B, 1, item_emb_module.embedding_dim, device=device)
            seq_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            seq_mean = torch.zeros(B, item_emb_module.embedding_dim, device=device)
            return seq_repr, seq_mask, seq_mean

        seq_repr_raw = item_emb_module(seq)
        seq_repr = seq_repr_raw
        if seq_pos_emb_module is not None:
            L = seq_repr.size(1)
            pos = torch.arange(L, device=device).clamp_max(seq_pos_emb_module.num_embeddings - 1)
            seq_repr = seq_repr + seq_pos_emb_module(pos).unsqueeze(0)
        if action_batch is not None and self.use_seq_action_type_embedding:
            action_emb = self.seq_action_emb(action_batch.clamp_min(0))
            if self.seq_action_fusion == "concat_mlp":
                seq_repr = self.seq_action_fuse(torch.cat([seq_repr, action_emb], dim=-1))
            else:
                seq_repr = seq_repr + action_emb

        seq_mask = (seq != 0)
        # B8 fix: MeanPool 必须用 raw item_emb,**不含** pos/action emb,
        # 否则 seq_mean 与 target_emb (纯 item) 不在同一空间,
        # 导致 TCPB c=[target, seq_mean] 的 gating/bias 学不好(KuaiRand 三个异常 case 实证)。
        denom = seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
        seq_mean = (seq_repr_raw * seq_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return seq_repr, seq_mask, seq_mean

    def _tcqg_gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self._gate_activation_mode == "2sigmoid":
            return 2.0 * torch.sigmoid(logits)
        elif self._gate_activation_mode == "residual_tanh":
            return 1.0 + self._target_gate_scale * torch.tanh(logits)
        else:  # "sigmoid"
            return torch.sigmoid(logits)

    def forward(self, batch: dict) -> torch.Tensor:
        device = next(self.parameters()).device

        # 推断 batch size
        if "sparse" in batch:
            B = batch["sparse"].size(0)
        elif "dense" in batch and batch["dense"] is not None:
            B = batch["dense"].size(0)
        elif "seq" in batch and batch["seq"] is not None:
            B = batch["seq"].size(0)
        elif "target" in batch and batch["target"] is not None:
            B = batch["target"].size(0)
        else:
            B = 1

        # ---- Step 1: 编码每条序列(原始 embedding,pre-encoder)----
        seq_repr_list: List[torch.Tensor] = []
        seq_mask_list: List[torch.Tensor] = []
        seq_mean_list: List[torch.Tensor] = []

        seq_repr, seq_mask, seq_mean = self._encode_sequence(
            seq=batch.get("seq"),
            item_emb_module=self.item_emb,
            seq_pos_emb_module=self.seq_pos_emb if self.use_abs_seq_pos_emb else None,
            action_batch=batch.get("seq_action"),
            device=device,
            B=B,
        )
        seq_repr_list.append(seq_repr)
        seq_mask_list.append(seq_mask)
        seq_mean_list.append(seq_mean)

        if self.has_seq2:
            seq_repr2, seq_mask2, seq_mean2 = self._encode_sequence(
                seq=batch.get("seq2"),
                item_emb_module=self.item_emb2,
                seq_pos_emb_module=self.seq_pos_emb2 if self.use_abs_seq_pos_emb else None,
                action_batch=None,
                device=device,
                B=B,
            )
            seq_repr_list.append(seq_repr2)
            seq_mask_list.append(seq_mask2)
            seq_mean_list.append(seq_mean2)

        # ---- Step 2: 取 target_emb(每条序列各一个)----
        target = batch.get("target")
        if target is not None:
            target_emb = self.item_emb(target)
            target_cond_item_emb = (
                self.target_condition_item_emb(target)
                if self.target_condition_item_emb is not None
                else target_emb
            )
        else:
            target_emb = torch.zeros(B, self.item_emb.embedding_dim, device=device)
            target_cond_item_emb = target_emb

        target_emb_list: List[torch.Tensor] = [target_emb]
        target_cond_emb_list: List[torch.Tensor] = [target_cond_item_emb]
        if self.has_seq2:
            target2 = batch.get("target2")
            if target2 is not None:
                target2_emb = self.item_emb2(target2)
                target_cond_item_emb2 = (
                    self.target_condition_item_emb2(target2)
                    if self.target_condition_item_emb2 is not None
                    else target2_emb
                )
            else:
                target2_emb = torch.zeros(B, self.item_emb2.embedding_dim, device=device)
                target_cond_item_emb2 = target2_emb
            target_emb_list.append(target2_emb)
            target_cond_emb_list.append(target_cond_item_emb2)

        # ---- Step 3: sparse/dense ----
        sparse_per = None  # (B, num_sparse, emb_dim)
        sparse_flat = None
        dense_raw = None
        if self.sparse_arch is not None and "sparse" in batch:
            sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])
            sparse_flat = sparse_per.flatten(start_dim=1)
        if self.has_dense and batch.get("dense") is not None:
            dense_raw = batch["dense"]

        # ---- Step 4: 构造 QueryGen 的 Global Info ----
        # B2 fix: __init__ 计算 feat_dim 时假设 sparse/dense 一定存在(若 model 启用)。
        # 当 batch 缺失对应字段时,这里用零填充,保持 feat_vec 维度与 feat_dim 严格一致。
        # 与 target_condition 路径(L1267-1278)的零填充防御行为对齐。
        feat_parts = []
        if self.sparse_arch is not None:
            if sparse_flat is not None:
                feat_parts.append(sparse_flat)
            else:
                feat_parts.append(torch.zeros(
                    B, self.num_sparse * self.sparse_arch.embedding_dim, device=device,
                ))
        if self.has_dense:
            if dense_raw is not None:
                feat_parts.append(dense_raw)
            else:
                feat_parts.append(torch.zeros(B, self.num_dense, device=device))
        for t_emb in target_emb_list:
            feat_parts.append(t_emb)
        if self.use_meanpool_in_qgen:
            for s_mean in seq_mean_list:
                feat_parts.append(s_mean)
        feat_vec = torch.cat(feat_parts, dim=1) if len(feat_parts) > 1 else feat_parts[0]

        # ---- Step 5: 构造 target_condition (TCQG/TCPB 用) ----
        # B4 注:LENS 论文只覆盖单序列。此处对 multi-seq 的扩展规则:
        #   - source="item":target_condition = target1_emb 仅 (target2 不参与;符合论文 c=t)
        #   - source="item_seq":[target1; target2; mean1; mean2] (双序列扩展 c=[t; \bar{s}])
        #   - source="item_side"/"item_side_seq":工程扩展,见 docstring
        # 若希望双序列时 c 包含 target2,请用 "item_seq" 或 "item_side*"。
        target_condition = target_cond_emb_list[0]
        if self.target_condition_source != "item":
            cond_parts = [target_cond_emb_list[0]]
            if self.has_seq2:
                cond_parts.append(target_cond_emb_list[1])
            if self.target_condition_source in ("item_side", "item_side_seq"):
                if sparse_flat is not None:
                    cond_parts.append(sparse_flat)
                elif self.num_sparse > 0:
                    cond_parts.append(torch.zeros(
                        B, self.num_sparse * self.sparse_arch.embedding_dim, device=device,
                    ))
                if self.has_dense:
                    if dense_raw is not None:
                        cond_parts.append(dense_raw)
                    else:
                        cond_parts.append(torch.zeros(B, self.num_dense, device=device))
            if self.target_condition_source in ("item_seq", "item_side_seq"):
                cond_parts.append(seq_mean_list[0])
                if self.has_seq2:
                    cond_parts.append(seq_mean_list[1])
            target_condition = torch.cat(cond_parts, dim=1)
        if self.target_condition_detach:
            target_condition = target_condition.detach()

        # ---- Step 6: 每条序列独立 QueryGen ----
        queries_list: List[torch.Tensor] = []
        for s in range(self.num_sequences):
            if self.use_target_gating:
                q_s = self.query_gens[s](feat_vec, target_emb=target_condition)
            else:
                q_s = self.query_gens[s](feat_vec)
            queries_list.append(q_s)

        # ---- Step 7: 构造 NS tokens(sparse_buckets + dense + targets)----
        ns_parts = []
        if sparse_per is not None:
            if self.sparse_field_pool is not None:
                sparse_pooled = self.sparse_field_pool(sparse_per)        # (B, num_buckets, emb_dim)
            else:
                sparse_pooled = sparse_per                                 # (B, num_sparse, emb_dim)
            ns_parts.append(self.sparse_proj(sparse_pooled))               # → (B, ?, hidden_dim)
        elif self.num_sparse > 0:
            # sparse 缺失但 num_sparse > 0,用 0 填充以保 ns_token 数恒定
            ns_parts.append(torch.zeros(B, self.num_sparse_buckets, self.item_emb.embedding_dim, device=device))
        if self.has_dense:
            if dense_raw is not None:
                ns_parts.append(self.dense_proj(dense_raw).unsqueeze(1))
            else:
                ns_parts.append(torch.zeros(B, 1, self.item_emb.embedding_dim, device=device))
        # target tokens (per sequence)
        for t_emb in target_emb_list:
            ns_parts.append(t_emb.unsqueeze(1))
        ns_tokens = (
            torch.cat(ns_parts, dim=1)
            if ns_parts
            else torch.zeros(B, 0, self.item_emb.embedding_dim, device=device)
        )
        # 一致性 sanity check
        assert ns_tokens.size(1) == self.num_ns_tokens, (
            f"NS tokens count mismatch: got {ns_tokens.size(1)}, expected {self.num_ns_tokens}"
        )

        # ---- Step 8: 可选 DIN shortcut interest ----
        interest_list: List[Optional[torch.Tensor]] = []
        if self.use_din_shortcut:
            for s in range(self.num_sequences):
                interest_list.append(
                    self.target_attn_list[s](
                        target_emb_list[s], seq_repr_list[s], seq_mask_list[s]
                    )
                )

        # ---- Step 9: HyFormer V3 Block 堆叠 ----
        for block in self.blocks:
            queries_list, ns_tokens = block(
                queries_list=queries_list,
                ns_tokens=ns_tokens,
                seq_repr_list=seq_repr_list,
                seq_mask_list=seq_mask_list,
                target_condition=target_condition,
            )

        # ---- Step 10: Final norm + 顶层输出 ----
        for s in range(self.num_sequences):
            queries_list[s] = self.final_norm_list[s](queries_list[s])

        final_parts = []
        for s in range(self.num_sequences):
            final_parts.append(queries_list[s].flatten(start_dim=1))
        if self.use_din_shortcut:
            for s in range(self.num_sequences):
                final_parts.append(interest_list[s])
                final_parts.append(target_emb_list[s])
        if self.use_mean_pool:
            for s in range(self.num_sequences):
                final_parts.append(seq_mean_list[s])
        final = torch.cat(final_parts, dim=1)
        logits = self.top_mlp(final).squeeze(-1)

        # ---- Aux losses ----
        aux_loss = None

        # diversity loss(扩展开关,默认关)
        # B11 注:loss 在 final_norm 后(per-sequence)计算,各序列的 LN 参数不共享。
        # cosine 对 scale 不敏感(F.normalize),但 logical comparability 是 per-sequence-then-average。
        if self.training and self.diversity_weight > 0:
            div_terms = []
            for s in range(self.num_sequences):
                q_s = queries_list[s]
                n_q = q_s.size(1)
                if n_q < 2:
                    continue
                q_norm = F.normalize(q_s, dim=-1)
                gram = torch.matmul(q_norm, q_norm.transpose(1, 2))
                eye = torch.eye(n_q, device=gram.device).unsqueeze(0)
                off_diag = gram * (1 - eye)
                aux = (off_diag ** 2).sum() / (gram.size(0) * n_q * (n_q - 1))
                div_terms.append(aux)
            if div_terms:
                aux_loss = self.diversity_weight * sum(div_terms) / len(div_terms)

        # cond_emb alignment loss
        if (
            self.training
            and self.target_condition_align_weight > 0.0
            and self.target_condition_embedding == "separate"
            and self.target_condition_item_emb is not None
            and target is not None
        ):
            with torch.no_grad():
                main_target_ref = self.item_emb(target).detach()
            cond_target = self.target_condition_item_emb(target)
            align = ((cond_target - main_target_ref) ** 2).mean()
            if self.has_seq2 and self.target_condition_item_emb2 is not None:
                target2 = batch.get("target2")
                if target2 is not None:
                    with torch.no_grad():
                        main_target2_ref = self.item_emb2(target2).detach()
                    cond_target2 = self.target_condition_item_emb2(target2)
                    align = align + ((cond_target2 - main_target2_ref) ** 2).mean()
            align_term = self.target_condition_align_weight * align
            aux_loss = align_term if aux_loss is None else aux_loss + align_term

        if aux_loss is not None:
            return logits, aux_loss
        return logits
