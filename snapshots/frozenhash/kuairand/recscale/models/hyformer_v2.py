"""
recscale.models.hyformer_v2 — HyFormer V2（论文正确实现）

严格遵循论文 "Revisiting the Roles of Sequence Modeling and Feature Interaction
in CTR Prediction" (arXiv:2601.12681) 的设计。

本次修订（2026-04-24 第二轮）修复了第一版 V2 的四个问题：
  P1: Query Generation 输入补充 MeanPool(Seq)（论文 Eq.4，消融 +0.20% AUC）
  P3: Query Boosting 外层 residual 丢失（Pre-LN 结构被破坏），改为返回 delta，
      外层 block 统一加 residual
  P4: ns_tokens 没过 Norm 导致 scale 不匹配，增加独立 LayerNorm `norm2_ns`
  P5: LONGER encoder 原来是 FFN，改为真正的短序列 cross-attention（Eq. 6）

其他已实现功能（第一版 V2 已做）:
  - Query Boosting concat (queries, NS-tokens) 一起做 token mixing
  - Per-Token Channel FFN（einsum 独立参数）
  - 可选序列编码器: None/transformer/longer/swiglu
  - Query Generation 两种模式: shared (默认) / independent

架构：
  seq → item_emb → seq_repr (B, L_seq, D)
      ↓ (optional) seq_encoder
  seq_mean = MeanPool(seq_repr)
  non-seq features + seq_mean → Query Generation → Global Tokens (B, N_q, D)
      ↓
  Stack of HyFormerBlockV2 × N  (Pre-LN + 外层 residual)
  ├── Query Decoding: queries = queries + CrossAttn(Norm(queries), seq_kv, seq_kv)
  └── Query Boosting V2:
        q_delta, ns_delta = Boosting(Norm_q(queries), Norm_ns(ns))
        queries = queries + q_delta
        ns     = ns + ns_delta
      ↓
  pool queries + target_attn + target_emb + MeanPool(seq) → MLP → logit
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, DenseArch, SparseArch


class QueryGenerationV2(nn.Module):
    """
    Generate global query tokens from non-sequential features.

    支持两种模式：
      - "shared": 共享 MLP + split（原实现，内存友好）
      - "independent": N 个独立 FFN（论文要求，但内存消耗大）

    可选：target-aware gating（A2 方向）
      - 如果构造时 `use_target_gating=True`，forward 要求同时传入 target_emb，
        `q_out = q_raw * sigmoid(W_t · target_emb)`，让 query 显式以 target 为条件
    """

    def __init__(
        self,
        input_dim: int,
        num_queries: int,
        hidden_dim: int,
        mode: str = "shared",
        use_target_gating: bool = False,
        target_emb_dim: int = None,
        gate_activation_mode: str = "2sigmoid",
        gate_network_mode: str = "linear",
        target_gate_hidden_dim: int = None,
        target_gate_scale: float = 1.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.use_target_gating = use_target_gating
        # gate_activation_mode: "2sigmoid" (default, gate range (0, 2), centered at 1.0)
        #                      "sigmoid"  (plain σ, gate range (0, 1), centered at 0.5)
        # "sigmoid" mode is an ablation to check whether the factor-2 centering
        # matters. Note: with plain σ and zero-init W_t, gate starts at 0.5 which
        # halves query magnitude on step 0 — less "gentle" start than 2σ.
        assert gate_activation_mode in ("2sigmoid", "sigmoid", "residual_tanh"), gate_activation_mode
        assert gate_network_mode in ("linear", "mlp"), gate_network_mode
        self.gate_activation_mode = gate_activation_mode
        self.gate_network_mode = gate_network_mode
        self.target_gate_scale = float(target_gate_scale)

        if mode == "independent":
            self.projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_queries)
            ])
        else:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * num_queries),
                nn.GELU(),
                nn.Linear(hidden_dim * num_queries, hidden_dim * num_queries),
            )

        if use_target_gating:
            # target -> (q * D) sigmoid gate, 让每个 slot 对 target 有自己的响应
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
                # Identity-start mask: the last projection starts at zero, so
                # residual_tanh gives gate=1 before learning instance effects.
                nn.init.zeros_(self.target_gate[-1].weight)
                nn.init.zeros_(self.target_gate[-1].bias)
            else:
                self.target_gate = nn.Linear(target_emb_dim, num_queries * hidden_dim)
                # 初始化偏向 1（不 gating 时等价于原始 query）
                nn.init.zeros_(self.target_gate.weight)
                nn.init.zeros_(self.target_gate.bias)

    def forward(self, features: torch.Tensor, target_emb: torch.Tensor = None) -> torch.Tensor:
        """
        features: (B, input_dim)
        target_emb: (B, target_emb_dim), 仅当 use_target_gating=True 时必需
        Returns: (B, N_q, D)
        """
        if self.mode == "independent":
            out = torch.stack(
                [proj(features) for proj in self.projs], dim=1
            )
        else:
            out = self.proj(features)
            out = out.view(-1, self.num_queries, self.hidden_dim)

        if self.use_target_gating:
            if target_emb is None:
                raise ValueError("target_emb required when use_target_gating=True")
            # (B, q*D) -> (B, q, D)
            gate_logits = self.target_gate(target_emb).view(-1, self.num_queries, self.hidden_dim)
            # Default "2sigmoid": gate range (0, 2), centered at 1.0 when W_t=0 →
            # query unchanged at step 0, model can learn to amplify or suppress.
            # Ablation "sigmoid": plain σ, gate range (0, 1), centered at 0.5 at W_t=0
            # → step-0 query magnitude halved (ablated design, §5.5).
            if self.gate_activation_mode == "2sigmoid":
                gate = 2.0 * torch.sigmoid(gate_logits)
            elif self.gate_activation_mode == "residual_tanh":
                gate = 1.0 + self.target_gate_scale * torch.tanh(gate_logits)
            else:  # "sigmoid"
                gate = torch.sigmoid(gate_logits)
            out = out * gate

        return out


class QueryDecoding(nn.Module):
    """
    Cross-attention: Query tokens attend to sequence K/V.

    可选：`use_position_bias=True` 添加可学的位置偏置（B1 方向），
    让不同 query 对"最近 item"有不同的近因偏好（近期 / 远期 / 均匀）。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 use_position_bias: bool = False,
                 use_global_position_bias: bool = False,
                 use_target_time_bias: bool = False,
                 num_queries: int = 8,
                 max_seq_len: int = 1024,
                 target_time_rank: int = 8,
                 target_time_input_dim: Optional[int] = None,
                 shared_position_bias: Optional[nn.Parameter] = None):
        super().__init__()
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
        self.shared_position_bias_used = (shared_position_bias is not None)
        if use_position_bias:
            # 每个 query 对每个位置（相对序列末尾的 "recency index"）一个偏置标量
            # shape: (num_queries, max_seq_len)
            if shared_position_bias is not None:
                # Ablation: all layers share one B_pos parameter (§5.5).
                # Store via object.__setattr__ so nn.Module doesn't register it
                # as this block's own Parameter — outer model owns the single
                # parameter instance; we just hold a reference.
                object.__setattr__(self, "position_bias", shared_position_bias)
            else:
                self.position_bias = nn.Parameter(torch.zeros(num_queries, max_seq_len))
        if use_global_position_bias:
            # Traditional baseline: one recency prior shared by all query slots and heads.
            self.global_position_bias = nn.Parameter(torch.zeros(max_seq_len))
        if use_target_time_bias:
            # Target-conditioned temporal bias:
            # target -> per-query temporal mixture, dotted with position basis.
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
        kv: (B, L_seq, D)
        kv_mask: (B, L_seq)
        Returns: (B, N_q, D)
        """
        B, N_q, D = queries.shape
        L = kv.size(1)
        H = self.num_heads

        q = self.q_proj(queries).view(B, N_q, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L, H, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Position bias (B1): 加一个 per-query, per-position 的可学偏置
        # 约定 seq 是 left-padded、右对齐，index=0 代表序列最早位置，index=L-1 代表最近位置
        # 但 pad token 会被 kv_mask 屏蔽，所以只有 valid 位置才起作用
        if self.use_position_bias:
            # position_bias: (N_q, max_seq_len) -> (1, 1, N_q, L)
            pos_bias = self.position_bias[:, :L]  # (N_q, L)
            # attn shape: (B, H, N_q, L); per-head 复用同一 bias（简化）
            attn = attn + pos_bias.unsqueeze(0).unsqueeze(0)
        if self.use_global_position_bias:
            # global_position_bias: (max_seq_len,) -> (1, 1, 1, L)
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
            attn = attn.masked_fill(
                ~kv_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)


class QueryBoostingV2(nn.Module):
    """
    Query Boosting V2: MLP-Mixer style token mixing on Concat(queries, NS-tokens).

    论文核心改进：NS-tokens 参与 token mixing，提供全局上下文信息。

    流程：
      1. Concat([queries, ns_tokens]) → (B, N_q + L_NS, D)
      2. Token Mixing: Split into H subspaces, concat per-subspace → FFN（内部 residual）
      3. Per-token Channel FFN（独立参数，einsum 实现，内部 residual）
      4. Split back to queries and ns_tokens，返回 **delta**（外层加 residual）

    注意：本模块返回的是增量 delta（相对输入的变化量），而不是带外层 residual 的输出。
    外层 block 负责 `queries = queries + q_delta`、`ns = ns + ns_delta`。
    这样符合 Pre-LN Transformer 的标准结构。
    """

    def __init__(
        self,
        num_queries: int,
        num_ns_tokens: int,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.0,
        use_per_token_ffn: bool = True,
        use_v1_residual: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_ns_tokens = num_ns_tokens
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_per_token_ffn = use_per_token_ffn
        self.use_v1_residual = use_v1_residual  # V1: queries += boosting(LN(q)) 含 LN(q) 自身

        total_tokens = num_queries + num_ns_tokens
        mixing_input_dim = total_tokens * self.head_dim

        # Token mixing FFN (on concatenated tokens)
        self.mixing_ffn = nn.Sequential(
            nn.Linear(mixing_input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, mixing_input_dim),
            nn.Dropout(dropout),
        )

        # Channel mixing FFN
        if use_per_token_ffn:
            # Per-token 独立参数 (V2 默认)
            self.channel_fc1 = nn.Parameter(
                torch.empty(total_tokens, hidden_dim, ffn_dim)
            )
            self.channel_fc2 = nn.Parameter(
                torch.empty(total_tokens, ffn_dim, hidden_dim)
            )
            self.channel_bias1 = nn.Parameter(torch.zeros(total_tokens, ffn_dim))
            self.channel_bias2 = nn.Parameter(torch.zeros(total_tokens, hidden_dim))
            nn.init.kaiming_uniform_(self.channel_fc1, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_uniform_(self.channel_fc2, a=0, mode="fan_in", nonlinearity="relu")
        else:
            # 共享 FFN (V1 行为)
            self.channel_ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        queries: torch.Tensor,
        ns_tokens: torch.Tensor,
    ) -> tuple:
        """
        queries: (B, N_q, D) — 已 Norm 的 queries
        ns_tokens: (B, L_NS, D) — 已 Norm 的 ns_tokens
        Returns: (q_delta, ns_delta) — 增量（相对输入的 transform 结果），
                 外层负责加未 Norm 的 residual: queries = queries + q_delta
        """
        B, N_q, D = queries.shape
        H = self.num_heads
        hd = self.head_dim

        # Concatenate queries and ns_tokens (ns may be empty if num_ns_tokens=0)
        if self.num_ns_tokens > 0:
            x0 = torch.cat([queries, ns_tokens], dim=1)  # (B, N_q + L_NS, D)
        else:
            x0 = queries  # (B, N_q, D)
        T = x0.size(1)

        # Token mixing with internal residual (MLP-Mixer 标准结构)
        x_heads = x0.view(B, T, H, hd)
        x_heads = x_heads.permute(0, 2, 1, 3)
        x_concat = x_heads.reshape(B * H, T * hd)
        x_mixed = self.mixing_ffn(x_concat)
        x_mixed = x_mixed.view(B, H, T, hd).permute(0, 2, 1, 3).reshape(B, T, D)
        x1 = x0 + x_mixed  # internal residual 1

        # Channel mixing with internal residual
        if self.use_per_token_ffn:
            h = F.gelu(
                torch.einsum("btd,tdf->btf", x1, self.channel_fc1) + self.channel_bias1
            )
            h = torch.einsum("btf,tfd->btd", h, self.channel_fc2) + self.channel_bias2
        else:
            h = self.channel_ffn(x1)
        x2 = x1 + h  # internal residual 2

        if self.use_v1_residual:
            # V1 模式：返回完整的 x2（包含 x0=LN(queries) 自身），外层 residual 等效于
            # queries = queries + LN(queries) + Δ（比标准 Pre-LN 多了 LN(queries) 项）
            q_out = x2[:, :N_q]
            ns_out = x2[:, N_q:] if self.num_ns_tokens > 0 else torch.zeros_like(queries[:, :0])
            return q_out, ns_out
        else:
            # V2 标准模式：返回纯增量 delta（外层 residual 只加变化量）
            delta = x2 - x0
            q_delta = delta[:, :N_q]
            ns_delta = delta[:, N_q:] if self.num_ns_tokens > 0 else torch.zeros_like(queries[:, :0])
            return q_delta, ns_delta


class SeqEncoder(nn.Module):
    """
    序列编码阶段（可选，论文提供 3 种可选方案）。

    类型：
      - None: 不使用序列编码（默认，保持向后兼容）
      - "transformer": 完整 Transformer 编码器
      - "longer": LONGER (论文 Eq. 6)，短序列对长序列做 cross-attention
        S_short = CrossAttn(Q_short, S, S)，其中 Q_short 是可学习的短 query 序列
        复杂度 O(L_H · L_S · D)，L_H 远小于 L_S
      - "swiglu": SwiGLU 变换
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
        self.head_dim = hidden_dim // num_heads

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
            # LONGER: 用可学习短序列 Q_short 对完整序列 S 做 cross-attention
            # 然后再把输出映射回 S 的长度空间（用另一次 cross-attention，长序列 query 短序列 KV）
            self.longer_num_tokens = longer_num_tokens
            self.longer_query = nn.Parameter(
                torch.empty(longer_num_tokens, hidden_dim)
            )
            nn.init.normal_(self.longer_query, std=0.02)
            self.longer_q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.longer_k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.longer_v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.longer_out_proj = nn.Linear(hidden_dim, hidden_dim)
            # 反向 attention：长序列作 query，压缩后的 S_short 作 KV
            self.longer_back_q = nn.Linear(hidden_dim, hidden_dim)
            self.longer_back_k = nn.Linear(hidden_dim, hidden_dim)
            self.longer_back_v = nn.Linear(hidden_dim, hidden_dim)
            self.longer_back_out = nn.Linear(hidden_dim, hidden_dim)
            self.longer_norm = nn.LayerNorm(hidden_dim)
            self.longer_dropout = nn.Dropout(dropout)
        elif encoder_type == "swiglu":
            self.w1 = nn.Linear(hidden_dim, ffn_dim)
            self.w2 = nn.Linear(hidden_dim, ffn_dim)
            self.w3 = nn.Linear(ffn_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
        else:
            self.encoder = None

    def _longer_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """通用 multi-head cross-attention。
        q: (B, Lq, D), k: (B, Lk, D), v: (B, Lk, D)
        mask: (B, Lk) True 表示 valid
        Returns: (B, Lq, D)
        """
        B, Lq, D = q.shape
        Lk = k.size(1)
        H = self.num_heads
        hd = self.head_dim
        scale = 1.0 / math.sqrt(hd)

        q_ = q_proj(q).view(B, Lq, H, hd).transpose(1, 2)
        k_ = k_proj(k).view(B, Lk, H, hd).transpose(1, 2)
        v_ = v_proj(v).view(B, Lk, H, hd).transpose(1, 2)

        attn = torch.matmul(q_, k_.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.longer_dropout(attn)

        out = torch.matmul(attn, v_).transpose(1, 2).reshape(B, Lq, D)
        return out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D)
        mask: (B, L) - True for valid positions
        """
        if self.encoder_type == "transformer":
            if mask is not None:
                return self.encoder(x, src_key_padding_mask=~mask)
            return self.encoder(x)
        elif self.encoder_type == "longer":
            # Step 1: S_short = CrossAttn(Q_short, S, S)
            # Q_short 是可学习参数，扩展到 batch
            B = x.size(0)
            q_short = self.longer_query.unsqueeze(0).expand(B, -1, -1)  # (B, L_H, D)
            s_short = self._longer_attn(
                q_short, x, x,
                self.longer_q_proj, self.longer_k_proj, self.longer_v_proj,
                self.longer_out_proj, mask,
            )  # (B, L_H, D)
            s_short = self.longer_norm(s_short)

            # Step 2: 把 S_short 回映射到 L 长度（长序列 query 短序列 KV）
            # 没有 mask（S_short 是压缩表示，全部 valid）
            out = self._longer_attn(
                x, s_short, s_short,
                self.longer_back_q, self.longer_back_k, self.longer_back_v,
                self.longer_back_out, None,
            )  # (B, L, D)
            return x + out  # residual
        elif self.encoder_type == "swiglu":
            return self.w3(F.silu(self.w1(x)) * self.w2(x))
        else:
            return x


class HyFormerBlockV2(nn.Module):
    """
    HyFormer Block V2 (Pre-LN + 外层 residual):

    1. Query Decoding: queries = queries + CrossAttn(Norm(queries), K, V)
    2. Query Boosting V2:
         q_delta, ns_delta = Boosting(Norm_q(queries), Norm_ns(ns_tokens))
         queries = queries + q_delta
         ns_tokens = ns_tokens + ns_delta
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_queries: int,
        num_ns_tokens: int,
        ffn_dim: int = 128,
        dropout: float = 0.0,
        use_per_token_ffn: bool = True,
        use_v1_residual: bool = False,
        use_position_bias: bool = False,
        use_global_position_bias: bool = False,
        use_target_time_bias: bool = False,
        max_seq_len: int = 1024,
        target_time_rank: int = 8,
        target_time_input_dim: Optional[int] = None,
        has_seq2: bool = False,
        max_seq_len2: int = 1024,
        shared_position_bias: Optional[nn.Parameter] = None,
        shared_position_bias_seq2: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.num_ns_tokens = num_ns_tokens
        self.has_seq2 = has_seq2
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = QueryDecoding(
            hidden_dim, num_heads, dropout,
            use_position_bias=use_position_bias,
            use_global_position_bias=use_global_position_bias,
            use_target_time_bias=use_target_time_bias,
            num_queries=num_queries,
            max_seq_len=max_seq_len,
            target_time_rank=target_time_rank,
            target_time_input_dim=target_time_input_dim,
            shared_position_bias=shared_position_bias,
        )
        # Dual-seq: 独立的第二路 cross-attention（独立 K/V projection + 独立 position_bias）
        if has_seq2:
            self.cross_attn_seq2 = QueryDecoding(
                hidden_dim, num_heads, dropout,
                use_position_bias=use_position_bias,
                use_global_position_bias=use_global_position_bias,
                use_target_time_bias=use_target_time_bias,
                num_queries=num_queries,
                max_seq_len=max_seq_len2,
                target_time_rank=target_time_rank,
                target_time_input_dim=target_time_input_dim,
                shared_position_bias=shared_position_bias_seq2,
            )
        # queries 和 ns_tokens 用独立的 LayerNorm（scale 对齐）
        self.norm2_q = nn.LayerNorm(hidden_dim)
        if num_ns_tokens > 0:
            self.norm2_ns = nn.LayerNorm(hidden_dim)
        self.boosting = QueryBoostingV2(
            num_queries, num_ns_tokens, hidden_dim, num_heads, ffn_dim, dropout,
            use_per_token_ffn, use_v1_residual,
        )

    def forward(
        self,
        queries: torch.Tensor,
        ns_tokens: torch.Tensor,
        seq_kv: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        seq2_kv: Optional[torch.Tensor] = None,
        seq2_mask: Optional[torch.Tensor] = None,
        target_emb: Optional[torch.Tensor] = None,
    ) -> tuple:
        # Query Decoding: 外层 residual
        q_normed = self.norm1(queries)
        decoded = self.cross_attn(q_normed, seq_kv, seq_mask, target_emb=target_emb)
        queries = queries + decoded

        # Dual-seq: 第二次 cross-attention（独立参数）
        if self.has_seq2 and seq2_kv is not None:
            q_normed2 = self.norm1(queries)
            decoded2 = self.cross_attn_seq2(q_normed2, seq2_kv, seq2_mask, target_emb=target_emb)
            queries = queries + decoded2

        # Query Boosting V2: 外层统一加 residual
        q_normed = self.norm2_q(queries)
        if self.num_ns_tokens > 0:
            ns_normed = self.norm2_ns(ns_tokens)
        else:
            ns_normed = ns_tokens  # dummy, won't be used
        q_delta, ns_delta = self.boosting(q_normed, ns_normed)
        queries = queries + q_delta
        if self.num_ns_tokens > 0:
            ns_tokens = ns_tokens + ns_delta

        return queries, ns_tokens


class TargetAttention(nn.Module):
    """DIN-style target attention shortcut."""

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
class HyFormerV2Model(RecModel):
    """
    HyFormer V2: Hybrid Transformer for CTR Prediction (论文正确实现).

    新超参数:
      - seq_encoder_type (str): 序列编码类型，None/"transformer"/"longer"/"swiglu"
      - q_gen_mode (str): Query Generation 模式，"shared" (默认) 或 "independent"
      - use_mean_pool (bool): 是否使用 MeanPool(Seq) 作为全局信息（默认 True）

    消融开关（默认全开=V2完整版，关掉=退化到V1行为）:
      - use_meanpool_in_qgen (bool): Query Generation 输入是否包含 MeanPool(Seq)，默认 True
      - use_ns_in_boosting (bool): NS-tokens 是否参与 Query Boosting，默认 True
      - use_per_token_ffn (bool): Channel FFN 是否用 per-token 独立参数，默认 False
      - use_v1_residual (bool): Boosting 是否用 V1 残差风格（返回含 LN(q) 的完整值），默认 False

    Supports batch keys: "sparse", "dense" (opt), "seq", "target" (opt), "label"
    """
    model_name = "hyformer_v2"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc.get("embedding_dim", 16)
        hidden_dim = mc.get("hidden_dim", 128)
        num_layers = mc.get("num_layers", 3)
        num_heads = mc.get("num_heads", 4)
        num_queries = mc.get("num_queries", 8)
        ffn_dim = mc.get("ffn_dim", hidden_dim * 2)
        dropout = mc.get("dropout", 0.1)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        self.use_seq_action_type_embedding = mc.get("use_seq_action_type_embedding", False)
        self.seq_action_fusion = mc.get("seq_action_fusion", "add")
        num_seq_action_types = int(mc.get("num_seq_action_types", 5))

        # 新超参数
        self.seq_encoder_type = mc.get("seq_encoder_type", None)
        self.q_gen_mode = mc.get("q_gen_mode", "shared")
        self.use_mean_pool = mc.get("use_mean_pool", True)

        # 消融开关
        self.use_meanpool_in_qgen = mc.get("use_meanpool_in_qgen", True)
        self.use_ns_in_boosting = mc.get("use_ns_in_boosting", True)
        self.use_per_token_ffn = mc.get("use_per_token_ffn", False)
        self.use_qgen_feature_gate = mc.get("use_qgen_feature_gate", False)
        self.qgen_feature_gate_mode = mc.get("qgen_feature_gate_mode", "token_scalar")

        # Interest Slot Transformer 新开关
        self.use_learnable_prior = mc.get("use_learnable_prior", False)
        self.diversity_weight = float(mc.get("diversity_weight", 0.0))
        self.use_din_shortcut = mc.get("use_din_shortcut", True)  # DIN target_attn shortcut

        # Round 2: Target-Aware QueryGen (A2)
        self.use_target_gating = mc.get("use_target_gating", False)
        self.tcqg_per_layer = mc.get("tcqg_per_layer", False)
        self._gate_activation_mode = mc.get("gate_activation_mode", "2sigmoid")
        self._target_gate_scale = float(mc.get("target_gate_scale", 1.0))
        self.target_condition_source = mc.get("target_condition_source", "item")
        if self.target_condition_source not in ("item", "item_seq", "item_side", "item_side_seq"):
            raise ValueError(
                "target_condition_source must be one of: item, item_seq, item_side, item_side_seq"
            )
        self.target_condition_detach = bool(mc.get("target_condition_detach", False))
        self.target_condition_embedding = mc.get("target_condition_embedding", "shared")
        if self.target_condition_embedding not in ("shared", "separate"):
            raise ValueError("target_condition_embedding must be one of: shared, separate")
        self.target_condition_embedding_frozen = bool(
            mc.get("target_condition_embedding_frozen", False)
        )
        # MSE auxiliary loss pulling separate cond emb toward main item emb's
        # semantic space. main_emb side is detached so the alignment only
        # affects cond_emb, not the main CTR path.
        self.target_condition_align_weight = float(
            mc.get("target_condition_align_weight", 0.0)
        )

        # Round 3: Position-Aware Query Decoding (B1)
        self.use_position_bias = mc.get("use_position_bias", False)
        self.use_global_position_bias = mc.get("use_global_position_bias", False)
        self.use_abs_seq_pos_emb = mc.get("use_abs_seq_pos_emb", False)
        self.use_target_time_bias = mc.get("use_target_time_bias", False)
        self.target_time_rank = int(mc.get("target_time_rank", 8))
        self.pos_bias_max_len = int(mc.get("pos_bias_max_len", dc.get("maxlen", 1024)))

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        # Dual-sequence support: 若 dataset 提供 num_items2 > 0 则开启第二路序列
        # （对齐 DIN 的 dual-seq 接口，如 TaobaoAd 的 cate_his + brand_his）
        # 关键约束：has_seq2=False 时所有新参数 / 新代码路径都不创建 / 不进入，
        # 与改动前逐比特等价。
        self.has_seq2 = dc.get("num_items2", 0) > 0
        num_items2 = dc.get("num_items2", 0)
        self.pos_bias_max_len2 = int(dc.get("maxlen2", dc.get("maxlen", 1024)))

        # num_ns_tokens 基础: num_sparse + (dense?) + target
        # dual-seq 开启时追加 1 个 target2 NS-token
        self.num_ns_tokens = num_sparse + (1 if num_dense > 0 else 0) + 1
        if self.has_seq2:
            self.num_ns_tokens += 1

        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None
        self.has_dense = num_dense > 0

        # 补充 sparse_proj 和 dense_proj（原实现缺失）
        self.sparse_proj = nn.Linear(emb_dim, hidden_dim) if emb_dim != hidden_dim else nn.Identity()
        if self.has_dense:
            self.dense_proj = nn.Sequential(
                nn.Linear(num_dense, hidden_dim),
                nn.ReLU(),
            )

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
        if self.use_qgen_feature_gate and qgen_side_dim > 0:
            if self.qgen_feature_gate_mode == "token_scalar":
                gate_out_dim = num_sparse + (1 if self.has_dense else 0)
            elif self.qgen_feature_gate_mode == "vector":
                gate_out_dim = qgen_side_dim
            else:
                raise ValueError(f"Unsupported qgen_feature_gate_mode={self.qgen_feature_gate_mode!r}")
            self.qgen_feature_gate = nn.Linear(qgen_side_dim + hidden_dim * 2, gate_out_dim)
            nn.init.zeros_(self.qgen_feature_gate.weight)
            nn.init.zeros_(self.qgen_feature_gate.bias)
        else:
            self.qgen_feature_gate = None

        # Query Generation 输入维度
        feat_dim = (
            num_sparse * emb_dim
            + (num_dense if self.has_dense else 0)
            + hidden_dim  # target_emb
            + (hidden_dim if self.use_meanpool_in_qgen else 0)  # MeanPool(Seq)
        )
        if self.has_seq2:
            # 多加 target2_emb；use_meanpool_in_qgen 时再多加一份 MeanPool(Seq2)
            feat_dim += hidden_dim
            if self.use_meanpool_in_qgen:
                feat_dim += hidden_dim

        self.query_gen = QueryGenerationV2(
            feat_dim, num_queries, hidden_dim, mode=self.q_gen_mode,
            use_target_gating=self.use_target_gating,
            target_emb_dim=target_condition_dim,
            gate_activation_mode=mc.get("gate_activation_mode", "2sigmoid"),
            gate_network_mode=mc.get("target_gate_network", "linear"),
            target_gate_hidden_dim=mc.get("target_gate_hidden_dim"),
            target_gate_scale=mc.get("target_gate_scale", 1.0),
        )
        self.num_queries = num_queries

        # Interest Slot Transformer: 每个 query 的可学全局先验 P_i
        # 让 query 不再完全依赖样本特征，而是有一个可学的"角色向量"
        if self.use_learnable_prior:
            self.query_prior = nn.Parameter(torch.empty(num_queries, hidden_dim))
            nn.init.normal_(self.query_prior, std=0.02)
        else:
            self.query_prior = None

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.item_emb.weight.data[0].zero_()
        if self.target_condition_embedding == "separate":
            rng_state = torch.get_rng_state()
            self.target_condition_item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
            torch.set_rng_state(rng_state)
            self.target_condition_item_emb.weight.data.copy_(self.item_emb.weight.data)
            if self.target_condition_embedding_frozen:
                self.target_condition_item_emb.weight.requires_grad_(False)
        else:
            self.target_condition_item_emb = None
        if self.use_abs_seq_pos_emb:
            self.seq_pos_emb = nn.Embedding(self.pos_bias_max_len, hidden_dim)
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

        self.seq_encoder = SeqEncoder(
            hidden_dim, num_heads, ffn_dim, dropout, self.seq_encoder_type,
        )

        # Dual-seq 独立 embedding / encoder / target attention
        if self.has_seq2:
            self.item_emb2 = nn.Embedding(num_items2 + 1, hidden_dim, padding_idx=0)
            nn.init.normal_(self.item_emb2.weight, std=0.02)
            self.item_emb2.weight.data[0].zero_()
            if self.target_condition_embedding == "separate":
                rng_state = torch.get_rng_state()
                self.target_condition_item_emb2 = nn.Embedding(num_items2 + 1, hidden_dim, padding_idx=0)
                torch.set_rng_state(rng_state)
                self.target_condition_item_emb2.weight.data.copy_(self.item_emb2.weight.data)
                if self.target_condition_embedding_frozen:
                    self.target_condition_item_emb2.weight.requires_grad_(False)
            else:
                self.target_condition_item_emb2 = None
            if self.use_abs_seq_pos_emb:
                self.seq_pos_emb2 = nn.Embedding(self.pos_bias_max_len2, hidden_dim)
                nn.init.normal_(self.seq_pos_emb2.weight, std=0.02)
            self.seq_encoder2 = SeqEncoder(
                hidden_dim, num_heads, ffn_dim, dropout, self.seq_encoder_type,
            )
            self.target_attn2 = TargetAttention(hidden_dim, hidden_dim)

        self.use_v1_residual = mc.get("use_v1_residual", False)

        # Ablation (§5.5): share a single B_pos parameter across all blocks
        # instead of per-block independent ones. Default False (per-block).
        self.share_pos_bias_across_layers = mc.get("share_pos_bias_across_layers", False)
        shared_pb = None
        shared_pb2 = None
        if self.use_position_bias and self.share_pos_bias_across_layers:
            shared_pb = nn.Parameter(torch.zeros(num_queries, self.pos_bias_max_len))
            self.shared_position_bias = shared_pb  # register on outer model so it appears once in state_dict
            if self.has_seq2:
                shared_pb2 = nn.Parameter(torch.zeros(num_queries, self.pos_bias_max_len2))
                self.shared_position_bias_seq2 = shared_pb2

        boosting_ns = self.num_ns_tokens if self.use_ns_in_boosting else 0
        self.blocks = nn.ModuleList([
            HyFormerBlockV2(
                hidden_dim, num_heads, num_queries, boosting_ns,
                ffn_dim, dropout, self.use_per_token_ffn, self.use_v1_residual,
                use_position_bias=self.use_position_bias,
                use_global_position_bias=self.use_global_position_bias,
                use_target_time_bias=self.use_target_time_bias,
                max_seq_len=self.pos_bias_max_len,
                target_time_rank=self.target_time_rank,
                target_time_input_dim=target_condition_dim,
                has_seq2=self.has_seq2,
                max_seq_len2=self.pos_bias_max_len2,
                shared_position_bias=shared_pb,
                shared_position_bias_seq2=shared_pb2,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Per-layer TCQG: re-gate queries after each decoder layer.
        # Default OFF (tcqg_per_layer=False) → original single-gate behavior.
        # When ON, each layer gets its own zero-init gate projection so the
        # identity-at-init contract is preserved.
        self.tcqg_per_layer_norm = mc.get("tcqg_per_layer_norm", False)
        if self.use_target_gating and self.tcqg_per_layer:
            self.per_layer_tcqg = nn.ModuleList([
                nn.Linear(target_condition_dim, num_queries * hidden_dim)
                for _ in range(num_layers)
            ])
            for proj in self.per_layer_tcqg:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            if self.tcqg_per_layer_norm:
                self.per_layer_tcqg_ln = nn.ModuleList([
                    nn.LayerNorm(hidden_dim)
                    for _ in range(num_layers)
                ])

        self.target_attn = TargetAttention(hidden_dim, hidden_dim) if self.use_din_shortcut else None

        top_layers = []
        in_dim = num_queries * hidden_dim
        if self.use_din_shortcut:
            in_dim += hidden_dim + hidden_dim  # interest + target_emb
        if self.use_mean_pool:
            in_dim += hidden_dim
        if self.has_seq2:
            # 追加 interest2 + target2_emb；use_mean_pool=True 时再加 seq2_mean
            if self.use_din_shortcut:
                in_dim += hidden_dim + hidden_dim
            if self.use_mean_pool:
                in_dim += hidden_dim
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def _tcqg_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation to TCQG gate logits.
        Reused by both the initial QueryGen gate and per-layer gates."""
        if self._gate_activation_mode == '2sigmoid':
            return 2.0 * torch.sigmoid(logits)
        elif self._gate_activation_mode == 'residual_tanh':
            return 1.0 + self._target_gate_scale * torch.tanh(logits)
        else:  # "sigmoid"
            return torch.sigmoid(logits)

    def forward(self, batch: dict) -> torch.Tensor:
        # Step 1: 先计算 seq_kv 和 seq_mean（后者需要提前加入 Query Generation 输入）
        seq = batch.get("seq")
        if seq is not None and seq.sum() > 0:
            seq_kv = self.item_emb(seq)
            if self.use_abs_seq_pos_emb:
                L = seq_kv.size(1)
                pos = torch.arange(L, device=seq_kv.device).clamp_max(self.pos_bias_max_len - 1)
                seq_kv = seq_kv + self.seq_pos_emb(pos).unsqueeze(0)
            if self.use_seq_action_type_embedding and "seq_action" in batch:
                action_emb = self.seq_action_emb(batch["seq_action"].clamp_min(0))
                if self.seq_action_fusion == "concat_mlp":
                    seq_kv = self.seq_action_fuse(torch.cat([seq_kv, action_emb], dim=-1))
                else:
                    seq_kv = seq_kv + action_emb
            seq_mask = (seq != 0)
            seq_kv = self.seq_encoder(seq_kv, seq_mask)
            # MeanPool(Seq) (论文 Eq.4 的 Global Info 组成部分)
            seq_mean = (seq_kv * seq_mask.unsqueeze(-1).float()).sum(dim=1) / \
                       seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
            has_seq = True
        else:
            has_seq = False
            seq_kv = None
            seq_mask = None
            seq_mean = None

        # Step 1b: Dual-seq 时编码第二路 seq2
        seq2_kv = None
        seq2_mask = None
        seq2_mean = None
        if self.has_seq2:
            seq2 = batch.get("seq2")
            if seq2 is not None and seq2.sum() > 0:
                seq2_kv = self.item_emb2(seq2)
                if self.use_abs_seq_pos_emb:
                    L2 = seq2_kv.size(1)
                    pos2 = torch.arange(L2, device=seq2_kv.device).clamp_max(self.pos_bias_max_len2 - 1)
                    seq2_kv = seq2_kv + self.seq_pos_emb2(pos2).unsqueeze(0)
                seq2_mask = (seq2 != 0)
                seq2_kv = self.seq_encoder2(seq2_kv, seq2_mask)
                seq2_mean = (seq2_kv * seq2_mask.unsqueeze(-1).float()).sum(dim=1) / \
                            seq2_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Step 2: 构造 non-seq 特征向量
        feat_parts = []
        sparse_flat = None
        dense_raw = None
        if self.sparse_arch is not None and "sparse" in batch:
            if self.qgen_feature_gate is not None and self.qgen_feature_gate_mode == "token_scalar":
                sparse_tokens_for_qgen = self.sparse_arch.forward_per_feature(batch["sparse"])
                sparse_flat = sparse_tokens_for_qgen.flatten(start_dim=1)
            else:
                sparse_flat = self.sparse_arch(batch["sparse"])
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            dense_raw = batch["dense"]

        target = batch.get("target")
        if target is not None:
            target_emb = self.item_emb(target)
            if self.target_condition_item_emb is not None:
                target_condition_item_emb = self.target_condition_item_emb(target)
            else:
                target_condition_item_emb = target_emb
        else:
            if sparse_flat is not None:
                B = sparse_flat.size(0)
            elif dense_raw is not None:
                B = dense_raw.size(0)
            elif has_seq:
                B = seq_kv.size(0)
            else:
                B = 1
            target_emb = torch.zeros(
                B, self.item_emb.embedding_dim, device=next(self.parameters()).device
            )
            target_condition_item_emb = target_emb

        # 补齐：无 seq 时用 0 填充 MeanPool 部分，保持 feat_dim 一致
        if not has_seq:
            B = target_emb.size(0)
            seq_kv = torch.zeros(B, 1, self.item_emb.embedding_dim,
                                 device=next(self.parameters()).device)
            seq_mask = torch.zeros(B, 1, dtype=torch.bool,
                                   device=next(self.parameters()).device)
            seq_mean = torch.zeros(B, self.item_emb.embedding_dim,
                                   device=next(self.parameters()).device)

        # P1 修复：Query Generation 输入包含 MeanPool(Seq)（可通过开关关掉）
        if self.use_meanpool_in_qgen:
            qgen_seq_mean = seq_mean
        else:
            qgen_seq_mean = torch.zeros_like(target_emb)

        if self.qgen_feature_gate is not None:
            side_for_gate = []
            if sparse_flat is not None:
                side_for_gate.append(sparse_flat)
            if dense_raw is not None:
                side_for_gate.append(dense_raw)
            side_vec = torch.cat(side_for_gate, dim=1)
            gate_input = torch.cat([side_vec, target_emb, qgen_seq_mean], dim=1)
            gate = 2.0 * torch.sigmoid(self.qgen_feature_gate(gate_input))
            if self.qgen_feature_gate_mode == "token_scalar":
                offset = 0
                if sparse_flat is not None:
                    num_sparse_tokens = self.sparse_arch.num_sparse
                    sparse_gate = gate[:, offset:offset + num_sparse_tokens]
                    offset += num_sparse_tokens
                    sparse_tokens = sparse_flat.view(sparse_flat.size(0), num_sparse_tokens, -1)
                    sparse_flat = (sparse_tokens * sparse_gate.unsqueeze(-1)).flatten(start_dim=1)
                if dense_raw is not None:
                    dense_gate = gate[:, offset:offset + 1]
                    dense_raw = dense_raw * dense_gate
            else:
                offset = 0
                if sparse_flat is not None:
                    sparse_dim = sparse_flat.size(1)
                    sparse_flat = sparse_flat * gate[:, offset:offset + sparse_dim]
                    offset += sparse_dim
                if dense_raw is not None:
                    dense_dim = dense_raw.size(1)
                    dense_raw = dense_raw * gate[:, offset:offset + dense_dim]

        if sparse_flat is not None:
            feat_parts.append(sparse_flat)
        if dense_raw is not None:
            feat_parts.append(dense_raw)
        feat_parts.append(target_emb)
        if self.use_meanpool_in_qgen:
            feat_parts.append(seq_mean)

        # Dual-seq: 追加 target2_emb 和（可选）seq2_mean。
        # 注意追加顺序必须与 __init__ 里 feat_dim 的计算顺序一致：
        #   sparse + dense + target_emb + [seq_mean] + [target2_emb] + [seq2_mean]
        target2_emb = None
        target_condition_item_emb2 = None
        if self.has_seq2:
            target2 = batch.get("target2")
            if target2 is not None:
                target2_emb = self.item_emb2(target2)
                if self.target_condition_item_emb2 is not None:
                    target_condition_item_emb2 = self.target_condition_item_emb2(target2)
                else:
                    target_condition_item_emb2 = target2_emb
            else:
                B = feat_parts[0].size(0)
                target2_emb = torch.zeros(
                    B, self.item_emb2.embedding_dim,
                    device=next(self.parameters()).device,
                )
                target_condition_item_emb2 = target2_emb
            feat_parts.append(target2_emb)

            # 若 seq2 缺失，用 0 填充 MeanPool(seq2) 以保持 feat_dim
            if seq2_kv is None:
                B = feat_parts[0].size(0)
                seq2_kv = torch.zeros(B, 1, self.item_emb2.embedding_dim,
                                      device=next(self.parameters()).device)
                seq2_mask = torch.zeros(B, 1, dtype=torch.bool,
                                        device=next(self.parameters()).device)
                seq2_mean = torch.zeros(B, self.item_emb2.embedding_dim,
                                        device=next(self.parameters()).device)

            if self.use_meanpool_in_qgen:
                feat_parts.append(seq2_mean)

        target_condition = target_condition_item_emb
        if self.target_condition_source != "item":
            B = target_emb.size(0)
            cond_parts = [target_condition_item_emb]
            if self.has_seq2:
                cond_parts.append(target_condition_item_emb2)
            if self.target_condition_source in ("item_side", "item_side_seq") and self.num_sparse > 0:
                if sparse_flat is not None:
                    cond_parts.append(sparse_flat)
                else:
                    cond_parts.append(torch.zeros(
                        B, self.num_sparse * self.sparse_arch.embedding_dim,
                        device=target_emb.device,
                    ))
            if self.target_condition_source in ("item_side", "item_side_seq") and self.has_dense:
                if dense_raw is not None:
                    cond_parts.append(dense_raw)
                else:
                    cond_parts.append(torch.zeros(B, self.num_dense, device=target_emb.device))
            if self.target_condition_source in ("item_seq", "item_side_seq"):
                cond_parts.append(seq_mean)
                if self.has_seq2:
                    cond_parts.append(seq2_mean)
            target_condition = torch.cat(cond_parts, dim=1)
        if self.target_condition_detach:
            target_condition = target_condition.detach()

        feat_vec = torch.cat(feat_parts, dim=1) if len(feat_parts) > 1 else feat_parts[0]

        # Round 2: target-aware gating 在 QueryGen 里做。默认只用 target item id
        # embedding；新实验可通过 target_condition_source 引入 side/sequence 表征。
        if self.use_target_gating:
            queries = self.query_gen(feat_vec, target_emb=target_condition)
        else:
            queries = self.query_gen(feat_vec)

        # Interest Slot Transformer: 叠加可学全局先验
        # q_i = QueryGen_i(feat) + P_i, 让每个 slot 具备可学"角色"
        if self.query_prior is not None:
            queries = queries + self.query_prior.unsqueeze(0)  # broadcast 到 batch

        # NS-tokens: sparse per-feature + (dense) + target + (target2 if has_seq2)
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])
        ns_tokens = [self.sparse_proj(sparse_per)]
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            ns_tokens.append(self.dense_proj(batch["dense"]).unsqueeze(1))
        ns_tokens.append(target_emb.unsqueeze(1))
        if self.has_seq2:
            ns_tokens.append(target2_emb.unsqueeze(1))
        ns = torch.cat(ns_tokens, dim=1)

        interest = self.target_attn(target_emb, seq_kv, seq_mask) if self.use_din_shortcut else None
        interest2 = None
        if self.has_seq2 and self.use_din_shortcut:
            interest2 = self.target_attn2(target2_emb, seq2_kv, seq2_mask)

        for layer_idx, block in enumerate(self.blocks):
            if self.use_ns_in_boosting:
                queries, ns = block(
                    queries, ns, seq_kv, seq_mask,
                    seq2_kv=seq2_kv, seq2_mask=seq2_mask,
                    target_emb=target_condition,
                )
            else:
                queries, _ = block(
                    queries, ns, seq_kv, seq_mask,
                    seq2_kv=seq2_kv, seq2_mask=seq2_mask,
                    target_emb=target_condition,
                )

            # Per-layer TCQG: re-gate queries after each decoder layer.
            # Zero-init guarantees gate=1 at step 0 (identity).
            if self.use_target_gating and self.tcqg_per_layer:
                gl = self.per_layer_tcqg[layer_idx](target_condition)
                gl = gl.view(-1, self.num_queries, queries.size(-1))
                queries = queries * self._tcqg_gate(gl)
                # Optional LayerNorm after gate to prevent magnitude drift
                # from multiplicative compounding across layers.
                if self.tcqg_per_layer_norm:
                    queries = self.per_layer_tcqg_ln[layer_idx](queries)

        queries = self.final_norm(queries)

        q_flat = queries.flatten(start_dim=1)
        final_parts = [q_flat]

        if self.use_din_shortcut:
            final_parts.append(interest)
            final_parts.append(target_emb)

        if self.use_mean_pool:
            final_parts.append(seq_mean)

        if self.has_seq2 and self.use_din_shortcut:
            final_parts.append(interest2)
            final_parts.append(target2_emb)
            if self.use_mean_pool:
                final_parts.append(seq2_mean)

        final = torch.cat(final_parts, dim=1)
        logits = self.top_mlp(final).squeeze(-1)

        aux_loss = None

        # Query Diversity Loss (仅训练时生效)：迫使 q 个 query 两两正交
        # 防止 slot 塌成同一模式；利用 final_norm 之后的 queries，它们已经是模型最终使用的 slot 表示
        if self.training and self.diversity_weight > 0 and self.num_queries > 1:
            # 把 queries (B, q, D) L2 归一化，计算两两 cos 的平方
            q_norm = F.normalize(queries, dim=-1)  # (B, q, D)
            gram = torch.matmul(q_norm, q_norm.transpose(1, 2))  # (B, q, q)
            # 去掉对角线：off-diagonal cos^2 求和，理想值是 0
            eye = torch.eye(self.num_queries, device=gram.device).unsqueeze(0)
            off_diag = gram * (1 - eye)
            # 除以 q*(q-1) 归一化，避免 q 变化时 loss 量级跳变
            aux = (off_diag ** 2).sum() / (gram.size(0) * self.num_queries * (self.num_queries - 1))
            aux_loss = self.diversity_weight * aux

        # Cond-emb alignment loss (仅训练时生效，且需要 separate cond emb)：
        # MSE(cond_emb(target), main_emb(target).detach())。验证 sepitem 跌的元凶是
        # 否在 cond_emb 和 main_emb 语义空间错位。main_emb 侧 detach，避免主路径
        # 反过来被对齐目标污染。
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
