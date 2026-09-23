"""
recscale.models.mixformer — MixFormer（严格按论文实现，带明确标注的 scope simplifications）

论文：MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders
      (ByteDance, arxiv:2602.14110)

核心贡献：把 dense 特征交互和用户序列建模统一到同一个 decoder-only backbone 里，
解决传统方法中两者竞争算力的 co-scaling 问题。

架构（每个 MixFormer Block 三个模块，严格对齐论文 Eq.）：

  1) Query Mixer（替代 self-attention）
       P = HeadMixing(Norm(X)) + X                                   # Eq. 3
       q_i = SwiGLUFFN_i(Norm(p_i)) + p_i,  i = 1..N                 # Eq. 4 (per-head FFN)

  2) Cross Attention（对序列做 cross-attention，query 来自 Query Mixer）
       h_t = SwiGLUFFN^(l)(Norm(s_t)) + s_t    ∈ R^(ND)              # Eq. 5 (per-layer, shared)
       h_t^i = h_t[iD:(i+1)D]                  ∈ R^D                 # Eq. 6
       k_t^i = W_k^i · h_t^i,   v_t^i = W_v^i · h_t^i                # Eq. 7
       z_i = Σ_t softmax((q_i^T k_t^i)/√D) · v_t^i + q_i             # Eq. 8

  3) Output Fusion
       o_i = SwiGLUFFN_i(Norm(z_i)) + z_i                            # Eq. 9 (per-head FFN)

Norm = RMSNorm（论文消融证明优于 Post-LN）。

UI-MixFormer（可选，用 ui_decoupled=True 启用）：
  HeadMixing 加 user→item 单向 mask（论文 4.2 节）：
       M[i,j] = 0, if i < N_U and j >= N_U · D/N
               1, otherwise
       HeadMixing_decouple = M ⊙ HeadMixing
  目的：让 user head 可以在一次请求内跨多个候选 item 复用（request-level batching）。

---

【Scope Simplifications（相对论文的明确偏差）】

两处简化在 MixFormerModel docstring 里有详细说明，这里仅索引：

  1) s_t 只用 item_id embedding，不含 action_type / timestamp / side_info
     → 和项目 HyFormer/OneTrans/DIN 等序列模型保持 scope 一致

  2) target 通过 batch["target"] 显式注入 e_ns（复用 self.seq_emb）
     → 项目里 target 单独拎出而不在 batch["sparse"] 里，需显式注入才能让 CTR 看到候选 item
     → 默认 use_target_in_dense=True；数据集无 target 或做消融可设 False

严格对齐论文的部分：
  - Eq. 2 x_j = W_j · e_ns[...]  ✓ 无 bias
  - Eq. 3-9                      ✓ 完全对齐
  - Pre-RMSNorm + 残差           ✓
  - Per-head SwiGLU (dense), Per-layer SwiGLU (seq)  ✓
  - D_ns % N == 0 强 assert, 不做 silent zero-pad

接口约定（与项目其他模型对齐）：
  batch["sparse"]:   (B, S) int64, required
  batch["seq"]:      (B, T_seq) int64 (item_id, 0 = padding), required
  batch["target"]:   (B,) int64, required iff use_target_in_dense=True (默认)
  batch["dense"]:    (B, D_dense) float, optional
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, SparseArch


# =============================================================================
# 论文 3.3.1 — HeadMixing (parameter-free cross-head information exchange)
# =============================================================================

class HeadMixing(nn.Module):
    """
    HeadMixing: 论文 3.3.1 节，parameter-free。

    步骤（严格按论文）:
      Input:  X ∈ R^(B, N, D)      （B 是 batch 维，论文中忽略）
      Step1:  reshape  → (B, N, N, D/N)       # 前两维都是 N
      Step2:  transpose(dim=1, dim=2)         # 交换前两维的 N
      Step3:  reshape  → (B, N, D)

    约束: D % N == 0
    """

    def __init__(self, num_heads: int, head_dim_full: int):
        """
        Args:
            num_heads: N
            head_dim_full: D（每个 head 的维度，也就是 reshape 后最后一维的 N × (D/N)）
        """
        super().__init__()
        assert head_dim_full % num_heads == 0, (
            f"head_dim_full D={head_dim_full} must be divisible by num_heads N={num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim_full = head_dim_full
        self.sub_dim = head_dim_full // num_heads  # D/N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        assert N == self.num_heads and D == self.head_dim_full
        # (B, N, D) → (B, N, N, D/N)
        x = x.reshape(B, N, N, self.sub_dim)
        # transpose 前两个 N 维
        x = x.transpose(1, 2).contiguous()
        # flatten 回 (B, N, D)
        return x.reshape(B, N, D)


class HeadMixingMasked(nn.Module):
    """
    UI-MixFormer 的 HeadMixing (论文 4.2 节, Eq.):

      M[i, j] = 0, if i < N_U  and  j >= N_U · D/N
               1, otherwise
      HeadMixing_decouple = M ⊙ HeadMixing(X)

    直观: 在 HeadMixing 结果上，user head (i<N_U) 位置且来自 item 子列的元素被置 0，
         → 等价于 "user 不从 item 拉信息"，只有 "user → item" 单向流。

    注：mask 是作用在 HeadMixing 之后的输出张量 (B, N, D) 上的元素级 mask，
    其中 j 维对应 D 轴（被切成 N 段，每段 D/N 维）。
    """

    def __init__(self, num_heads: int, head_dim_full: int, num_user_heads: int):
        super().__init__()
        assert head_dim_full % num_heads == 0
        assert 0 < num_user_heads < num_heads
        self.num_heads = num_heads
        self.head_dim_full = head_dim_full
        self.sub_dim = head_dim_full // num_heads
        self.num_user_heads = num_user_heads

        self.head_mixing = HeadMixing(num_heads, head_dim_full)

        # 预计算 mask: (1, N, D)
        N = num_heads
        D = head_dim_full
        N_U = num_user_heads
        sub = self.sub_dim
        mask = torch.ones(1, N, D)
        # 前 N_U 行（user heads），列 j 在 [N_U·sub, D) 的位置置 0
        mask[:, :N_U, N_U * sub:] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head_mixing(x)          # (B, N, D)
        return out * self.mask              # element-wise mask


# =============================================================================
# 论文 Eq. 4 / 9 — Per-Head SwiGLU FFN
# =============================================================================

class PerHeadSwiGLU(nn.Module):
    """
    每个 head 独立参数的 SwiGLU FFN（论文 "per-head SwiGLU-activated FFN"）:
      SwiGLU(p_i) = W_down^i · ( Swish(W_gate^i · p_i) ⊙ (W_up^i · p_i) ) + b_down^i

    每个 i ∈ {1..N} 有自己的 (W_gate^i, W_up^i, W_down^i)，einsum 实现。
    """

    def __init__(self, num_heads: int, head_dim: int,
                 ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.w_gate = nn.Parameter(torch.empty(num_heads, head_dim, ffn_dim))
        self.b_gate = nn.Parameter(torch.zeros(num_heads, ffn_dim))
        self.w_up = nn.Parameter(torch.empty(num_heads, head_dim, ffn_dim))
        self.b_up = nn.Parameter(torch.zeros(num_heads, ffn_dim))
        self.w_down = nn.Parameter(torch.empty(num_heads, ffn_dim, head_dim))
        self.b_down = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.w_gate, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_up, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_down, a=0, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        gate = torch.einsum("bnd,ndf->bnf", x, self.w_gate) + self.b_gate
        gate = F.silu(gate)
        up = torch.einsum("bnd,ndf->bnf", x, self.w_up) + self.b_up
        h = gate * up
        out = torch.einsum("bnf,nfd->bnd", h, self.w_down) + self.b_down
        return self.dropout(out)


# =============================================================================
# 论文 Eq. 5 — Per-Layer (shared across tokens) SwiGLU for sequence
# =============================================================================

class PerLayerSwiGLU(nn.Module):
    """
    整条序列共享的 per-layer SwiGLU（论文 Cross Attention 里的 SwiGLUFFN^(l)）:
      h_t = SwiGLUFFN^(l)(Norm(s_t)) + s_t

    对所有 time step t 共享同一组参数（但每层 l 有独立的 SwiGLUFFN^(l)）。
    """

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, ffn_dim, bias=True)
        self.w_up = nn.Linear(dim, ffn_dim, bias=True)
        self.w_down = nn.Linear(ffn_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
        Returns:
            (..., dim)
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# =============================================================================
# 论文 3.3.1 — Query Mixer
# =============================================================================

class QueryMixer(nn.Module):
    """
    Query Mixer（论文 Eq. 3, 4）:
      P = HeadMixing(Norm(X)) + X                     # Eq. 3
      q_i = SwiGLUFFN_i(Norm(p_i)) + p_i             # Eq. 4

    Norm = RMSNorm（Pre-Norm）
    """

    def __init__(self, num_heads: int, head_dim_full: int, ffn_dim: int,
                 dropout: float = 0.0, ui_decoupled: bool = False,
                 num_user_heads: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.RMSNorm(head_dim_full)
        if ui_decoupled:
            assert num_user_heads is not None and 0 < num_user_heads < num_heads
            self.head_mixing = HeadMixingMasked(num_heads, head_dim_full, num_user_heads)
        else:
            self.head_mixing = HeadMixing(num_heads, head_dim_full)

        self.norm2 = nn.RMSNorm(head_dim_full)
        self.per_head_ffn = PerHeadSwiGLU(num_heads, head_dim_full, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        p = self.head_mixing(self.norm1(x)) + x          # Eq. 3
        q = self.per_head_ffn(self.norm2(p)) + p          # Eq. 4
        return q


# =============================================================================
# 论文 3.3.2 — Cross Attention
# =============================================================================

class CrossAttention(nn.Module):
    """
    Cross Attention（论文 Eq. 5-8）:

      h_t = SwiGLUFFN^(l)(Norm(s_t)) + s_t    ∈ R^(ND)              # Eq. 5
      h_t^i = h_t[iD:(i+1)D]                  ∈ R^D                 # Eq. 6
      k_t^i = W_k^i · h_t^i,   v_t^i = W_v^i · h_t^i                # Eq. 7
      z_i = Σ_t softmax((q_i^T k_t^i)/√D) · v_t^i + q_i             # Eq. 8

    注：
    - SwiGLUFFN^(l) 是 per-layer 的（所有 token 共享，每层独立）—— 论文明确强调
    - W_k^i, W_v^i 是 per-head 的（每个 i 独立）—— 论文 Eq. 7
    - 残差作用在 cross-attention 的输出上，加回 q_i

    【隐藏行为: 某行 seq 全 pad 时的 fallback】
      若 seq_mask 某一行全 False（序列全部是 padding），softmax(-inf, -inf, ...) 会产生 NaN，
      我们用 `nan_to_num(nan=0.0)` 把 attention 权重置 0 → Σ_t 0·v = 0 → z_i = 0 + q_i = q_i
      相当于 cross-attn 对这行样本退化为恒等映射（加 residual），后续 Output Fusion 照常跑。
      这是合理的 fallback，不会抛 NaN，但该样本不会从序列获得任何增量信息。
    """

    def __init__(self, num_heads: int, head_dim: int,
                 ffn_dim: int, dropout: float = 0.0):
        """
        Args:
            num_heads: N
            head_dim: D（每 head 的维度，同时是 q_i、k_t^i、v_t^i 的维度）
            ffn_dim: 序列 SwiGLUFFN 的中间维度
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        full_dim = num_heads * head_dim          # ND

        # Eq. 5: per-layer SwiGLU for sequence (所有 token 共享，每层独立)
        self.seq_norm = nn.RMSNorm(full_dim)
        self.seq_ffn = PerLayerSwiGLU(full_dim, ffn_dim, dropout)

        # Eq. 7: per-head K/V projection
        self.w_k = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        self.w_v = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        self.scale = 1.0 / math.sqrt(head_dim)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q: torch.Tensor, seq: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None,
                pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q:        (B, N, D) — Query Mixer 的输出
            seq:      (B, T, N*D) — 序列 token embedding（每个 s_t 是 ND 维）
            seq_mask: (B, T) — True/1 = valid, False/0 = padding（可选）
            pos_bias: (B, N, T) — TCPB target-conditioned position bias（可选）
                      added to scores BEFORE mask + softmax so that padded
                      positions remain at -inf and TCPB does not "leak" into
                      pad slots.
        Returns:
            z: (B, N, D) — Eq. 8
        """
        B, N, D = q.shape
        assert N == self.num_heads and D == self.head_dim
        _, T, ND = seq.shape
        assert ND == N * D, f"seq last dim {ND} != N*D = {N*D}"

        # Eq. 5: h_t = SwiGLUFFN(Norm(s_t)) + s_t
        h = self.seq_ffn(self.seq_norm(seq)) + seq        # (B, T, ND)

        # Eq. 6: reshape 成 per-head (B, T, N, D)
        h = h.reshape(B, T, N, D)

        # Eq. 7: k_t^i = W_k^i · h_t^i,  v_t^i = W_v^i · h_t^i
        # einsum: (B,T,N,D) × (N,D,D) → (B,T,N,D)
        k = torch.einsum("btnd,nde->btne", h, self.w_k)
        v = torch.einsum("btnd,nde->btne", h, self.w_v)

        # Eq. 8: attention scores per head, per query
        # q: (B, N, D),  k: (B, T, N, D)
        # scores[b, n, t] = q[b, n, :] · k[b, t, n, :] / √D
        scores = torch.einsum("bnd,btnd->bnt", q, k) * self.scale     # (B, N, T)

        # TCPB (target-conditioned position bias) — applied BEFORE mask so
        # that padded positions still get -inf overwrite from `masked_fill`.
        if pos_bias is not None:
            scores = scores + pos_bias

        # mask
        if seq_mask is not None:
            # seq_mask: (B, T) → (B, 1, T)
            mask = seq_mask.unsqueeze(1)
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        attn = F.softmax(scores, dim=-1)        # (B, N, T)
        # 处理全 mask 行（避免 NaN）
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        # Σ_t attn[b,n,t] · v[b,t,n,:]
        # v: (B, T, N, D), attn: (B, N, T) → (B, N, D)
        z_no_res = torch.einsum("bnt,btnd->bnd", attn, v)

        # 残差: + q_i
        return z_no_res + q


# =============================================================================
# 论文 3.3.3 — Output Fusion
# =============================================================================

class OutputFusion(nn.Module):
    """
    Output Fusion（论文 Eq. 9）:
      o_i = SwiGLUFFN_i(Norm(z_i)) + z_i

    又一个 per-head SwiGLU（和 Query Mixer 里的是不同参数）。
    """

    def __init__(self, num_heads: int, head_dim: int,
                 ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.RMSNorm(head_dim)
        self.per_head_ffn = PerHeadSwiGLU(num_heads, head_dim, ffn_dim, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N, D) → (B, N, D)"""
        return self.per_head_ffn(self.norm(z)) + z


# =============================================================================
# MixFormer Block
# =============================================================================

class MixFormerBlock(nn.Module):
    """
    单个 MixFormer Block = Query Mixer + Cross Attention + Output Fusion

    输入:
      X:    (B, N, D) — dense 特征的 head 表示
      seq:  (B, T, N*D) — 序列 token embedding
      seq_mask: (B, T) — 可选
    输出:
      (B, N, D)
    """

    def __init__(self, num_heads: int, head_dim: int,
                 ffn_dim_dense: int, ffn_dim_seq: int,
                 dropout: float = 0.0, ui_decoupled: bool = False,
                 num_user_heads: Optional[int] = None):
        super().__init__()
        # Query Mixer 作用在 (B, N, D) 上，HeadMixing 的 "D" 就是 head_dim
        # 它把 (N, D) reshape 成 (N, N, D/N)，所以 head_dim 必须能被 num_heads 整除
        self.query_mixer = QueryMixer(
            num_heads=num_heads,
            head_dim_full=head_dim,
            ffn_dim=ffn_dim_dense,
            dropout=dropout,
            ui_decoupled=ui_decoupled,
            num_user_heads=num_user_heads,
        )
        self.cross_attn = CrossAttention(
            num_heads=num_heads, head_dim=head_dim,
            ffn_dim=ffn_dim_seq, dropout=dropout,
        )
        self.output_fusion = OutputFusion(
            num_heads=num_heads, head_dim=head_dim,
            ffn_dim=ffn_dim_dense, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, seq: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None,
                pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.query_mixer(x)                          # (B, N, D)
        z = self.cross_attn(q, seq, seq_mask, pos_bias=pos_bias)  # (B, N, D)
        o = self.output_fusion(z)                        # (B, N, D)
        return o


# =============================================================================
# MixFormer Model
# =============================================================================

@register_model
class MixFormerModel(RecModel):
    """
    MixFormer: 统一 dense 特征交互 + 用户序列建模的 decoder-only 模型。

    严格按论文（arxiv 2602.14110）实现，有两处 scope simplifications 已明确标注：

    【Scope Simplification 1】序列 token s_t 只用 item_id embedding
      - 论文原文: s_t = concat(item_id_emb, action_type_emb, timestamp_emb, side_info_embs...)
                 ∈ R^(ND), 包含丰富的行为侧信息
      - 本实现: s_t = linear(item_id_emb) ∈ R^(ND), 只有 item_id
      - 原因: 和项目中 HyFormer/OneTrans/DIN 等其他序列模型的 scope 保持一致，
              数据集 pipeline 当前只提供 item_id 序列
      - 影响: 论文里 "rich action representation + recency via timestamp" 的优势在此缺失

    【Scope Simplification 2】target 注入方式
      - 论文原文: target item features 本来就在 e_ns 的非序列特征里 (某些特征如 cate_id 对应 target item)
      - 本实现: 若 batch 提供 batch["target"] 且 use_target_in_dense=True (默认),
              复用 self.seq_emb 得到 target_emb 并 concat 进 e_ns
      - 原因: 项目里 batch["target"] 是单独拎出的 item_id, 不在 batch["sparse"] 里。
              为保证模型能看到候选 item, 需显式注入
      - 如果数据集不提供 target, 或者你想消融, 可以设 use_target_in_dense=False

    【隐藏行为 1: target ID = 0 会被当作 padding】
      self.seq_emb 用 padding_idx=0, 且 weight[0] 强制为 0 向量。
      如果某些 batch 的 target=0, 该样本的 target_emb 就是全 0 → dense 路径上完全丢失 target 信号。
      - 实践中 target=0 通常代表数据脏 (pad/unknown), 此时丢失信号是期望行为。
      - 但如果你的数据集把 0 当成有效的 target item ID, 请在 dataset 侧做 +1 偏移, 或改用
        独立的 target embedding 表。
      - 该限制和项目中 HyFormer/DIN 等用 padding_idx=0 的模型一致。

    【隐藏行为 2: 序列全 pad 时 cross-attn 退化为恒等】
      若某样本的 seq 全是 pad (seq_mask 全 False), CrossAttention 会用 nan_to_num 把 attention
      置 0 → z_i = q_i (cross-attn 退化为恒等+residual)。模型仍可 forward, 但该样本不会从
      序列获得任何增量信息。详见 CrossAttention docstring。

    输入 (batch keys):
      batch["sparse"]: (B, S) int64, required
      batch["seq"]:    (B, T_seq) int64 (item_id, 0=padding), required
      batch["target"]: (B,) int64, required iff use_target_in_dense=True
      batch["dense"]:  (B, D_dense) float, optional

    Config (mc = config["model"]):
      num_heads:             N（论文默认 16）
      head_dim:              D（每 head 维度，论文 small=386, medium=768）
      num_layers:            L（论文默认 4）
      ffn_dim:               per-head SwiGLU 中间维度（Query Mixer + Output Fusion 共用）
      seq_ffn_dim:           per-layer SwiGLU 中间维度（Cross Attention 里序列 FFN）
                             默认 N*D（1:1 gate factor，对 CTR 小 hidden 场景较保守）；
                             大模型可设为 2*N*D 或 4*N*D 以提升容量
      seq_emb_dim:           序列 item embedding 维度（会 linear 投到 N*D）。默认 = head_dim
      sparse_emb_dim:        每个 sparse 特征的 embedding 维度
      use_target_in_dense:   True（默认, 复用 seq_emb 取 target 注入 e_ns, 对齐论文）
                             / False（消融 / 数据集无 target 时关闭）
      ui_decoupled:          False（完整 MixFormer）/ True（UI-MixFormer）
      num_user_heads:        N_U，ui_decoupled=True 时生效（默认 N//2）
      mlp_dims:              顶层 MLP 维度
      dropout:               dropout rate

    注意: 必须满足 D_ns % N == 0（这里 D_ns 包括 target embedding if use_target_in_dense=True）。
    如果不满足会在 __init__ 中 assert 报错, 请调整 sparse_emb_dim / num_sparse / seq_emb_dim。
    """
    model_name = "mixformer"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        N = mc["num_heads"]
        D = mc["head_dim"]
        L = mc.get("num_layers", 4)
        ffn_dim = mc.get("ffn_dim", D * 2)
        # 默认改成 N*D (1:1 gate factor)，对 CTR 小 hidden 场景更保守；大模型可显式设更大
        seq_ffn_dim = mc.get("seq_ffn_dim", N * D)
        seq_emb_dim = mc.get("seq_emb_dim", D)
        use_target_in_dense = mc.get("use_target_in_dense", True)
        ui_decoupled = mc.get("ui_decoupled", False)
        num_user_heads = mc.get("num_user_heads", N // 2)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)

        # ----- TCQD-MF knobs (target-conditioned modulation, all default OFF) -----
        # use_target_gating  (TCQG): per-(N, D) channel gate on dense heads X
        # use_target_time_bias (TCPB): per-layer target-conditioned position bias
        #                              on cross-attention scores (B, N, T)
        # target_condition_source ∈ {"item", "item_seq"}: maturity rule
        # gate_activation_mode  ∈ {"2sigmoid", "sigmoid", "residual_tanh"}
        # tcpb_share_layers     : True => one (W_τ, P_emb) shared by all layers
        # tcpb_max_seq_len      : sets the per-layer position-emb table size
        self.use_target_gating = mc.get("use_target_gating", False)
        self.use_target_time_bias = mc.get("use_target_time_bias", False)
        self.use_position_bias = mc.get("use_position_bias", False)
        self.pos_bias_max_len = int(mc.get("pos_bias_max_len", 1024))
        self.target_time_rank = mc.get("target_time_rank", 8)
        self.target_condition_source = mc.get("target_condition_source", "item")
        self.gate_activation_mode = mc.get("gate_activation_mode", "2sigmoid")
        self.tcpb_share_layers = mc.get("tcpb_share_layers", False)
        self.tcpb_max_seq_len = mc.get("tcpb_max_seq_len", 1024)
        assert self.target_condition_source in ("item", "item_seq"), (
            f"target_condition_source must be 'item' or 'item_seq', "
            f"got {self.target_condition_source}"
        )
        assert self.gate_activation_mode in (
            "2sigmoid", "sigmoid", "residual_tanh"
        ), self.gate_activation_mode

        assert D > 0, "head_dim must be > 0"
        assert D % N == 0, f"head_dim D={D} must be divisible by num_heads N={N}"
        if ui_decoupled:
            assert 0 < num_user_heads < N, (
                f"num_user_heads={num_user_heads} must be in (0, {N})"
            )

        self.num_heads = N
        self.head_dim = D
        self.num_layers = L
        self.ui_decoupled = ui_decoupled
        self.use_target_in_dense = use_target_in_dense

        # ----- Dense 特征 embedding: sparse [+ dense] [+ target] → concat → 切 N head -----
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        # 每个 sparse 特征用同一个 emb_dim = sparse_emb_dim
        sparse_emb_dim = mc.get("sparse_emb_dim", 16)
        self.sparse_arch = SparseArch(num_sparse, sparse_emb_dim, cardinalities)

        num_dense = len(dc.get("dense_cols") or [])
        self.num_dense = num_dense

        # ----- 序列 embedding (先建, 因为 target 注入要复用它) -----
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)
        self.seq_emb = nn.Embedding(num_items + 1, seq_emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.seq_emb.weight)
        self.seq_emb.weight.data[0].zero_()

        # 将 seq_emb_dim 投到 N*D (s_t ∈ R^(ND))
        if seq_emb_dim != N * D:
            self.seq_proj = nn.Linear(seq_emb_dim, N * D)
        else:
            self.seq_proj = nn.Identity()

        # ----- D_ns 计算 (论文 Eq. 2, 含 target 若启用) -----
        D_ns = num_sparse * sparse_emb_dim + num_dense
        if use_target_in_dense:
            D_ns += seq_emb_dim   # 复用 seq_emb 得到 target embedding

        # 严格 assert, 不 silent pad
        assert D_ns % N == 0, (
            f"D_ns = {D_ns} (num_sparse * sparse_emb_dim [+ num_dense] [+ seq_emb_dim]) "
            f"must be divisible by num_heads N={N}. "
            f"组成: num_sparse*sparse_emb_dim = {num_sparse}*{sparse_emb_dim} = "
            f"{num_sparse * sparse_emb_dim}"
            f"{' + num_dense=' + str(num_dense) if num_dense > 0 else ''}"
            f"{' + seq_emb_dim=' + str(seq_emb_dim) if use_target_in_dense else ''}. "
            f"请调整 sparse_emb_dim, num_sparse 或 seq_emb_dim 使 D_ns 被 N 整除。"
        )
        self.sub_dim = D_ns // N        # 论文 d = D_ns / N
        self.D_ns = D_ns

        # 论文 Eq. 2: x_j = W_j · e_ns[d·(j-1):d·j],  W_j ∈ R^(D × d) (no bias)
        self.head_projs = nn.Parameter(torch.empty(N, self.sub_dim, D))
        nn.init.xavier_uniform_(self.head_projs)

        # ----- L 个 MixFormer Block -----
        self.blocks = nn.ModuleList([
            MixFormerBlock(
                num_heads=N, head_dim=D,
                ffn_dim_dense=ffn_dim,
                ffn_dim_seq=seq_ffn_dim,
                dropout=dropout,
                ui_decoupled=ui_decoupled,
                num_user_heads=num_user_heads if ui_decoupled else None,
            )
            for _ in range(L)
        ])

        # ----- TCQD-MF modules (built only when knobs are True) -----
        # Static Slot-Specific Position Bias (SlotPos): per-head, per-position
        # learnable bias shared across samples. Same design as HyFormer's B_pos.
        if self.use_position_bias:
            self.slot_pos_bias = nn.Parameter(
                torch.zeros(N, self.pos_bias_max_len)
            )

        # condition vector dim: target embedding only, or +seq_emb mean
        cond_dim = seq_emb_dim
        if self.target_condition_source == "item_seq":
            cond_dim = seq_emb_dim + seq_emb_dim
        self._cond_dim = cond_dim
        self.seq_emb_dim = seq_emb_dim

        if self.use_target_gating:
            # gate ∈ ℝ^(B, N*D) -> reshape (B, N, D) and act on X
            self.tcqg_proj = nn.Linear(cond_dim, N * D)
            # zero-init so g(c) starts at neutral (1.0 for 2sigmoid / residual_tanh,
            # 0.5 for sigmoid — the latter halves X on step 0 and is the explicit
            # ablation; identical convention to HyFormer's QueryGenerationV2).
            nn.init.zeros_(self.tcqg_proj.weight)
            nn.init.zeros_(self.tcqg_proj.bias)

        if self.use_target_time_bias:
            R = self.target_time_rank
            # Per-layer params (HyFormer convention) unless tcpb_share_layers=True.
            num_modules = 1 if self.tcpb_share_layers else L
            self.tcpb_proj = nn.ModuleList([
                nn.Linear(cond_dim, N * R) for _ in range(num_modules)
            ])
            self.tcpb_pos_emb = nn.ModuleList([
                nn.Embedding(self.tcpb_max_seq_len, R) for _ in range(num_modules)
            ])
            for proj in self.tcpb_proj:
                # Zero-init W_τ → bias = 0 at step 0, identity start.
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            for emb in self.tcpb_pos_emb:
                nn.init.normal_(emb.weight, std=0.02)

        # ----- Top MLP -----
        # 最终 output (B, N, D) → flatten → MLP → logit
        in_dim = N * D
        top_layers = []
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

        variant = "UI-MixFormer" if ui_decoupled else "MixFormer"
        nu_str = f", N_U={num_user_heads}" if ui_decoupled else ""
        tgt_str = f", target_in_dense={use_target_in_dense}"
        # TCQD-MF status line: only printed when at least one of TCQG / TCPB on.
        tcqd_bits = []
        if self.use_target_gating:
            tcqd_bits.append(f"TCQG({self.gate_activation_mode})")
        if self.use_target_time_bias:
            share_str = "shared" if self.tcpb_share_layers else "per-layer"
            tcqd_bits.append(
                f"TCPB(R={self.target_time_rank},{share_str})"
            )
        if tcqd_bits:
            tcqd_str = (
                f", TCQD=[{', '.join(tcqd_bits)}], "
                f"cond={self.target_condition_source}({self._cond_dim}d)"
            )
        else:
            tcqd_str = ""
        print(f"[{variant}] N={N}, D={D}, L={L}, ffn={ffn_dim}, seq_ffn={seq_ffn_dim}, "
              f"seq_emb={seq_emb_dim}, sparse_emb={sparse_emb_dim}, "
              f"D_ns={D_ns}, sub_dim={self.sub_dim}{tgt_str}{nu_str}{tcqd_str}")

    def _build_dense_heads(self, batch: dict) -> torch.Tensor:
        """
        构造论文 Eq. 2 的 X = [x_1, ..., x_N] ∈ R^(B, N, D)

        e_ns = concat(sparse_embs, dense [, target_emb]) → split 成 N 段, 每段 d=D_ns/N
        x_j = W_j · e_ns[d·(j-1):d·j]     (论文 Eq. 2, 无 bias)
        """
        parts = []
        sparse_flat = self.sparse_arch(batch["sparse"])   # (B, S*sparse_emb)
        parts.append(sparse_flat)
        if self.num_dense > 0 and "dense" in batch and batch["dense"] is not None:
            parts.append(batch["dense"])

        # Scope Simplification 2: 显式注入 target
        if self.use_target_in_dense:
            target = batch.get("target", None)
            if target is None:
                raise ValueError(
                    "MixFormer: use_target_in_dense=True 但 batch 未提供 'target' 字段。"
                    " 请在 dataset/dataloader 中提供 batch['target'] (B,) int64, 或在 config "
                    "中设置 use_target_in_dense=False"
                )
            target_emb = self.seq_emb(target)              # (B, seq_emb_dim)
            parts.append(target_emb)

        e_ns = torch.cat(parts, dim=1)                     # (B, D_ns)

        # split into N heads: (B, N, sub_dim)
        e_ns = e_ns.reshape(-1, self.num_heads, self.sub_dim)

        # per-head projection: (B, N, sub_dim) × (N, sub_dim, D) → (B, N, D)  (无 bias)
        X = torch.einsum("bnd,nde->bne", e_ns, self.head_projs)
        return X

    def _build_seq_tokens(self, batch: dict):
        """
        Returns (seq_tokens, seq_mask, seq_raw_mean):
          seq_tokens:    (B, T, N*D) — projected sequence tokens for cross-attn
          seq_mask:      (B, T), bool (True=valid)
          seq_raw_mean:  (B, seq_emb_dim) — masked mean of *raw* item embeddings
                         (before seq_proj). Used as part of condition vector
                         when target_condition_source='item_seq', to keep the
                         maturity rule's c-dim small and aligned with HyFormer.
        """
        seq_ids = batch["seq"]                              # (B, T)
        seq_mask = (seq_ids != 0)                            # (B, T)
        h_raw = self.seq_emb(seq_ids)                        # (B, T, seq_emb_dim)
        # masked mean of raw item embeddings
        m = seq_mask.float().unsqueeze(-1)                   # (B, T, 1)
        denom = m.sum(dim=1).clamp_min(1.0)                  # (B, 1)
        seq_raw_mean = (h_raw * m).sum(dim=1) / denom        # (B, seq_emb_dim)
        h = self.seq_proj(h_raw)                             # (B, T, N*D)
        return h, seq_mask, seq_raw_mean

    def _build_condition(self, batch: dict, seq_raw_mean: torch.Tensor) -> torch.Tensor:
        """
        Build the target-condition vector c ∈ ℝ^(B, _cond_dim).

        Sources:
          - "item":      c = target_emb           (KuaiRec / TaobaoAd, mature item regime)
          - "item_seq":  c = [target_emb; s̄]     (KuaiRand / TAAC, sparse target regime)

        Raises if `batch['target']` is missing because TCQD modules are
        active only when target conditioning is meaningful.
        """
        target = batch.get("target", None)
        if target is None:
            raise ValueError(
                "MixFormer TCQD: condition source needs batch['target']."
                " Either provide target or disable use_target_gating /"
                " use_target_time_bias."
            )
        target_emb = self.seq_emb(target)                    # (B, seq_emb_dim)
        if self.target_condition_source == "item":
            return target_emb
        # "item_seq"
        return torch.cat([target_emb, seq_raw_mean], dim=1)

    def _gate_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the chosen activation to gate logits (HyFormer convention)."""
        if self.gate_activation_mode == "2sigmoid":
            return 2.0 * torch.sigmoid(logits)
        if self.gate_activation_mode == "residual_tanh":
            return 1.0 + torch.tanh(logits)
        return torch.sigmoid(logits)

    def forward(self, batch: dict) -> torch.Tensor:
        X = self._build_dense_heads(batch)                   # (B, N, D)
        seq, seq_mask, seq_raw_mean = self._build_seq_tokens(batch)
        # ^ (B, T, N*D), (B, T), (B, seq_emb_dim)
        B = X.size(0)

        # Build c only if any TCQD module is on (avoid extra cost otherwise)
        cond = None
        if self.use_target_gating or self.use_target_time_bias:
            cond = self._build_condition(batch, seq_raw_mean)  # (B, cond_dim)

        # TCQG: gate dense heads X by g(c) once before the block stack.
        if self.use_target_gating:
            gate_logits = self.tcqg_proj(cond).view(B, self.num_heads, self.head_dim)
            gate = self._gate_activation(gate_logits)
            X = X * gate

        x = X
        T_seq = seq.size(1)
        for layer_idx, block in enumerate(self.blocks):
            pos_bias = None
            # Static SlotPos: per-head position bias shared across samples
            if self.use_position_bias:
                pos_bias = self.slot_pos_bias[:, :T_seq].unsqueeze(0)  # (1, N, T)
            # TCPB: target-conditioned position bias (additive on top of SlotPos)
            if self.use_target_time_bias:
                # Per-layer params unless `tcpb_share_layers` is True.
                idx = 0 if self.tcpb_share_layers else layer_idx
                R = self.target_time_rank
                # M(c) ∈ ℝ^(B, N, R)
                M = self.tcpb_proj[idx](cond).view(B, self.num_heads, R)
                # P ∈ ℝ^(T, R) — clamp positions to embedding table size
                pos_idx = torch.arange(T_seq, device=x.device).clamp_max(
                    self.tcpb_pos_emb[idx].num_embeddings - 1
                )
                P = self.tcpb_pos_emb[idx](pos_idx)           # (T, R)
                tcpb_bias = torch.einsum("bnr,tr->bnt", M, P)  # (B, N, T)
                if pos_bias is not None:
                    pos_bias = pos_bias + tcpb_bias
                else:
                    pos_bias = tcpb_bias
            x = block(x, seq, seq_mask, pos_bias=pos_bias)    # (B, N, D)

        # flatten → MLP
        x = x.reshape(x.size(0), -1)                          # (B, N*D)
        return self.top_mlp(x).squeeze(-1)
