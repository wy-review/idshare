"""
recscale.models.unimixer_v2 — UniMixer V2（论文正确实现）

严格遵循论文 "UniMixer: A Unified Architecture for Scaling Laws in
Recommendation Systems" (https://arxiv.org/abs/2604.00590) 的设计。

核心创新：
  1. UniMixing: 两层可学习交互（局部 W_B^i + 全局 W_G），替代 RankMixer 的纯 permute
  2. 移除 H=T 约束: 用 block_size B 控制交互粒度，B 和 T 独立
  3. Sinkhorn-Knopp 双随机归一化 + 对称约束 + 温度退火
  4. SiameseNorm: 双流耦合残差，改善深层网络梯度传播
  5. Per-token SwiGLU: channel mixing 用门控结构

支持完整版 UniMixer 和 UniMixer-Lite 两种变体（由 config 切换）：
  - UniMixer-Full (默认):
      W_B^i: 每个 block 独立学习 (N_b 个 B×B 矩阵)
      W_G:   完整 (N_b × N_b) 矩阵
  - UniMixer-Lite (公式 18):
      W_B^i = Σ_ℓ ω_ℓ^i · Z_ℓ  (basis-composed, b 个共享 basis + 每个 block 的权重向量)
      W_G   = A_G @ B_G          (low-rank, 秩 r)

架构：
  sparse(S fields) → SparseArch → (B, S, D)
                 → FeatureFieldPooling → (B, T, D)
       ↓
  Stack of UniMixerV2Block × M (with SiameseNorm dual-stream)
  ├── UniMixing: 局部 W_B^i (B×B) + 全局 W_G (N_b×N_b)，Sinkhorn 归一化
  └── Per-Token SwiGLU (每个 token 独立 gate/up/down)
       ↓
  SiameseNorm 融合 → flatten / mean pooling → MLP → logit
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import FeatureFieldPooling, RecModel, SparseArch


# =============================================================================
# Sinkhorn-Knopp 迭代（双随机矩阵归一化）
# =============================================================================

def sinkhorn_knopp(logits: torch.Tensor, num_iters: int = 5,
                   tau: float = 1.0) -> torch.Tensor:
    """
    Sinkhorn-Knopp 迭代：将 logits 归一化为近似双随机矩阵。

    Args:
        logits: (..., N, N) 原始 logits
        num_iters: 迭代次数（默认 5，论文未指定具体值）
        tau: 温度系数，越小越稀疏
    Returns:
        (..., N, N) 近似双随机矩阵
    """
    M = logits / tau
    for _ in range(num_iters):
        M = M - torch.logsumexp(M, dim=-1, keepdim=True)  # 行归一化
        M = M - torch.logsumexp(M, dim=-2, keepdim=True)  # 列归一化
    return M.exp()


# =============================================================================
# UniMixing 模块（论文公式 11-15）
# =============================================================================

class UniMixing(nn.Module):
    """
    UniMixing: 两层可学习特征交互。

    论文公式 11:
      UniMixing(X) = reshape((W_G ⊗ {W_B^i}) * flatten(X), 1, L)

    实际分两步：
      Step 1 (局部交互, 公式 13):
        L = T*D, 分成 N_b = L/B 个 block，每个 block 大小 B
        每个 block i 有独立的 W_B^i ∈ R^(B×B)
        H[i] = x_i @ W_B^i

      Step 2 (全局交互, 公式 14):
        在 block 维度乘 W_G ∈ R^(N_b × N_b)
        输出 = W_G @ H (在 block 维度上)

    约束 (公式 15):
      - 对称: W̃ = (W + W^T) / 2
      - Sinkhorn-Knopp 双随机归一化
      - 温度 τ 控制稀疏度

    --- 完整版 vs Lite 版的区别 ---

    全局交互 W_G:
      - 完整版: W_G ∈ R^(N_b×N_b) 直接学习
      - Lite 版 (公式 18): W_G = A_G @ B_G, low-rank, A_G ∈ R^(N_b×r), B_G ∈ R^(r×N_b)

    局部交互 W_B^i:
      - 完整版: 每个 block 独立学习 W_B^i ∈ R^(B×B)（共 N_b 个矩阵，参数量 N_b*B²）
      - Lite 版 (公式 18, basis-composed module):
          共享 basis 矩阵 {Z_ℓ}_{ℓ=1}^b, 每个 Z_ℓ ∈ R^(B×B)
          每个 block 的权重向量 ω^i ∈ R^b
          W_B^{*i} = Sinkhorn-Knopp( Σ_ℓ ω_ℓ^i · Z_ℓ )
          参数量: b*B² + N_b*b，当 b << N_b 时显著减少

    Args:
        num_basis: basis 矩阵数量 b（Lite 版开关）。
                   <= 0 或 None: 使用完整版（每个 block 独立 W_B^i）
                   > 0: 使用 Lite 版（basis-composed module）
        global_rank: W_G 的低秩维度。
                     None: 使用完整 W_G
                     > 0 且 < N_b: 使用低秩分解 A_G @ B_G
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        block_size: int = 0,
        global_rank: Optional[int] = None,
        num_basis: Optional[int] = None,
        sinkhorn_iters: int = 5,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        symmetric: bool = True,
    ):
        super().__init__()
        L = num_tokens * token_dim  # 总展平维度

        # 自动选 block_size: 默认 = token_dim (每个 token 一个 block)
        if block_size <= 0:
            block_size = token_dim
        assert L % block_size == 0, (
            f"L={L} (T={num_tokens} * D={token_dim}) must be divisible by "
            f"block_size={block_size}"
        )

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.block_size = block_size
        self.num_blocks = L // block_size  # N_b
        self.sinkhorn_iters = sinkhorn_iters
        self.symmetric = symmetric

        # 当前温度（通过 set_tau 外部更新，支持退火）
        self.register_buffer("tau", torch.tensor(tau_start))
        self.tau_start = tau_start
        self.tau_end = tau_end

        # ----- 局部交互 W_B -----
        # Lite 版: basis-composed module (公式 18)
        #   共享 basis Z ∈ R^(b, B, B) + block 权重 omega ∈ R^(N_b, b)
        #   W_B^i = Σ_ℓ omega[i,ℓ] * Z[ℓ]
        # 完整版: 每个 block 独立 W_B^i ∈ R^(N_b, B, B)
        if num_basis is not None and num_basis > 0:
            self.num_basis = num_basis
            self.Z_basis = nn.Parameter(torch.empty(num_basis, block_size, block_size))
            self.omega = nn.Parameter(torch.empty(self.num_blocks, num_basis))
            self.W_B = None
        else:
            self.num_basis = None
            self.Z_basis = None
            self.omega = None
            self.W_B = nn.Parameter(torch.empty(self.num_blocks, block_size, block_size))

        # ----- 全局交互 W_G -----
        # Lite 版: low-rank (公式 18), W_G = A_G @ B_G
        # 完整版: (N_b, N_b) 直接学习
        self.global_rank = global_rank
        if global_rank is not None and 0 < global_rank < self.num_blocks:
            self.W_G_left = nn.Parameter(torch.empty(self.num_blocks, global_rank))
            self.W_G_right = nn.Parameter(torch.empty(global_rank, self.num_blocks))
            self.W_G = None
        else:
            self.W_G = nn.Parameter(torch.empty(self.num_blocks, self.num_blocks))
            self.W_G_left = None
            self.W_G_right = None
            self.global_rank = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.W_B is not None:
            nn.init.xavier_uniform_(self.W_B)
        else:
            # basis-composed 初始化
            nn.init.xavier_uniform_(self.Z_basis)
            # omega 初始化为均值 1/b、方差小的分布，让初始 W_B^i ≈ 各 basis 的平均
            nn.init.normal_(self.omega, mean=1.0 / self.num_basis, std=0.02)
        if self.W_G is not None:
            nn.init.xavier_uniform_(self.W_G)
        else:
            nn.init.xavier_uniform_(self.W_G_left)
            nn.init.xavier_uniform_(self.W_G_right)

    def set_tau(self, tau: float):
        """外部调用，用于温度退火"""
        self.tau.fill_(tau)

    def _get_global_logits(self) -> torch.Tensor:
        """返回 (N_b, N_b) 的全局交互 logits"""
        if self.W_G is not None:
            logits = self.W_G
        else:
            logits = self.W_G_left @ self.W_G_right  # (N_b, N_b)
        if self.symmetric:
            logits = (logits + logits.T) / 2
        return logits

    def _get_local_logits(self) -> torch.Tensor:
        """
        返回 (N_b, B, B) 的局部交互 logits。

        - 完整版: 直接返回学习好的 W_B
        - Lite 版: 由 basis Z 和权重 omega 动态组合:
            W_B^i[j,k] = Σ_ℓ omega[i,ℓ] * Z[ℓ,j,k]
        """
        if self.W_B is not None:
            logits = self.W_B
        else:
            # basis-composed: omega @ Z → (N_b, B, B)
            # einsum: (N_b, b) × (b, B, B) → (N_b, B, B)
            logits = torch.einsum("nb,bij->nij", self.omega, self.Z_basis)
        if self.symmetric:
            logits = (logits + logits.transpose(-1, -2)) / 2
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B_batch, T, D) → (B_batch, T, D)

        实现:
          flatten → 分 block → 局部 W_B^i → 全局 W_G → reshape
        """
        B_batch, T, D = x.shape
        B_s = self.block_size
        N_b = self.num_blocks
        tau = self.tau.item()

        # flatten: (B_batch, T*D) → (B_batch, N_b, B_s)
        x_flat = x.reshape(B_batch, N_b, B_s)

        # Step 1: 局部交互 — 每个 block 乘独立的 W_B^i
        # W_B_norm: (N_b, B_s, B_s), Sinkhorn 归一化
        local_logits = self._get_local_logits()
        W_B_norm = sinkhorn_knopp(local_logits, self.sinkhorn_iters, tau)
        # (B_batch, N_b, B_s) @ (N_b, B_s, B_s) → (B_batch, N_b, B_s)
        h = torch.einsum("bns,nsd->bnd", x_flat, W_B_norm)

        # Step 2: 全局交互 — block 间乘 W_G
        # W_G_norm: (N_b, N_b), Sinkhorn 归一化
        global_logits = self._get_global_logits()
        W_G_norm = sinkhorn_knopp(global_logits, self.sinkhorn_iters, tau)
        # (N_b, N_b) @ (B_batch, N_b, B_s) → (B_batch, N_b, B_s)
        out = torch.einsum("mn,bns->bms", W_G_norm, h)

        # reshape 回 (B_batch, T, D)
        return out.reshape(B_batch, T, D)


# =============================================================================
# Per-Token SwiGLU（einsum 实现）
# =============================================================================

class PerTokenSwiGLU(nn.Module):
    """
    Per-Token SwiGLU（论文公式 19）：
      pSwiGLU(o_i) = W_down^i @ (Swish(W_gate^i @ o_i) * (W_up^i @ o_i))

    每个 token 独立参数，einsum 实现。
    """

    def __init__(self, num_tokens: int, token_dim: int,
                 ffn_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, ffn_dim))
        self.w_up = nn.Parameter(torch.empty(num_tokens, token_dim, ffn_dim))
        self.b_up = nn.Parameter(torch.zeros(num_tokens, ffn_dim))
        self.w_down = nn.Parameter(torch.empty(num_tokens, ffn_dim, token_dim))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, token_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.w_gate, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_up, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_down, a=0, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        gate = torch.einsum("btd,tdf->btf", x, self.w_gate) + self.b_gate
        gate = F.silu(gate)
        up = torch.einsum("btd,tdf->btf", x, self.w_up) + self.b_up
        hidden = gate * up
        out = torch.einsum("btf,tfd->btd", hidden, self.w_down) + self.b_down
        return self.dropout(out)


# =============================================================================
# SiameseNorm（论文公式 20）
# =============================================================================

class SiameseNormLayer(nn.Module):
    """
    SiameseNorm 的单层组件（每层的 norm_y 和 norm_x）。

    论文公式 20:
      Ỹ_ℓ = RMSNorm(Ȳ_ℓ)
      O_ℓ = Block(X̄_ℓ + Ỹ_ℓ)
      X̄_{ℓ+1} = RMSNorm(X̄_ℓ + O_ℓ)
      Ȳ_{ℓ+1} = Ȳ_ℓ + O_ℓ
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm_y = nn.RMSNorm(dim)
        self.norm_x = nn.RMSNorm(dim)

    def pre_block(self, x_bar: torch.Tensor, y_bar: torch.Tensor):
        """block 前：计算 block 的输入 = X̄ + RMSNorm(Ȳ)"""
        return x_bar + self.norm_y(y_bar)

    def post_block(self, x_bar: torch.Tensor, y_bar: torch.Tensor,
                   block_out: torch.Tensor):
        """block 后：更新两条流"""
        x_bar_next = self.norm_x(x_bar + block_out)
        y_bar_next = y_bar + block_out
        return x_bar_next, y_bar_next


# =============================================================================
# UniMixer V2 Block（UniMixing + Per-Token SwiGLU）
# =============================================================================

class UniMixerV2Block(nn.Module):
    """
    Single UniMixer V2 block:
      O = UniMixing(input) + PerTokenSwiGLU(UniMixing(input))

    注意：残差连接由外部 SiameseNorm 管理，block 本身不做残差。
    """

    def __init__(self, num_tokens: int, token_dim: int,
                 block_size: int = 0, ffn_dim: int = 256,
                 global_rank: Optional[int] = None,
                 num_basis: Optional[int] = None,
                 sinkhorn_iters: int = 5,
                 tau_start: float = 1.0, tau_end: float = 0.1,
                 symmetric: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(token_dim)
        self.unimixing = UniMixing(
            num_tokens=num_tokens,
            token_dim=token_dim,
            block_size=block_size,
            global_rank=global_rank,
            num_basis=num_basis,
            sinkhorn_iters=sinkhorn_iters,
            tau_start=tau_start,
            tau_end=tau_end,
            symmetric=symmetric,
        )
        self.norm2 = nn.RMSNorm(token_dim)
        self.channel_mixing = PerTokenSwiGLU(
            num_tokens=num_tokens,
            token_dim=token_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        h = x + self.unimixing(self.norm1(x))
        out = h + self.channel_mixing(self.norm2(h))
        return out

    def set_tau(self, tau: float):
        self.unimixing.set_tau(tau)


# =============================================================================
# UniMixer V2 Model
# =============================================================================

def _find_valid_num_tokens(num_sparse: int, emb_dim: int) -> int:
    """找满足 D%T==0 的最大 T（T <= num_sparse）"""
    for t in range(min(num_sparse, emb_dim), 0, -1):
        if emb_dim % t == 0:
            return t
    return 1


@register_model
class UniMixerV2Model(RecModel):
    """
    UniMixer V2：严格遵循论文设计。

    核心特性：
      - UniMixing 两层交互（局部 W_B^i + 全局 W_G）
      - Sinkhorn-Knopp 双随机归一化 + 对称约束 + 温度退火
      - SiameseNorm 双流耦合残差
      - Per-Token SwiGLU (门控 channel mixing)
      - 不强制 H=T，用 block_size 控制交互粒度

    Config keys:
        embedding_dim:      D
        num_mixer_layers:   block 层数 M
        num_feature_fields: 手动指定 T；不指定则自动选择
        block_size:         UniMixing 的 block 大小 B（默认=D，即每个 token 一个 block）
        global_rank:        全局交互 W_G 的低秩维度（None=full-rank, Lite 版特性）
        num_basis:          局部交互 basis 数量 b（None/0=完整版 W_B^i, >0=Lite 版 basis-composed W_B）
        ffn_dim:            Per-Token SwiGLU 中间层维度
        sinkhorn_iters:     Sinkhorn-Knopp 迭代次数
        tau_start:          温度初始值
        tau_end:            温度最终值
        symmetric:          是否强制对称约束
        mlp_dims:           top MLP 层维度
        dropout:            dropout rate
        pool_output:        True=mean pooling, False=flatten

    版本切换:
        - 完整版 UniMixer: num_basis=None, global_rank=None
        - UniMixer-Lite:   num_basis=b (e.g. 4), global_rank=r (e.g. N_b/4)
        - 混合:            只启用其中一个也可以（论文 Table 4 也做了消融）
    """
    model_name = "unimixer_v2"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_mixer_layers", 2)
        block_size = mc.get("block_size", 0)  # 0 = 自动(=D)
        global_rank = mc.get("global_rank", None)
        num_basis = mc.get("num_basis", None)
        ffn_dim = mc.get("ffn_dim", 256)
        sinkhorn_iters = mc.get("sinkhorn_iters", 5)
        tau_start = mc.get("tau_start", 1.0)
        tau_end = mc.get("tau_end", 0.1)
        symmetric = mc.get("symmetric", True)
        mlp_dims = mc.get("mlp_dims", [128, 64])
        dropout = mc.get("dropout", 0.0)
        pool_output = mc.get("pool_output", False)

        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse

        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)

        # 确定 T（不再要求 D%T==0，因为 UniMixer 没有 H=T 约束）
        num_feature_fields = mc.get("num_feature_fields", None)
        if num_feature_fields is not None:
            T = num_feature_fields
        else:
            # 默认仍用 _find_valid_num_tokens，保证 T*D 能被 block_size 整除
            T = _find_valid_num_tokens(num_sparse, emb_dim)

        if T < num_sparse:
            self.feature_field_pool = FeatureFieldPooling(
                num_features=num_sparse, num_fields=T,
                seed=config.get("seed", 42),
            )
        else:
            self.feature_field_pool = None

        # 确保 T*D 能被 block_size 整除
        effective_bs = block_size if block_size > 0 else emb_dim
        L = T * emb_dim
        assert L % effective_bs == 0, (
            f"L = T*D = {T}*{emb_dim} = {L} must be divisible by block_size={effective_bs}"
        )
        N_b = L // effective_bs

        self.num_tokens = T
        self.emb_dim = emb_dim
        self.pool_output = pool_output
        self.tau_start = tau_start
        self.tau_end = tau_end

        rank_str = f"global_rank={global_rank}" if global_rank else "full-rank"
        basis_str = f"num_basis={num_basis}" if num_basis else "per-block-WB"
        variant = "UniMixer-Lite" if (num_basis and num_basis > 0) else "UniMixer-Full"
        print(f"[UniMixerV2/{variant}] T={T}, D={emb_dim}, block_size={effective_bs}, "
              f"N_b={N_b}, L={num_layers}, ffn={ffn_dim}, {rank_str}, {basis_str}, "
              f"tau={tau_start}->{tau_end}, sinkhorn={sinkhorn_iters} "
              f"(field_pool: {num_sparse}->{T})")

        self.mixer_blocks = nn.ModuleList([
            UniMixerV2Block(
                num_tokens=T, token_dim=emb_dim, block_size=block_size,
                ffn_dim=ffn_dim, global_rank=global_rank,
                num_basis=num_basis,
                sinkhorn_iters=sinkhorn_iters,
                tau_start=tau_start, tau_end=tau_end,
                symmetric=symmetric, dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # SiameseNorm: 每层一个 layer norm，最终融合用一个额外的 norm
        self.siamese_norms = nn.ModuleList([
            SiameseNormLayer(emb_dim) for _ in range(num_layers)
        ])
        self.final_fuse_norm = nn.RMSNorm(emb_dim)

        top_layers = []
        in_dim = emb_dim if pool_output else T * emb_dim
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def set_tau_for_step(self, current_step: int, total_steps: int):
        """温度退火：外部 trainer 每步调用"""
        tau = max(
            self.tau_start - (self.tau_start - self.tau_end) * current_step / max(total_steps, 1),
            self.tau_end,
        )
        for block in self.mixer_blocks:
            block.set_tau(tau)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.sparse_arch.forward_per_feature(batch["sparse"])  # (B, S, D)
        if self.feature_field_pool is not None:
            x = self.feature_field_pool(x)  # (B, T, D)

        # SiameseNorm 双流
        x_bar = x
        y_bar = x.clone()

        for block, sn in zip(self.mixer_blocks, self.siamese_norms):
            block_input = sn.pre_block(x_bar, y_bar)
            block_out = block(block_input)
            # block 内部已有残差（h = x + mixing; out = h + ffn），
            # SiameseNorm 用的是 block 的增量 = block_out - block_input
            delta = block_out - block_input
            x_bar, y_bar = sn.post_block(x_bar, y_bar, delta)

        # 最终融合: X_output = X̄_M + RMSNorm(Ȳ_M)
        x = x_bar + self.final_fuse_norm(y_bar)

        if self.pool_output:
            x = x.mean(dim=1)
        else:
            x = x.flatten(start_dim=1)
        return self.top_mlp(x).squeeze(-1)
