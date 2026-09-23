"""
recscale.models.onetrans — OneTrans (One Transformer for All)

"OneTrans: One Transformer for All" - ByteDance 2026
Paper: https://arxiv.org/abs/2510.26104

核心思想: 将所有特征（sparse, dense, sequence）统一 tokenize 为 flat token 序列,
用单个 Transformer backbone 处理。关键创新是 Mixed Parameterization:
- S-tokens (序列行为): 共享 Q/K/V + FFN 参数 (同质, 减少冗余)
- NS-tokens (非序列特征): 每个 token 独立 Q/K/V + FFN 参数 (异质, 保持表达力)

Architecture:
  sparse/dense → NS-tokens (B, L_NS, D)
  seq → item_emb → S-tokens (B, L_S, D)
  concat → [S-tokens || NS-tokens]
      ↓
  Stack of MixedTransformerBlock × N
  ├── RMSNorm + MixedMHA (S: shared, NS: per-token)
  └── RMSNorm + MixedFFN
  + Residual
      ↓
  pool NS-tokens → MLP → logit

Pyramid Token Pruning (可选, 默认关闭):
  每层按 pyramid_prune_ratio 逐步裁减 S-tokens，减少长序列的计算量。
  默认关闭 (pyramid_prune_ratio=0) 保持效果最优；
  长序列场景（L_S > 256）可开启以换取推理速度。
  配置项: model.pyramid_prune_ratio (0=关闭, 0.5=每层裁一半)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, DenseArch, SparseArch


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLMs)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MixedMultiHeadAttention(nn.Module):
    """
    Mixed Parameterization Multi-Head Attention.

    S-tokens 共享一套 Q/K/V 投影参数;
    NS-tokens 每个 token 有独立的 Q/K/V 投影参数。

    注意: 这里简化实现 — NS-tokens 共享一套参数但加上 per-token bias,
    因为完全独立的参数在 token 数量大时内存爆炸。
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_ns_tokens: int,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_ns_tokens = num_ns_tokens

        # S-tokens: shared Q/K/V projections
        self.s_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)

        # NS-tokens: per-token Q/K/V (shared base + per-token bias)
        self.ns_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        if num_ns_tokens > 0:
            self.ns_bias = nn.Parameter(
                torch.zeros(num_ns_tokens, 3 * hidden_dim)
            )
            nn.init.normal_(self.ns_bias, std=0.02)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, num_s_tokens: int) -> torch.Tensor:
        """
        x: (B, L, D) where L = L_S + L_NS
        num_s_tokens: how many tokens at the beginning are S-tokens
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        L_S = num_s_tokens
        L_NS = L - L_S

        # Split S and NS tokens
        s_tokens = x[:, :L_S]   # (B, L_S, D)
        ns_tokens = x[:, L_S:]  # (B, L_NS, D)

        # Compute Q/K/V for S-tokens
        s_qkv = self.s_qkv(s_tokens)  # (B, L_S, 3D)

        # Compute Q/K/V for NS-tokens with per-token bias
        ns_qkv = self.ns_qkv(ns_tokens)  # (B, L_NS, 3D)
        if L_NS > 0 and self.num_ns_tokens > 0:
            bias = self.ns_bias[:L_NS]  # (L_NS, 3D)
            ns_qkv = ns_qkv + bias.unsqueeze(0)

        # Concatenate back
        qkv = torch.cat([s_qkv, ns_qkv], dim=1)  # (B, L, 3D)

        # Reshape to multi-head
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)

        return self.out_proj(out)


class MixedFFN(nn.Module):
    """
    Mixed Parameterization FFN.
    S-tokens 共享 FFN, NS-tokens 用独立 bias。
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_ns_tokens: int,
                 dropout: float = 0.0):
        super().__init__()
        # Shared FFN
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # NS-token specific bias
        if num_ns_tokens > 0:
            self.ns_bias1 = nn.Parameter(torch.zeros(num_ns_tokens, ffn_dim))
            self.ns_bias2 = nn.Parameter(torch.zeros(num_ns_tokens, hidden_dim))
            nn.init.normal_(self.ns_bias1, std=0.02)
            nn.init.normal_(self.ns_bias2, std=0.02)
        self.num_ns_tokens = num_ns_tokens

    def forward(self, x: torch.Tensor, num_s_tokens: int) -> torch.Tensor:
        B, L, D = x.shape
        L_S = num_s_tokens
        L_NS = L - L_S

        h = F.gelu(self.fc1(x))  # (B, L, ffn_dim)

        # Add per-token bias for NS tokens — out-of-place to avoid autograd issues
        if L_NS > 0 and self.num_ns_tokens > 0:
            bias1 = self.ns_bias1[:L_NS].unsqueeze(0)  # (1, L_NS, ffn_dim)
            h = torch.cat([h[:, :L_S], h[:, L_S:] + bias1], dim=1)

        h = self.dropout(h)
        out = self.fc2(h)  # (B, L, hidden_dim)

        if L_NS > 0 and self.num_ns_tokens > 0:
            bias2 = self.ns_bias2[:L_NS].unsqueeze(0)  # (1, L_NS, hidden_dim)
            out = torch.cat([out[:, :L_S], out[:, L_S:] + bias2], dim=1)

        return self.dropout(out)


class MixedTransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with Mixed Parameterization.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 num_ns_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = MixedMultiHeadAttention(hidden_dim, num_heads, num_ns_tokens, dropout)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = MixedFFN(hidden_dim, ffn_dim, num_ns_tokens, dropout)

    def forward(self, x: torch.Tensor, num_s_tokens: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), num_s_tokens)
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

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
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
class OneTransModel(RecModel):
    """
    OneTrans: One Transformer for All features.

    v2: Added DIN-style target attention shortcut for stronger target-sequence matching.

    Supports batch keys: "sparse", "dense" (opt), "seq" (opt), "target" (opt), "label"
    """
    model_name = "onetrans"

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

        # Pyramid Token Pruning
        # 0 = 关闭（默认，效果优先）；0.5 = 每层裁一半（效率优先，长序列推荐）
        self.pyramid_prune_ratio = mc.get("pyramid_prune_ratio", 0)

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        # NS-token count: sparse features + optional dense token + target token
        self.num_ns_tokens = num_sparse + (1 if num_dense > 0 else 0) + 1  # +1 for target
        self.has_dense = num_dense > 0
        self.has_seq = "seq" in (dc.get("batch_keys") or []) or dc.get("num_items", 0) > 0

        # Sparse embedding → per-feature tokens, project to hidden_dim
        self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        self.sparse_proj = nn.Linear(emb_dim, hidden_dim) if emb_dim != hidden_dim else nn.Identity()

        # Dense → 1 NS-token
        if self.has_dense:
            self.dense_proj = nn.Sequential(
                nn.Linear(num_dense, hidden_dim),
                nn.ReLU(),
            )

        # Sequence item embedding → S-tokens
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.item_emb.weight.data[0].zero_()

        # DIN-style target attention shortcut
        self.target_attn = TargetAttentionOT(hidden_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MixedTransformerBlock(hidden_dim, num_heads, ffn_dim,
                                 self.num_ns_tokens, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)

        # 预计算每层 S-tokens 数量（pyramid 开启时使用）
        # 注意: 实际序列长度在 forward 时确定，这里只存 prune_ratio 供 forward 使用
        self.num_layers = num_layers

        # Top MLP: NS-tokens + interest shortcut + target embedding
        top_layers = []
        in_dim = self.num_ns_tokens * hidden_dim + hidden_dim + hidden_dim  # ns + interest + target
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        # NS-tokens: sparse per-feature
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])  # (B, S, emb)
        ns_tokens = [self.sparse_proj(sparse_per)]

        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            ns_tokens.append(self.dense_proj(batch["dense"]).unsqueeze(1))

        # Target token: use item_emb (shared with sequence) for target-aware attention
        target = batch.get("target")
        if target is not None:
            target_emb = self.item_emb(target)  # (B, hidden)
            target_token = target_emb.unsqueeze(1)  # (B, 1, hidden)
            ns_tokens.append(target_token)
        else:
            # No target: use zero token as placeholder
            B = sparse_per.size(0)
            target_emb = torch.zeros(B, self.item_emb.embedding_dim,
                                     device=sparse_per.device)
            ns_tokens.append(target_emb.unsqueeze(1))

        ns = torch.cat(ns_tokens, dim=1)  # (B, L_NS, hidden)
        L_NS = ns.size(1)

        # S-tokens: sequence items
        seq = batch.get("seq")
        if seq is not None and seq.sum() > 0:
            s_tokens = self.item_emb(seq)  # (B, L_seq, hidden)
            seq_mask = (seq != 0)  # (B, L_seq)
            s_tokens = s_tokens * seq_mask.unsqueeze(-1).float()
            L_S = s_tokens.size(1)
            # Concat: [S-tokens, NS-tokens]
            x = torch.cat([s_tokens, ns], dim=1)  # (B, L_S + L_NS, hidden)

            # DIN-style target attention shortcut
            interest = self.target_attn(target_emb, s_tokens, seq_mask)  # (B, hidden)
        else:
            L_S = 0
            x = ns
            interest = torch.zeros_like(target_emb)

        # Transformer blocks（支持 Pyramid Token Pruning）
        current_L_S = L_S
        for block in self.blocks:
            x = block(x, num_s_tokens=current_L_S)

            # Pyramid: 按比例裁减 S-tokens（NS-tokens 始终保留）
            if self.pyramid_prune_ratio > 0 and current_L_S > 1:
                target_L_S = max(1, int(current_L_S * (1 - self.pyramid_prune_ratio)))
                if target_L_S < current_L_S:
                    # 右对齐序列: padding 在左，最新行为在右端（index 最大）
                    # → 保留最后 target_L_S 个 S-tokens = 保留最近的行为
                    keep_start = current_L_S - target_L_S
                    x_s = x[:, keep_start:current_L_S]  # (B, target_L_S, D)
                    x_ns = x[:, current_L_S:]            # (B, L_NS, D)
                    x = torch.cat([x_s, x_ns], dim=1)
                    current_L_S = target_L_S

        x = self.final_norm(x)

        # Pool NS-tokens only for prediction
        ns_out = x[:, current_L_S:]  # (B, L_NS, hidden)
        ns_flat = ns_out.flatten(start_dim=1)  # (B, L_NS * hidden)

        # Concat with direct shortcuts
        final = torch.cat([ns_flat, interest, target_emb], dim=1)
        return self.top_mlp(final).squeeze(-1)

