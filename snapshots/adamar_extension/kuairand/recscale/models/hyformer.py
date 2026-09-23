"""
recscale.models.hyformer — HyFormer (Hybrid Transformer)

"Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction"
Paper: https://arxiv.org/abs/2601.12681

核心思想: 用 Query Decoding + Query Boosting 迭代交替，统一序列建模与特征交互。
- Query Decoding: 用 non-seq features 生成的 global tokens 对 seq 做 cross-attention
- Query Boosting: MLP-Mixer style token mixing 增强 decoded queries

Architecture:
  seq → item_emb → K/V representations (B, L_seq, D)
  non-seq features → Query Generation → Global Tokens (B, N_q, D)
      ↓
  Stack of HyFormerBlock × N
  ├── Query Decoding: CrossAttn(Q=queries, K=seq_kv, V=seq_kv)
  └── Query Boosting: token mixing (split heads → concat across tokens → FFN)
  + Residual + LayerNorm
      ↓
  pool queries → MLP → logit
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, DenseArch, SparseArch


class QueryGeneration(nn.Module):
    """
    Generate global query tokens from non-sequential features.

    Non-seq features (sparse emb + optional dense) → concat → MLP → N_q tokens of dim D
    """

    def __init__(self, input_dim: int, num_queries: int, hidden_dim: int):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * num_queries),
            nn.GELU(),
            nn.Linear(hidden_dim * num_queries, hidden_dim * num_queries),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, input_dim) — flattened non-seq features
        Returns: (B, N_q, D) — query tokens
        """
        out = self.proj(features)  # (B, N_q * D)
        return out.view(-1, self.num_queries, self.hidden_dim)


class QueryDecoding(nn.Module):
    """
    Cross-attention: Query tokens attend to sequence K/V.

    Q = global tokens (from non-seq features)
    K, V = sequence representations (from item embeddings)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
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

    def forward(self, queries: torch.Tensor, kv: torch.Tensor,
                kv_mask: torch.Tensor = None) -> torch.Tensor:
        """
        queries: (B, N_q, D) — global query tokens
        kv: (B, L_seq, D) — sequence representations
        kv_mask: (B, L_seq) — True for valid positions
        Returns: (B, N_q, D)
        """
        B, N_q, D = queries.shape
        L = kv.size(1)
        H = self.num_heads

        q = self.q_proj(queries).view(B, N_q, H, self.head_dim).transpose(1, 2)  # (B,H,N_q,hd)
        k = self.k_proj(kv).view(B, L, H, self.head_dim).transpose(1, 2)  # (B,H,L,hd)
        v = self.v_proj(kv).view(B, L, H, self.head_dim).transpose(1, 2)  # (B,H,L,hd)

        # Attention: (B,H,N_q,L)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if kv_mask is not None:
            # kv_mask: (B, L) → (B, 1, 1, L)
            attn = attn.masked_fill(~kv_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-masked
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,N_q,hd)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)


class QueryBoosting(nn.Module):
    """
    MLP-Mixer style token mixing on query tokens.

    1. Split each token into H subspaces
    2. For each subspace h: concat all tokens' h-th part → FFN
    3. Residual connection

    This enables cross-query interactions without attention.
    """

    def __init__(self, num_tokens: int, hidden_dim: int, num_heads: int = 4,
                 ffn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Token mixing FFN (shared across heads)
        mixing_input_dim = num_tokens * self.head_dim
        self.mixing_ffn = nn.Sequential(
            nn.Linear(mixing_input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, mixing_input_dim),
            nn.Dropout(dropout),
        )

        # Per-token FFN (channel mixing)
        self.channel_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_q, D)
        Returns: (B, N_q, D)
        """
        B, T, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        # Token mixing
        x_heads = x.view(B, T, H, hd)  # (B, T, H, hd)
        x_heads = x_heads.permute(0, 2, 1, 3)  # (B, H, T, hd)
        x_concat = x_heads.reshape(B * H, T * hd)
        x_mixed = self.mixing_ffn(x_concat)  # (B*H, T*hd)
        x_mixed = x_mixed.view(B, H, T, hd).permute(0, 2, 1, 3).reshape(B, T, D)

        x = x + x_mixed  # residual

        # Channel mixing
        x = x + self.channel_ffn(x)

        return x


class HyFormerBlock(nn.Module):
    """
    Single HyFormer block:
    1. Query Decoding: cross-attention to sequence
    2. Query Boosting: token mixing among queries + NS tokens

    Dual-seq 支持：当 has_seq2=True 时增加一个独立的 cross_attn_seq2，forward 里
    对 seq2_kv 追加一次 cross-attention（与 seq_kv 的 cross-attention 串联，
    queries 依次吸收两路序列信息）。single-seq 时 cross_attn_seq2 不创建，
    forward 走原路径（bit-identical）。
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_queries: int,
                 ffn_dim: int = 128, dropout: float = 0.0,
                 has_seq2: bool = False):
        super().__init__()
        self.has_seq2 = has_seq2
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = QueryDecoding(hidden_dim, num_heads, dropout)
        if has_seq2:
            self.cross_attn_seq2 = QueryDecoding(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.boosting = QueryBoosting(num_queries, hidden_dim, num_heads, ffn_dim, dropout)

    def forward(self, queries: torch.Tensor, seq_kv: torch.Tensor,
                seq_mask: torch.Tensor = None,
                seq2_kv: torch.Tensor = None,
                seq2_mask: torch.Tensor = None) -> torch.Tensor:
        """
        queries: (B, N_q, D)
        seq_kv: (B, L_seq, D)
        seq_mask: (B, L_seq)
        seq2_kv/seq2_mask: optional, 仅在 has_seq2=True 时使用
        Returns: (B, N_q, D) — refined queries
        """
        # Query Decoding
        q_normed = self.norm1(queries)
        decoded = self.cross_attn(q_normed, seq_kv, seq_mask)
        queries = queries + decoded

        # Dual-seq: 第二路 cross-attention（独立参数，共享 norm1）
        if self.has_seq2 and seq2_kv is not None:
            q_normed2 = self.norm1(queries)
            decoded2 = self.cross_attn_seq2(q_normed2, seq2_kv, seq2_mask)
            queries = queries + decoded2

        # Query Boosting
        queries = queries + self.boosting(self.norm2(queries))

        return queries


class TargetAttention(nn.Module):
    """
    DIN-style target attention shortcut.
    score = MLP(concat(query, key, query-key, query*key))
    output = weighted_sum(values, softmax(scores))
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        query: (B, E) — target embedding
        keys: (B, L, E) — sequence embeddings
        mask: (B, L) — True for valid
        Returns: (B, E)
        """
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
class HyFormerModel(RecModel):
    """
    HyFormer: Hybrid Transformer for CTR Prediction.

    Unifies sequence modeling (via cross-attention) and feature interaction
    (via MLP-Mixer token mixing) in an iterative architecture.

    v2: Added DIN-style target attention shortcut for direct target-sequence matching.

    Supports batch keys: "sparse", "dense" (opt), "seq", "target" (opt), "label"
    """
    model_name = "hyformer"

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

        num_dense = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        # Dual-sequence support: 和 HyFormer V2 对齐，dc["num_items2"] > 0 触发。
        # has_seq2=False 时不创建任何 seq2 参数，state_dict 与改动前一致（bit-identical）。
        self.has_seq2 = dc.get("num_items2", 0) > 0
        num_items2 = dc.get("num_items2", 0)

        # Sparse embedding
        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None
        self.has_dense = num_dense > 0

        # Feature input dim for query generation (加上 target embedding dim)
        feat_dim = num_sparse * emb_dim + (num_dense if self.has_dense else 0) + hidden_dim
        if self.has_seq2:
            feat_dim += hidden_dim   # target2_emb 追加

        # Query Generation: non-seq features → global tokens
        self.query_gen = QueryGeneration(feat_dim, num_queries, hidden_dim)
        self.num_queries = num_queries

        # Sequence item embedding
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.item_emb.weight.data[0].zero_()
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

        if self.has_seq2:
            self.item_emb2 = nn.Embedding(num_items2 + 1, hidden_dim, padding_idx=0)
            nn.init.normal_(self.item_emb2.weight, std=0.02)
            self.item_emb2.weight.data[0].zero_()
            self.target_attn2 = TargetAttention(hidden_dim, hidden_dim)

        # HyFormer blocks
        self.blocks = nn.ModuleList([
            HyFormerBlock(hidden_dim, num_heads, num_queries, ffn_dim, dropout,
                          has_seq2=self.has_seq2)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # DIN-style target attention shortcut (direct target-sequence matching)
        self.target_attn = TargetAttention(hidden_dim, hidden_dim)

        # Top MLP: query output + target attention output + target embedding
        top_layers = []
        in_dim = num_queries * hidden_dim + hidden_dim + hidden_dim  # queries + interest + target
        if self.has_seq2:
            in_dim += hidden_dim + hidden_dim  # interest2 + target2_emb
        for dim in mlp_dims:
            top_layers.append(nn.Linear(in_dim, dim))
            top_layers.append(nn.ReLU())
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_dim = dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        # Non-sequential features → flat vector
        feat_parts = []
        if self.sparse_arch is not None and "sparse" in batch:
            sparse_flat = self.sparse_arch(batch["sparse"])  # (B, S*E)
            feat_parts.append(sparse_flat)
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            feat_parts.append(batch["dense"])

        # Target item embedding (shared with sequence for target-aware queries)
        target = batch.get("target")
        if target is not None:
            target_emb = self.item_emb(target)  # (B, hidden)
            feat_parts.append(target_emb)
        else:
            # infer B from seq or sparse
            if len(feat_parts) > 0:
                B = feat_parts[0].size(0)
            else:
                B = batch.get("seq").size(0)
            target_emb = torch.zeros(B, self.item_emb.embedding_dim,
                                     device=next(self.parameters()).device)
            feat_parts.append(target_emb)

        # Dual-seq: target2_emb 也进 feat_vec (顺序对齐 __init__ 里的 feat_dim 计算)
        target2_emb = None
        if self.has_seq2:
            target2 = batch.get("target2")
            if target2 is not None:
                target2_emb = self.item_emb2(target2)
            else:
                B = feat_parts[0].size(0)
                target2_emb = torch.zeros(B, self.item_emb2.embedding_dim,
                                          device=next(self.parameters()).device)
            feat_parts.append(target2_emb)

        if len(feat_parts) > 1:
            feat_vec = torch.cat(feat_parts, dim=1)  # (B, feat_dim)
        else:
            feat_vec = feat_parts[0]

        # Generate query tokens
        queries = self.query_gen(feat_vec)  # (B, N_q, hidden)

        # Sequence K/V
        seq = batch.get("seq")
        if seq is not None and seq.sum() > 0:
            seq_kv = self.item_emb(seq)  # (B, L, hidden)
            if self.use_seq_action_type_embedding and "seq_action" in batch:
                action_emb = self.seq_action_emb(batch["seq_action"].clamp_min(0))
                if self.seq_action_fusion == "concat_mlp":
                    seq_kv = self.seq_action_fuse(torch.cat([seq_kv, action_emb], dim=-1))
                else:
                    seq_kv = seq_kv + action_emb
            seq_mask = (seq != 0)  # (B, L)
        else:
            # No sequence: create dummy single token
            B = queries.size(0)
            seq_kv = torch.zeros(B, 1, queries.size(2), device=queries.device)
            seq_mask = torch.zeros(B, 1, dtype=torch.bool, device=queries.device)

        # Second sequence K/V (dual-seq)
        seq2_kv = None
        seq2_mask = None
        interest2 = None
        if self.has_seq2:
            seq2 = batch.get("seq2")
            if seq2 is not None and seq2.sum() > 0:
                seq2_kv = self.item_emb2(seq2)
                seq2_mask = (seq2 != 0)
            else:
                B = queries.size(0)
                seq2_kv = torch.zeros(B, 1, queries.size(2), device=queries.device)
                seq2_mask = torch.zeros(B, 1, dtype=torch.bool, device=queries.device)
            interest2 = self.target_attn2(target2_emb, seq2_kv, seq2_mask)

        # DIN-style target attention shortcut (direct target-seq matching)
        interest = self.target_attn(target_emb, seq_kv, seq_mask)  # (B, hidden)

        # Iterative refinement through HyFormer blocks
        for block in self.blocks:
            queries = block(queries, seq_kv, seq_mask,
                            seq2_kv=seq2_kv, seq2_mask=seq2_mask)

        queries = self.final_norm(queries)

        # Pool and predict: queries + interest shortcut + target (+ interest2 + target2 if dual-seq)
        q_flat = queries.flatten(start_dim=1)  # (B, N_q * hidden)
        final_parts = [q_flat, interest, target_emb]
        if self.has_seq2:
            final_parts.append(interest2)
            final_parts.append(target2_emb)
        final = torch.cat(final_parts, dim=1)
        return self.top_mlp(final).squeeze(-1)
