"""
recscale.models.hybrid — 混合模型: Sparse 交互 + DIN 序列 Attention

Wukong+DIN: Wukong FMB+LCB 做 sparse 高阶交互 + DIN target attention 做序列建模
DCN+DIN:    DCN Cross Network 做 sparse 交叉 + DIN target attention 做序列建模

核心思想: Wukong/DCN 和 DIN 不冲突。
- Wukong/DCN: 负责 sparse 特征间的高阶交互建模 (user 画像 × ad 属性)
- DIN: 负责行为序列的 target-aware attention (用候选 item 查询历史行为)
- 两者输出 concat → MLP → logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, DenseArch, SparseArch, CrossNetwork
from .din import DINAttention


# ============================================================
# Wukong components (inlined to avoid circular imports)
# ============================================================

class _FMB(nn.Module):
    """Factorization Machine Block (from Wukong)"""
    def __init__(self, input_features, output_features, embedding_dim,
                 rank_k=8, mlp_dims=None, dropout=0.0):
        super().__init__()
        self.output_features = output_features
        self.embedding_dim = embedding_dim
        if rank_k > 0 and rank_k < input_features:
            self.proj_Y = nn.Parameter(torch.randn(input_features, rank_k) * 0.01)
            flat_dim = input_features * rank_k
        else:
            self.proj_Y = None
            flat_dim = input_features * input_features
        self.layer_norm = nn.LayerNorm(flat_dim)
        dims = mlp_dims or [flat_dim // 2]
        layers = []
        in_d = flat_dim
        for d in dims:
            layers += [nn.Linear(in_d, d), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = d
        layers.append(nn.Linear(in_d, output_features * embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.proj_Y is not None:
            projected = torch.matmul(x.transpose(1, 2), self.proj_Y)
            fm_matrix = torch.bmm(x, projected)
        else:
            fm_matrix = torch.bmm(x, x.transpose(1, 2))
        flat = fm_matrix.flatten(start_dim=1)
        normed = self.layer_norm(flat)
        out = self.mlp(normed)
        return out.view(-1, self.output_features, self.embedding_dim)


class _LCB(nn.Module):
    """Linear Compression Block (from Wukong)"""
    def __init__(self, input_features, output_features, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


class _WukongLayer(nn.Module):
    """Single Wukong interaction layer"""
    def __init__(self, input_features, lcb_features, fmb_features,
                 embedding_dim, fmb_rank_k=8, fmb_mlp_dims=None, dropout=0.0):
        super().__init__()
        self.output_features = lcb_features + fmb_features
        self.fmb = _FMB(input_features, fmb_features, embedding_dim,
                        fmb_rank_k, fmb_mlp_dims, dropout)
        self.lcb = _LCB(input_features, lcb_features, embedding_dim)
        if input_features != self.output_features:
            self.residual_proj = nn.Linear(input_features, self.output_features)
        else:
            self.residual_proj = None
        self.layer_norm = nn.LayerNorm([self.output_features, embedding_dim])

    def forward(self, x):
        fmb_out = self.fmb(x)
        lcb_out = self.lcb(x)
        concat_out = torch.cat([fmb_out, lcb_out], dim=1)
        if self.residual_proj is not None:
            res = self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
        else:
            res = x
        return self.layer_norm(concat_out + res)


# ============================================================
# DCN + DIN
# ============================================================

@register_model
class DCNDINModel(RecModel):
    """
    DCN + DIN: Cross Network 做 sparse 交互 + DIN target attention 做序列建模

    Architecture:
      sparse → SparseArch → concat → [CrossNetwork || DeepMLP] → sparse_repr
      seq → item_emb → DIN_attention(target) → interest_repr
      concat(sparse_repr, interest, target_emb, [interest2, target2]) → MLP → logit
    """
    model_name = "dcn_din"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_cross_layers = mc.get("num_cross_layers", mc.get("cross_layers", 3))
        mlp_dims = mc.get("mlp_dims", [256, 128, 64])
        dropout = mc.get("dropout", 0.0)
        attn_hidden = mc.get("attention_dim", 64)

        num_dense = len(dc.get("dense_cols") or [])
        num_sparse = len(dc.get("sparse_cols") or [])
        cardinalities = dc.get("cardinalities", [10000] * num_sparse)
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        # Dual-sequence support
        self.dual_seq = dc.get("num_items2", 0) > 0
        num_items2 = dc.get("num_items2", num_items)

        # === Sparse path: DCN ===
        dense_out_dim = emb_dim if num_dense > 0 else 0
        self.dense_arch = DenseArch(num_dense, [dense_out_dim] if num_dense > 0 else [], dropout)
        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None

        sparse_dim = self.dense_arch.output_dim
        if self.sparse_arch is not None:
            sparse_dim += self.sparse_arch.output_dim

        # Cross Network path
        self.cross_net = CrossNetwork(sparse_dim, num_cross_layers)

        # Deep path
        deep_layers = []
        in_d = sparse_dim
        deep_dims = mc.get("deep_dims", [256, 128])
        for out_d in deep_dims:
            deep_layers += [nn.Linear(in_d, out_d), nn.ReLU()]
            if dropout > 0:
                deep_layers.append(nn.Dropout(dropout))
            in_d = out_d
        self.deep_net = nn.Sequential(*deep_layers)
        dcn_raw_dim = sparse_dim + deep_dims[-1]  # cross + deep concat

        # feat_repr_dim: compress DCN output to fixed dim before concat with seq
        # 0 = no compression (use raw DCN output)
        feat_repr_dim = mc.get("feat_repr_dim", 0)
        if feat_repr_dim > 0:
            self.feat_proj = nn.Linear(dcn_raw_dim, feat_repr_dim)
            dcn_output_dim = feat_repr_dim
        else:
            self.feat_proj = None
            dcn_output_dim = dcn_raw_dim

        # === Sequence path: DIN ===
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()
        self.attention = DINAttention(emb_dim, attn_hidden)

        if self.dual_seq:
            self.item_emb2 = nn.Embedding(num_items2 + 1, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(self.item_emb2.weight)
            self.item_emb2.weight.data[0].zero_()
            self.attention2 = DINAttention(emb_dim, attn_hidden)

        # Multimodal
        mm_dim = dc.get("mm_dim", 0)
        self.has_mm = mm_dim > 0
        if self.has_mm:
            self.mm_proj = nn.Linear(mm_dim, emb_dim)

        # === Top MLP ===
        top_input = dcn_output_dim + emb_dim * 2  # interest + target
        if self.dual_seq:
            top_input += emb_dim * 2  # interest2 + target2
        if self.has_mm:
            top_input += emb_dim

        top_layers = []
        in_d = top_input
        for out_d in mlp_dims:
            top_layers += [nn.Linear(in_d, out_d), nn.ReLU()]
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_d = out_d
        top_layers.append(nn.Linear(in_d, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        # === Sparse: DCN ===
        sparse_parts = []
        if "dense" in batch and batch["dense"] is not None:
            sparse_parts.append(self.dense_arch(batch["dense"]))
        if self.sparse_arch is not None and "sparse" in batch:
            sparse_parts.append(self.sparse_arch(batch["sparse"]))

        x_sparse = torch.cat(sparse_parts, dim=1) if len(sparse_parts) > 1 else sparse_parts[0]
        cross_out = self.cross_net(x_sparse)
        deep_out = self.deep_net(x_sparse)
        dcn_repr = torch.cat([cross_out, deep_out], dim=1)
        if self.feat_proj is not None:
            dcn_repr = F.relu(self.feat_proj(dcn_repr))

        # === Sequence: DIN ===
        parts = [dcn_repr]

        seq = batch.get("seq")
        target = batch.get("target")
        if seq is not None and target is not None:
            seq_emb = self.item_emb(seq)
            target_emb = self.item_emb(target)
            interest = self.attention(target_emb, seq_emb, seq != 0)
            parts.extend([interest, target_emb])

        if self.dual_seq:
            seq2 = batch.get("seq2")
            target2 = batch.get("target2")
            if seq2 is not None and target2 is not None:
                seq_emb2 = self.item_emb2(seq2)
                target_emb2 = self.item_emb2(target2)
                interest2 = self.attention2(target_emb2, seq_emb2, seq2 != 0)
                parts.extend([interest2, target_emb2])

        if self.has_mm and "mm_emb" in batch and batch["mm_emb"] is not None:
            parts.append(self.mm_proj(batch["mm_emb"]))

        x = torch.cat(parts, dim=1)
        return self.top_mlp(x).squeeze(-1)


# ============================================================
# Wukong + DIN
# ============================================================

@register_model
class WukongDINModel(RecModel):
    """
    Wukong + DIN: FMB+LCB 做 sparse 高阶交互 + DIN target attention 做序列建模

    Architecture:
      sparse → per_feature_emb → [WukongLayer × L] → flatten → sparse_repr
      seq → item_emb → DIN_attention(target) → interest_repr
      concat(sparse_repr, interest, target_emb, [interest2, target2]) → MLP → logit
    """
    model_name = "wukong_din"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        num_layers = mc.get("num_wukong_layers", 3)
        lcb_features = mc.get("lcb_features", 20)
        fmb_features = mc.get("fmb_features", 20)
        fmb_rank_k = mc.get("fmb_rank_k", 8)
        fmb_mlp_dims = mc.get("fmb_mlp_dims", [32, 32])
        mlp_dims = mc.get("mlp_dims", [256, 128, 64])
        dropout = mc.get("dropout", 0.0)
        attn_hidden = mc.get("attention_dim", 64)

        num_dense = len(dc.get("dense_cols") or [])
        num_sparse = len(dc.get("sparse_cols") or [])
        cardinalities = dc.get("cardinalities", [10000] * num_sparse)
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        self.dual_seq = dc.get("num_items2", 0) > 0
        num_items2 = dc.get("num_items2", num_items)

        # === Sparse path: Wukong ===
        self.has_dense = num_dense > 0
        if self.has_dense:
            self.dense_proj = nn.Linear(num_dense, emb_dim)

        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None

        num_tokens = num_sparse + (1 if self.has_dense else 0)

        # Wukong layers
        self.wukong_layers = nn.ModuleList()
        in_features = num_tokens
        for _ in range(num_layers):
            layer = _WukongLayer(in_features, lcb_features, fmb_features,
                                 emb_dim, fmb_rank_k, fmb_mlp_dims, dropout)
            self.wukong_layers.append(layer)
            in_features = lcb_features + fmb_features

        wukong_raw_dim = in_features * emb_dim  # flatten

        # feat_repr_dim: compress Wukong output to fixed dim before concat with seq
        feat_repr_dim = mc.get("feat_repr_dim", 0)
        if feat_repr_dim > 0:
            self.feat_proj = nn.Linear(wukong_raw_dim, feat_repr_dim)
            wukong_output_dim = feat_repr_dim
        else:
            self.feat_proj = None
            wukong_output_dim = wukong_raw_dim

        # === Sequence path: DIN ===
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()
        self.attention = DINAttention(emb_dim, attn_hidden)

        if self.dual_seq:
            self.item_emb2 = nn.Embedding(num_items2 + 1, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(self.item_emb2.weight)
            self.item_emb2.weight.data[0].zero_()
            self.attention2 = DINAttention(emb_dim, attn_hidden)

        # Multimodal
        mm_dim = dc.get("mm_dim", 0)
        self.has_mm = mm_dim > 0
        if self.has_mm:
            self.mm_proj = nn.Linear(mm_dim, emb_dim)

        # === Top MLP ===
        top_input = wukong_output_dim + emb_dim * 2  # interest + target
        if self.dual_seq:
            top_input += emb_dim * 2
        if self.has_mm:
            top_input += emb_dim

        top_layers = []
        in_d = top_input
        for out_d in mlp_dims:
            top_layers += [nn.Linear(in_d, out_d), nn.ReLU()]
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_d = out_d
        top_layers.append(nn.Linear(in_d, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        # === Sparse: Wukong ===
        tokens = []
        if self.sparse_arch is not None and "sparse" in batch:
            tokens.append(self.sparse_arch.forward_per_feature(batch["sparse"]))
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            tokens.append(self.dense_proj(batch["dense"]).unsqueeze(1))

        x = torch.cat(tokens, dim=1)  # (B, N, E)
        for layer in self.wukong_layers:
            x = layer(x)
        wukong_repr = x.flatten(start_dim=1)  # (B, N*E)
        if self.feat_proj is not None:
            wukong_repr = F.relu(self.feat_proj(wukong_repr))

        # === Sequence: DIN ===
        parts = [wukong_repr]

        seq = batch.get("seq")
        target = batch.get("target")
        if seq is not None and target is not None:
            seq_emb = self.item_emb(seq)
            target_emb = self.item_emb(target)
            interest = self.attention(target_emb, seq_emb, seq != 0)
            parts.extend([interest, target_emb])

        if self.dual_seq:
            seq2 = batch.get("seq2")
            target2 = batch.get("target2")
            if seq2 is not None and target2 is not None:
                seq_emb2 = self.item_emb2(seq2)
                target_emb2 = self.item_emb2(target2)
                interest2 = self.attention2(target_emb2, seq_emb2, seq2 != 0)
                parts.extend([interest2, target_emb2])

        if self.has_mm and "mm_emb" in batch and batch["mm_emb"] is not None:
            parts.append(self.mm_proj(batch["mm_emb"]))

        x = torch.cat(parts, dim=1)
        return self.top_mlp(x).squeeze(-1)


# ============================================================
# UniMixer + DIN
# ============================================================

@register_model
class UniMixerDINModel(RecModel):
    """
    UniMixer + DIN:
      sparse/dense item stat features → SparseArch → UniMixer blocks → feat_repr
      seq → item_emb → DIN target attention → interest
      concat(feat_repr, interest, target_emb) → MLP → logit

    UniMixer 负责 item stat 特征的可学习 token 交互，
    DIN 负责序列的 target-aware attention，两者互补。
    """
    model_name = "unimixer_din"

    def __init__(self, config: dict):
        super().__init__(config)
        from .unimixer import UniMixerBlock
        mc = config["model"]
        dc = config["dataset"]

        emb_dim     = mc["embedding_dim"]
        num_layers  = mc.get("num_mixer_layers", 2)
        num_heads   = mc.get("num_heads", 4)
        ffn_dim     = mc.get("ffn_dim", 128)
        num_rank    = mc.get("num_rank", None)
        mlp_dims    = mc.get("mlp_dims", [256, 128, 64])
        dropout     = mc.get("dropout", 0.1)
        attn_hidden = mc.get("attention_dim", 64)
        feat_repr_dim = mc.get("feat_repr_dim", 64)  # project mixer output to fixed dim

        num_dense   = len(dc.get("dense_cols") or [])
        cardinalities = dc.get("cardinalities", [])
        num_sparse  = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_items   = dc.get("num_items", 100000)

        # === Sparse path: UniMixer ===
        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None

        self.has_dense = num_dense > 0
        if self.has_dense:
            self.dense_proj = nn.Linear(num_dense, emb_dim)

        num_tokens = (num_sparse if num_sparse > 0 else 0) + (1 if self.has_dense else 0)
        self.has_feat = num_tokens > 0

        if self.has_feat and num_tokens > 0:
            self.mixer_blocks = nn.Sequential(*[
                UniMixerBlock(num_tokens, emb_dim, num_heads, ffn_dim, num_rank, dropout)
                for _ in range(num_layers)
            ])
            mixer_out_dim = num_tokens * emb_dim
            self.feat_proj = nn.Linear(mixer_out_dim, feat_repr_dim)
        else:
            self.mixer_blocks = None
            self.feat_proj = None
            feat_repr_dim = 0

        # === Sequence path: DIN ===
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()
        self.attention = DINAttention(emb_dim, attn_hidden)

        # === Top MLP ===
        top_input = feat_repr_dim + emb_dim * 2  # feat_repr + interest + target
        top_layers = []
        in_d = top_input
        for out_d in mlp_dims:
            top_layers += [nn.Linear(in_d, out_d), nn.ReLU()]
            if dropout > 0:
                top_layers.append(nn.Dropout(dropout))
            in_d = out_d
        top_layers.append(nn.Linear(in_d, 1))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        parts = []

        # === UniMixer: item stat features ===
        if self.has_feat and self.mixer_blocks is not None:
            tokens = []
            if self.sparse_arch is not None and "sparse" in batch:
                tokens.append(self.sparse_arch.forward_per_feature(batch["sparse"]))
            if self.has_dense and "dense" in batch and batch["dense"] is not None:
                tokens.append(F.relu(self.dense_proj(batch["dense"])).unsqueeze(1))
            if tokens:
                x = torch.cat(tokens, dim=1)
                x = self.mixer_blocks(x)
                feat_repr = F.relu(self.feat_proj(x.flatten(start_dim=1)))
                parts.append(feat_repr)

        # === DIN: sequence attention ===
        seq    = batch.get("seq")
        target = batch.get("target")
        if seq is not None and target is not None:
            seq_emb    = self.item_emb(seq)
            target_emb = self.item_emb(target)
            interest   = self.attention(target_emb, seq_emb, seq != 0)
            parts.extend([interest, target_emb])

        x = torch.cat(parts, dim=1)
        return self.top_mlp(x).squeeze(-1)
