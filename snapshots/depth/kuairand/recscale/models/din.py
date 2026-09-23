"""
recscale.models.din — DIN (Deep Interest Network)

Target Attention: 用 target item 对历史序列做 attention，提取与 target 相关的兴趣表示。
支持 batch["seq"] + batch["target"] 输入。

Architecture:
  sparse → SparseArch → sparse_emb
  seq → item_emb → attention(query=target_emb, keys=seq_emb) → interest_repr
  concat(sparse_emb, interest_repr) → MLP → logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, DenseArch, SparseArch


class DINAttention(nn.Module):
    """
    DIN Target Attention:
    score = MLP(concat(query, key, query-key, query*key))
    output = weighted_sum(values, softmax(scores))
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        # query, key, query-key, query*key → 4 * emb_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: (B, E) — target item embedding
            keys: (B, L, E) — sequence item embeddings
            mask: (B, L) — True for valid positions
        Returns:
            (B, E) — attention-weighted sequence representation
        """
        B, L, E = keys.shape
        query_exp = query.unsqueeze(1).expand(-1, L, -1)  # (B, L, E)

        # DIN attention features
        attn_input = torch.cat([
            query_exp,
            keys,
            query_exp - keys,
            query_exp * keys,
        ], dim=-1)  # (B, L, 4E)

        scores = self.attn_mlp(attn_input).squeeze(-1)  # (B, L)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (B, L)

        # Handle all-masked rows (avoid NaN)
        weights = torch.nan_to_num(weights, nan=0.0)

        output = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, E)
        return output


@register_model
class DINModel(RecModel):
    """
    DIN: sparse features + target attention over behavior sequence → MLP → logit

    Supports single-sequence mode (batch keys: "sparse", "seq", "target")
    and dual-sequence mode (batch keys: "sparse", "seq", "seq2", "target", "target2").

    Dual-sequence mode aligns with TaobaoAd's original DIN paper:
    cate_his + brand_his each have independent embedding + attention modules,
    query comes from sparse embedding of cate_id / brand respectively.

    Optional: "dense", "mm_emb"
    """
    model_name = "din"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        mlp_dims = mc.get("mlp_dims", [256, 128, 64])
        dropout = mc.get("dropout", 0.0)
        attn_hidden = mc.get("attention_dim", 64)
        self.use_seq_action_type_embedding = mc.get("use_seq_action_type_embedding", False)
        self.seq_action_fusion = mc.get("seq_action_fusion", "add")
        num_seq_action_types = int(mc.get("num_seq_action_types", 5))

        num_dense = len(dc.get("dense_cols") or [])
        use_sparse_features = mc.get("use_sparse_features", True)
        num_sparse = len(dc.get("sparse_cols") or []) if use_sparse_features else 0
        cardinalities = dc.get("cardinalities", [10000] * num_sparse)

        # 序列 item 的 vocab size
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)

        # Dual-sequence support: second seq field (e.g. brand_his)
        self.dual_seq = dc.get("num_items2", 0) > 0
        num_items2 = dc.get("num_items2", num_items)

        # Dense
        dense_out_dim = emb_dim if num_dense > 0 else 0
        self.dense_arch = DenseArch(num_dense, [dense_out_dim] if num_dense > 0 else [], dropout)

        # Sparse (user/context features)
        # feat_repr_dim: if > 0, compress sparse embeddings to fixed dim via pooling+linear
        #   mode="mean": mean-pool (N, E) → (E,) → Linear(E, feat_repr_dim)   [lightweight]
        #   mode="flat": flatten (N*E,) → Linear(N*E, feat_repr_dim)           [standard]
        # feat_repr_dim=0 (default): keep original flatten behavior (N*E concat)
        feat_repr_dim = mc.get("feat_repr_dim", 0)
        feat_pool_mode = mc.get("feat_pool_mode", "mean")  # "mean" or "flat"
        self.feat_repr_dim = feat_repr_dim
        self.feat_pool_mode = feat_pool_mode

        # user_emb_dim: use smaller emb dim for user/stat features (default = emb_dim)
        # This decouples user feature resolution from item embedding size
        user_emb_dim = mc.get("user_emb_dim", emb_dim)
        self.user_emb_dim = user_emb_dim

        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, user_emb_dim, cardinalities)
            if feat_repr_dim > 0:
                if feat_pool_mode == "mean":
                    # mean-pool across fields → (B, user_emb_dim) → proj
                    self.feat_proj = nn.Linear(user_emb_dim, feat_repr_dim)
                else:
                    # flatten → proj
                    self.feat_proj = nn.Linear(num_sparse * user_emb_dim, feat_repr_dim)
                sparse_out_dim = feat_repr_dim
            else:
                self.feat_proj = None
                sparse_out_dim = num_sparse * user_emb_dim
        else:
            self.sparse_arch = None
            self.feat_proj = None
            sparse_out_dim = 0

        # Item embedding for primary sequence (e.g. cate_his)
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()
        if self.use_seq_action_type_embedding:
            self.seq_action_emb = nn.Embedding(num_seq_action_types, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(self.seq_action_emb.weight)
            self.seq_action_emb.weight.data[0].zero_()
            if self.seq_action_fusion == "concat_mlp":
                self.seq_action_fuse = nn.Sequential(
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                )
            elif self.seq_action_fusion != "add":
                raise ValueError(f"Unsupported seq_action_fusion={self.seq_action_fusion!r}")

        # item_side_info: item stat features → Linear → add to item_emb
        # Allows item statistics to enrich per-item representation used in attention
        # num_item_dense: number of dense item stat features (from dc["num_dense"])
        num_item_dense = dc.get("num_dense", 0)
        self.use_item_side_info = mc.get("use_item_side_info", False) and num_item_dense > 0
        if self.use_item_side_info:
            self.item_side_proj = nn.Sequential(
                nn.Linear(num_item_dense, emb_dim),
                nn.ReLU(),
            )
            print(f"[DIN] item_side_info: {num_item_dense}d dense → {emb_dim}d side embedding")

        # DIN attention for primary sequence
        self.attention = DINAttention(emb_dim, attn_hidden)

        # Second sequence embedding + attention (e.g. brand_his)
        if self.dual_seq:
            self.item_emb2 = nn.Embedding(num_items2 + 1, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(self.item_emb2.weight)
            self.item_emb2.weight.data[0].zero_()
            self.attention2 = DINAttention(emb_dim, attn_hidden)

        # Sparse interaction blocks (legacy, used when feat_repr_dim=0)
        num_interaction_blocks = mc.get("num_interaction_blocks", 0)
        self.interaction_blocks = nn.ModuleList()
        if num_interaction_blocks > 0 and self.sparse_arch is not None and feat_repr_dim == 0:
            for _ in range(num_interaction_blocks):
                self.interaction_blocks.append(nn.Sequential(
                    nn.Linear(sparse_out_dim, sparse_out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ))

        # Multimodal embedding projection (optional, e.g. Taobao-MM SCL 128d)
        mm_dim = dc.get("mm_dim", 0)
        self.has_mm = mm_dim > 0
        if self.has_mm:
            self.mm_proj = nn.Linear(mm_dim, emb_dim)

        # Interaction term: interest * target_emb (align with old TaobaoMM code)
        self.use_interaction = mc.get("use_interaction", False)

        # Top MLP
        input_dim = self.dense_arch.output_dim
        if self.sparse_arch is not None:
            input_dim += sparse_out_dim  # either feat_repr_dim or N*user_emb_dim
        input_dim += emb_dim  # primary attention output
        input_dim += emb_dim  # primary target embedding
        if self.use_interaction:
            input_dim += emb_dim  # interest * target interaction
        if self.dual_seq:
            input_dim += emb_dim  # second attention output
            input_dim += emb_dim  # second target embedding
            if self.use_interaction:
                input_dim += emb_dim  # second interaction
        if self.has_mm:
            input_dim += emb_dim  # mm embedding projection

        self.use_feature_path_gate = mc.get("use_feature_path_gate", False)
        self.feature_path_gate_mode = mc.get("feature_path_gate_mode", "scalar")
        side_input_dim = self.dense_arch.output_dim + (sparse_out_dim if self.sparse_arch is not None else 0)
        self.side_input_dim = side_input_dim
        if self.use_feature_path_gate and side_input_dim > 0:
            gate_out_dim = side_input_dim if self.feature_path_gate_mode == "vector" else 1
            self.feature_path_gate = nn.Linear(side_input_dim + emb_dim * 2, gate_out_dim)
            nn.init.zeros_(self.feature_path_gate.weight)
            nn.init.zeros_(self.feature_path_gate.bias)
        else:
            self.feature_path_gate = None

        layers = []
        in_dim = input_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*layers)

        # Debug controls for testing whether DIN can ignore sparse feature paths.
        self.zero_sparse_input = mc.get("zero_sparse_input", False)
        sparse_first_layer_init = mc.get("sparse_first_layer_init", None)
        freeze_sparse_first_layer = mc.get("freeze_sparse_first_layer", False)
        if (
            self.sparse_arch is not None
            and sparse_out_dim > 0
            and sparse_first_layer_init == "zero"
            and len(self.top_mlp) > 0
            and isinstance(self.top_mlp[0], nn.Linear)
        ):
            sparse_start = self.dense_arch.output_dim
            sparse_end = sparse_start + sparse_out_dim
            with torch.no_grad():
                self.top_mlp[0].weight[:, sparse_start:sparse_end].zero_()
            if freeze_sparse_first_layer:
                def _zero_sparse_grad(grad):
                    grad = grad.clone()
                    grad[:, sparse_start:sparse_end].zero_()
                    return grad
                self.top_mlp[0].weight.register_hook(_zero_sparse_grad)

    def forward(self, batch: dict) -> torch.Tensor:
        parts = []
        side_parts = []

        # Dense
        if "dense" in batch and batch["dense"] is not None:
            side_parts.append(self.dense_arch(batch["dense"]))

        # Sparse
        if self.sparse_arch is not None and "sparse" in batch:
            # sparse_arch returns (B, N*user_emb_dim) when forward()
            # but we need per-field view for mean pooling
            if self.feat_repr_dim > 0 and self.feat_pool_mode == "mean":
                # Use per_feature forward: (B, N, user_emb_dim)
                feat_tokens = self.sparse_arch.forward_per_feature(batch["sparse"])  # (B, N, E)
                feat_flat = feat_tokens.mean(dim=1)  # (B, user_emb_dim)
                feat_repr = F.relu(self.feat_proj(feat_flat))  # (B, feat_repr_dim)
            elif self.feat_repr_dim > 0:
                # flat mode: (B, N*E) → proj
                feat_flat = self.sparse_arch(batch["sparse"])  # (B, N*E)
                feat_repr = F.relu(self.feat_proj(feat_flat))  # (B, feat_repr_dim)
            else:
                feat_repr = self.sparse_arch(batch["sparse"])  # (B, N*E)
                h = feat_repr
                for block in self.interaction_blocks:
                    h = block(h) + h
                feat_repr = h
            if self.zero_sparse_input:
                feat_repr = torch.zeros_like(feat_repr)
            side_parts.append(feat_repr)

        # Primary sequence attention (e.g. cate_his)
        seq = batch.get("seq")  # (B, L)
        target = batch.get("target")  # (B,)

        if seq is not None and target is not None:
            seq_emb = self.item_emb(seq)      # (B, L, E)
            if self.use_seq_action_type_embedding and "seq_action" in batch:
                action_emb = self.seq_action_emb(batch["seq_action"].clamp_min(0))
                if self.seq_action_fusion == "concat_mlp":
                    seq_emb = self.seq_action_fuse(torch.cat([seq_emb, action_emb], dim=-1))
                else:
                    seq_emb = seq_emb + action_emb
            target_emb = self.item_emb(target) # (B, E)

            # item side info: add stat features to item embeddings
            if self.use_item_side_info and "dense" in batch and batch["dense"] is not None:
                side = self.item_side_proj(batch["dense"])  # (B, E)
                target_emb = target_emb + side              # enrich target with item stats
                # seq positions don't have individual dense features (shared per item),
                # so we add the same side info to all seq positions as a context bias
                seq_emb = seq_emb + side.unsqueeze(1)       # (B, L, E)

            seq_mask = (seq != 0)  # (B, L)
            interest = self.attention(target_emb, seq_emb, seq_mask)  # (B, E)
            if side_parts:
                side_repr = torch.cat(side_parts, dim=1)
                if self.feature_path_gate is not None:
                    gate_input = torch.cat([side_repr, target_emb, interest], dim=1)
                    gate = 2.0 * torch.sigmoid(self.feature_path_gate(gate_input))
                    side_repr = side_repr * gate
                parts.append(side_repr)
            parts.append(interest)
            parts.append(target_emb)
            if self.use_interaction:
                parts.append(interest * target_emb)
        elif target is not None:
            target_emb = self.item_emb(target)
            if self.use_item_side_info and "dense" in batch and batch["dense"] is not None:
                target_emb = target_emb + self.item_side_proj(batch["dense"])
            if side_parts:
                side_repr = torch.cat(side_parts, dim=1)
                if self.feature_path_gate is not None:
                    zeros = torch.zeros_like(target_emb)
                    gate_input = torch.cat([side_repr, target_emb, zeros], dim=1)
                    gate = 2.0 * torch.sigmoid(self.feature_path_gate(gate_input))
                    side_repr = side_repr * gate
                parts.append(side_repr)
            parts.append(target_emb)
        elif side_parts:
            parts.append(torch.cat(side_parts, dim=1))

        # Second sequence attention (e.g. brand_his)
        if self.dual_seq:
            seq2 = batch.get("seq2")  # (B, L)
            target2 = batch.get("target2")  # (B,)

            if seq2 is not None and target2 is not None:
                seq_emb2 = self.item_emb2(seq2)  # (B, L, E)
                target_emb2 = self.item_emb2(target2)  # (B, E)
                seq_mask2 = (seq2 != 0)  # (B, L)

                interest2 = self.attention2(target_emb2, seq_emb2, seq_mask2)  # (B, E)
                parts.append(interest2)
                parts.append(target_emb2)
                if self.use_interaction:
                    parts.append(interest2 * target_emb2)
            elif target2 is not None:
                target_emb2 = self.item_emb2(target2)
                parts.append(target_emb2)

        # Multimodal embedding (e.g. SCL visual/text features)
        if self.has_mm and "mm_emb" in batch and batch["mm_emb"] is not None:
            mm_feat = self.mm_proj(batch["mm_emb"])  # (B, E)
            parts.append(mm_feat)

        x = torch.cat(parts, dim=1)
        return self.top_mlp(x).squeeze(-1)
