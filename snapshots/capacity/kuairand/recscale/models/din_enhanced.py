"""
recscale.models.din_enhanced — Enhanced DIN with dual-sequence attention

Improvements over current DIN:
1. Support multiple sequence fields (cate_his + brand_his)
2. Independent DINAttention modules per sequence field
3. Proper query alignment (cate_id→cate_his, brand→brand_his)
4. Optional interaction blocks for feature interactions
5. Configurable architecture with backward compatibility

Architecture:
  sparse → SparseArch → sparse_emb
  seq_1 → item_emb → attention(query=target_1, keys=seq_1_emb) → interest_1
  seq_2 → item_emb → attention(query=target_2, keys=seq_2_emb) → interest_2
  concat(sparse_emb, interest_1, interest_2, target_1, target_2) → [interaction_blocks] → MLP → logit
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


class MLPInteractionBlock(nn.Module):
    """Standard MLP interaction block with residual connection."""
    def __init__(self, dim, hidden_mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))


class MixerBlock(nn.Module):
    """MLPMixer-style interaction block: token mixing + channel mixing."""
    def __init__(self, num_features, feature_dim, dropout=0.0):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, num_features, feature_dim)
        # Token mixing: (B, num_features, feature_dim) → (B, num_features, feature_dim)
        h = x + self.token_mixing(x.transpose(1, 2)).transpose(1, 2)
        # Channel mixing: per-feature MLP
        h = h + self.channel_mixing(h)
        return h


class TransformerBlock(nn.Module):
    """Transformer-style interaction block: self-attention + FFN."""
    def __init__(self, feature_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, num_features, feature_dim)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


@register_model
class DINEnhancedModel(RecModel):
    """
    Enhanced DIN: dual-sequence attention + optional interaction blocks
    
    Supports both single-sequence (backward compatible) and multi-sequence modes.
    
    Config keys:
        embedding_dim: embedding dimension
        mlp_dims: MLP layer dimensions
        dropout: dropout rate
        attention_dim: attention MLP hidden dimension
        num_interaction_blocks: number of interaction blocks (0 for none)
        interaction_type: 'mlp', 'mixer', 'transformer'
        num_heads: for transformer interaction blocks
        use_target_emb: whether to include target embedding in final MLP (default: True)
    """
    model_name = "din_enhanced"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        dc = config["dataset"]

        emb_dim = mc["embedding_dim"]
        mlp_dims = mc.get("mlp_dims", [256, 128, 64])
        dropout = mc.get("dropout", 0.0)
        attn_hidden = mc.get("attention_dim", 64)
        
        # New parameters for enhanced model
        num_interaction_blocks = mc.get("num_interaction_blocks", 0)
        interaction_type = mc.get("interaction_type", "mlp")
        num_heads = mc.get("num_heads", 4)
        self.use_target_emb = mc.get("use_target_emb", True)

        num_dense = len(dc.get("dense_cols") or [])
        num_sparse = len(dc.get("sparse_cols") or [])
        cardinalities = dc.get("cardinalities", [10000] * num_sparse)

        # Sequence configuration
        num_items = dc.get("num_items", max(cardinalities) if cardinalities else 100000)
        num_seq_fields = dc.get("num_seq_fields", 1)  # Support multiple sequence fields
        seq_field_names = dc.get("seq_field_names", ["seq"])  # e.g., ["cate_his", "brand_his"]

        # Dense
        dense_out_dim = emb_dim if num_dense > 0 else 0
        self.dense_arch = DenseArch(num_dense, [dense_out_dim] if num_dense > 0 else [], dropout)

        # Sparse (user/context features)
        if num_sparse > 0:
            self.sparse_arch = SparseArch(num_sparse, emb_dim, cardinalities)
        else:
            self.sparse_arch = None

        # Item embedding (shared between target and all sequences)
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.item_emb.weight.data[0].zero_()

        # Multiple DIN attention modules (one per sequence field)
        self.num_seq_fields = num_seq_fields
        self.seq_field_names = seq_field_names
        self.attentions = nn.ModuleDict()
        for seq_name in seq_field_names:
            self.attentions[seq_name] = DINAttention(emb_dim, attn_hidden)

        # Target embeddings (one per sequence field, or shared)
        # For TaobaoAd: target_cate and target_brand from sparse features
        self.target_embs = nn.ModuleDict()
        # Could be mapped from sparse embeddings or standalone
        
        # Multimodal embedding projection (optional)
        mm_dim = dc.get("mm_dim", 0)
        self.has_mm = mm_dim > 0
        if self.has_mm:
            self.mm_proj = nn.Linear(mm_dim, emb_dim)

        # Interaction blocks
        self.num_interaction_blocks = num_interaction_blocks
        self.interaction_blocks = nn.ModuleList()
        
        if num_interaction_blocks > 0:
            # Calculate input dimension for interaction blocks
            interaction_input_dim = self.dense_arch.output_dim
            if self.sparse_arch is not None:
                interaction_input_dim += self.sparse_arch.output_dim
            interaction_input_dim += num_seq_fields * emb_dim  # attention outputs
            if self.use_target_emb:
                interaction_input_dim += num_seq_fields * emb_dim  # target embeddings
            
            for _ in range(num_interaction_blocks):
                if interaction_type == "mixer":
                    # For mixer: need num_features
                    num_features = interaction_input_dim // emb_dim
                    self.interaction_blocks.append(
                        MixerBlock(num_features, emb_dim, dropout)
                    )
                elif interaction_type == "transformer":
                    self.interaction_blocks.append(
                        TransformerBlock(emb_dim, num_heads, dropout)
                    )
                else:  # mlp
                    self.interaction_blocks.append(
                        MLPInteractionBlock(interaction_input_dim, hidden_mult=4, dropout=dropout)
                    )

        # Top MLP
        input_dim = self.dense_arch.output_dim
        if self.sparse_arch is not None:
            input_dim += self.sparse_arch.output_dim
        input_dim += num_seq_fields * emb_dim  # attention outputs
        if self.use_target_emb:
            input_dim += num_seq_fields * emb_dim  # target embeddings
        if self.has_mm:
            input_dim += emb_dim  # mm embedding projection

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

    def forward(self, batch: dict) -> torch.Tensor:
        parts = []

        # Dense
        if "dense" in batch and batch["dense"] is not None:
            parts.append(self.dense_arch(batch["dense"]))

        # Sparse
        if self.sparse_arch is not None and "sparse" in batch:
            parts.append(self.sparse_arch(batch["sparse"]))

        # Sequence attention (support both single and multiple sequences)
        interests = []
        targets = []
        
        if self.num_seq_fields == 1:
            # Backward compatible: single sequence mode
            seq = batch.get("seq")
            target = batch.get("target")
            
            if seq is not None and target is not None:
                seq_emb = self.item_emb(seq)
                target_emb = self.item_emb(target)
                seq_mask = (seq != 0)
                
                interest = self.attentions[self.seq_field_names[0]](
                    target_emb, seq_emb, seq_mask
                )
                interests.append(interest)
                targets.append(target_emb)
            elif target is not None:
                target_emb = self.item_emb(target)
                targets.append(target_emb)
        else:
            # Multi-sequence mode
            seqs = batch.get("seqs")  # (B, num_seq_fields, maxlen)
            targets_dict = batch.get("targets")  # dict of targets per sequence
            
            if seqs is not None:
                for i, seq_name in enumerate(self.seq_field_names):
                    seq = seqs[:, i, :]  # (B, maxlen)
                    seq_emb = self.item_emb(seq)  # (B, maxlen, E)
                    seq_mask = (seq != 0)  # (B, maxlen)
                    
                    # Get target for this sequence
                    if isinstance(targets_dict, dict) and seq_name in targets_dict:
                        target = targets_dict[seq_name]
                    elif isinstance(targets_dict, torch.Tensor):
                        # Assume targets_dict is (B, num_seq_fields)
                        target = targets_dict[:, i]
                    else:
                        target = None
                    
                    if target is not None:
                        target_emb = self.item_emb(target)  # (B, E)
                        interest = self.attentions[seq_name](
                            target_emb, seq_emb, seq_mask
                        )
                        interests.append(interest)
                        targets.append(target_emb)

        # Add attention outputs
        parts.extend(interests)
        
        # Add target embeddings
        if self.use_target_emb:
            parts.extend(targets)

        # Multimodal embedding
        if self.has_mm and "mm_emb" in batch and batch["mm_emb"] is not None:
            mm_feat = self.mm_proj(batch["mm_emb"])
            parts.append(mm_feat)

        x = torch.cat(parts, dim=1)
        
        # Apply interaction blocks
        if self.num_interaction_blocks > 0:
            for block in self.interaction_blocks:
                x = block(x)
        
        return self.top_mlp(x).squeeze(-1)
