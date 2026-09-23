"""
Projected controls for Fuxi-style T13D104L2 comparisons.

The original RankMixerV2 and TokenMixerLargeV3 use ``embedding_dim`` as the token
width, so they cannot run ``emb16, T13, D104`` directly. These controls add only
the missing input projection from field embeddings to ``T`` tokens of width
``d_model``, then reuse the original mixer blocks and output style.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, SparseArch
from .rankmixer_v2 import RankMixerV2Block
from .tokenmixer_large_v3 import TokenMixerLargeV3Block


class _ProjectedTokenizer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        dc = config["dataset"]

        embedding_dim = mc.get("embedding_dim", 16)
        num_tokens = mc.get("num_tokens", mc.get("num_feature_fields", 13))
        d_model = mc.get("d_model", 104)
        tokenize_mode = mc.get("tokenize_mode", "uniform")
        uniform_proj = mc.get("uniform_proj", "split")
        field_shuffle = mc.get("field_shuffle", False)

        cardinalities = dc.get("cardinalities", [])
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        num_dense = len(dc.get("dense_cols") or [])

        self.sparse_arch = SparseArch(num_sparse, embedding_dim, cardinalities)
        self.has_dense = num_dense > 0
        self.dense_proj = nn.Linear(num_dense, embedding_dim) if self.has_dense else None
        num_fields = num_sparse + (1 if self.has_dense else 0)

        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.tokenize_mode = tokenize_mode
        self.uniform_proj = uniform_proj

        if tokenize_mode == "uniform":
            self.num_tokens = num_tokens
            if uniform_proj == "shared":
                self.token_proj = nn.Linear(num_fields * embedding_dim, num_tokens * d_model)
            elif uniform_proj == "split":
                if num_fields % num_tokens != 0:
                    raise ValueError(
                        f"uniform split requires num_fields={num_fields} divisible by num_tokens={num_tokens}"
                    )
                self.chunk_size = (num_fields // num_tokens) * embedding_dim
                self.token_proj_W = nn.Parameter(torch.empty(num_tokens, self.chunk_size, d_model))
                self.token_proj_b = nn.Parameter(torch.zeros(num_tokens, d_model))
                nn.init.xavier_uniform_(self.token_proj_W)
            else:
                raise ValueError(f"uniform_proj must be 'shared' or 'split', got {uniform_proj}")

            if field_shuffle:
                gen = torch.Generator()
                gen.manual_seed(int(config.get("seed", 42)))
                self.register_buffer("field_perm", torch.randperm(num_fields, generator=gen))
            else:
                self.field_perm = None
        elif tokenize_mode == "per_field":
            self.num_tokens = num_fields
            if uniform_proj == "shared":
                self.token_proj = nn.Linear(embedding_dim, d_model)
            elif uniform_proj == "split":
                self.token_proj_W = nn.Parameter(torch.empty(num_fields, embedding_dim, d_model))
                self.token_proj_b = nn.Parameter(torch.zeros(num_fields, d_model))
                nn.init.xavier_uniform_(self.token_proj_W)
            else:
                raise ValueError(f"uniform_proj must be 'shared' or 'split', got {uniform_proj}")
            self.field_perm = None
        else:
            raise ValueError(f"tokenize_mode must be 'uniform' or 'per_field', got {tokenize_mode}")

        if d_model % self.num_tokens != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_tokens={self.num_tokens}")

        self.num_fields = num_fields

    def _field_embeddings(self, batch: dict) -> torch.Tensor:
        sparse_per = self.sparse_arch.forward_per_feature(batch["sparse"])
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            dense_token = self.dense_proj(batch["dense"]).unsqueeze(1)
            return torch.cat([sparse_per, dense_token], dim=1)
        return sparse_per

    def forward(self, batch: dict) -> torch.Tensor:
        feat_emb = self._field_embeddings(batch)
        if self.tokenize_mode == "uniform":
            if self.field_perm is not None:
                feat_emb = feat_emb[:, self.field_perm, :]
            flat = feat_emb.flatten(start_dim=1)
            if self.uniform_proj == "shared":
                return self.token_proj(flat).view(-1, self.num_tokens, self.d_model)
            chunks = flat.view(-1, self.num_tokens, self.chunk_size)
            return torch.einsum("btc,tcd->btd", chunks, self.token_proj_W) + self.token_proj_b

        if self.uniform_proj == "shared":
            return self.token_proj(feat_emb)
        return torch.einsum("bne,ned->bnd", feat_emb, self.token_proj_W) + self.token_proj_b


@register_model
class RankMixerV2ProjectedModel(RecModel):
    """RankMixerV2 blocks with an emb16 -> T/D projected tokenizer control."""

    model_name = "rankmixer_v2_projected"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        self.tokenizer = _ProjectedTokenizer(config)

        num_layers = mc.get("num_mixer_layers", mc.get("num_layers", 2))
        ffn_dim = mc.get("ffn_dim", 416)
        dropout = mc.get("dropout", 0.0)
        self.pool_output = mc.get("pool_output", False)
        mlp_dims = mc.get("mlp_dims", [512, 256])

        self.mixer_blocks = nn.ModuleList([
            RankMixerV2Block(self.tokenizer.num_tokens, self.tokenizer.d_model, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        in_dim = self.tokenizer.d_model if self.pool_output else self.tokenizer.num_tokens * self.tokenizer.d_model
        head_layers = []
        for dim in mlp_dims:
            head_layers.append(nn.Linear(in_dim, dim))
            head_layers.append(nn.ReLU())
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            in_dim = dim
        head_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*head_layers)

        print(
            f"[RankMixerV2Projected] fields={self.tokenizer.num_fields}, T={self.tokenizer.num_tokens}, "
            f"emb={self.tokenizer.embedding_dim}, d_model={self.tokenizer.d_model}, L={num_layers}, "
            f"ffn={ffn_dim}, tokenize={self.tokenizer.tokenize_mode}/{self.tokenizer.uniform_proj}"
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.tokenizer(batch)
        for block in self.mixer_blocks:
            x = block(x)
        if self.pool_output:
            x = x.mean(dim=1)
        else:
            x = x.flatten(start_dim=1)
        return self.top_mlp(x).squeeze(-1)


class _AuxPredictor(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tokens * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1)).squeeze(-1)


@register_model
class TokenMixerLargeV3ProjectedModel(RecModel):
    """TokenMixerLargeV3 blocks with an emb16 -> T/D projected tokenizer control."""

    model_name = "tokenmixer_large_v3_projected"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]
        self.tokenizer = _ProjectedTokenizer(config)

        num_layers = mc.get("num_mixer_layers", mc.get("num_layers", 2))
        ffn_dim = mc.get("ffn_dim", 416)
        dropout = mc.get("dropout", 0.0)
        self.pool_output = mc.get("pool_output", False)
        mlp_dims = mc.get("mlp_dims", [512, 256])

        self.inter_layer_residual = mc.get("inter_layer_residual", True)
        self.residual_skip_stride = mc.get("residual_skip_stride", 2)
        self.exclude_last_inter_residual = mc.get("exclude_last_inter_layer_residual", True)

        self.aux_loss_weight = float(mc.get("aux_loss_weight", 0.0))
        self.aux_pred_hidden_dim = int(mc.get("aux_pred_hidden_dim", 64))
        self.aux_exclude_last_layer = bool(mc.get("aux_exclude_last_layer", True))

        self.mixer_blocks = nn.ModuleList([
            TokenMixerLargeV3Block(
                num_tokens=self.tokenizer.num_tokens,
                token_dim=self.tokenizer.d_model,
                ffn_dim=ffn_dim,
                dropout=dropout,
                channel_mixer_type=mc.get("channel_mixer_type", "dense"),
                num_experts=mc.get("num_experts", 4),
                top_k=mc.get("top_k", 2),
                num_shared=mc.get("num_shared", 1),
                init_mix_scale=mc.get("init_mix_scale", 1.0),
                init_ffn_scale=mc.get("init_ffn_scale", 1.0),
                init_inter_scale=mc.get("init_inter_scale", 0.1),
                learnable_scale=mc.get("learnable_scale", True),
                down_init_std=mc.get("down_init_std", 0.01),
                router_scale=mc.get("router_scale", 1.0),
            )
            for _ in range(num_layers)
        ])

        self.num_aux = 0
        if self.aux_loss_weight > 0.0:
            self.num_aux = num_layers - 1 if self.aux_exclude_last_layer else num_layers
            if self.num_aux > 0:
                self.aux_predictors = nn.ModuleList([
                    _AuxPredictor(self.tokenizer.num_tokens, self.tokenizer.d_model, self.aux_pred_hidden_dim)
                    for _ in range(self.num_aux)
                ])

        in_dim = self.tokenizer.d_model if self.pool_output else self.tokenizer.num_tokens * self.tokenizer.d_model
        head_layers = []
        for dim in mlp_dims:
            head_layers.append(nn.Linear(in_dim, dim))
            head_layers.append(nn.ReLU())
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            in_dim = dim
        head_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*head_layers)

        print(
            f"[TokenMixerLargeV3Projected] fields={self.tokenizer.num_fields}, T={self.tokenizer.num_tokens}, "
            f"emb={self.tokenizer.embedding_dim}, d_model={self.tokenizer.d_model}, L={num_layers}, "
            f"ffn={ffn_dim}, tokenize={self.tokenizer.tokenize_mode}/{self.tokenizer.uniform_proj}, "
            f"channel={mc.get('channel_mixer_type', 'dense')}"
        )

    def _run_blocks(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        hidden_states = []
        last_idx = len(self.mixer_blocks) - 1
        for layer_idx, block in enumerate(self.mixer_blocks):
            skip = None
            use_inter = self.inter_layer_residual and not (
                self.exclude_last_inter_residual and layer_idx == last_idx
            )
            if use_inter and layer_idx >= self.residual_skip_stride:
                skip = hidden_states[layer_idx - self.residual_skip_stride]
            x = block(x, skip=skip)
            hidden_states.append(x)

        aux_loss = None
        if (
            self.training
            and label is not None
            and self.aux_loss_weight > 0.0
            and self.num_aux > 0
        ):
            aux_loss = x.new_zeros(())
            for i, predictor in enumerate(self.aux_predictors):
                aux_logit = predictor(hidden_states[i])
                aux_loss = aux_loss + F.binary_cross_entropy_with_logits(aux_logit, label.float())
            aux_loss = self.aux_loss_weight * aux_loss / self.num_aux
        return x, aux_loss

    def forward(self, batch: dict):
        x = self.tokenizer(batch)
        x, aux_loss = self._run_blocks(x, label=batch.get("label"))
        if self.pool_output:
            x = x.mean(dim=1)
        else:
            x = x.flatten(start_dim=1)
        logits = self.top_mlp(x).squeeze(-1)
        if aux_loss is not None:
            return logits, aux_loss
        return logits
