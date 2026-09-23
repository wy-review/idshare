"""
recscale.models.s2drec — Sparse-to-Dense Tokenized Recommender.

S2DRec decomposes a recommender model into:
  field embeddings -> pluggable sparse-to-dense tokenizer -> dense token backbone -> head

The tokenizer is the main experimental axis: it compresses many sparse field
embeddings into a compact set of dense tokens before the scalable backbone.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .base import RecModel, SparseArch
from .k1_identity_router import K1IdentityRouter
from .offline_tail_id_collapse import OfflineFrequencyTailIdCollapse
from .rankmixer_v2 import RankMixerV2Block, TokenMixingV2
from .tokenmixer_large_v3 import TokenMixerLargeV3Block, _RMSNorm, _make_channel_mixer
from .zero_anchor_identity_quantizer import ZeroAnchorIdentityQuantizer


class _FieldEmbeddingEncoder(nn.Module):
    """DLRM-compatible sparse/dense feature encoder returning field embeddings."""

    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        dc = config["dataset"]

        self.embedding_dim = int(mc.get("embedding_dim", 16))
        cardinalities = list(dc.get("cardinalities", []))
        num_sparse = len(dc.get("sparse_cols") or cardinalities)
        if not cardinalities:
            cardinalities = [10000] * num_sparse
        if len(cardinalities) != num_sparse:
            raise ValueError(
                f"cardinalities length ({len(cardinalities)}) must match num_sparse ({num_sparse})"
            )

        collapse_cfg = mc.get(OfflineFrequencyTailIdCollapse.CONFIG_KEY, {})
        k1_routing_cfg = mc.get(K1IdentityRouter.CONFIG_KEY, {})
        zero_anchor_cfg = mc.get(ZeroAnchorIdentityQuantizer.CONFIG_KEY, {})
        if not isinstance(zero_anchor_cfg, dict):
            raise ValueError(
                "model.zero_anchor_identity_quantization must be a mapping"
            )
        if not isinstance(k1_routing_cfg, dict):
            raise ValueError("model.k1_identity_routing must be a mapping")
        self.k1_identity_router = None
        if bool(k1_routing_cfg.get("enabled", False)):
            self.k1_identity_router = K1IdentityRouter(
                config,
                cardinalities=cardinalities,
                field_names=list(dc.get("sparse_cols") or []),
                embedding_dim=self.embedding_dim,
            )
        self.offline_tail_id_collapse = None
        if not isinstance(collapse_cfg, dict):
            raise ValueError(
                "model.offline_frequency_tail_collapse must be a mapping"
            )
        if isinstance(collapse_cfg, dict) and bool(collapse_cfg.get("enabled", False)):
            if self.k1_identity_router is not None:
                raise ValueError(
                    "k1_identity_routing and offline_frequency_tail_collapse "
                    "cannot be enabled together"
                )
            self.offline_tail_id_collapse = OfflineFrequencyTailIdCollapse(
                config,
                cardinalities=cardinalities,
                field_names=list(dc.get("sparse_cols") or []),
                embedding_dim=self.embedding_dim,
            )
        self.zero_anchor_identity_quantizer = None
        if bool(zero_anchor_cfg.get("enabled", False)):
            if (
                self.k1_identity_router is not None
                or self.offline_tail_id_collapse is not None
            ):
                raise ValueError(
                    "zero_anchor_identity_quantization cannot be combined with "
                    "K1 frequency routing or offline tail collapse"
                )

        num_dense = len(dc.get("dense_cols") or [])
        self.embedding_init = str(mc.get("embedding_init", "uniform"))
        self.embedding_init_std = float(mc.get("embedding_init_std", 0.01))
        self.sparse_arch = SparseArch(
            num_sparse,
            self.embedding_dim,
            cardinalities,
            embedding_init=self.embedding_init,
            embedding_init_std=self.embedding_init_std,
        )
        zero_rows_cfg = mc.get(
            "sparse_embedding_zero_init_rows_by_field", {}
        )
        if not isinstance(zero_rows_cfg, dict):
            raise ValueError(
                "model.sparse_embedding_zero_init_rows_by_field must be "
                "a mapping"
            )
        field_names = list(dc.get("sparse_cols") or [])
        name_to_index = {
            str(name): index for index, name in enumerate(field_names)
        }
        unknown_zero_row_fields = sorted(
            set(zero_rows_cfg) - set(name_to_index)
        )
        if unknown_zero_row_fields:
            raise ValueError(
                "zero-init sparse rows reference unknown fields: "
                f"{unknown_zero_row_fields}"
            )
        self.sparse_embedding_zero_init_rows_by_field = {}
        with torch.no_grad():
            for field, raw_rows in zero_rows_cfg.items():
                if not isinstance(raw_rows, (list, tuple)):
                    raise ValueError(
                        "zero-init sparse rows must be lists or tuples"
                    )
                rows = tuple(int(row) for row in raw_rows)
                if len(set(rows)) != len(rows) or any(row < 0 for row in rows):
                    raise ValueError(
                        f"invalid zero-init sparse rows for {field}: {rows}"
                    )
                embedding = self.sparse_arch.embeddings[
                    name_to_index[field]
                ]
                if any(row >= embedding.num_embeddings for row in rows):
                    raise ValueError(
                        f"zero-init sparse row exceeds cardinality for "
                        f"{field}: {rows}"
                    )
                if rows:
                    index = torch.tensor(
                        rows,
                        dtype=torch.long,
                        device=embedding.weight.device,
                    )
                    embedding.weight.index_fill_(0, index, 0.0)
                self.sparse_embedding_zero_init_rows_by_field[
                    str(field)
                ] = rows
        if bool(zero_anchor_cfg.get("enabled", False)):
            isolate_quantizer_rng = bool(
                zero_anchor_cfg.get("isolate_initialization_rng", False)
            )
            quantizer_seed = int(
                zero_anchor_cfg.get(
                    "initialization_seed",
                    mc.get("tokenizer_seed", config.get("seed", 2021)),
                )
            )
            if isolate_quantizer_rng:
                fork_devices = (
                    list(range(torch.cuda.device_count()))
                    if torch.cuda.is_available()
                    else []
                )
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(quantizer_seed)
                    self.zero_anchor_identity_quantizer = (
                        ZeroAnchorIdentityQuantizer(
                            config,
                            cardinalities=cardinalities,
                            field_names=list(dc.get("sparse_cols") or []),
                            embedding_dim=self.embedding_dim,
                        )
                    )
            else:
                self.zero_anchor_identity_quantizer = (
                    ZeroAnchorIdentityQuantizer(
                        config,
                        cardinalities=cardinalities,
                        field_names=list(dc.get("sparse_cols") or []),
                        embedding_dim=self.embedding_dim,
                    )
                )
            self.zero_anchor_identity_quantizer.initialize_identity_embeddings(
                self.sparse_arch.embeddings
            )
        self.has_dense = num_dense > 0
        self.dense_proj = nn.Linear(num_dense, self.embedding_dim) if self.has_dense else None
        self.num_fields = num_sparse + (1 if self.has_dense else 0)

    def forward(self, batch: dict) -> torch.Tensor:
        sparse = batch["sparse"]
        k1_masks = None
        if self.k1_identity_router is not None:
            sparse, k1_masks = self.k1_identity_router(sparse)
        collapse_masks = None
        if self.offline_tail_id_collapse is not None:
            sparse, collapse_masks = self.offline_tail_id_collapse(sparse)
        if self.zero_anchor_identity_quantizer is not None:
            self.zero_anchor_identity_quantizer.initialize_train_first_touch_rows(
                self.sparse_arch.embeddings,
                sparse,
            )
        fields = self.sparse_arch.forward_per_feature(sparse)
        if self.k1_identity_router is not None:
            fields = self.k1_identity_router.replace_routed_embeddings(
                fields, k1_masks
            )
        if self.offline_tail_id_collapse is not None:
            fields = self.offline_tail_id_collapse.replace_collapsed_embeddings(
                fields, collapse_masks
            )
        if self.zero_anchor_identity_quantizer is not None:
            fields = self.zero_anchor_identity_quantizer(fields, batch["sparse"])
        if self.has_dense and "dense" in batch and batch["dense"] is not None:
            dense_token = self.dense_proj(batch["dense"]).unsqueeze(1)
            fields = torch.cat([fields, dense_token], dim=1)
        return fields


class S2DRecTokenizer(nn.Module, ABC):
    """Base interface for sparse-to-dense tokenizers."""

    def __init__(self, num_fields: int, embedding_dim: int, num_tokens: int, d_model: int):
        super().__init__()
        self.num_fields = int(num_fields)
        self.embedding_dim = int(embedding_dim)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.aux_loss: Optional[torch.Tensor] = None
        self._diagnostics = {}

    @abstractmethod
    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        """field_emb: (B, num_fields, embedding_dim) -> (B, num_tokens, d_model)."""
        ...

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self.aux_loss

    def get_diagnostics(self) -> dict[str, float]:
        diagnostics = {}
        for name, value in self._diagnostics.items():
            if isinstance(value, torch.Tensor):
                diagnostics[name] = float(value.detach().float().cpu().item())
            else:
                diagnostics[name] = float(value)
        return diagnostics


class UniformProjectionTokenizer(S2DRecTokenizer):
    """Fuxi-style flattened projection tokenizer with shared or split projection."""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        proj_mode: str = "split",
        field_shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if proj_mode not in ("shared", "split"):
            raise ValueError(f"uniform_proj_mode must be 'shared' or 'split', got {proj_mode}")
        self.proj_mode = proj_mode

        if proj_mode == "shared":
            self.token_proj = nn.Linear(num_fields * embedding_dim, num_tokens * d_model)
        else:
            if num_fields % num_tokens != 0:
                raise ValueError(
                    f"uniform split requires num_fields={num_fields} divisible by num_tokens={num_tokens}"
                )
            self.chunk_size = (num_fields // num_tokens) * embedding_dim
            self.token_proj_W = nn.Parameter(torch.empty(num_tokens, self.chunk_size, d_model))
            self.token_proj_b = nn.Parameter(torch.zeros(num_tokens, d_model))
            nn.init.xavier_uniform_(self.token_proj_W)

        if field_shuffle:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            self.register_buffer("field_perm", torch.randperm(num_fields, generator=gen))
        else:
            self.field_perm = None

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        if self.field_perm is not None:
            field_emb = field_emb[:, self.field_perm, :]
        flat = field_emb.flatten(start_dim=1)
        if self.proj_mode == "shared":
            return self.token_proj(flat).view(-1, self.num_tokens, self.d_model)
        chunks = flat.view(-1, self.num_tokens, self.chunk_size)
        return torch.einsum("btc,tcd->btd", chunks, self.token_proj_W) + self.token_proj_b


class GroupPoolingTokenizer(S2DRecTokenizer):
    """Deterministic field-group pooling followed by per-token projection."""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        pooling: str = "mean",
        seed: int = 42,
        group_strategy: str = "seeded_round_robin",
        projection: str = "linear",
        field_cardinalities: Optional[list[int]] = None,
        field_names: Optional[list[str]] = None,
        field_groups: Optional[list[list[int]]] = None,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if pooling not in ("mean", "sum"):
            raise ValueError(f"group_pooling must be 'mean' or 'sum', got {pooling}")
        if projection not in ("linear", "identity"):
            raise ValueError(f"group_projection must be 'linear' or 'identity', got {projection}")
        if projection == "identity" and embedding_dim != d_model:
            raise ValueError(
                "group_projection='identity' requires embedding_dim == d_model, "
                f"got embedding_dim={embedding_dim}, d_model={d_model}"
            )
        self.pooling = pooling
        self.group_strategy = group_strategy
        self.projection = projection

        groups = self._build_groups(
            num_fields=num_fields,
            num_tokens=num_tokens,
            seed=seed,
            group_strategy=group_strategy,
            field_cardinalities=field_cardinalities,
            field_names=field_names,
            field_groups=field_groups,
        )

        max_group_size = max(len(g) for g in groups)
        group_indices = torch.zeros(num_tokens, max_group_size, dtype=torch.long)
        group_mask = torch.zeros(num_tokens, max_group_size, dtype=torch.bool)
        for token_idx, feature_indices in enumerate(groups):
            if feature_indices:
                group_indices[token_idx, : len(feature_indices)] = torch.tensor(feature_indices)
                group_mask[token_idx, : len(feature_indices)] = True

        self.register_buffer("group_indices", group_indices)
        self.register_buffer("group_mask", group_mask)
        self.token_proj = nn.Identity() if projection == "identity" else nn.Linear(embedding_dim, d_model)

    @staticmethod
    def _validate_groups(groups: list[list[int]], num_fields: int, num_tokens: int) -> list[list[int]]:
        if len(groups) != num_tokens:
            raise ValueError(f"field_groups must contain {num_tokens} groups, got {len(groups)}")
        flat = [int(i) for group in groups for i in group]
        if len(flat) != num_fields:
            raise ValueError(f"field_groups must cover {num_fields} fields exactly once, got {len(flat)}")
        if sorted(flat) != list(range(num_fields)):
            raise ValueError("field_groups must be a permutation of field indices [0, num_fields)")
        return [[int(i) for i in group] for group in groups]

    @classmethod
    def _build_groups(
        cls,
        num_fields: int,
        num_tokens: int,
        seed: int,
        group_strategy: str,
        field_cardinalities: Optional[list[int]],
        field_names: Optional[list[str]],
        field_groups: Optional[list[list[int]]],
    ) -> list[list[int]]:
        if field_groups is not None:
            return cls._validate_groups(field_groups, num_fields, num_tokens)

        if group_strategy in ("seeded_round_robin", "random_round_robin"):
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            perm = torch.randperm(num_fields, generator=gen).tolist()
            groups = [[] for _ in range(num_tokens)]
            for idx, feat_idx in enumerate(perm):
                groups[idx % num_tokens].append(feat_idx)
            return groups

        if group_strategy == "contiguous":
            groups = []
            start = 0
            for token_idx in range(num_tokens):
                remaining_fields = num_fields - start
                remaining_tokens = num_tokens - token_idx
                size = (remaining_fields + remaining_tokens - 1) // remaining_tokens
                groups.append(list(range(start, start + size)))
                start += size
            return cls._validate_groups(groups, num_fields, num_tokens)

        if group_strategy == "cardinality_balanced":
            if field_cardinalities is None or len(field_cardinalities) != num_fields:
                raise ValueError("cardinality_balanced requires field_cardinalities for every field")
            order = sorted(range(num_fields), key=lambda i: (int(field_cardinalities[i]), i), reverse=True)
            groups = [[] for _ in range(num_tokens)]
            for idx, feat_idx in enumerate(order):
                groups[idx % num_tokens].append(feat_idx)
            return cls._validate_groups(groups, num_fields, num_tokens)

        if group_strategy == "dense_sparse_balanced":
            if field_names is None or len(field_names) != num_fields:
                raise ValueError("dense_sparse_balanced requires field_names for every field")
            dense_indices = [
                i for i, name in enumerate(field_names)
                if str(name).startswith("I") or str(name).endswith("_bucket")
            ]
            sparse_indices = [i for i in range(num_fields) if i not in set(dense_indices)]
            if len(dense_indices) != num_tokens:
                raise ValueError(
                    "dense_sparse_balanced expects exactly one dense/bucket field per token; "
                    f"got dense={len(dense_indices)}, tokens={num_tokens}"
                )
            groups = [[dense_indices[token_idx]] for token_idx in range(num_tokens)]
            for idx, feat_idx in enumerate(sparse_indices):
                groups[idx % num_tokens].append(feat_idx)
            return cls._validate_groups(groups, num_fields, num_tokens)

        raise ValueError(
            "group_strategy must be one of "
            "'seeded_round_robin', 'contiguous', 'cardinality_balanced', 'dense_sparse_balanced'"
        )

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        grouped = field_emb[:, self.group_indices, :]
        mask = self.group_mask.view(1, self.num_tokens, -1, 1).to(dtype=field_emb.dtype)
        pooled = (grouped * mask).sum(dim=2)
        if self.pooling == "mean":
            denom = mask.sum(dim=2).clamp_min(1.0)
            pooled = pooled / denom
        return self.token_proj(pooled)


class PerFieldProjectionTokenizer(S2DRecTokenizer):
    """Project fields first, then compress/expand field tokens to the target token count."""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        proj_mode: str = "split",
        seed: int = 42,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if proj_mode not in ("shared", "split"):
            raise ValueError(f"per_field_proj_mode must be 'shared' or 'split', got {proj_mode}")
        self.proj_mode = proj_mode
        if proj_mode == "shared":
            self.field_proj = nn.Linear(embedding_dim, d_model)
        else:
            self.field_proj_W = nn.Parameter(torch.empty(num_fields, embedding_dim, d_model))
            self.field_proj_b = nn.Parameter(torch.zeros(num_fields, d_model))
            nn.init.xavier_uniform_(self.field_proj_W)

        if num_tokens < num_fields:
            self.pooler = GroupPoolingTokenizer(num_fields, d_model, num_tokens, d_model, seed=seed)
        elif num_tokens > num_fields:
            self.extra_tokens = nn.Parameter(torch.empty(num_tokens - num_fields, d_model))
            nn.init.normal_(self.extra_tokens, std=1.0 / math.sqrt(d_model))
        else:
            self.pooler = None
            self.extra_tokens = None

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        if self.proj_mode == "shared":
            tokens = self.field_proj(field_emb)
        else:
            tokens = torch.einsum("bne,ned->bnd", field_emb, self.field_proj_W) + self.field_proj_b

        if self.num_tokens < self.num_fields:
            return self.pooler(tokens)
        if self.num_tokens > self.num_fields:
            extra = self.extra_tokens.unsqueeze(0).expand(tokens.size(0), -1, -1)
            return torch.cat([tokens, extra], dim=1)
        return tokens


class PureDiscretePerFieldTokenizer(S2DRecTokenizer):
    """Replace every continuous field embedding with hierarchical discrete codes.

    Raw ID embeddings only select field-specific codewords. The downstream
    projection receives exactly the sum of selected codebook vectors; no
    continuous ID residual is added back. Code 0 is fixed at zero and
    terminates all deeper levels for that field.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        *,
        num_levels: int = 3,
        codebook_size: int = 64,
        temperature: float = 0.2,
        code_init_scale: float = 0.05,
        zero_margin: float = 0.15,
        quantization_loss_weight: float = 0.2,
        commitment_weight: float = 0.25,
        proj_mode: str = "split",
        seed: int = 42,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if num_tokens != num_fields:
            raise ValueError(
                "pure_discrete_per_field requires num_tokens == num_fields, "
                f"got num_tokens={num_tokens}, num_fields={num_fields}"
            )
        if num_levels <= 0:
            raise ValueError(
                f"pure_discrete_num_levels must be positive, got {num_levels}"
            )
        if codebook_size < 2:
            raise ValueError(
                "pure_discrete_codebook_size must include zero and at least one "
                f"nonzero code, got {codebook_size}"
            )
        if temperature <= 0:
            raise ValueError(
                f"pure_discrete_temperature must be positive, got {temperature}"
            )
        if code_init_scale <= 0:
            raise ValueError(
                f"pure_discrete_code_init_scale must be positive, got {code_init_scale}"
            )
        if zero_margin < 0:
            raise ValueError(
                f"pure_discrete_zero_margin must be non-negative, got {zero_margin}"
            )
        if quantization_loss_weight < 0 or commitment_weight < 0:
            raise ValueError(
                "pure discrete quantization loss weights must be non-negative"
            )

        self.num_levels = int(num_levels)
        self.codebook_size = int(codebook_size)
        self.temperature = float(temperature)
        self.code_init_scale = float(code_init_scale)
        self.zero_margin = float(zero_margin)
        self.quantization_loss_weight = float(quantization_loss_weight)
        self.commitment_weight = float(commitment_weight)
        self.proj_mode = proj_mode

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.nonzero_codebooks = nn.ParameterList()
        for level in range(self.num_levels):
            scale = self.code_init_scale / math.sqrt(level + 1)
            codebook = torch.randn(
                num_fields,
                self.codebook_size - 1,
                embedding_dim,
                generator=generator,
            )
            self.nonzero_codebooks.append(nn.Parameter(codebook * scale))

        self.projector = PerFieldProjectionTokenizer(
            num_fields,
            embedding_dim,
            num_tokens,
            d_model,
            proj_mode=proj_mode,
            seed=seed,
        )

    def codebook(self, level: int) -> torch.Tensor:
        nonzero = self.nonzero_codebooks[level]
        if self.zero_margin > 0:
            minimum_norm = self.zero_margin / math.sqrt(level + 1)
            norms = torch.linalg.vector_norm(nonzero, dim=-1, keepdim=True)
            scale = (minimum_norm / norms.clamp_min(1e-12)).clamp_min(1.0)
            nonzero = nonzero * scale
        zero = nonzero.new_zeros(self.num_fields, 1, self.embedding_dim)
        return torch.cat((zero, nonzero), dim=1)

    @staticmethod
    def _squared_distance(
        values: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        return (
            values.square().sum(dim=-1, keepdim=True)
            + codebook.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * torch.einsum("bfe,fke->bfk", values, codebook)
        ).clamp_min(0.0)

    def quantize_fields(
        self,
        field_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if field_emb.ndim != 3 or field_emb.shape[1:] != (
            self.num_fields,
            self.embedding_dim,
        ):
            raise ValueError(
                "pure discrete input must have shape "
                f"(B, {self.num_fields}, {self.embedding_dim}), "
                f"got {tuple(field_emb.shape)}"
            )

        remaining_error = field_emb.float()
        quantized = torch.zeros_like(remaining_error)
        active = torch.ones(
            field_emb.size(0),
            self.num_fields,
            dtype=torch.bool,
            device=field_emb.device,
        )
        field_index = torch.arange(
            self.num_fields, device=field_emb.device
        ).unsqueeze(0).expand(field_emb.size(0), -1)
        codes = []

        for level in range(self.num_levels):
            codebook = self.codebook(level).float()
            distances = self._squared_distance(remaining_error, codebook)
            probabilities = torch.softmax(
                -distances / self.temperature,
                dim=-1,
            )
            indices = distances.argmin(dim=-1)
            indices = torch.where(active, indices, torch.zeros_like(indices))
            hard_value = codebook[field_index, indices]
            soft_value = torch.einsum(
                "bfk,fke->bfe", probabilities, codebook
            )
            level_value = soft_value + (hard_value - soft_value).detach()
            level_value = level_value * active.unsqueeze(-1)

            quantized = quantized + level_value
            remaining_error = remaining_error - level_value
            codes.append(indices)
            active = active & indices.ne(0)

        code_tensor = torch.stack(codes, dim=-1)
        commitment = F.mse_loss(field_emb.float(), quantized.detach())
        codebook_fit = F.mse_loss(field_emb.detach().float(), quantized)
        quantization_loss = (
            self.commitment_weight * commitment + codebook_fit
        )
        return quantized.to(dtype=field_emb.dtype), code_tensor, quantization_loss

    def reconstruct_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-2:] != (self.num_fields, self.num_levels):
            raise ValueError(
                "pure discrete codes must have shape "
                f"(B, {self.num_fields}, {self.num_levels}), "
                f"got {tuple(codes.shape)}"
            )
        field_index = torch.arange(
            self.num_fields, device=codes.device
        ).unsqueeze(0).expand(codes.size(0), -1)
        reconstructed = self.codebook(0)[field_index, codes[..., 0]]
        for level in range(1, self.num_levels):
            reconstructed = reconstructed + self.codebook(level)[
                field_index, codes[..., level]
            ]
        return reconstructed

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        discrete_fields, codes, quantization_loss = self.quantize_fields(field_emb)
        self.aux_loss = (
            self.quantization_loss_weight * quantization_loss
            if self.training and self.quantization_loss_weight > 0
            else None
        )

        with torch.no_grad():
            zero_tuple = codes.eq(0).all(dim=-1)
            active_depth = codes.ne(0).sum(dim=-1).float()
            error = torch.linalg.vector_norm(
                field_emb.detach().float() - discrete_fields.detach().float(),
                dim=-1,
            )
            diagnostics = {
                "discrete_zero_tuple_fraction": zero_tuple.float().mean(),
                "discrete_active_depth_mean": active_depth.mean(),
                "discrete_active_depth_max": active_depth.max(),
                "discrete_quantization_error_mean": error.mean(),
                "discrete_quantization_loss": quantization_loss.detach(),
            }
            for level in range(self.num_levels):
                diagnostics[f"discrete_level_{level + 1}_zero_fraction"] = (
                    codes[..., level].eq(0).float().mean()
                )
            self._diagnostics = diagnostics

        return self.projector(discrete_fields)


class SelectiveSingleLevelDiscreteTokenizer(S2DRecTokenizer):
    """Quantize only selected high-cardinality fields with one large codebook.

    Unselected fields keep their continuous embeddings. A selected non-missing
    field is replaced completely by one codebook vector, with no continuous
    residual. Exact nearest-code search is chunked for bounded memory; gradients
    use a local soft straight-through estimator over the nearest candidates.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        *,
        selected_field_indices: list[int],
        codebook_size: int = 10000,
        candidate_top_m: int = 32,
        distance_chunk_size: int = 2048,
        temperature: float = 0.2,
        temperature_start: Optional[float] = None,
        code_init_scale: float = 0.05,
        quantization_loss_weight: float = 0.2,
        commitment_weight: float = 0.25,
        warmup_fraction: float = 0.0,
        transition_fraction: float = 0.0,
        codebook_init: str = "random",
        proj_mode: str = "split",
        seed: int = 42,
        audit_enabled: bool = False,
        audit_assignment_every: int = 50,
        audit_gradient_every: int = 250,
        audit_anchor_observations: int = 65536,
        assignment_mode: str = "nearest_ste",
        gumbel_scale: float = 1.0,
        usage_ema_decay: float = 0.99,
        hot_threshold_ratio: float = 1.5,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if num_tokens != num_fields:
            raise ValueError(
                "selective_single_level_discrete requires num_tokens == num_fields, "
                f"got num_tokens={num_tokens}, num_fields={num_fields}"
            )
        selected = [int(index) for index in selected_field_indices]
        if not selected:
            raise ValueError("selective_discrete_field_indices must not be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("selective_discrete_field_indices must be unique")
        if min(selected) < 0 or max(selected) >= num_fields:
            raise ValueError(
                "selective_discrete_field_indices must be within "
                f"[0, {num_fields}), got {selected}"
            )
        if codebook_size <= 1:
            raise ValueError(
                f"selective_discrete_codebook_size must be > 1, got {codebook_size}"
            )
        if candidate_top_m <= 0 or candidate_top_m > codebook_size:
            raise ValueError(
                "selective_discrete_candidate_top_m must be in "
                f"[1, {codebook_size}], got {candidate_top_m}"
            )
        if distance_chunk_size <= 0:
            raise ValueError(
                "selective_discrete_distance_chunk_size must be positive, "
                f"got {distance_chunk_size}"
            )
        if temperature <= 0:
            raise ValueError(
                f"selective_discrete_temperature must be positive, got {temperature}"
            )
        if temperature_start is not None and temperature_start <= 0:
            raise ValueError(
                "selective_discrete_temperature_start must be positive, "
                f"got {temperature_start}"
            )
        if code_init_scale <= 0:
            raise ValueError(
                "selective_discrete_code_init_scale must be positive, "
                f"got {code_init_scale}"
            )
        if quantization_loss_weight < 0 or commitment_weight < 0:
            raise ValueError("selective discrete loss weights must be non-negative")
        if not 0.0 <= warmup_fraction < 1.0:
            raise ValueError(
                "selective_discrete_warmup_fraction must be in [0, 1), "
                f"got {warmup_fraction}"
            )
        if transition_fraction < 0.0 or warmup_fraction + transition_fraction > 1.0:
            raise ValueError(
                "selective_discrete_transition_fraction must be non-negative and "
                "warmup_fraction + transition_fraction must be <= 1, got "
                f"{warmup_fraction} + {transition_fraction}"
            )
        if codebook_init not in (
            "random",
            "warmup_samples",
            "warmup_unique_ids",
        ):
            raise ValueError(
                "selective_discrete_codebook_init must be 'random', "
                "'warmup_samples', or 'warmup_unique_ids', got "
                f"{codebook_init!r}"
            )
        if codebook_init in ("warmup_samples", "warmup_unique_ids") and (
            warmup_fraction <= 0.0
        ):
            raise ValueError(
                f"selective_discrete_codebook_init={codebook_init!r} requires a "
                "positive warmup_fraction"
            )
        if audit_assignment_every <= 0:
            raise ValueError(
                "selective_discrete_audit_assignment_every must be positive, "
                f"got {audit_assignment_every}"
            )
        if audit_gradient_every <= 0:
            raise ValueError(
                "selective_discrete_audit_gradient_every must be positive, "
                f"got {audit_gradient_every}"
            )
        if audit_anchor_observations <= 0:
            raise ValueError(
                "selective_discrete_audit_anchor_observations must be positive, "
                f"got {audit_anchor_observations}"
            )
        if assignment_mode not in ("nearest_ste", "hotcode_gumbel"):
            raise ValueError(
                "selective_discrete_assignment_mode must be 'nearest_ste' or "
                f"'hotcode_gumbel', got {assignment_mode!r}"
            )
        if gumbel_scale < 0.0:
            raise ValueError(
                "selective_discrete_gumbel_scale must be non-negative, "
                f"got {gumbel_scale}"
            )
        if not 0.0 <= usage_ema_decay < 1.0:
            raise ValueError(
                "selective_discrete_usage_ema_decay must be in [0, 1), "
                f"got {usage_ema_decay}"
            )
        if hot_threshold_ratio <= 0.0:
            raise ValueError(
                "selective_discrete_hot_threshold_ratio must be positive, "
                f"got {hot_threshold_ratio}"
            )

        self.selected_field_indices = tuple(selected)
        self.codebook_size = int(codebook_size)
        self.candidate_top_m = int(candidate_top_m)
        self.distance_chunk_size = int(distance_chunk_size)
        self.temperature = float(temperature)
        self.temperature_start = float(
            temperature if temperature_start is None else temperature_start
        )
        self.current_temperature = self.temperature_start
        self.code_init_scale = float(code_init_scale)
        self.quantization_loss_weight = float(quantization_loss_weight)
        self.commitment_weight = float(commitment_weight)
        self.warmup_fraction = float(warmup_fraction)
        self.transition_fraction = float(transition_fraction)
        self.codebook_init = codebook_init
        self.quantization_mix = 1.0 if warmup_fraction == 0.0 else 0.0
        self._pending_warmup_init = False
        self._warmup_initialized = codebook_init == "random"
        self._warmup_buffer_write = [0 for _ in self.selected_field_indices]
        self._warmup_buffer_count = [0 for _ in self.selected_field_indices]
        self.proj_mode = proj_mode
        self.seed = int(seed)
        self.gumbel_seed = self.seed ^ 0x5DEECE66D
        self._gumbel_generators: dict[str, torch.Generator] = {}
        self.audit_enabled = bool(audit_enabled)
        self.audit_assignment_every = int(audit_assignment_every)
        self.audit_gradient_every = int(audit_gradient_every)
        self.audit_anchor_observations = int(audit_anchor_observations)
        self.assignment_mode = str(assignment_mode)
        self.gumbel_scale = float(gumbel_scale)
        self.usage_ema_decay = float(usage_ema_decay)
        self.hot_threshold_ratio = float(hot_threshold_ratio)
        self._audit_current_step = 0
        self._audit_phase_names = ("warmup", "soft_transition", "hard")
        self._audit_anchor_observation_count = [
            0 for _ in self.selected_field_indices
        ]
        self._audit_anchor_count = [0 for _ in self.selected_field_indices]
        self._audit_anchor_frozen = [False for _ in self.selected_field_indices]
        self._audit_last_active_ids: list[Optional[torch.Tensor]] = [
            None for _ in self.selected_field_indices
        ]
        self.register_buffer(
            "_code_usage_ema",
            torch.full(
                (len(self.selected_field_indices), self.codebook_size),
                1.0 / self.codebook_size,
                dtype=torch.float32,
            ),
        )

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.codebooks = nn.ParameterList()
        for _ in self.selected_field_indices:
            codebook = torch.empty(
                self.codebook_size,
                embedding_dim,
            )
            codebook.uniform_(
                -self.code_init_scale,
                self.code_init_scale,
                generator=generator,
            )
            self.codebooks.append(nn.Parameter(codebook))
        if self.audit_enabled:
            num_selected = len(self.selected_field_indices)
            num_phases = len(self._audit_phase_names)
            self.register_buffer(
                "_audit_assignment_counts",
                torch.zeros(
                    num_phases,
                    num_selected,
                    self.codebook_size,
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_audit_assignment_samples",
                torch.zeros(num_phases, num_selected, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_audit_soft_metric_sums",
                torch.zeros(num_phases, num_selected, 4, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_audit_soft_metric_counts",
                torch.zeros(num_phases, num_selected, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_audit_anchor_ids",
                torch.full(
                    (num_selected, self.audit_anchor_observations),
                    -1,
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_audit_anchor_last_codes",
                torch.full(
                    (num_selected, self.audit_anchor_observations),
                    -1,
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_audit_churn_comparisons",
                torch.zeros(num_phases, num_selected, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_audit_churn_changes",
                torch.zeros(num_phases, num_selected, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_audit_gradient_sums",
                torch.zeros(num_phases, num_selected, 5, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_audit_gradient_maxima",
                torch.zeros(num_phases, num_selected, 5, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_audit_gradient_counts",
                torch.zeros(num_phases, num_selected, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_audit_exploration_sums",
                torch.zeros(num_phases, num_selected, 4, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_audit_exploration_counts",
                torch.zeros(num_phases, num_selected, dtype=torch.long),
                persistent=False,
            )
        if self.codebook_init in ("warmup_samples", "warmup_unique_ids"):
            for selected_pos in range(len(self.selected_field_indices)):
                self.register_buffer(
                    f"_warmup_buffer_{selected_pos}",
                    torch.empty(self.codebook_size, embedding_dim),
                    persistent=False,
                )
                if self.codebook_init == "warmup_unique_ids":
                    self.register_buffer(
                        f"_warmup_id_buffer_{selected_pos}",
                        torch.full((self.codebook_size,), -1, dtype=torch.long),
                        persistent=False,
                    )

        self.projector = PerFieldProjectionTokenizer(
            num_fields,
            embedding_dim,
            num_tokens,
            d_model,
            proj_mode=proj_mode,
            seed=seed,
        )

    def set_progress(self, current_step: int, total_steps: int) -> None:
        """Update the train-only continuous-to-discrete schedule."""
        self._audit_current_step = int(current_step)
        denominator = max(int(total_steps) - 1, 1)
        progress = min(max(float(current_step) / denominator, 0.0), 1.0)
        if progress < self.warmup_fraction:
            mix = 0.0
        elif self.transition_fraction > 0.0 and progress < (
            self.warmup_fraction + self.transition_fraction
        ):
            mix = (progress - self.warmup_fraction) / self.transition_fraction
        else:
            mix = 1.0
        self.quantization_mix = min(max(mix, 0.0), 1.0)
        self.current_temperature = (
            self.temperature_start
            + self.quantization_mix * (self.temperature - self.temperature_start)
        )
        if (
            self.codebook_init in ("warmup_samples", "warmup_unique_ids")
            and not self._warmup_initialized
            and progress >= self.warmup_fraction
        ):
            self._pending_warmup_init = True

    def _audit_phase_index(self) -> int:
        if self.quantization_mix <= 0.0:
            return 0
        if self.quantization_mix < 1.0:
            return 1
        return 2

    def _should_audit_assignments(self) -> bool:
        return (
            self.audit_enabled
            and self.training
            and self._audit_current_step % self.audit_assignment_every == 0
        )

    @torch.no_grad()
    def _collect_audit_anchor_observations(
        self,
        selected_pos: int,
        ids: Optional[torch.Tensor],
    ) -> None:
        if (
            not self.audit_enabled
            or self._audit_anchor_frozen[selected_pos]
            or ids is None
        ):
            return
        ids = ids.detach().reshape(-1).to(
            device=self._audit_anchor_ids.device,
            dtype=torch.long,
        )
        ids = ids[ids.gt(0)]
        if ids.numel() == 0:
            return
        count = self._audit_anchor_observation_count[selected_pos]
        available = self.audit_anchor_observations - count
        if available <= 0:
            return
        take = min(available, ids.numel())
        self._audit_anchor_ids[selected_pos, count : count + take].copy_(ids[:take])
        self._audit_anchor_observation_count[selected_pos] = count + take

    @torch.no_grad()
    def _freeze_audit_anchors(
        self,
        selected_pos: int,
        fallback_ids: Optional[torch.Tensor],
    ) -> None:
        if not self.audit_enabled or self._audit_anchor_frozen[selected_pos]:
            return
        self._collect_audit_anchor_observations(selected_pos, fallback_ids)
        observed = self._audit_anchor_observation_count[selected_pos]
        if observed:
            unique_ids = torch.unique(
                self._audit_anchor_ids[selected_pos, :observed],
                sorted=True,
            )
            count = min(unique_ids.numel(), self.audit_anchor_observations)
            self._audit_anchor_ids[selected_pos].fill_(-1)
            self._audit_anchor_ids[selected_pos, :count].copy_(unique_ids[:count])
            self._audit_anchor_count[selected_pos] = int(count)
        self._audit_anchor_frozen[selected_pos] = True

    def _sample_gumbel(self, logits: torch.Tensor) -> torch.Tensor:
        device_key = str(logits.device)
        generator = self._gumbel_generators.get(device_key)
        if generator is None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(self.gumbel_seed)
            self._gumbel_generators[device_key] = generator
        uniform = torch.rand(
            logits.shape,
            dtype=logits.dtype,
            device=logits.device,
            generator=generator,
        ).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    @torch.no_grad()
    def _update_code_usage_ema(
        self,
        selected_pos: int,
        deterministic_indices: torch.Tensor,
    ) -> None:
        counts = torch.bincount(
            deterministic_indices,
            minlength=self.codebook_size,
        ).to(dtype=self._code_usage_ema.dtype)
        current = counts / counts.sum().clamp_min(1.0)
        self._code_usage_ema[selected_pos].mul_(self.usage_ema_decay).add_(
            current,
            alpha=1.0 - self.usage_ema_decay,
        )

    def _assignment_distribution(
        self,
        *,
        selected_pos: int,
        candidate_indices: torch.Tensor,
        candidate_distances: torch.Tensor,
        effective_mix: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = -candidate_distances
        deterministic_positions = torch.zeros(
            candidate_indices.size(0),
            dtype=torch.long,
            device=candidate_indices.device,
        )
        exploration_mask = torch.zeros_like(deterministic_positions, dtype=torch.bool)
        changed = torch.zeros_like(exploration_mask)
        use_exploration = bool(
            self.training
            and self.assignment_mode == "hotcode_gumbel"
            and 0.0 < effective_mix < 1.0
        )
        if not use_exploration:
            probabilities = torch.softmax(
                logits / self.current_temperature,
                dim=-1,
            )
            return probabilities, deterministic_positions, exploration_mask, changed

        deterministic_indices = candidate_indices[:, 0]
        self._update_code_usage_ema(selected_pos, deterministic_indices)
        assigned_usage = self._code_usage_ema[selected_pos].index_select(
            0, deterministic_indices
        )
        hot_threshold = self.hot_threshold_ratio / self.codebook_size
        exploration_mask = assigned_usage.gt(hot_threshold)

        deterministic_probabilities = torch.softmax(
            logits / self.current_temperature,
            dim=-1,
        )
        noisy_probabilities = torch.softmax(
            (logits + self.gumbel_scale * self._sample_gumbel(logits))
            / self.current_temperature,
            dim=-1,
        )
        probabilities = torch.where(
            exploration_mask.unsqueeze(-1),
            noisy_probabilities,
            deterministic_probabilities,
        )
        sampled_positions = noisy_probabilities.argmax(dim=-1)
        hard_positions = torch.where(
            exploration_mask,
            sampled_positions,
            deterministic_positions,
        )
        changed = exploration_mask & hard_positions.ne(0)
        return probabilities, hard_positions, exploration_mask, changed

    @torch.no_grad()
    def _record_assignment_audit(
        self,
        *,
        selected_pos: int,
        active_ids: Optional[torch.Tensor],
        hard_indices: torch.Tensor,
        probabilities: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_centers: torch.Tensor,
        hard_values: torch.Tensor,
        soft_values: torch.Tensor,
        hard_candidate_positions: torch.Tensor,
        exploration_mask: torch.Tensor,
        exploration_changed: torch.Tensor,
    ) -> None:
        if not self._should_audit_assignments():
            return
        phase = self._audit_phase_index()
        counts = torch.bincount(hard_indices, minlength=self.codebook_size)
        self._audit_assignment_counts[phase, selected_pos].add_(counts)
        self._audit_assignment_samples[phase, selected_pos].add_(hard_indices.numel())

        soft_argmax_agreement = probabilities.argmax(dim=-1).eq(
            hard_candidate_positions
        ).float().sum()
        proxy_distances = (
            soft_values.unsqueeze(1) - candidate_centers.detach()
        ).square().sum(dim=-1)
        proxy_codes = candidate_indices.gather(
            1, proxy_distances.argmin(dim=-1, keepdim=True)
        ).squeeze(1)
        proxy_agreement = proxy_codes.eq(hard_indices).float().sum()
        hard_probability = probabilities.gather(
            1, hard_candidate_positions.unsqueeze(-1)
        ).sum()
        soft_hard_gap = torch.linalg.vector_norm(
            soft_values.detach() - hard_values.detach(), dim=-1
        ).sum()
        metrics = torch.stack(
            (
                soft_argmax_agreement,
                proxy_agreement,
                hard_probability,
                soft_hard_gap,
            )
        ).to(dtype=torch.float64)
        self._audit_soft_metric_sums[phase, selected_pos].add_(metrics)
        self._audit_soft_metric_counts[phase, selected_pos].add_(hard_indices.numel())

        usage_ema = self._code_usage_ema[selected_pos]
        deterministic_indices = candidate_indices[:, 0]
        assigned_usage_mean = usage_ema.index_select(
            0, deterministic_indices
        ).mean()
        exploration_metrics = torch.stack(
            (
                exploration_mask.float().sum(),
                exploration_changed.float().sum(),
                usage_ema.max() * hard_indices.numel(),
                assigned_usage_mean * hard_indices.numel(),
            )
        ).to(dtype=torch.float64)
        self._audit_exploration_sums[phase, selected_pos].add_(
            exploration_metrics
        )
        self._audit_exploration_counts[phase, selected_pos].add_(hard_indices.numel())

        self._freeze_audit_anchors(selected_pos, active_ids)
        anchor_count = self._audit_anchor_count[selected_pos]
        if active_ids is None or anchor_count == 0:
            return
        active_ids = active_ids.detach().reshape(-1).to(dtype=torch.long)
        order = torch.argsort(active_ids, stable=True)
        sorted_ids = active_ids[order]
        sorted_codes = hard_indices[order]
        unique_ids, duplicate_counts = torch.unique_consecutive(
            sorted_ids, return_counts=True
        )
        unique_codes = sorted_codes[duplicate_counts.cumsum(dim=0) - 1]
        anchors = self._audit_anchor_ids[selected_pos, :anchor_count]
        positions = torch.searchsorted(anchors, unique_ids)
        in_bounds = positions.lt(anchor_count)
        matched = torch.zeros_like(in_bounds)
        if in_bounds.any():
            matched[in_bounds] = anchors[positions[in_bounds]].eq(unique_ids[in_bounds])
        if not matched.any():
            return
        positions = positions[matched]
        unique_codes = unique_codes[matched]
        previous = self._audit_anchor_last_codes[selected_pos, positions]
        comparable = previous.ge(0)
        self._audit_churn_comparisons[phase, selected_pos].add_(comparable.sum())
        self._audit_churn_changes[phase, selected_pos].add_(
            (comparable & previous.ne(unique_codes)).sum()
        )
        self._audit_anchor_last_codes[selected_pos, positions] = unique_codes

    @torch.no_grad()
    def record_gradient_audit(
        self,
        selected_pos: int,
        *,
        raw_grad_l2: torch.Tensor,
        raw_grad_rms: torch.Tensor,
        codebook_grad_l2: torch.Tensor,
        codebook_grad_rms: torch.Tensor,
        active_rows: int,
    ) -> None:
        if not self.audit_enabled:
            return
        phase = self._audit_phase_index()
        values = torch.stack(
            (
                raw_grad_l2.detach(),
                raw_grad_rms.detach(),
                codebook_grad_l2.detach(),
                codebook_grad_rms.detach(),
                raw_grad_l2.new_tensor(float(active_rows)),
            )
        ).to(dtype=torch.float64)
        self._audit_gradient_sums[phase, selected_pos].add_(values)
        self._audit_gradient_maxima[phase, selected_pos] = torch.maximum(
            self._audit_gradient_maxima[phase, selected_pos], values
        )
        self._audit_gradient_counts[phase, selected_pos].add_(1)

    @staticmethod
    def _assignment_summary(counts: torch.Tensor) -> dict[str, float | int | None]:
        counts = counts.detach().to(dtype=torch.float64, device="cpu")
        total = float(counts.sum().item())
        if total <= 0.0:
            return {
                "sampled_occurrences": 0,
                "active_codes": 0,
                "codebook_utilization": 0.0,
                "assignment_entropy_nats": None,
                "normalized_assignment_entropy": None,
                "assignment_perplexity": None,
                "effective_code_fraction": None,
                "max_center_share": None,
                "gini": None,
            }
        probabilities = counts / total
        positive = probabilities.gt(0)
        entropy = float(
            -(probabilities[positive] * probabilities[positive].log()).sum().item()
        )
        perplexity = math.exp(entropy)
        sorted_counts = counts.sort().values
        n = sorted_counts.numel()
        indices = torch.arange(1, n + 1, dtype=torch.float64)
        gini = float(
            ((2.0 * indices - n - 1.0) * sorted_counts).sum().item()
            / (n * total)
        )
        active_codes = int(positive.sum().item())
        return {
            "sampled_occurrences": int(total),
            "active_codes": active_codes,
            "codebook_utilization": active_codes / n,
            "assignment_entropy_nats": entropy,
            "normalized_assignment_entropy": entropy / math.log(n),
            "assignment_perplexity": perplexity,
            "effective_code_fraction": perplexity / n,
            "max_center_share": float(probabilities.max().item()),
            "gini": gini,
        }

    def get_audit_report(self) -> Optional[dict]:
        if not self.audit_enabled:
            return None
        soft_names = (
            "soft_argmax_hard_agreement",
            "soft_proxy_nearest_candidate_hard_agreement",
            "hard_code_probability_mean",
            "soft_hard_center_gap_mean",
        )
        gradient_names = (
            "raw_active_grad_l2",
            "raw_active_grad_rms",
            "codebook_grad_l2",
            "codebook_grad_rms",
            "active_raw_rows",
        )
        fields = []
        for selected_pos, field_index in enumerate(self.selected_field_indices):
            phase_reports = {}
            for phase, phase_name in enumerate(self._audit_phase_names):
                assignment = self._assignment_summary(
                    self._audit_assignment_counts[phase, selected_pos]
                )
                soft_count = int(
                    self._audit_soft_metric_counts[phase, selected_pos].item()
                )
                assignment["soft_hard"] = {
                    name: (
                        float(
                            self._audit_soft_metric_sums[
                                phase, selected_pos, metric_index
                            ].item()
                            / soft_count
                        )
                        if soft_count
                        else None
                    )
                    for metric_index, name in enumerate(soft_names)
                }
                comparisons = int(
                    self._audit_churn_comparisons[phase, selected_pos].item()
                )
                changes = int(self._audit_churn_changes[phase, selected_pos].item())
                assignment["assignment_churn"] = {
                    "comparisons": comparisons,
                    "changes": changes,
                    "fraction": changes / comparisons if comparisons else None,
                }
                exploration_count = int(
                    self._audit_exploration_counts[phase, selected_pos].item()
                )
                hot_count = float(
                    self._audit_exploration_sums[phase, selected_pos, 0].item()
                )
                changed_count = float(
                    self._audit_exploration_sums[phase, selected_pos, 1].item()
                )
                assignment["exploration"] = {
                    "sampled_occurrences": exploration_count,
                    "gumbel_applied_fraction": (
                        hot_count / exploration_count if exploration_count else None
                    ),
                    "gumbel_changed_fraction": (
                        changed_count / exploration_count
                        if exploration_count
                        else None
                    ),
                    "gumbel_changed_given_applied": (
                        changed_count / hot_count if hot_count else None
                    ),
                    "usage_ema_max_mean": (
                        float(
                            self._audit_exploration_sums[
                                phase, selected_pos, 2
                            ].item()
                            / exploration_count
                        )
                        if exploration_count
                        else None
                    ),
                    "assigned_usage_ema_mean": (
                        float(
                            self._audit_exploration_sums[
                                phase, selected_pos, 3
                            ].item()
                            / exploration_count
                        )
                        if exploration_count
                        else None
                    ),
                }
                gradient_count = int(
                    self._audit_gradient_counts[phase, selected_pos].item()
                )
                assignment["gradient"] = {
                    "sampled_steps": gradient_count,
                    "mean": {
                        name: (
                            float(
                                self._audit_gradient_sums[
                                    phase, selected_pos, metric_index
                                ].item()
                                / gradient_count
                            )
                            if gradient_count
                            else None
                        )
                        for metric_index, name in enumerate(gradient_names)
                    },
                    "max": {
                        name: (
                            float(
                                self._audit_gradient_maxima[
                                    phase, selected_pos, metric_index
                                ].item()
                            )
                            if gradient_count
                            else None
                        )
                        for metric_index, name in enumerate(gradient_names)
                    },
                }
                phase_reports[phase_name] = assignment

            overall_counts = self._audit_assignment_counts[:, selected_pos].sum(dim=0)
            fields.append(
                {
                    "field_index": int(field_index),
                    "codebook_size": self.codebook_size,
                    "anchor_observations": self._audit_anchor_observation_count[
                        selected_pos
                    ],
                    "unique_frozen_anchors": self._audit_anchor_count[selected_pos],
                    "anchors_with_assignment": int(
                        self._audit_anchor_last_codes[selected_pos].ge(0).sum().item()
                    ),
                    "overall_assignment": self._assignment_summary(overall_counts),
                    "phases": phase_reports,
                }
            )
        return {
            "schema_version": 2,
            "assignment_sampling_every_steps": self.audit_assignment_every,
            "gradient_sampling_every_steps": self.audit_gradient_every,
            "anchor_observation_capacity": self.audit_anchor_observations,
            "phase_rule": {
                "warmup": "quantization_mix == 0",
                "soft_transition": "0 < quantization_mix < 1",
                "hard": "quantization_mix == 1",
            },
            "assignment_weighting": "sampled_training_occurrences",
            "gradient_point": (
                "after_backward_before_gradient_clipping_and_optimizer_weight_decay"
            ),
            "assignment_mode": self.assignment_mode,
            "train_forward_assignment": (
                "hotcode_gumbel_hard_forward_soft_backward"
                if self.assignment_mode == "hotcode_gumbel"
                else "hard_nearest_code_with_soft_straight_through_gradient"
            ),
            "gumbel_scope": "soft_transition_only",
            "gumbel_scale": self.gumbel_scale,
            "gumbel_seed": self.gumbel_seed,
            "gumbel_rng": "tokenizer_local_generator",
            "usage_ema_decay": self.usage_ema_decay,
            "hot_threshold_ratio": self.hot_threshold_ratio,
            "hot_threshold": self.hot_threshold_ratio / self.codebook_size,
            "eval_assignment": "hard_nearest_code",
            "fields": fields,
        }

    def _warmup_buffer(self, selected_pos: int) -> torch.Tensor:
        return getattr(self, f"_warmup_buffer_{selected_pos}")

    def _warmup_id_buffer(self, selected_pos: int) -> torch.Tensor:
        return getattr(self, f"_warmup_id_buffer_{selected_pos}")

    def _warmup_fill_fraction(self) -> float:
        if self.codebook_init not in ("warmup_samples", "warmup_unique_ids"):
            return 0.0
        return sum(self._warmup_buffer_count) / (
            len(self.selected_field_indices) * self.codebook_size
        )

    def _warmup_unique_id_count(self) -> float:
        if self.codebook_init != "warmup_unique_ids":
            return 0.0
        return sum(self._warmup_buffer_count) / len(self.selected_field_indices)

    def _warmup_id_hash(
        self,
        selected_pos: int,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministically rank IDs for a bounded uniform key sample."""
        modulus = 2_147_483_647
        increment = (self.seed + 104_729 * (selected_pos + 1)) % modulus
        return (
            ids.remainder(modulus) * 1_103_515_245 + increment
        ).remainder(modulus)

    @staticmethod
    def _deduplicate_latest(
        ids: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(ids, stable=True)
        sorted_ids = ids[order]
        sorted_values = values[order]
        unique_ids, counts = torch.unique_consecutive(
            sorted_ids,
            return_counts=True,
        )
        latest_indices = counts.cumsum(dim=0) - 1
        return unique_ids, sorted_values[latest_indices]

    @torch.no_grad()
    def _collect_warmup_values(
        self,
        selected_pos: int,
        values: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
    ) -> None:
        if self.codebook_init == "random" or values.numel() == 0:
            return
        buffer = self._warmup_buffer(selected_pos)
        values = values.detach().to(device=buffer.device, dtype=buffer.dtype)
        if self.codebook_init == "warmup_unique_ids":
            if ids is None:
                raise RuntimeError(
                    "warmup_unique_ids initialization requires raw sparse IDs"
                )
            ids = ids.detach().to(device=buffer.device, dtype=torch.long).reshape(-1)
            if ids.numel() != values.size(0):
                raise ValueError(
                    "warmup IDs and values must have the same batch size, got "
                    f"ids={ids.numel()}, values={values.size(0)}"
                )
            nonmissing = ids.gt(0)
            if not nonmissing.any():
                return
            ids = ids[nonmissing]
            values = values[nonmissing]
            id_buffer = self._warmup_id_buffer(selected_pos)
            count = self._warmup_buffer_count[selected_pos]
            candidate_ids = torch.cat((id_buffer[:count], ids))
            candidate_values = torch.cat((buffer[:count], values), dim=0)
            unique_ids, unique_values = self._deduplicate_latest(
                candidate_ids,
                candidate_values,
            )
            if unique_ids.numel() > self.codebook_size:
                hashes = self._warmup_id_hash(selected_pos, unique_ids)
                keep = torch.topk(
                    hashes,
                    self.codebook_size,
                    largest=False,
                    sorted=False,
                ).indices
                unique_ids = unique_ids[keep]
                unique_values = unique_values[keep]
            next_count = unique_ids.numel()
            id_buffer[:next_count].copy_(unique_ids)
            buffer[:next_count].copy_(unique_values)
            self._warmup_buffer_count[selected_pos] = next_count
            return

        if values.size(0) >= self.codebook_size:
            buffer.copy_(values[-self.codebook_size :])
            self._warmup_buffer_write[selected_pos] = 0
            self._warmup_buffer_count[selected_pos] = self.codebook_size
            return
        write = self._warmup_buffer_write[selected_pos]
        first = min(values.size(0), self.codebook_size - write)
        buffer[write : write + first].copy_(values[:first])
        remaining = values.size(0) - first
        if remaining > 0:
            buffer[:remaining].copy_(values[first:])
        self._warmup_buffer_write[selected_pos] = (
            write + values.size(0)
        ) % self.codebook_size
        self._warmup_buffer_count[selected_pos] = min(
            self.codebook_size,
            self._warmup_buffer_count[selected_pos] + values.size(0),
        )

    @torch.no_grad()
    def _initialize_codebooks_from_warmup(self) -> None:
        for selected_pos, codebook in enumerate(self.codebooks):
            count = self._warmup_buffer_count[selected_pos]
            if count <= 0:
                raise RuntimeError(
                    "warmup-sample codebook initialization has no observed values"
                )
            observed = self._warmup_buffer(selected_pos)[:count]
            if count < self.codebook_size:
                repeats = math.ceil(self.codebook_size / count)
                centers = observed.repeat(repeats, 1)[: self.codebook_size]
            else:
                centers = observed
            codebook.copy_(centers.to(device=codebook.device, dtype=codebook.dtype))
        self._warmup_initialized = True
        self._pending_warmup_init = False

    @staticmethod
    def _squared_distance(
        values: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        return (
            values.square().sum(dim=-1, keepdim=True)
            + codebook.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * values @ codebook.transpose(0, 1)
        ).clamp_min(0.0)

    def _nearest_candidates(
        self,
        values: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact global top-M indices without retaining full-K autograd."""
        batch_size = values.size(0)
        best_distances = values.new_full(
            (batch_size, self.candidate_top_m),
            float("inf"),
        )
        best_indices = torch.zeros(
            batch_size,
            self.candidate_top_m,
            dtype=torch.long,
            device=values.device,
        )
        with torch.no_grad():
            detached_values = values.detach().float()
            detached_codebook = codebook.detach().float()
            for start in range(0, self.codebook_size, self.distance_chunk_size):
                end = min(start + self.distance_chunk_size, self.codebook_size)
                chunk_distances = self._squared_distance(
                    detached_values,
                    detached_codebook[start:end],
                )
                chunk_indices = torch.arange(
                    start,
                    end,
                    device=values.device,
                ).unsqueeze(0).expand(batch_size, -1)
                merged_distances = torch.cat((best_distances, chunk_distances), dim=1)
                merged_indices = torch.cat((best_indices, chunk_indices), dim=1)
                best_distances, positions = torch.topk(
                    merged_distances,
                    self.candidate_top_m,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                best_indices = merged_indices.gather(1, positions)
        return best_indices

    def quantize_fields(
        self,
        field_emb: torch.Tensor,
        field_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if field_emb.ndim != 3 or field_emb.shape[1:] != (
            self.num_fields,
            self.embedding_dim,
        ):
            raise ValueError(
                "selective discrete input must have shape "
                f"(B, {self.num_fields}, {self.embedding_dim}), "
                f"got {tuple(field_emb.shape)}"
            )
        if field_ids is not None:
            if field_ids.ndim != 2 or field_ids.size(0) != field_emb.size(0):
                raise ValueError(
                    "selective discrete field_ids must have shape (B, F), got "
                    f"{tuple(field_ids.shape)}"
                )
            if max(self.selected_field_indices) >= field_ids.size(1):
                raise ValueError(
                    "selective discrete selected fields must be present in field_ids, "
                    f"got selected={self.selected_field_indices}, "
                    f"field_ids shape={tuple(field_ids.shape)}"
                )

        mixed_fields = field_emb.clone()
        codes = torch.full(
            (field_emb.size(0), len(self.selected_field_indices)),
            -1,
            dtype=torch.long,
            device=field_emb.device,
        )
        commitments = []
        codebook_fits = []
        candidate_entropies = []
        errors = []
        active_code_fractions = []
        top_code_fractions = []
        missing_fractions = []
        gumbel_applied_fractions = []
        gumbel_changed_fractions = []
        effective_mix = 1.0 if not self.training else self.quantization_mix

        if self.training and effective_mix <= 0.0:
            for selected_pos, field_index in enumerate(self.selected_field_indices):
                values = field_emb[:, field_index, :].float()
                missing = values.eq(0).all(dim=-1)
                selected_ids = (
                    field_ids[:, field_index][~missing]
                    if field_ids is not None
                    else None
                )
                if self.audit_enabled:
                    self._audit_last_active_ids[selected_pos] = (
                        selected_ids.detach() if selected_ids is not None else None
                    )
                    if self._should_audit_assignments():
                        self._collect_audit_anchor_observations(
                            selected_pos, selected_ids
                        )
                self._collect_warmup_values(
                    selected_pos,
                    values[~missing],
                    selected_ids,
                )
                missing_fractions.append(missing.float().mean())
            zero = sum(codebook.sum() for codebook in self.codebooks) * 0.0
            with torch.no_grad():
                self._diagnostics = {
                    "selective_discrete_selected_fields": zero.new_tensor(
                        len(self.selected_field_indices)
                    ),
                    "selective_discrete_codebook_size": zero.new_tensor(
                        self.codebook_size
                    ),
                    "selective_discrete_missing_fraction": torch.stack(
                        missing_fractions
                    ).mean(),
                    "selective_discrete_quantization_mix": zero.new_tensor(0.0),
                    "selective_discrete_current_temperature": zero.new_tensor(
                        self.current_temperature
                    ),
                    "selective_discrete_warmup_buffer_fraction": zero.new_tensor(
                        self._warmup_fill_fraction()
                    ),
                    "selective_discrete_warmup_unique_id_count": zero.new_tensor(
                        self._warmup_unique_id_count()
                    ),
                    "selective_discrete_warmup_initialized": zero.new_tensor(
                        float(self._warmup_initialized)
                    ),
                    "selective_discrete_quantization_loss": zero.detach(),
                    "selective_discrete_gumbel_applied_fraction": zero.detach(),
                    "selective_discrete_gumbel_changed_fraction": zero.detach(),
                    "selective_discrete_usage_ema_max": self._code_usage_ema.max(),
                }
            return mixed_fields, codes, zero

        if self.training and self._pending_warmup_init:
            self._initialize_codebooks_from_warmup()

        for selected_pos, field_index in enumerate(self.selected_field_indices):
            values = field_emb[:, field_index, :].float()
            codebook = self.codebooks[selected_pos].float()
            missing = values.eq(0).all(dim=-1)
            nonmissing = ~missing
            replacement = torch.zeros_like(values)
            active_ids = (
                field_ids[:, field_index][nonmissing]
                if field_ids is not None
                else None
            )
            if self.audit_enabled:
                self._audit_last_active_ids[selected_pos] = (
                    active_ids.detach() if active_ids is not None else None
                )

            if nonmissing.any():
                active_values = values[nonmissing]
                candidate_indices = self._nearest_candidates(active_values, codebook)
                candidate_centers = codebook[candidate_indices]
                candidate_distances = (
                    active_values.unsqueeze(1) - candidate_centers
                ).square().sum(dim=-1)
                (
                    probabilities,
                    hard_candidate_positions,
                    exploration_mask,
                    exploration_changed,
                ) = self._assignment_distribution(
                    selected_pos=selected_pos,
                    candidate_indices=candidate_indices,
                    candidate_distances=candidate_distances,
                    effective_mix=effective_mix,
                )
                hard_indices = candidate_indices.gather(
                    1, hard_candidate_positions.unsqueeze(-1)
                ).squeeze(-1)
                hard_values = codebook[hard_indices]
                soft_values = torch.einsum(
                    "bm,bme->be",
                    probabilities,
                    candidate_centers,
                )
                self._record_assignment_audit(
                    selected_pos=selected_pos,
                    active_ids=active_ids,
                    hard_indices=hard_indices,
                    probabilities=probabilities,
                    candidate_indices=candidate_indices,
                    candidate_centers=candidate_centers,
                    hard_values=hard_values,
                    soft_values=soft_values,
                    hard_candidate_positions=hard_candidate_positions,
                    exploration_mask=exploration_mask,
                    exploration_changed=exploration_changed,
                )
                discrete_values = soft_values + (hard_values - soft_values).detach()
                quantized_values = discrete_values
                if effective_mix < 1.0:
                    quantized_values = active_values + effective_mix * (
                        discrete_values - active_values
                    )
                replacement[nonmissing] = quantized_values
                codes[nonmissing, selected_pos] = hard_indices

                commitments.append(
                    F.mse_loss(active_values, discrete_values.detach())
                )
                codebook_fits.append(
                    F.mse_loss(active_values.detach(), discrete_values)
                )
                with torch.no_grad():
                    probability_entropy = -(
                        probabilities
                        * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1)
                    candidate_entropies.append(probability_entropy.mean())
                    errors.append(
                        torch.linalg.vector_norm(
                            active_values.detach() - hard_values.detach(),
                            dim=-1,
                        ).mean()
                    )
                    _, counts = hard_indices.unique(return_counts=True)
                    active_code_fractions.append(
                        counts.new_tensor(
                            counts.numel() / self.codebook_size,
                            dtype=torch.float32,
                        )
                    )
                    top_code_fractions.append(counts.max().float() / counts.sum())
                    gumbel_applied_fractions.append(exploration_mask.float().mean())
                    gumbel_changed_fractions.append(exploration_changed.float().mean())
            missing_fractions.append(missing.float().mean())
            mixed_fields[:, field_index, :] = replacement.to(dtype=field_emb.dtype)

        if commitments:
            commitment = torch.stack(commitments).mean()
            codebook_fit = torch.stack(codebook_fits).mean()
            quantization_loss = self.commitment_weight * commitment + codebook_fit
        else:
            quantization_loss = sum(codebook.sum() for codebook in self.codebooks) * 0.0

        with torch.no_grad():
            zero = field_emb.new_zeros(())
            self._diagnostics = {
                "selective_discrete_selected_fields": zero.new_tensor(
                    len(self.selected_field_indices)
                ),
                "selective_discrete_codebook_size": zero.new_tensor(self.codebook_size),
                "selective_discrete_missing_fraction": torch.stack(
                    missing_fractions
                ).mean(),
                "selective_discrete_quantization_error_mean": (
                    torch.stack(errors).mean() if errors else zero
                ),
                "selective_discrete_candidate_entropy": (
                    torch.stack(candidate_entropies).mean()
                    if candidate_entropies
                    else zero
                ),
                "selective_discrete_batch_active_code_fraction": (
                    torch.stack(active_code_fractions).mean()
                    if active_code_fractions
                    else zero
                ),
                "selective_discrete_batch_top_code_fraction": (
                    torch.stack(top_code_fractions).mean()
                    if top_code_fractions
                    else zero
                ),
                "selective_discrete_quantization_loss": quantization_loss.detach(),
                "selective_discrete_quantization_mix": zero.new_tensor(effective_mix),
                "selective_discrete_current_temperature": zero.new_tensor(
                    self.current_temperature
                ),
                "selective_discrete_warmup_buffer_fraction": zero.new_tensor(
                    self._warmup_fill_fraction()
                ),
                "selective_discrete_warmup_unique_id_count": zero.new_tensor(
                    self._warmup_unique_id_count()
                ),
                "selective_discrete_warmup_initialized": zero.new_tensor(
                    float(self._warmup_initialized)
                ),
                "selective_discrete_gumbel_applied_fraction": (
                    torch.stack(gumbel_applied_fractions).mean()
                    if gumbel_applied_fractions
                    else zero
                ),
                "selective_discrete_gumbel_changed_fraction": (
                    torch.stack(gumbel_changed_fractions).mean()
                    if gumbel_changed_fractions
                    else zero
                ),
                "selective_discrete_usage_ema_max": self._code_usage_ema.max(),
            }
        return mixed_fields, codes, quantization_loss

    def forward(
        self,
        field_emb: torch.Tensor,
        field_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mixed_fields, _, quantization_loss = self.quantize_fields(
            field_emb,
            field_ids,
        )
        effective_mix = 1.0 if not self.training else self.quantization_mix
        self.aux_loss = (
            self.quantization_loss_weight * effective_mix * quantization_loss
            if self.training and self.quantization_loss_weight > 0
            else None
        )
        return self.projector(mixed_fields)


class ContextualFallbackTokenizer(S2DRecTokenizer):
    """Keep observed ID tokens exact and reconstruct only missing field tokens.

    A learned field query attends to all other observed field tokens. During
    training, a self-excluded distillation loss teaches this contextual path to
    reconstruct each observed token without changing the embedding/backbone
    path. At inference, the reconstruction replaces a token only when its raw
    sparse ID is zero.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        *,
        enabled: bool = True,
        distill_weight: float = 0.1,
        distill_beta: float = 0.1,
        proj_mode: str = "split",
        seed: int = 42,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if num_tokens != num_fields:
            raise ValueError(
                "contextual_fallback requires num_tokens == num_fields, "
                f"got num_tokens={num_tokens}, num_fields={num_fields}"
            )
        if proj_mode not in ("shared", "split"):
            raise ValueError(
                "contextual_fallback_proj_mode must be 'shared' or 'split', "
                f"got {proj_mode}"
            )
        if distill_weight < 0:
            raise ValueError(
                f"contextual_fallback_distill_weight must be non-negative, got {distill_weight}"
            )
        if distill_beta <= 0:
            raise ValueError(
                f"contextual_fallback_distill_beta must be positive, got {distill_beta}"
            )

        self.enabled = bool(enabled)
        self.distill_weight = float(distill_weight)
        self.distill_beta = float(distill_beta)
        self.proj_mode = proj_mode

        if proj_mode == "shared":
            self.field_proj = nn.Linear(embedding_dim, d_model)
        else:
            self.field_proj_W = nn.Parameter(torch.empty(num_fields, embedding_dim, d_model))
            self.field_proj_b = nn.Parameter(torch.zeros(num_fields, d_model))
            nn.init.xavier_uniform_(self.field_proj_W)

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        query = torch.randn(num_fields, d_model, generator=generator) / math.sqrt(d_model)
        self.field_query = nn.Parameter(query)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1_000_003)
            self.context_key = nn.Linear(d_model, d_model, bias=False)
            self.context_value = nn.Linear(d_model, d_model, bias=False)
            self.context_out = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.context_out.weight)

        self.register_buffer(
            "self_exclusion_mask",
            ~torch.eye(num_fields, dtype=torch.bool),
            persistent=False,
        )

    def _project_fields(self, field_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proj_mode == "shared":
            tokens = self.field_proj(field_emb)
            bias = self.field_proj.bias.view(1, 1, -1).expand(
                field_emb.size(0), self.num_fields, -1
            )
            return tokens, bias
        tokens = torch.einsum("bne,ned->bnd", field_emb, self.field_proj_W)
        bias = self.field_proj_b.unsqueeze(0).expand(field_emb.size(0), -1, -1)
        return tokens + bias, bias

    def _contextual_prediction(
        self,
        source_tokens: torch.Tensor,
        field_bias: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = F.layer_norm(source_tokens.detach(), (self.d_model,))
        keys = self.context_key(normalized)
        values = self.context_value(normalized)
        queries = self.field_query.unsqueeze(0).expand(source_tokens.size(0), -1, -1)
        scores = torch.einsum("bid,bjd->bij", queries, keys) / math.sqrt(self.d_model)

        observed_keys = (~missing_mask).unsqueeze(1)
        valid = observed_keys & self.self_exclusion_mask.unsqueeze(0)
        masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        score_max = masked_scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(masked_scores - score_max) * valid.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

        context = torch.einsum("bij,bjd->bid", weights, values)
        context_delta = self.context_out(context)
        prediction = field_bias + context_delta
        return prediction, weights, context_delta

    def contextual_consistency(
        self,
        field_emb: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compare each exact ID token with its self-excluded context prediction."""
        if missing_mask.shape != field_emb.shape[:2]:
            raise ValueError(
                "contextual consistency missing_mask must match field_emb[:2], "
                f"got mask={tuple(missing_mask.shape)}, "
                f"fields={tuple(field_emb.shape[:2])}"
            )
        missing_mask = missing_mask.to(device=field_emb.device, dtype=torch.bool)
        base_tokens, field_bias = self._project_fields(field_emb)
        prediction, attention, _ = self._contextual_prediction(
            base_tokens, field_bias, missing_mask
        )
        difference = base_tokens - prediction
        l2 = torch.linalg.vector_norm(difference, dim=-1)
        mean_norm = 0.5 * (
            torch.linalg.vector_norm(base_tokens, dim=-1)
            + torch.linalg.vector_norm(prediction, dim=-1)
        )
        relative_l2 = l2 / mean_norm.clamp_min(torch.finfo(l2.dtype).eps)
        cosine_distance = 1.0 - F.cosine_similarity(
            base_tokens, prediction, dim=-1, eps=1e-8
        )
        smooth_l1 = F.smooth_l1_loss(
            prediction,
            base_tokens,
            beta=self.distill_beta,
            reduction="none",
        ).mean(dim=-1)
        return {
            "base_tokens": base_tokens,
            "prediction": prediction,
            "attention": attention,
            "l2": l2,
            "relative_l2": relative_l2,
            "cosine_distance": cosine_distance,
            "smooth_l1": smooth_l1,
        }

    def forward(
        self,
        field_emb: torch.Tensor,
        missing_mask: torch.Tensor,
        replacement_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if missing_mask.shape != field_emb.shape[:2]:
            raise ValueError(
                "contextual_fallback missing_mask must match field_emb[:2], "
                f"got mask={tuple(missing_mask.shape)}, fields={tuple(field_emb.shape[:2])}"
            )
        missing_mask = missing_mask.to(device=field_emb.device, dtype=torch.bool)
        if replacement_mask is None:
            replacement_mask = missing_mask
        else:
            if replacement_mask.shape != missing_mask.shape:
                raise ValueError(
                    "contextual_fallback replacement_mask must match missing_mask, "
                    f"got replacement={tuple(replacement_mask.shape)}, "
                    f"missing={tuple(missing_mask.shape)}"
                )
            replacement_mask = replacement_mask.to(
                device=field_emb.device, dtype=torch.bool
            )
            if bool((replacement_mask & ~missing_mask).any()):
                raise ValueError(
                    "contextual_fallback replacement_mask must be a subset of missing_mask"
                )
        base_tokens, field_bias = self._project_fields(field_emb)
        prediction, attention, context_delta = self._contextual_prediction(
            base_tokens, field_bias, missing_mask
        )

        self.aux_loss = None
        if self.training and self.distill_weight > 0:
            observed_targets = ~missing_mask
            if observed_targets.any():
                distill_prediction = context_delta + field_bias.detach()
                distill = F.smooth_l1_loss(
                    distill_prediction[observed_targets],
                    base_tokens.detach()[observed_targets],
                    beta=self.distill_beta,
                )
            else:
                distill = prediction.sum() * 0.0
            self.aux_loss = self.distill_weight * distill

        output = (
            torch.where(replacement_mask.unsqueeze(-1), prediction, base_tokens)
            if self.enabled
            else base_tokens
        )

        with torch.no_grad():
            fallback_delta = torch.linalg.vector_norm(prediction - field_bias, dim=-1)
            safe_attention = attention.clamp_min(torch.finfo(attention.dtype).eps)
            entropy = -(attention * safe_attention.log()).sum(dim=-1)
            self._diagnostics = {
                "contextual_missing_fraction": missing_mask.float().mean(),
                "contextual_replacement_fraction": (
                    replacement_mask.float().mean()
                    if self.enabled
                    else missing_mask.new_zeros(())
                ),
                "contextual_fallback_delta_norm_mean": fallback_delta.mean(),
                "contextual_fallback_delta_norm_max": fallback_delta.max(),
                "contextual_attention_entropy_mean": entropy.mean(),
                "contextual_distill_loss": (
                    self.aux_loss.detach()
                    if self.aux_loss is not None
                    else fallback_delta.new_zeros(())
                ),
            }
        return output


class BoundedResidualTokenizer(S2DRecTokenizer):
    """Represent each field as a shared anchor plus a bounded ID residual.

    Sparse embedding row 0 maps exactly to the field anchor. With zero-initialized
    sparse tables, unseen IDs also start at the anchor and only observed IDs can
    learn an ID-specific residual. The residual is smoothly bounded in L2 norm,
    so no offline frequency feature is needed at inference time.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        residual_norm: float = 0.5,
        anchor_init_scale: float = 1.0,
        proj_mode: str = "split",
        seed: int = 42,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if num_tokens != num_fields:
            raise ValueError(
                "bounded_residual currently requires num_tokens == num_fields, "
                f"got num_tokens={num_tokens}, num_fields={num_fields}"
            )
        if residual_norm <= 0:
            raise ValueError(f"bounded_residual_norm must be positive, got {residual_norm}")
        if anchor_init_scale <= 0:
            raise ValueError(
                f"bounded_anchor_init_scale must be positive, got {anchor_init_scale}"
            )
        if proj_mode not in ("shared", "split"):
            raise ValueError(
                "bounded_residual_proj_mode must be 'shared' or 'split', "
                f"got {proj_mode}"
            )

        self.residual_norm = float(residual_norm)
        self.anchor_init_scale = float(anchor_init_scale)
        self.proj_mode = proj_mode

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        anchor = torch.randn(num_fields, d_model, generator=generator)
        anchor.mul_(self.anchor_init_scale / math.sqrt(d_model))
        self.field_anchor = nn.Parameter(anchor)

        if proj_mode == "shared":
            self.residual_proj = nn.Linear(embedding_dim, d_model, bias=False)
        else:
            self.residual_proj_W = nn.Parameter(torch.empty(num_fields, embedding_dim, d_model))
            nn.init.xavier_uniform_(self.residual_proj_W)

    def _project_residual(self, field_emb: torch.Tensor) -> torch.Tensor:
        if self.proj_mode == "shared":
            return self.residual_proj(field_emb)
        return torch.einsum("bne,ned->bnd", field_emb, self.residual_proj_W)

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        raw_residual = self._project_residual(field_emb)
        raw_norm = torch.linalg.vector_norm(raw_residual, dim=-1, keepdim=True)
        eps = torch.finfo(raw_residual.dtype).eps
        smooth_scale = self.residual_norm * torch.tanh(raw_norm / self.residual_norm)
        smooth_scale = smooth_scale / raw_norm.clamp_min(eps)
        scale = torch.where(raw_norm > eps, smooth_scale, torch.ones_like(raw_norm))
        residual = raw_residual * scale
        output = self.field_anchor.unsqueeze(0) + residual

        with torch.no_grad():
            bounded_norm = torch.linalg.vector_norm(residual, dim=-1)
            anchor_norm = torch.linalg.vector_norm(self.field_anchor, dim=-1)
            self._diagnostics = {
                "bounded_residual_norm_mean": bounded_norm.mean(),
                "bounded_residual_norm_max": bounded_norm.max(),
                "bounded_residual_saturation_fraction": (
                    bounded_norm >= 0.95 * self.residual_norm
                ).float().mean(),
                "bounded_anchor_norm_mean": anchor_norm.mean(),
                "bounded_residual_limit": bounded_norm.new_tensor(self.residual_norm),
            }
        return output


class LearnableQueryTokenizer(S2DRecTokenizer):
    """Perceiver/Q-Former style learnable-query sparse-to-dense tokenizer."""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        query_init_scale: float = 1.0,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by query_num_heads={num_heads}")
        self.field_proj = nn.Linear(embedding_dim, d_model)
        self.queries = nn.Parameter(torch.empty(num_tokens, d_model))
        nn.init.normal_(self.queries, std=float(query_init_scale) / math.sqrt(d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        values = self.field_proj(field_emb)
        queries = self.queries.unsqueeze(0).expand(field_emb.size(0), -1, -1)
        tokens, _ = self.attn(queries, values, values, need_weights=False)
        return self.out_ln(tokens)


class GatedRoutingTokenizer(S2DRecTokenizer):
    """Softly route each field embedding into K dense token slots."""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        normalize: bool = True,
        aux_loss_weight: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__(num_fields, embedding_dim, num_tokens, d_model)
        if temperature <= 0:
            raise ValueError(f"routing_temperature must be positive, got {temperature}")
        self.value_proj = nn.Linear(embedding_dim, d_model)
        self.routing_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_tokens),
        )
        self.normalize = normalize
        self.aux_loss_weight = float(aux_loss_weight)
        self.temperature = float(temperature)

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        logits = self.routing_net(field_emb)
        gates = F.softmax(logits / self.temperature, dim=-1)
        values = self.value_proj(field_emb)
        tokens = torch.einsum("bnk,bnd->bkd", gates, values)
        if self.normalize:
            denom = gates.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
            tokens = tokens / denom

        self._update_diagnostics(gates)
        self.aux_loss = None
        if self.aux_loss_weight > 0:
            load = gates.mean(dim=(0, 1))
            target = torch.full_like(load, 1.0 / self.num_tokens)
            self.aux_loss = self.aux_loss_weight * F.mse_loss(load, target)
        return tokens

    def _update_diagnostics(self, gates: torch.Tensor) -> None:
        eps = 1e-8
        with torch.no_grad():
            load = gates.sum(dim=1).mean(dim=0)
            load_prob = load / load.sum().clamp_min(eps)
            log_k = math.log(max(self.num_tokens, 2))
            load_entropy = -(load_prob * load_prob.clamp_min(eps).log()).sum() / log_k
            assignment_entropy = -(gates * gates.clamp_min(eps).log()).sum(dim=-1).mean() / log_k
            load_mean = load.mean().clamp_min(eps)
            target = torch.full_like(load_prob, 1.0 / self.num_tokens)
            self._diagnostics = {
                "routing_assignment_entropy": assignment_entropy.detach(),
                "routing_load_entropy": load_entropy.detach(),
                "routing_load_cv": (load.std(unbiased=False) / load_mean).detach(),
                "routing_load_mse": F.mse_loss(load_prob, target).detach(),
                "routing_max_gate_mean": gates.max(dim=-1).values.mean().detach(),
                "routing_temperature": load_prob.new_tensor(self.temperature),
                "routing_top_token_load": load_prob.max().detach(),
            }


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


class S2DRecCarryPathBlock(nn.Module):
    """TokenMixer branch with a configurable carry-path merge.

    This block is for surgical ablation: keep the branch function fixed, and
    change only how the branch is merged back into the carry path.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        channel_mixer_type: str = "dense",
        num_experts: int = 4,
        top_k: int = 2,
        num_shared: int = 1,
        init_mix_scale: float = 1.0,
        init_ffn_scale: float = 1.0,
        init_inter_scale: float = 0.1,
        learnable_scale: bool = True,
        down_init_std: float = 0.01,
        router_scale: float = 1.0,
        carry_merge: str = "additive",
        per_token_init_v2: bool = False,
    ):
        super().__init__()
        if carry_merge not in ("additive", "multiplicative_x0"):
            raise ValueError(
                f"carry_merge must be 'additive' or 'multiplicative_x0', got {carry_merge}"
            )
        self.carry_merge = carry_merge
        self.norm_mix = _RMSNorm(token_dim)
        self.token_mixing = TokenMixingV2(num_tokens, token_dim)
        _mixer_kw = dict(
            mixer_type=channel_mixer_type,
            num_tokens=num_tokens,
            token_dim=token_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            down_init_std=down_init_std,
            num_experts=num_experts,
            top_k=top_k,
            num_shared=num_shared,
            router_scale=router_scale,
            per_token_init_v2=per_token_init_v2,
        )
        self.mixed_channel_mixing = _make_channel_mixer(**_mixer_kw)
        self.norm_ffn = _RMSNorm(token_dim)
        self.channel_mixing = _make_channel_mixer(**_mixer_kw)

        if learnable_scale:
            self.mix_scale = nn.Parameter(torch.tensor(float(init_mix_scale)))
            self.ffn_scale = nn.Parameter(torch.tensor(float(init_ffn_scale)))
            self.inter_scale = nn.Parameter(torch.tensor(float(init_inter_scale)))
        else:
            self.register_buffer("mix_scale", torch.tensor(float(init_mix_scale)))
            self.register_buffer("ffn_scale", torch.tensor(float(init_ffn_scale)))
            self.register_buffer("inter_scale", torch.tensor(float(init_inter_scale)))

    def _branch(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixing(self.norm_mix(x))
        mixed = self.mixed_channel_mixing(mixed)
        reverted = self.token_mixing(mixed)
        mix_branch = self.mix_scale * reverted
        transformed = self.channel_mixing(self.norm_ffn(x + mix_branch))
        return mix_branch + self.ffn_scale * transformed

    def forward(
        self,
        x: torch.Tensor,
        carry_base: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        branch = self._branch(x)
        if self.carry_merge == "additive":
            out = x + branch
        else:
            out = x + carry_base * branch
        if skip is not None:
            out = out + self.inter_scale * skip
        return out


@register_model
class S2DRecModel(RecModel):
    """Sparse-to-Dense Tokenized Recommender with pluggable tokenizer/backbone."""

    model_name = "s2drec"

    def __init__(self, config: dict):
        super().__init__(config)
        mc = config["model"]

        self.encoder = _FieldEmbeddingEncoder(config)
        num_fields = self.encoder.num_fields
        embedding_dim = self.encoder.embedding_dim
        num_tokens = int(mc.get("num_tokens", 13))
        d_model = int(mc.get("d_model", embedding_dim))
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.tokenizer_type = mc.get("tokenizer_type", "uniform_proj")
        if (
            self.tokenizer_type == "bounded_residual"
            and bool(mc.get("bounded_residual_require_zero_init", True))
            and self.encoder.embedding_init != "zero"
        ):
            raise ValueError(
                "bounded_residual requires model.embedding_init='zero' so unseen IDs "
                "start at the shared field anchor"
            )
        self.tokenizer = self._build_tokenizer(mc, num_fields, embedding_dim, num_tokens, d_model)

        self.backbone_type = mc.get("backbone_type", "tokenmixer_v3")
        self.carry_merge = mc.get("carry_merge", "additive")
        self.carry_path_ablation = bool(mc.get("carry_path_ablation", False)) or self.carry_merge != "additive"
        if d_model % num_tokens != 0:
            raise ValueError(
                f"backbone={self.backbone_type} requires d_model={d_model} divisible by num_tokens={num_tokens}"
            )
        num_layers = int(mc.get("num_mixer_layers", mc.get("num_layers", 2)))
        ffn_dim = int(mc.get("ffn_dim", 256))
        dropout = float(mc.get("dropout", mc.get("net_dropout", 0.0)))

        if self.backbone_type == "tokenmixer_v3":
            block_cls = S2DRecCarryPathBlock if self.carry_path_ablation else TokenMixerLargeV3Block
            block_kwargs = dict(
                num_tokens=num_tokens,
                token_dim=d_model,
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
                per_token_init_v2=bool(mc.get("per_token_init_v2", False)),
            )
            if self.carry_path_ablation:
                block_kwargs["carry_merge"] = self.carry_merge
            self.blocks = nn.ModuleList([
                block_cls(**block_kwargs)
                for _ in range(num_layers)
            ])
        elif self.backbone_type == "rankmixer_v2":
            self.blocks = nn.ModuleList([
                RankMixerV2Block(num_tokens, d_model, ffn_dim, dropout)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(
                f"Unknown backbone_type={self.backbone_type!r}; expected 'tokenmixer_v3' or 'rankmixer_v2'"
            )

        self.input_ln = nn.LayerNorm(d_model) if mc.get("use_input_ln", True) else None
        self.final_ln = nn.LayerNorm(d_model) if mc.get("use_final_ln", True) else None
        self.pool_output = bool(mc.get("pool_output", True))
        self.inter_layer_residual = bool(mc.get("inter_layer_residual", self.backbone_type == "tokenmixer_v3"))
        self.residual_skip_stride = int(mc.get("residual_skip_stride", 2))
        self.exclude_last_inter_residual = bool(mc.get("exclude_last_inter_layer_residual", True))

        self.aux_loss_weight = float(mc.get("aux_loss_weight", 0.0))
        self.aux_pred_hidden_dim = int(mc.get("aux_pred_hidden_dim", 64))
        self.aux_exclude_last_layer = bool(mc.get("aux_exclude_last_layer", True))
        self.num_aux = 0
        if self.aux_loss_weight > 0:
            self.num_aux = num_layers - 1 if self.aux_exclude_last_layer else num_layers
            if self.num_aux > 0:
                self.aux_predictors = nn.ModuleList([
                    _AuxPredictor(num_tokens, d_model, self.aux_pred_hidden_dim)
                    for _ in range(self.num_aux)
                ])

        head_hidden_units = mc.get("head_hidden_units", mc.get("mlp_dims", [512, 256]))
        head_activation = mc.get("head_activation", "relu")
        head_dropout = float(mc.get("head_dropout", 0.0))
        in_dim = d_model if self.pool_output else num_tokens * d_model
        self.head = self._build_head(in_dim, head_hidden_units, head_activation, head_dropout)

        print(
            f"[S2DRec] fields={num_fields}, emb={embedding_dim}, T={num_tokens}, "
            f"d_model={d_model}, tokenizer={self.tokenizer_type}, backbone={self.backbone_type}, "
            f"embedding_init={self.encoder.embedding_init}, "
            f"carry_merge={self.carry_merge if self.carry_path_ablation else 'native'}, "
            f"L={num_layers}, ffn={ffn_dim}, pool={self.pool_output}"
        )
        if self.encoder.offline_tail_id_collapse is not None:
            metadata = self.encoder.offline_tail_id_collapse.get_metadata()
            field_summary = [
                {
                    "index": field["index"],
                    "name": field["name"],
                    "tail_key_virtual_id": field["tail_key_virtual_id"],
                    "collapsed_ids": field["collapsed_id_count"],
                }
                for field in metadata["fields"]
            ]
            print(
                "[S2DRec] Offline frequency-oracle tail-ID collapse: "
                f"max_count={metadata['max_count']}, "
                f"include_unseen={metadata['include_unseen']}, "
                f"missing_id_collapsed={metadata['missing_id_collapsed']}, "
                f"fields={field_summary}, production_ready=False"
            )
        if self.encoder.k1_identity_router is not None:
            metadata = self.encoder.k1_identity_router.get_metadata()
            print(
                "[S2DRec] K1 identity routing: "
                f"side_enabled={metadata['side_enabled']}, "
                f"tail_share_fields={metadata['tail_share_fields']}, "
                f"max_count={metadata['max_count']}, "
                f"special_token_ids={metadata['special_token_ids']}, "
                "production_ready=False"
            )
        if self.encoder.zero_anchor_identity_quantizer is not None:
            metadata = self.encoder.zero_anchor_identity_quantizer.get_metadata()
            print(
                "[S2DRec] SHRED-ZA identity quantization: "
                f"fields={metadata['identity_fields']}, "
                f"codes={metadata['codebook_size']}, "
                f"subspaces={metadata['num_subspaces']}, "
                f"residual_levels={metadata['num_residual_levels']}, "
                f"multi_codebook_mode={metadata['multi_codebook_mode']}, "
                f"level_scale_decay="
                f"{metadata['residual_level_scale_decay']:g}, "
                f"level_temperature_decay="
                f"{metadata['residual_level_temperature_decay']:g}, "
                f"base_embedding_mode={metadata['base_embedding_mode']}, "
                f"margin={metadata['margin']:g}, "
                f"temperature={metadata['temperature_start']:g}"
                f"->{metadata['temperature_end']:g}, "
                f"codebook_loss={metadata['codebook_loss_weight']:g}, "
                f"codebook_schedule="
                f"{metadata['codebook_schedule']['mode']}"
                f"(release="
                f"{metadata['codebook_schedule']['release_fraction']:g},"
                f"ramp="
                f"{metadata['codebook_schedule']['ramp_fraction']:g}), "
                f"commitment={metadata['commitment_weight']:g}, "
                f"regularization_schedule="
                f"{metadata['regularization_schedule']['mode']}"
                f"(release="
                f"{metadata['regularization_schedule']['release_fraction']:g},"
                f"ramp="
                f"{metadata['regularization_schedule']['ramp_fraction']:g}), "
                f"zero_l2={metadata['zero_l2_weight']:g}, "
                "hard_forward=True, continuous_residual=False, "
                "production_ready=False"
            )
        if self.tokenizer_type == "bounded_residual":
            print(
                "[S2DRec] Bounded residual tokenizer: "
                f"proj={self.tokenizer.proj_mode}, "
                f"residual_norm={self.tokenizer.residual_norm:g}, "
                f"anchor_init_scale={self.tokenizer.anchor_init_scale:g}"
            )
        if self.tokenizer_type == "contextual_fallback":
            print(
                "[S2DRec] Contextual missing-ID fallback: "
                f"enabled={self.tokenizer.enabled}, "
                f"proj={self.tokenizer.proj_mode}, "
                f"distill_weight={self.tokenizer.distill_weight:g}, "
                f"distill_beta={self.tokenizer.distill_beta:g}, self_excluded=True"
            )
        if self.tokenizer_type == "pure_discrete_per_field":
            print(
                "[S2DRec] Pure discrete per-field tokenizer: "
                f"levels={self.tokenizer.num_levels}, "
                f"codes_per_level={self.tokenizer.codebook_size}, "
                f"temperature={self.tokenizer.temperature:g}, "
                f"zero_margin={self.tokenizer.zero_margin:g}, "
                f"quantization_loss_weight={self.tokenizer.quantization_loss_weight:g}, "
                f"proj={self.tokenizer.proj_mode}, continuous_residual=False"
            )
        if self.tokenizer_type == "selective_single_level_discrete":
            print(
                "[S2DRec] Selective single-level discrete tokenizer: "
                f"fields={list(self.tokenizer.selected_field_indices)}, "
                f"codes={self.tokenizer.codebook_size}, "
                f"candidate_top_m={self.tokenizer.candidate_top_m}, "
                f"distance_chunk={self.tokenizer.distance_chunk_size}, "
                f"temperature={self.tokenizer.temperature:g}, "
                f"temperature_start={self.tokenizer.temperature_start:g}, "
                f"quantization_loss_weight={self.tokenizer.quantization_loss_weight:g}, "
                f"warmup_fraction={self.tokenizer.warmup_fraction:g}, "
                f"transition_fraction={self.tokenizer.transition_fraction:g}, "
                f"codebook_init={self.tokenizer.codebook_init}, "
                f"proj={self.tokenizer.proj_mode}, continuous_residual=False, "
                f"assignment_mode={self.tokenizer.assignment_mode}, "
                f"gumbel_scale={self.tokenizer.gumbel_scale:g}, "
                f"usage_ema_decay={self.tokenizer.usage_ema_decay:g}, "
                f"hot_threshold_ratio={self.tokenizer.hot_threshold_ratio:g}, "
                f"audit_enabled={self.tokenizer.audit_enabled}"
            )

    def _build_tokenizer(
        self,
        mc: dict,
        num_fields: int,
        embedding_dim: int,
        num_tokens: int,
        d_model: int,
    ) -> S2DRecTokenizer:
        seed = int(mc.get("tokenizer_seed", self.config.get("seed", 42)))
        if self.tokenizer_type == "uniform_proj":
            return UniformProjectionTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                proj_mode=mc.get("uniform_proj_mode", mc.get("uniform_proj", "split")),
                field_shuffle=mc.get("field_shuffle", False),
                seed=seed,
            )
        if self.tokenizer_type == "per_field_proj":
            return PerFieldProjectionTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                proj_mode=mc.get("per_field_proj_mode", mc.get("uniform_proj", "split")),
                seed=seed,
            )
        if self.tokenizer_type == "pure_discrete_per_field":
            return PureDiscretePerFieldTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                num_levels=int(mc.get("pure_discrete_num_levels", 3)),
                codebook_size=int(mc.get("pure_discrete_codebook_size", 64)),
                temperature=float(mc.get("pure_discrete_temperature", 0.2)),
                code_init_scale=float(mc.get("pure_discrete_code_init_scale", 0.05)),
                zero_margin=float(mc.get("pure_discrete_zero_margin", 0.15)),
                quantization_loss_weight=float(
                    mc.get("pure_discrete_quantization_loss_weight", 0.2)
                ),
                commitment_weight=float(
                    mc.get("pure_discrete_commitment_weight", 0.25)
                ),
                proj_mode=mc.get("pure_discrete_proj_mode", "split"),
                seed=seed,
            )
        if self.tokenizer_type == "selective_single_level_discrete":
            selected_fields = mc.get("selective_discrete_fields")
            selected_indices = mc.get("selective_discrete_field_indices")
            if selected_fields is not None and selected_indices is not None:
                raise ValueError(
                    "configure either selective_discrete_fields or "
                    "selective_discrete_field_indices, not both"
                )
            if selected_fields is not None:
                if not isinstance(selected_fields, list) or not all(
                    isinstance(name, str) for name in selected_fields
                ):
                    raise ValueError("selective_discrete_fields must be a list of names")
                field_names = self.config.get("dataset", {}).get("sparse_cols") or []
                name_to_index = {str(name): index for index, name in enumerate(field_names)}
                missing_names = [
                    name for name in selected_fields if name not in name_to_index
                ]
                if missing_names:
                    raise ValueError(
                        "selective_discrete_fields are absent from dataset.sparse_cols: "
                        f"{missing_names}"
                    )
                selected_indices = [name_to_index[name] for name in selected_fields]
            if selected_indices is None:
                raise ValueError(
                    "selective_single_level_discrete requires "
                    "selective_discrete_fields or selective_discrete_field_indices"
                )
            if not isinstance(selected_indices, list):
                raise ValueError("selective_discrete_field_indices must be a list")
            return SelectiveSingleLevelDiscreteTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                selected_field_indices=selected_indices,
                codebook_size=int(mc.get("selective_discrete_codebook_size", 10000)),
                candidate_top_m=int(mc.get("selective_discrete_candidate_top_m", 32)),
                distance_chunk_size=int(
                    mc.get("selective_discrete_distance_chunk_size", 2048)
                ),
                temperature=float(mc.get("selective_discrete_temperature", 0.2)),
                temperature_start=float(
                    mc.get(
                        "selective_discrete_temperature_start",
                        mc.get("selective_discrete_temperature", 0.2),
                    )
                ),
                code_init_scale=float(
                    mc.get("selective_discrete_code_init_scale", 0.05)
                ),
                quantization_loss_weight=float(
                    mc.get("selective_discrete_quantization_loss_weight", 0.2)
                ),
                commitment_weight=float(
                    mc.get("selective_discrete_commitment_weight", 0.25)
                ),
                warmup_fraction=float(
                    mc.get("selective_discrete_warmup_fraction", 0.0)
                ),
                transition_fraction=float(
                    mc.get("selective_discrete_transition_fraction", 0.0)
                ),
                codebook_init=str(
                    mc.get("selective_discrete_codebook_init", "random")
                ),
                proj_mode=mc.get("selective_discrete_proj_mode", "split"),
                seed=seed,
                audit_enabled=bool(
                    mc.get("selective_discrete_audit_enabled", False)
                ),
                audit_assignment_every=int(
                    mc.get("selective_discrete_audit_assignment_every", 50)
                ),
                audit_gradient_every=int(
                    mc.get("selective_discrete_audit_gradient_every", 250)
                ),
                audit_anchor_observations=int(
                    mc.get("selective_discrete_audit_anchor_observations", 65536)
                ),
                assignment_mode=str(
                    mc.get("selective_discrete_assignment_mode", "nearest_ste")
                ),
                gumbel_scale=float(
                    mc.get("selective_discrete_gumbel_scale", 1.0)
                ),
                usage_ema_decay=float(
                    mc.get("selective_discrete_usage_ema_decay", 0.99)
                ),
                hot_threshold_ratio=float(
                    mc.get("selective_discrete_hot_threshold_ratio", 1.5)
                ),
            )
        if self.tokenizer_type == "bounded_residual":
            return BoundedResidualTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                residual_norm=float(mc.get("bounded_residual_norm", 0.5)),
                anchor_init_scale=float(mc.get("bounded_anchor_init_scale", 1.0)),
                proj_mode=mc.get("bounded_residual_proj_mode", "split"),
                seed=seed,
            )
        if self.tokenizer_type == "contextual_fallback":
            return ContextualFallbackTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                enabled=bool(mc.get("contextual_fallback_enabled", True)),
                distill_weight=float(mc.get("contextual_fallback_distill_weight", 0.1)),
                distill_beta=float(mc.get("contextual_fallback_distill_beta", 0.1)),
                proj_mode=mc.get("contextual_fallback_proj_mode", "split"),
                seed=seed,
            )
        if self.tokenizer_type == "group_pooling":
            return GroupPoolingTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                pooling=mc.get("group_pooling", "mean"),
                seed=seed,
                group_strategy=mc.get("group_strategy", "seeded_round_robin"),
                projection=mc.get("group_projection", "linear"),
                field_cardinalities=self.config.get("dataset", {}).get("cardinalities"),
                field_names=self.config.get("dataset", {}).get("sparse_cols"),
                field_groups=mc.get("field_groups"),
            )
        if self.tokenizer_type == "learnable_query":
            return LearnableQueryTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                num_heads=int(mc.get("query_num_heads", 4)),
                attn_dropout=float(mc.get("query_attn_dropout", 0.0)),
                query_init_scale=float(mc.get("query_init_scale", 1.0)),
            )
        if self.tokenizer_type == "gated_routing":
            return GatedRoutingTokenizer(
                num_fields,
                embedding_dim,
                num_tokens,
                d_model,
                hidden_dim=int(mc.get("routing_mlp_hidden", 64)),
                dropout=float(mc.get("routing_dropout", 0.0)),
                normalize=bool(mc.get("routing_normalize", True)),
                aux_loss_weight=float(mc.get("routing_aux_loss_weight", 0.0)),
                temperature=float(mc.get("routing_temperature", 1.0)),
            )
        raise ValueError(
            f"Unknown tokenizer_type={self.tokenizer_type!r}; expected one of "
            "'uniform_proj', 'per_field_proj', 'pure_discrete_per_field', "
            "'selective_single_level_discrete', "
            "'bounded_residual', 'contextual_fallback', "
            "'group_pooling', "
            "'learnable_query', 'gated_routing'"
        )

    @staticmethod
    def _build_head(in_dim: int, hidden_units: list[int], activation: str, dropout: float) -> nn.Sequential:
        layers = []
        for dim in hidden_units:
            layers.append(nn.Linear(in_dim, dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported head_activation={activation}")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def set_tau_for_step(self, current_step: int, total_steps: int) -> None:
        set_progress = getattr(self.tokenizer, "set_progress", None)
        if set_progress is not None:
            set_progress(current_step, total_steps)
        zero_anchor = self.encoder.zero_anchor_identity_quantizer
        if zero_anchor is not None:
            zero_anchor.set_progress(current_step, total_steps)

    def get_tokenizer_diagnostics(self) -> dict[str, float]:
        diagnostics = self.tokenizer.get_diagnostics()
        collapse = self.encoder.offline_tail_id_collapse
        if collapse is not None:
            diagnostics.update(collapse.get_diagnostics())
        k1_router = self.encoder.k1_identity_router
        if k1_router is not None:
            diagnostics.update(k1_router.get_diagnostics())
        zero_anchor = self.encoder.zero_anchor_identity_quantizer
        if zero_anchor is not None:
            diagnostics.update(zero_anchor.get_diagnostics())
        return diagnostics

    def get_k1_identity_routing_metadata(self) -> Optional[dict]:
        router = self.encoder.k1_identity_router
        if router is None:
            return None
        return router.get_metadata()

    def get_zero_anchor_identity_quantization_metadata(self) -> Optional[dict]:
        quantizer = self.encoder.zero_anchor_identity_quantizer
        if quantizer is None:
            return None
        return quantizer.get_metadata()

    def get_zero_anchor_assignment_audit_report(self) -> Optional[dict]:
        quantizer = self.encoder.zero_anchor_identity_quantizer
        if quantizer is None:
            return None
        return quantizer.get_assignment_audit_report()

    def get_zero_anchor_health_report(self, **thresholds) -> Optional[dict]:
        quantizer = self.encoder.zero_anchor_identity_quantizer
        if quantizer is None:
            return None
        return quantizer.get_health_report(**thresholds)

    def get_offline_tail_id_collapse_metadata(self) -> Optional[dict]:
        collapse = self.encoder.offline_tail_id_collapse
        if collapse is None:
            return None
        return collapse.get_metadata()

    def should_record_discrete_gradient_audit(self, global_step: int) -> bool:
        return bool(
            self.tokenizer_type == "selective_single_level_discrete"
            and self.tokenizer.audit_enabled
            and global_step % self.tokenizer.audit_gradient_every == 0
        )

    @torch.no_grad()
    def record_discrete_gradient_audit(self) -> None:
        if self.tokenizer_type != "selective_single_level_discrete":
            return
        tokenizer = self.tokenizer
        if not tokenizer.audit_enabled:
            return
        embeddings = self.encoder.sparse_arch.embeddings
        for selected_pos, field_index in enumerate(
            tokenizer.selected_field_indices
        ):
            active_ids = tokenizer._audit_last_active_ids[selected_pos]
            raw_grad = embeddings[field_index].weight.grad
            if active_ids is None or raw_grad is None:
                continue
            active_ids = torch.unique(active_ids.detach().to(dtype=torch.long))
            active_ids = active_ids[active_ids.gt(0)]
            if active_ids.numel() == 0:
                continue
            active_raw_grad = raw_grad.index_select(0, active_ids).float()
            raw_grad_l2 = torch.linalg.vector_norm(active_raw_grad)
            raw_grad_rms = active_raw_grad.square().mean().sqrt()

            codebook_grad = tokenizer.codebooks[selected_pos].grad
            if codebook_grad is None:
                codebook_grad_l2 = raw_grad_l2.new_zeros(())
                codebook_grad_rms = raw_grad_l2.new_zeros(())
            else:
                codebook_grad = codebook_grad.float()
                codebook_grad_l2 = torch.linalg.vector_norm(codebook_grad)
                codebook_grad_rms = codebook_grad.square().mean().sqrt()
            tokenizer.record_gradient_audit(
                selected_pos,
                raw_grad_l2=raw_grad_l2,
                raw_grad_rms=raw_grad_rms,
                codebook_grad_l2=codebook_grad_l2,
                codebook_grad_rms=codebook_grad_rms,
                active_rows=int(active_ids.numel()),
            )

    def get_discrete_training_audit(self) -> Optional[dict]:
        if self.tokenizer_type != "selective_single_level_discrete":
            return None
        return self.tokenizer.get_audit_report()

    def forward_stages(self, batch: dict) -> dict[str, torch.Tensor | None]:
        """Run the model while exposing sparse-to-dense transmission stages."""
        field_emb = self.encoder(batch)
        if self.tokenizer_type == "contextual_fallback":
            missing_mask = batch["sparse"].eq(0)
            replacement_mask = batch.get("contextual_fallback_mask")
            if replacement_mask is not None:
                replacement_mask = replacement_mask.to(
                    device=missing_mask.device, dtype=torch.bool
                )
                if replacement_mask.shape != missing_mask.shape:
                    raise ValueError(
                        "batch contextual_fallback_mask must match sparse shape, "
                        f"got mask={tuple(replacement_mask.shape)}, "
                        f"sparse={tuple(missing_mask.shape)}"
                    )
            if missing_mask.size(1) < field_emb.size(1):
                dense_present = torch.zeros(
                    missing_mask.size(0),
                    field_emb.size(1) - missing_mask.size(1),
                    dtype=torch.bool,
                    device=missing_mask.device,
                )
                missing_mask = torch.cat([missing_mask, dense_present], dim=1)
                if replacement_mask is not None:
                    replacement_mask = torch.cat(
                        [replacement_mask, dense_present], dim=1
                    )
            tokenizer_output = self.tokenizer(
                field_emb, missing_mask, replacement_mask=replacement_mask
            )
        elif self.tokenizer_type == "selective_single_level_discrete":
            tokenizer_output = self.tokenizer(field_emb, batch["sparse"])
        else:
            tokenizer_output = self.tokenizer(field_emb)
        backbone_input = (
            self.input_ln(tokenizer_output)
            if self.input_ln is not None
            else tokenizer_output
        )
        backbone_output, aux_loss = self._run_blocks(
            backbone_input, label=batch.get("label")
        )
        zero_anchor = self.encoder.zero_anchor_identity_quantizer
        if zero_anchor is not None:
            zero_anchor_loss = zero_anchor.get_aux_loss()
            if zero_anchor_loss is not None:
                aux_loss = (
                    zero_anchor_loss
                    if aux_loss is None
                    else aux_loss + zero_anchor_loss
                )
        if self.final_ln is not None:
            backbone_output = self.final_ln(backbone_output)
        if self.pool_output:
            representation = backbone_output.mean(dim=1)
        else:
            representation = backbone_output.flatten(start_dim=1)
        logits = self.head(representation).squeeze(-1)
        if zero_anchor is not None:
            code_logit_bias = zero_anchor.get_logit_bias()
            if code_logit_bias is not None:
                if code_logit_bias.shape != logits.shape:
                    raise RuntimeError(
                        "zero-anchor code logit bias must match logits shape"
                    )
                logits = logits + code_logit_bias.to(dtype=logits.dtype)
        return {
            "field_embedding": field_emb,
            "tokenizer_output": tokenizer_output,
            "backbone_input": backbone_input,
            "backbone_output": backbone_output,
            "representation": representation,
            "logits": logits,
            "aux_loss": aux_loss,
        }

    def _run_blocks(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        hidden_states = []
        carry_base = x
        last_idx = len(self.blocks) - 1
        for layer_idx, block in enumerate(self.blocks):
            if self.backbone_type == "tokenmixer_v3":
                skip = None
                use_inter = self.inter_layer_residual and not (
                    self.exclude_last_inter_residual and layer_idx == last_idx
                )
                if use_inter and layer_idx >= self.residual_skip_stride:
                    skip = hidden_states[layer_idx - self.residual_skip_stride]
                if self.carry_path_ablation:
                    x = block(x, carry_base=carry_base, skip=skip)
                else:
                    x = block(x, skip=skip)
            else:
                x = block(x)
            hidden_states.append(x)

        aux_loss = self.tokenizer.get_aux_loss()
        if (
            self.training
            and label is not None
            and self.aux_loss_weight > 0.0
            and self.num_aux > 0
        ):
            pred_aux = x.new_zeros(())
            for i, predictor in enumerate(self.aux_predictors):
                aux_logit = predictor(hidden_states[i])
                pred_aux = pred_aux + F.binary_cross_entropy_with_logits(aux_logit, label.float())
            pred_aux = self.aux_loss_weight * pred_aux / self.num_aux
            aux_loss = pred_aux if aux_loss is None else aux_loss + pred_aux
        return x, aux_loss

    def forward(self, batch: dict):
        stages = self.forward_stages(batch)
        logits = stages["logits"]
        aux_loss = stages["aux_loss"]
        if aux_loss is not None:
            return logits, aux_loss
        return logits
