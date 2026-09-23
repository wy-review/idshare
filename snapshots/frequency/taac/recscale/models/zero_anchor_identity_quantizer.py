"""Zero-anchor, residual-free quantization for high-cardinality identities.

Each selected identity keeps a zero-initialized private residual in the sparse
embedding table. The downstream model only receives a hard codebook center;
the private residual never bypasses quantization. Code zero is a fixed buffer,
while missing and reserved-zero tokens use independent trainable values.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ZeroAnchorIdentityQuantizer(nn.Module):
    """Quantize selected identity residuals through a fixed zero anchor."""

    CONFIG_KEY = "zero_anchor_identity_quantization"
    SHARED_TRANSFORM_MIN_SINGULAR_RATIO = 1e-3
    DEFAULT_SPECIAL_IDS = {
        "padding": 0,
        "oov": 1,
        "missing": 2,
        "reserved_zero": 3,
    }

    def __init__(
        self,
        config: dict,
        *,
        cardinalities: list[int],
        field_names: list[str],
        embedding_dim: int,
    ):
        super().__init__()
        cfg = config["model"].get(self.CONFIG_KEY, {})
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            raise ValueError(
                "ZeroAnchorIdentityQuantizer requires "
                "model.zero_anchor_identity_quantization.enabled=true"
            )

        self.field_names = tuple(str(name) for name in field_names)
        self.name_to_index = {
            name: index for index, name in enumerate(self.field_names)
        }
        self.identity_fields = tuple(
            cfg.get("identity_fields", ["video_id", "author_id", "music_id"])
        )
        if not self.identity_fields or len(set(self.identity_fields)) != len(
            self.identity_fields
        ):
            raise ValueError(
                "zero_anchor_identity_quantization.identity_fields must be "
                "non-empty and unique"
            )
        missing_fields = [
            name for name in self.identity_fields if name not in self.name_to_index
        ]
        if missing_fields:
            raise ValueError(
                "zero-anchor identity fields absent from dataset.sparse_cols: "
                f"{missing_fields}"
            )

        self.embedding_dim = int(embedding_dim)
        legacy_codebook_size = int(cfg.get("codebook_size", 64))
        self.num_subspaces = int(cfg.get("num_subspaces", 1))
        self.num_residual_levels = int(cfg.get("num_residual_levels", 1))
        raw_codebook_sizes_by_field = cfg.get("codebook_size_by_field")
        if raw_codebook_sizes_by_field is None:
            self.codebook_sizes_by_field = tuple(
                legacy_codebook_size for _ in self.identity_fields
            )
            self.codebook_size_by_field_is_explicit = False
        else:
            if not isinstance(raw_codebook_sizes_by_field, dict):
                raise ValueError(
                    "zero-anchor codebook_size_by_field must be a mapping"
                )
            unknown_codebook_size_fields = sorted(
                set(raw_codebook_sizes_by_field) - set(self.identity_fields)
            )
            missing_codebook_size_fields = sorted(
                set(self.identity_fields) - set(raw_codebook_sizes_by_field)
            )
            if unknown_codebook_size_fields or missing_codebook_size_fields:
                raise ValueError(
                    "zero-anchor codebook_size_by_field must exactly cover "
                    "identity_fields; "
                    f"missing={missing_codebook_size_fields}, "
                    f"unknown={unknown_codebook_size_fields}"
                )
            invalid_codebook_sizes = {
                field: value
                for field, value in raw_codebook_sizes_by_field.items()
                if isinstance(value, bool)
                or not isinstance(value, int)
                or value < 2
            }
            if invalid_codebook_sizes:
                raise ValueError(
                    "zero-anchor codebook_size_by_field values must be "
                    "integers >= 2: "
                    f"{invalid_codebook_sizes}"
                )
            self.codebook_sizes_by_field = tuple(
                int(raw_codebook_sizes_by_field[field])
                for field in self.identity_fields
            )
            if (
                "codebook_size" in cfg
                and legacy_codebook_size != max(self.codebook_sizes_by_field)
            ):
                raise ValueError(
                    "zero-anchor codebook_size must equal max("
                    "codebook_size_by_field) when both are configured"
                )
            self.codebook_size_by_field_is_explicit = True
        # Parameters use a rectangular max-K backing tensor. Every route and
        # diagnostic slices it to the current field's active K_f.
        self.codebook_size = max(self.codebook_sizes_by_field)
        self.multi_codebook_mode = str(
            cfg.get("multi_codebook_mode", "serial_residual")
        ).lower()
        self.residual_level_scale_decay = float(
            cfg.get("residual_level_scale_decay", 1.0)
        )
        self.residual_level_temperature_decay = float(
            cfg.get("residual_level_temperature_decay", 1.0)
        )
        self.base_embedding_mode = str(
            cfg.get("base_embedding_mode", "learned")
        ).lower()
        raw_base_modes_by_field = cfg.get(
            "base_embedding_mode_by_field", {}
        )
        if not isinstance(raw_base_modes_by_field, dict):
            raise ValueError(
                "zero-anchor base_embedding_mode_by_field must be a mapping"
            )
        unknown_base_mode_fields = sorted(
            set(raw_base_modes_by_field) - set(self.identity_fields)
        )
        if unknown_base_mode_fields:
            raise ValueError(
                "zero-anchor base_embedding_mode_by_field contains unknown "
                f"identity fields: {unknown_base_mode_fields}"
            )
        self.base_embedding_modes_by_field = {
            field: str(
                raw_base_modes_by_field.get(field, self.base_embedding_mode)
            ).lower()
            for field in self.identity_fields
        }
        self.distance_backend = str(
            cfg.get("distance_backend", "broadcast")
        ).lower()
        self.distance_row_chunk_size = int(
            cfg.get("distance_row_chunk_size", 0)
        )
        self.compact_distance_outputs = bool(
            cfg.get("compact_distance_outputs", False)
        )
        self.sparse_batch_diagnostics = bool(
            cfg.get(
                "sparse_batch_diagnostics",
                self.compact_distance_outputs,
            )
        )
        self.defer_cumulative_diagnostics = bool(
            cfg.get(
                "defer_cumulative_diagnostics",
                self.compact_distance_outputs,
            )
        )
        self.margin = float(cfg.get("margin", 0.1))
        legacy_temperature = float(cfg.get("temperature", 1.0))
        self.temperature_start = float(
            cfg.get("temperature_start", legacy_temperature)
        )
        self.temperature_end = float(
            cfg.get("temperature_end", legacy_temperature)
        )
        self.current_temperature = self.temperature_start
        self.code_init_radius = float(cfg.get("code_init_radius", 0.2))
        self.codebook_loss_weight = float(cfg.get("codebook_loss_weight", 1.0))
        self.codebook_transform_mode = str(
            cfg.get("codebook_transform_mode", "none")
        ).lower()
        self.task_codebook_gradient_mode = str(
            cfg.get(
                "task_codebook_gradient_mode",
                "legacy_hard_plus_soft",
            )
        ).lower()
        codebook_release_is_explicit = "codebook_release_fraction" in cfg
        codebook_ramp_is_explicit = "codebook_ramp_fraction" in cfg
        if codebook_release_is_explicit != codebook_ramp_is_explicit:
            raise ValueError(
                "zero-anchor codebook release and ramp fractions must be "
                "configured together"
            )
        self.codebook_schedule_is_explicit = bool(
            codebook_release_is_explicit
        )
        if self.codebook_schedule_is_explicit:
            self.codebook_release_fraction = float(
                cfg["codebook_release_fraction"]
            )
            self.codebook_ramp_fraction = float(
                cfg["codebook_ramp_fraction"]
            )
            self.codebook_schedule_mode = "release_then_linear_ramp"
        else:
            self.codebook_release_fraction = 0.0
            self.codebook_ramp_fraction = 0.0
            self.codebook_schedule_mode = "constant"
        self.current_codebook_multiplier = (
            1.0
            if (
                self.codebook_release_fraction == 0.0
                and self.codebook_ramp_fraction == 0.0
            )
            else 0.0
        )
        legacy_code_bias_enabled = bool(
            cfg.get("code_logit_bias_enabled", False)
        )
        self.code_logit_bias_mode = str(
            cfg.get(
                "code_logit_bias_mode",
                "all_codes" if legacy_code_bias_enabled else "none",
            )
        ).lower()
        if self.code_logit_bias_mode not in {
            "none",
            "all_codes",
            "zero_state",
        }:
            raise ValueError(
                "zero-anchor code_logit_bias_mode must be "
                "'none', 'all_codes', or 'zero_state'"
            )
        self.code_logit_bias_enabled = (
            self.code_logit_bias_mode != "none"
        )
        self.commitment_weight = float(cfg.get("commitment_weight", 0.0))
        raw_commitment_weights_by_field = cfg.get(
            "commitment_weight_by_field", {}
        )
        if not isinstance(raw_commitment_weights_by_field, dict):
            raise ValueError(
                "zero-anchor commitment_weight_by_field must be a mapping"
            )
        unknown_commitment_fields = sorted(
            set(raw_commitment_weights_by_field) - set(self.identity_fields)
        )
        if unknown_commitment_fields:
            raise ValueError(
                "zero-anchor commitment_weight_by_field contains unknown "
                f"identity fields: {unknown_commitment_fields}"
            )
        self.commitment_weights_by_field = {
            field: float(
                raw_commitment_weights_by_field.get(
                    field, self.commitment_weight
                )
            )
            for field in self.identity_fields
        }
        if any(
            weight < 0.0
            for weight in self.commitment_weights_by_field.values()
        ):
            raise ValueError(
                "zero-anchor commitment weights must be non-negative"
            )
        self.commitment_warmup_fraction = float(
            cfg.get("commitment_warmup_fraction", 0.0)
        )
        release_is_explicit = "regularization_release_fraction" in cfg
        ramp_is_explicit = "regularization_ramp_fraction" in cfg
        if release_is_explicit != ramp_is_explicit:
            raise ValueError(
                "zero-anchor regularization release and ramp fractions must "
                "be configured together"
            )
        self.regularization_schedule_is_explicit = bool(release_is_explicit)
        if self.regularization_schedule_is_explicit:
            self.regularization_release_fraction = float(
                cfg["regularization_release_fraction"]
            )
            self.regularization_ramp_fraction = float(
                cfg["regularization_ramp_fraction"]
            )
            self.regularization_schedule_mode = "release_then_linear_ramp"
        else:
            self.regularization_release_fraction = 0.0
            self.regularization_ramp_fraction = self.commitment_warmup_fraction
            self.regularization_schedule_mode = "legacy_linear_warmup"
        self.current_regularization_multiplier = (
            1.0
            if (
                self.regularization_release_fraction == 0.0
                and self.regularization_ramp_fraction == 0.0
            )
            else 0.0
        )
        self.current_commitment_multiplier = (
            self.current_regularization_multiplier
        )
        self.zero_l2_weight = float(cfg.get("zero_l2_weight", 0.0))
        self.seed = int(cfg.get("seed", config["model"].get("tokenizer_seed", 2021)))
        self.private_row_initialization_mode = str(
            cfg.get("private_row_initialization_mode", "zero")
        ).lower()
        self.assignment_stability_mode = str(
            cfg.get("assignment_stability_mode", "free_nearest")
        ).lower()
        self.assignment_freeze_fraction = float(
            cfg.get("assignment_freeze_fraction", 0.0)
        )
        self.assignment_switch_relative_improvement = float(
            cfg.get("assignment_switch_relative_improvement", 0.0)
        )
        self.hard_routing_control_mode = str(
            cfg.get("hard_routing_control_mode", "learned_nearest")
        ).lower()
        self.stable_assignment_eval_source = "stored"
        self.current_progress = 0.0
        self.audit_enabled = bool(cfg.get("audit_enabled", False))
        self.audit_assignment_every = int(cfg.get("audit_assignment_every", 100))
        self.audit_anchor_observations = int(
            cfg.get("audit_anchor_observations", 4096)
        )
        self._audit_current_step = 0

        if self.embedding_dim <= 0:
            raise ValueError("zero-anchor embedding_dim must be positive")
        if self.num_subspaces <= 0:
            raise ValueError("zero-anchor num_subspaces must be positive")
        if self.num_residual_levels <= 0:
            raise ValueError("zero-anchor num_residual_levels must be positive")
        if self.multi_codebook_mode not in {
            "serial_residual",
            "parallel_additive",
            "product_residual",
        }:
            raise ValueError(
                "zero-anchor multi_codebook_mode must be "
                "'serial_residual', 'parallel_additive', or "
                "'product_residual'"
            )
        self.product_residual_enabled = bool(
            self.multi_codebook_mode == "product_residual"
        )
        if self.product_residual_enabled and not (
            self.num_subspaces > 1 and self.num_residual_levels > 1
        ):
            raise ValueError(
                "product_residual requires num_subspaces > 1 and "
                "num_residual_levels > 1"
            )
        if (
            self.num_subspaces > 1
            and self.num_residual_levels > 1
            and not self.product_residual_enabled
        ):
            raise ValueError(
                "product subspaces and residual levels cannot be enabled "
                "together unless multi_codebook_mode='product_residual'"
            )
        if (
            self.multi_codebook_mode == "parallel_additive"
            and self.num_residual_levels <= 1
        ):
            raise ValueError(
                "parallel_additive requires num_residual_levels > 1"
            )
        if (
            self.multi_codebook_mode == "parallel_additive"
            and self.num_subspaces > 1
        ):
            raise ValueError(
                "parallel_additive cannot be combined with product subspaces"
            )
        if self.embedding_dim % self.num_subspaces:
            raise ValueError(
                "zero-anchor embedding_dim must be divisible by num_subspaces"
            )
        if not 0.0 < self.residual_level_scale_decay <= 1.0:
            raise ValueError(
                "zero-anchor residual_level_scale_decay must be in (0,1]"
            )
        if not 0.0 < self.residual_level_temperature_decay <= 1.0:
            raise ValueError(
                "zero-anchor residual_level_temperature_decay must be in "
                "(0,1]"
            )
        allowed_base_embedding_modes = {
            "learned",
            "fixed_zero",
            "zero_center_gated",
            "split_zero_base",
            "split_oov_zero_base",
        }
        if self.base_embedding_mode not in allowed_base_embedding_modes:
            raise ValueError(
                "zero-anchor base_embedding_mode must be 'learned', "
                "'fixed_zero', 'zero_center_gated', 'split_zero_base', "
                "or 'split_oov_zero_base'"
            )
        invalid_field_modes = {
            field: mode
            for field, mode in self.base_embedding_modes_by_field.items()
            if mode not in allowed_base_embedding_modes
        }
        if invalid_field_modes:
            raise ValueError(
                "zero-anchor base_embedding_mode_by_field contains "
                f"unsupported modes: {invalid_field_modes}"
            )
        self.subspace_dim = self.embedding_dim // self.num_subspaces
        if min(self.codebook_sizes_by_field) < 2:
            raise ValueError(
                "zero-anchor codebook_size must include zero and nonzero"
            )
        if self.codebook_size_by_field_is_explicit and (
            self.num_subspaces != 1 or self.num_residual_levels != 1
        ):
            raise ValueError(
                "zero-anchor codebook_size_by_field currently supports M1 only"
            )
        if self.distance_backend not in {"broadcast", "gemm"}:
            raise ValueError(
                "zero-anchor distance_backend must be 'broadcast' or 'gemm'"
            )
        if self.distance_row_chunk_size < 0:
            raise ValueError(
                "zero-anchor distance_row_chunk_size must be non-negative"
            )
        if self.distance_row_chunk_size and self.distance_backend != "gemm":
            raise ValueError(
                "row-chunked zero-anchor distance requires distance_backend='gemm'"
            )
        if self.compact_distance_outputs != bool(
            self.distance_row_chunk_size
        ):
            raise ValueError(
                "compact_distance_outputs must be enabled exactly when "
                "distance_row_chunk_size is positive"
            )
        if self.compact_distance_outputs and (
            self.num_subspaces != 1 or self.num_residual_levels != 1
        ):
            raise ValueError(
                "compact row-chunked distance currently supports M1 only"
            )
        if self.compact_distance_outputs and self.code_logit_bias_enabled:
            raise ValueError(
                "compact row-chunked distance does not support code logit bias"
            )
        if self.compact_distance_outputs and self.audit_enabled:
            raise ValueError(
                "compact row-chunked distance uses the external frequency "
                "audit and cannot enable dense assignment audit"
            )
        allowed_task_codebook_gradient_modes = {
            "legacy_hard_plus_soft",
            "soft_only",
            "codebook_fit_only",
        }
        if (
            self.task_codebook_gradient_mode
            not in allowed_task_codebook_gradient_modes
        ):
            raise ValueError(
                "zero-anchor task_codebook_gradient_mode must be one of "
                f"{sorted(allowed_task_codebook_gradient_modes)}"
            )
        if (
            self.task_codebook_gradient_mode != "legacy_hard_plus_soft"
            and (self.num_subspaces != 1 or self.num_residual_levels != 1)
        ):
            raise ValueError(
                "non-legacy task-codebook gradient routing supports M1 only"
            )
        allowed_codebook_transform_modes = {
            "none",
            "shared_linear_residual",
        }
        if self.codebook_transform_mode not in allowed_codebook_transform_modes:
            raise ValueError(
                "zero-anchor codebook_transform_mode must be one of "
                f"{sorted(allowed_codebook_transform_modes)}"
            )
        if (
            self.codebook_transform_mode != "none"
            and (self.num_subspaces != 1 or self.num_residual_levels != 1)
        ):
            raise ValueError(
                "zero-anchor shared codebook transform supports M1 only"
            )
        allowed_private_row_initialization_modes = {
            "zero",
            "train_first_touch_deterministic_code",
            "train_first_touch_frozen_deterministic_code",
        }
        if (
            self.private_row_initialization_mode
            not in allowed_private_row_initialization_modes
        ):
            raise ValueError(
                "zero-anchor private_row_initialization_mode must be one of "
                f"{sorted(allowed_private_row_initialization_modes)}"
            )
        if (
            self.private_row_initialization_mode
            in {
                "train_first_touch_deterministic_code",
                "train_first_touch_frozen_deterministic_code",
            }
            and (self.num_subspaces != 1 or self.num_residual_levels != 1)
        ):
            raise ValueError(
                "train-first-touch private-row initialization supports M1 only"
            )
        allowed_assignment_stability_modes = {
            "free_nearest",
            "freeze_after_fraction",
            "relative_hysteresis",
        }
        if self.assignment_stability_mode not in allowed_assignment_stability_modes:
            raise ValueError(
                "zero-anchor assignment_stability_mode must be one of "
                f"{sorted(allowed_assignment_stability_modes)}"
            )
        if self.assignment_stability_mode != "free_nearest" and (
            self.private_row_initialization_mode
            != "train_first_touch_deterministic_code"
        ):
            raise ValueError(
                "stable assignment modes require "
                "private_row_initialization_mode="
                "'train_first_touch_deterministic_code'"
            )
        if not 0.0 <= self.assignment_freeze_fraction <= 1.0:
            raise ValueError(
                "zero-anchor assignment_freeze_fraction must be in [0,1]"
            )
        if self.assignment_stability_mode == "freeze_after_fraction" and (
            self.assignment_freeze_fraction <= 0.0
        ):
            raise ValueError(
                "freeze_after_fraction requires a positive "
                "assignment_freeze_fraction"
            )
        if not 0.0 <= self.assignment_switch_relative_improvement < 1.0:
            raise ValueError(
                "zero-anchor assignment_switch_relative_improvement must be "
                "in [0,1)"
            )
        if self.assignment_stability_mode == "relative_hysteresis" and (
            self.assignment_switch_relative_improvement <= 0.0
        ):
            raise ValueError(
                "relative_hysteresis requires a positive "
                "assignment_switch_relative_improvement"
            )
        allowed_hard_routing_control_modes = {
            "learned_nearest",
            "externally_configured_fixed_zero_gate",
        }
        if (
            self.hard_routing_control_mode
            not in allowed_hard_routing_control_modes
        ):
            raise ValueError(
                "zero-anchor hard_routing_control_mode must be one of "
                f"{sorted(allowed_hard_routing_control_modes)}"
            )
        if (
            self.hard_routing_control_mode
            == "externally_configured_fixed_zero_gate"
            and (
                self.num_subspaces != 1
                or self.num_residual_levels != 1
                or self.assignment_stability_mode != "free_nearest"
                or self.private_row_initialization_mode
                == "train_first_touch_frozen_deterministic_code"
            )
        ):
            raise ValueError(
                "externally configured fixed-zero routing requires M1 with "
                "stateless free-nearest assignments"
            )
        if self.margin <= 0:
            raise ValueError("zero-anchor margin must be positive")
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("zero-anchor temperatures must be positive")
        if not 0.0 <= self.codebook_release_fraction <= 1.0:
            raise ValueError(
                "zero-anchor codebook_release_fraction must be in [0,1]"
            )
        if not 0.0 <= self.codebook_ramp_fraction <= 1.0:
            raise ValueError(
                "zero-anchor codebook_ramp_fraction must be in [0,1]"
            )
        if (
            self.codebook_release_fraction
            + self.codebook_ramp_fraction
            > 1.0
        ):
            raise ValueError(
                "zero-anchor codebook release_fraction + ramp_fraction "
                "must be <= 1"
            )
        if not 0.0 <= self.commitment_warmup_fraction <= 1.0:
            raise ValueError(
                "zero-anchor commitment_warmup_fraction must be in [0,1]"
            )
        if not 0.0 <= self.regularization_release_fraction <= 1.0:
            raise ValueError(
                "zero-anchor regularization_release_fraction must be in [0,1]"
            )
        if not 0.0 <= self.regularization_ramp_fraction <= 1.0:
            raise ValueError(
                "zero-anchor regularization_ramp_fraction must be in [0,1]"
            )
        if (
            self.regularization_release_fraction
            + self.regularization_ramp_fraction
            > 1.0
        ):
            raise ValueError(
                "zero-anchor release_fraction + ramp_fraction must be <= 1"
            )
        if self.code_init_radius <= self.margin:
            raise ValueError(
                "zero-anchor code_init_radius must be strictly greater than margin"
            )
        if min(
            self.codebook_loss_weight,
            self.commitment_weight,
            self.zero_l2_weight,
        ) < 0:
            raise ValueError("zero-anchor loss weights must be non-negative")
        if self.codebook_loss_weight <= 0:
            raise ValueError(
                "zero-anchor codebook_loss_weight must stay positive in every arm"
            )
        if self.num_residual_levels > 1 and self.code_logit_bias_enabled:
            raise ValueError(
                "multi-level residual quantization does not support code logit bias"
            )
        if self.audit_assignment_every <= 0:
            raise ValueError(
                "zero-anchor audit_assignment_every must be positive"
            )
        if self.audit_anchor_observations <= 0:
            raise ValueError(
                "zero-anchor audit_anchor_observations must be positive"
            )
        self.assignment_axis_count = (
            self.num_subspaces * self.num_residual_levels
            if self.product_residual_enabled
            else (
                self.num_subspaces
                if self.num_subspaces > 1
                else self.num_residual_levels
            )
        )
        self.assignment_axis_kind = (
            "product_residual"
            if self.product_residual_enabled
            else (
                "subspace"
                if self.num_subspaces > 1
                else (
                    (
                        "additive_codebook"
                        if self.multi_codebook_mode == "parallel_additive"
                        else "level"
                    )
                    if self.num_residual_levels > 1
                    else "single"
                )
            )
        )
        audit_slots = len(self.identity_fields) * self.assignment_axis_count
        self._audit_anchor_observation_count = [0 for _ in range(audit_slots)]
        self._audit_anchor_count = [0 for _ in range(audit_slots)]
        self._audit_anchor_frozen = [False for _ in range(audit_slots)]

        special_ids = dict(self.DEFAULT_SPECIAL_IDS)
        special_ids.update(cfg.get("special_token_ids", {}))
        if sorted(special_ids.values()) != list(range(len(special_ids))):
            raise ValueError(
                "zero-anchor special token IDs must be distinct contiguous values "
                "starting at zero"
            )
        self.special_token_ids = special_ids
        self.first_private_id = len(special_ids)

        selected_cardinalities = []
        for field in self.identity_fields:
            cardinality = int(cardinalities[self.name_to_index[field]])
            if cardinality <= self.first_private_id:
                raise ValueError(
                    f"zero-anchor field {field} has no private rows: {cardinality}"
                )
            selected_cardinalities.append(cardinality)
        self.selected_cardinalities = tuple(selected_cardinalities)
        self._train_first_touch_seen_buffer_names: list[str] = []
        self._stable_assignment_buffer_names: list[str] = []
        self._fixed_zero_gate_buffer_names: list[str] = []
        self._fixed_zero_gate_reports: dict[str, dict] = {}
        if (
            self.hard_routing_control_mode
            == "externally_configured_fixed_zero_gate"
        ):
            for position in range(len(self.identity_fields)):
                buffer_name = f"_fixed_zero_gate_{position}"
                self.register_buffer(
                    buffer_name,
                    torch.empty(0, dtype=torch.bool),
                    persistent=False,
                )
                self._fixed_zero_gate_buffer_names.append(buffer_name)
        if (
            self.private_row_initialization_mode
            in {
                "train_first_touch_deterministic_code",
                "train_first_touch_frozen_deterministic_code",
            }
        ):
            for position, cardinality in enumerate(self.selected_cardinalities):
                buffer_name = f"_train_first_touch_seen_{position}"
                self.register_buffer(
                    buffer_name,
                    torch.zeros(cardinality, dtype=torch.bool),
                    persistent=True,
                )
                self._train_first_touch_seen_buffer_names.append(buffer_name)
                if self.assignment_stability_mode != "free_nearest":
                    assignment_buffer_name = f"_stable_assignment_{position}"
                    self.register_buffer(
                        assignment_buffer_name,
                        torch.full(
                            (cardinality,),
                            -1,
                            dtype=torch.int32,
                        ),
                        persistent=True,
                    )
                    self._stable_assignment_buffer_names.append(
                        assignment_buffer_name
                    )

        field_count = len(self.identity_fields)
        code_shape = (
            (
                field_count,
                self.num_subspaces,
                self.num_residual_levels,
                1,
                self.subspace_dim,
            )
            if self.product_residual_enabled
            else (
                (
                    field_count,
                    self.num_subspaces,
                    1,
                    self.subspace_dim,
                )
                if self.num_subspaces > 1
                else (
                    (
                        field_count,
                        self.num_residual_levels,
                        1,
                        self.embedding_dim,
                    )
                    if self.num_residual_levels > 1
                    else (field_count, 1, self.embedding_dim)
                )
            )
        )
        self.register_buffer("zero_codes", torch.zeros(code_shape))
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        direction_shape = (
            (
                field_count,
                self.num_subspaces,
                self.num_residual_levels,
                self.codebook_size - 1,
                self.subspace_dim,
            )
            if self.product_residual_enabled
            else (
                (
                    field_count,
                    self.num_subspaces,
                    self.codebook_size - 1,
                    self.subspace_dim,
                )
                if self.num_subspaces > 1
                else (
                    (
                        field_count,
                        self.num_residual_levels,
                        self.codebook_size - 1,
                        self.embedding_dim,
                    )
                    if self.num_residual_levels > 1
                    else (
                        field_count,
                        self.codebook_size - 1,
                        self.embedding_dim,
                    )
                )
            )
        )
        directions = torch.randn(*direction_shape, generator=generator)
        self.nonzero_directions = nn.Parameter(directions)
        if self.num_subspaces == 1 and self.num_residual_levels == 1:
            initial_norms = torch.linalg.vector_norm(
                directions, dim=-1, keepdim=True
            )
            initial_fallback = torch.zeros_like(directions)
            initial_fallback[..., 0] = 1.0
            initial_units = torch.where(
                initial_norms.gt(1e-12),
                directions / initial_norms.clamp_min(1e-12),
                initial_fallback,
            )
            initial_nonzero_centers = initial_units * self.code_init_radius
        else:
            initial_nonzero_centers = torch.empty(0)
        self.register_buffer(
            "_initial_m1_nonzero_centers",
            initial_nonzero_centers,
            persistent=False,
        )
        radius_offset = self.code_init_radius - self.margin
        inverse_softplus = math.log(math.expm1(radius_offset))
        radius_shape = (
            (
                field_count,
                self.num_subspaces,
                self.num_residual_levels,
                self.codebook_size - 1,
            )
            if self.product_residual_enabled
            else (
                (
                    field_count,
                    self.num_subspaces,
                    self.codebook_size - 1,
                )
                if self.num_subspaces > 1
                else (
                    (
                        field_count,
                        self.num_residual_levels,
                        self.codebook_size - 1,
                    )
                    if self.num_residual_levels > 1
                    else (field_count, self.codebook_size - 1)
                )
            )
        )
        self.nonzero_radius_logits = nn.Parameter(
            torch.full(radius_shape, inverse_softplus)
        )
        if self.codebook_transform_mode == "shared_linear_residual":
            self.shared_codebook_transform_delta = nn.Parameter(
                torch.zeros(
                    field_count,
                    self.embedding_dim,
                    self.embedding_dim,
                )
            )
        else:
            self.register_parameter(
                "shared_codebook_transform_delta",
                None,
            )
        if self.num_residual_levels > 1:
            with torch.no_grad():
                for level in range(self.num_residual_levels):
                    scale = self._multi_codebook_scale(level)
                    level_offset = (
                        self.code_init_radius * scale - self.margin * scale
                    )
                    level_inverse_softplus = math.log(
                        math.expm1(level_offset)
                    )
                    if self.product_residual_enabled:
                        self.nonzero_radius_logits[:, :, level].fill_(
                            level_inverse_softplus
                        )
                    else:
                        self.nonzero_radius_logits[:, level].fill_(
                            level_inverse_softplus
                        )
        self.base_embeddings = nn.Parameter(
            torch.zeros(field_count, self.embedding_dim)
        )
        self.zero_state_base_embeddings = nn.Parameter(
            torch.zeros(field_count, self.embedding_dim)
        )
        self.oov_base_embeddings = nn.Parameter(
            torch.zeros(field_count, self.embedding_dim)
        )
        self.missing_embeddings = nn.Parameter(
            torch.zeros(field_count, self.embedding_dim)
        )
        self.reserved_zero_embeddings = nn.Parameter(
            torch.zeros(field_count, self.embedding_dim)
        )
        self.code_logit_biases = nn.Parameter(
            torch.zeros(
                field_count,
                self.num_subspaces,
                self.codebook_size,
            ),
            requires_grad=self.code_logit_bias_mode == "all_codes",
        )
        self.missing_logit_biases = nn.Parameter(
            torch.zeros(field_count),
            requires_grad=self.code_logit_bias_mode == "all_codes",
        )
        self.reserved_zero_logit_biases = nn.Parameter(
            torch.zeros(field_count),
            requires_grad=self.code_logit_bias_mode == "all_codes",
        )
        self.zero_state_logit_biases = nn.Parameter(
            torch.zeros(field_count),
            requires_grad=self.code_logit_bias_mode == "zero_state",
        )

        assignment_shape = (
            (field_count, self.codebook_size)
            if self.assignment_axis_count == 1
            else (
                field_count,
                self.assignment_axis_count,
                self.codebook_size,
            )
        )
        self.register_buffer(
            "assignment_counts",
            torch.zeros(*assignment_shape, dtype=torch.long),
            persistent=True,
        )
        self._multilevel_prefix_buffer_names: list[str] = []
        if self.num_residual_levels > 1:
            for prefix_length in range(1, self.num_residual_levels + 1):
                name = f"multilevel_prefix_counts_{prefix_length}"
                self.register_buffer(
                    name,
                    torch.zeros(
                        field_count,
                        *(
                            (
                                self.num_subspaces,
                                self.codebook_size**prefix_length,
                            )
                            if self.product_residual_enabled
                            else (self.codebook_size**prefix_length,)
                        ),
                        dtype=torch.long,
                    ),
                    persistent=True,
                )
                self._multilevel_prefix_buffer_names.append(name)
        if self.audit_enabled:
            self.register_buffer(
                "_audit_soft_metric_sums",
                torch.zeros(audit_slots, 3, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_audit_soft_metric_counts",
                torch.zeros(audit_slots, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_audit_anchor_ids",
                torch.full(
                    (audit_slots, self.audit_anchor_observations),
                    -1,
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_audit_anchor_last_codes",
                torch.full(
                    (audit_slots, self.audit_anchor_observations),
                    -1,
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_audit_churn_comparisons",
                torch.zeros(audit_slots, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_audit_churn_changes",
                torch.zeros(audit_slots, dtype=torch.long),
                persistent=False,
            )
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_logit_bias: Optional[torch.Tensor] = None
        self._last_loss_components: dict[str, torch.Tensor] = {}
        self._diagnostics: dict[str, torch.Tensor] = {}
        self._metadata = {
            "enabled": True,
            "identity_fields": list(self.identity_fields),
            "field_indices": [
                self.name_to_index[name] for name in self.identity_fields
            ],
            "selected_cardinalities": list(self.selected_cardinalities),
            "embedding_dim": self.embedding_dim,
            "codebook_size": self.codebook_size,
            "codebook_size_by_field": {
                field: self.codebook_sizes_by_field[position]
                for position, field in enumerate(self.identity_fields)
            },
            "codebook_size_by_field_is_explicit": (
                self.codebook_size_by_field_is_explicit
            ),
            "codebook_storage_size": self.codebook_size,
            "codebook_storage_layout": "max_k_padded_active_prefix",
            "active_nonzero_centers_total": sum(
                size - 1 for size in self.codebook_sizes_by_field
            ),
            "allocated_nonzero_centers_total": (
                len(self.identity_fields) * (self.codebook_size - 1)
            ),
            "num_subspaces": self.num_subspaces,
            "num_residual_levels": self.num_residual_levels,
            "multi_codebook_mode": self.multi_codebook_mode,
            "residual_level_scale_decay": self.residual_level_scale_decay,
            "residual_level_temperature_decay": (
                self.residual_level_temperature_decay
            ),
            "base_embedding_mode": self.base_embedding_mode,
            "base_embedding_mode_by_field": dict(
                self.base_embedding_modes_by_field
            ),
            "zero_state_base_is_separate": (
                any(
                    mode in {"split_zero_base", "split_oov_zero_base"}
                    for mode in self.base_embedding_modes_by_field.values()
                )
            ),
            "oov_base_is_separate": (
                any(
                    mode == "split_oov_zero_base"
                    for mode in self.base_embedding_modes_by_field.values()
                )
            ),
            "oov_uses_quantized_zero_path_by_field": {
                field: mode != "split_oov_zero_base"
                for field, mode in self.base_embedding_modes_by_field.items()
            },
            "assignment_axis_kind": self.assignment_axis_kind,
            "subspace_dim": self.subspace_dim,
            "maximum_code_combinations_per_field": (
                self.codebook_size**self.assignment_axis_count
            ),
            "maximum_code_combinations_by_field": {
                field: (
                    self.codebook_sizes_by_field[position]
                    ** self.assignment_axis_count
                )
                for position, field in enumerate(self.identity_fields)
            },
            "distance_backend": self.distance_backend,
            "distance_row_chunk_size": self.distance_row_chunk_size,
            "compact_distance_outputs": self.compact_distance_outputs,
            "sparse_batch_diagnostics": self.sparse_batch_diagnostics,
            "defer_cumulative_diagnostics": (
                self.defer_cumulative_diagnostics
            ),
            "margin": self.margin,
            "temperature_start": self.temperature_start,
            "temperature_end": self.temperature_end,
            "code_init_radius": self.code_init_radius,
            "initialization_rng_isolated": bool(
                cfg.get("isolate_initialization_rng", False)
            ),
            "initialization_seed": int(
                cfg.get(
                    "initialization_seed",
                    config["model"].get(
                        "tokenizer_seed", config.get("seed", 2021)
                    ),
                )
            ),
            "codebook_loss_weight": self.codebook_loss_weight,
            "codebook_transform": {
                "mode": self.codebook_transform_mode,
                "applies_to": ["nonzero_center_directions_only"],
                "cannot_change_radii": True,
                "zero_center_fixed": True,
                "inference_assignment": "stateless_nearest_center",
                "parameters_per_field": (
                    self.embedding_dim * self.embedding_dim
                    if self.codebook_transform_mode
                    == "shared_linear_residual"
                    else 0
                ),
                "effective_parameters_per_field": (
                    self.embedding_dim * self.embedding_dim - 1
                    if self.codebook_transform_mode
                    == "shared_linear_residual"
                    else 0
                ),
                "isotropic_scale_null_dimensions_per_field": (
                    1
                    if self.codebook_transform_mode
                    == "shared_linear_residual"
                    else 0
                ),
                "minimum_singular_ratio_tripwire": (
                    self.SHARED_TRANSFORM_MIN_SINGULAR_RATIO
                    if self.codebook_transform_mode
                    == "shared_linear_residual"
                    else None
                ),
            },
            "task_codebook_gradient_mode": (
                self.task_codebook_gradient_mode
            ),
            "codebook_schedule": {
                "mode": self.codebook_schedule_mode,
                "explicit": self.codebook_schedule_is_explicit,
                "release_fraction": self.codebook_release_fraction,
                "ramp_fraction": self.codebook_ramp_fraction,
                "applies_to": ["codebook_loss"],
            },
            "code_logit_bias_enabled": self.code_logit_bias_enabled,
            "code_logit_bias_mode": self.code_logit_bias_mode,
            "commitment_weight": self.commitment_weight,
            "commitment_weight_by_field": dict(
                self.commitment_weights_by_field
            ),
            "commitment_warmup_fraction": self.commitment_warmup_fraction,
            "regularization_schedule": {
                "mode": self.regularization_schedule_mode,
                "explicit": self.regularization_schedule_is_explicit,
                "release_fraction": self.regularization_release_fraction,
                "ramp_fraction": self.regularization_ramp_fraction,
                "applies_to": [
                    "commitment",
                    "touched_row_l2_if_enabled",
                    "whole_table_l2_if_enabled",
                ],
            },
            "zero_l2_weight": self.zero_l2_weight,
            "private_row_initialization": {
                "mode": self.private_row_initialization_mode,
                "training_only_mutation": (
                    self.private_row_initialization_mode
                    in {
                        "train_first_touch_deterministic_code",
                        "train_first_touch_frozen_deterministic_code",
                    }
                ),
                "hard_assignment_route": (
                    "deterministic_hash"
                    if self.private_row_initialization_mode
                    == "train_first_touch_frozen_deterministic_code"
                    else "nearest_center"
                ),
                "backward_surrogate_route": (
                    "assigned_center_identity_st"
                    if self.private_row_initialization_mode
                    == "train_first_touch_frozen_deterministic_code"
                    else "nearest_center_soft_st"
                ),
                "validation_unseen_stays_zero": True,
                "uses_offline_frequency": False,
                "uses_tail_key_oracle": False,
            },
            "assignment_stability": {
                "mode": self.assignment_stability_mode,
                "freeze_fraction": self.assignment_freeze_fraction,
                "switch_relative_improvement": (
                    self.assignment_switch_relative_improvement
                ),
                "persistent_per_id_state": (
                    self.assignment_stability_mode != "free_nearest"
                ),
                "default_eval_source": self.stable_assignment_eval_source,
                "backward_surrogate": "nearest_center_soft_st",
            },
            "hard_routing_control": {
                "mode": self.hard_routing_control_mode,
                "configured": (
                    self.hard_routing_control_mode == "learned_nearest"
                ),
                "persistent_per_id_state": False,
                "soft_st_distribution_unchanged": True,
                "special_rows_unchanged": True,
                "fields": {},
            },
            "audit_enabled": self.audit_enabled,
            "audit_assignment_every": self.audit_assignment_every,
            "audit_anchor_observations": self.audit_anchor_observations,
            "special_token_ids": dict(self.special_token_ids),
            "forward": "hard_center",
            "backward": (
                "assigned_center_identity_st"
                if self.private_row_initialization_mode
                == "train_first_touch_frozen_deterministic_code"
                else "soft_straight_through"
            ),
            "continuous_residual": False,
            "final_remaining_residual_enters_main_network": False,
            "zero_code_is_buffer": True,
            "production_ready": False,
            "offline_unseen_approximation": "shared_oov_zero_row",
        }

    def codebook_size_for_field(self, field_position: int) -> int:
        """Return the active center count K_f for one identity field."""
        if not 0 <= int(field_position) < len(self.identity_fields):
            raise ValueError("invalid zero-anchor identity field position")
        return self.codebook_sizes_by_field[int(field_position)]

    def configure_fixed_zero_gate(
        self,
        field: str,
        gate_mask: torch.Tensor,
        *,
        policy_report: Optional[dict] = None,
    ) -> dict:
        """Install a non-persistent, diagnostic-only per-ID zero gate."""
        if (
            self.hard_routing_control_mode
            != "externally_configured_fixed_zero_gate"
        ):
            raise RuntimeError(
                "fixed zero gates require the externally configured routing "
                "control mode"
            )
        if field not in self.identity_fields:
            raise KeyError(f"unknown zero-anchor identity field {field!r}")
        position = self.identity_fields.index(field)
        cardinality = self.selected_cardinalities[position]
        if gate_mask.ndim != 1 or gate_mask.numel() != cardinality:
            raise ValueError(
                f"fixed zero gate for {field} must have shape "
                f"({cardinality},), got {tuple(gate_mask.shape)}"
            )
        if gate_mask.dtype != torch.bool:
            raise ValueError("fixed zero gate must use torch.bool")
        normalized = gate_mask.detach().to(
            device=self.zero_codes.device,
            dtype=torch.bool,
        ).contiguous()
        if normalized[: self.first_private_id].any():
            raise ValueError("fixed zero gate cannot modify special rows")
        buffer_name = self._fixed_zero_gate_buffer_names[position]
        setattr(self, buffer_name, normalized)
        raw = bytes(normalized.cpu().to(dtype=torch.uint8))
        report = dict(policy_report or {})
        report.update(
            {
                "cardinality": cardinality,
                "first_private_id": self.first_private_id,
                "gated_private_ids": int(
                    normalized[self.first_private_id :].sum().item()
                ),
                "ungated_private_ids": int(
                    (~normalized[self.first_private_id :]).sum().item()
                ),
                "gate_mask_sha256": hashlib.sha256(raw).hexdigest(),
                "buffer_persistent": False,
            }
        )
        self._fixed_zero_gate_reports[field] = report
        routing_metadata = self._metadata["hard_routing_control"]
        routing_metadata["configured"] = (
            len(self._fixed_zero_gate_reports) == len(self.identity_fields)
        )
        routing_metadata["fields"] = dict(self._fixed_zero_gate_reports)
        return json.loads(json.dumps(report))

    def hard_routing_control_masks(
        self,
        ids: torch.Tensor,
        field_position: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return forced-zero and forced-nonzero masks for one ID batch."""
        if self.hard_routing_control_mode == "learned_nearest":
            return None, None
        if not 0 <= int(field_position) < len(self.identity_fields):
            raise ValueError("invalid zero-anchor identity field position")
        cardinality = self.selected_cardinalities[int(field_position)]
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError("routing-control IDs must be a rank-1 long tensor")
        if ids.numel() and (
            int(ids.min().item()) < 0 or int(ids.max().item()) >= cardinality
        ):
            raise ValueError("routing-control ID exceeds field cardinality")
        gate = getattr(
            self,
            self._fixed_zero_gate_buffer_names[int(field_position)],
        )
        if gate.numel() != cardinality:
            field = self.identity_fields[int(field_position)]
            raise RuntimeError(
                f"fixed zero gate for {field} was not configured"
            )
        force_zero = gate.index_select(0, ids)
        regular = ids.ge(self.first_private_id)
        if (force_zero & ~regular).any():
            raise RuntimeError(
                "fixed zero gate unexpectedly selected a special row"
            )
        force_nonzero = regular & ~force_zero
        return force_zero, force_nonzero

    def _multi_codebook_scale(self, position: int) -> float:
        if self.multi_codebook_mode == "parallel_additive":
            return 1.0 / float(self.num_residual_levels)
        return self.residual_level_scale_decay**position

    def initialize_identity_embeddings(
        self, embeddings: nn.ModuleList
    ) -> None:
        """Zero every selected private table without changing other fields."""
        for field in self.identity_fields:
            index = self.name_to_index[field]
            embedding = embeddings[index]
            if embedding.weight.size(0) != self.selected_cardinalities[
                self.identity_fields.index(field)
            ]:
                raise ValueError(f"embedding cardinality changed for {field}")
            with torch.no_grad():
                embedding.weight.zero_()

    def deterministic_private_code_indices(
        self,
        ids: torch.Tensor,
        field_position: int = 0,
    ) -> torch.Tensor:
        """Map private IDs deterministically onto nonzero M1 centers."""
        private_ids = ids.to(dtype=torch.long) - self.first_private_id
        hashed = (
            private_ids * 1_103_515_245 + (self.seed & 0x7FFFFFFF)
        ) % 2_147_483_647
        return 1 + hashed.remainder(
            self.codebook_size_for_field(field_position) - 1
        )

    def _first_touch_code_indices(
        self,
        ids: torch.Tensor,
        field_position: int = 0,
    ) -> torch.Tensor:
        """Backward-compatible alias for the deterministic initializer."""
        return self.deterministic_private_code_indices(
            ids,
            field_position=field_position,
        )

    def initialize_train_first_touch_rows(
        self,
        embeddings: nn.ModuleList,
        sparse: torch.Tensor,
    ) -> None:
        """Initialize newly observed training IDs at deterministic nonzero codes."""
        if (
            self.private_row_initialization_mode
            not in {
                "train_first_touch_deterministic_code",
                "train_first_touch_frozen_deterministic_code",
            }
            or not self.training
        ):
            return
        if sparse.ndim != 2 or sparse.size(1) != len(self.field_names):
            raise ValueError(
                "train-first-touch sparse IDs must have shape "
                f"(B,{len(self.field_names)}), got {tuple(sparse.shape)}"
            )

        with torch.no_grad():
            for position, field in enumerate(self.identity_fields):
                field_index = self.name_to_index[field]
                embedding = embeddings[field_index]
                cardinality = self.selected_cardinalities[position]
                if embedding.weight.size(0) != cardinality:
                    raise ValueError(f"embedding cardinality changed for {field}")

                ids = sparse[:, field_index].to(dtype=torch.long)
                regular = ids.ge(self.first_private_id) & ids.lt(cardinality)
                if not regular.any():
                    continue
                unique_ids = torch.unique(ids[regular])
                seen = getattr(
                    self, self._train_first_touch_seen_buffer_names[position]
                )
                new_ids = unique_ids[~seen.index_select(0, unique_ids)]
                if new_ids.numel() == 0:
                    continue

                centers = self.codebook(position).detach().to(
                    device=embedding.weight.device,
                    dtype=embedding.weight.dtype,
                )
                code_indices = self._first_touch_code_indices(
                    new_ids,
                    field_position=position,
                )
                embedding.weight.index_copy_(
                    0,
                    new_ids,
                    centers.index_select(0, code_indices),
                )
                seen.index_fill_(0, new_ids, True)
                if self.assignment_stability_mode != "free_nearest":
                    assignments = getattr(
                        self,
                        self._stable_assignment_buffer_names[position],
                    )
                    assignments.index_copy_(
                        0,
                        new_ids,
                        code_indices.to(dtype=assignments.dtype),
                    )

    def _apply_stable_assignments(
        self,
        *,
        field_position: int,
        ids: torch.Tensor,
        regular_mask: torch.Tensor,
        residuals: torch.Tensor,
        quantized: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply persistent assignment policy while retaining soft-ST gradients."""
        if self.assignment_stability_mode == "free_nearest":
            zero = residuals.new_zeros(())
            return quantized, zero, zero

        assignment_buffer = getattr(
            self,
            self._stable_assignment_buffer_names[field_position],
        )
        stored = assignment_buffer.index_select(0, ids).to(dtype=torch.long)
        nearest = quantized["indices"]
        if nearest.ndim != 1:
            raise RuntimeError("stable assignment modes support M1 only")
        selected = torch.where(stored.ge(0), stored, torch.zeros_like(stored))
        considered = regular_mask & stored.ge(0)
        switched = torch.zeros_like(regular_mask)

        if self.training:
            if (
                self.assignment_stability_mode == "freeze_after_fraction"
                and self.current_progress < self.assignment_freeze_fraction
            ):
                selected = torch.where(regular_mask, nearest, selected)
                switched = considered & selected.ne(stored)
            elif self.assignment_stability_mode == "relative_hysteresis":
                codebook = quantized["codebook"]
                safe_stored = stored.clamp_min(0)
                current_values = codebook.index_select(0, safe_stored)
                candidate_values = codebook.index_select(0, nearest)
                current_distance = (
                    residuals.float() - current_values.float()
                ).square().sum(dim=-1)
                candidate_distance = (
                    residuals.float() - candidate_values.float()
                ).square().sum(dim=-1)
                switch_allowed = candidate_distance.le(
                    current_distance
                    * (1.0 - self.assignment_switch_relative_improvement)
                )
                switched = (
                    considered
                    & nearest.ne(stored)
                    & switch_allowed
                )
                selected = torch.where(switched, nearest, selected)

            update_mask = regular_mask & selected.ge(0)
            if update_mask.any():
                with torch.no_grad():
                    assignment_buffer.index_copy_(
                        0,
                        ids[update_mask],
                        selected[update_mask].to(
                            dtype=assignment_buffer.dtype
                        ),
                    )
        elif self.stable_assignment_eval_source == "nearest":
            selected = torch.where(regular_mask, nearest, selected)

        hard = quantized["codebook"].index_select(0, selected)
        hard_for_task = (
            hard
            if self.task_codebook_gradient_mode == "legacy_hard_plus_soft"
            else hard.detach()
        )
        soft = quantized["soft_value"].to(dtype=hard.dtype)
        quantized = dict(quantized)
        quantized.update(
            {
                "value": (
                    hard_for_task + (soft - soft.detach())
                ).to(dtype=residuals.dtype),
                "hard_value": hard.to(dtype=residuals.dtype),
                "indices": selected,
            }
        )
        denominator = considered.sum().clamp_min(1)
        switch_fraction = switched.sum().to(dtype=residuals.dtype) / denominator
        considered_fraction = considered.float().mean().to(dtype=residuals.dtype)
        return quantized, switch_fraction, considered_fraction

    def set_stable_assignment_eval_source(self, source: str) -> None:
        """Select stored or recomputed-nearest inference for an audit pass."""
        source = str(source).lower()
        if source not in {"stored", "nearest"}:
            raise ValueError(
                "stable assignment eval source must be 'stored' or 'nearest'"
            )
        if self.assignment_stability_mode == "free_nearest" and source != "nearest":
            raise RuntimeError(
                "free-nearest assignment has no persistent stored eval route"
            )
        self.stable_assignment_eval_source = source

    def codebook(
        self,
        field_position: int,
        subspace_position: Optional[int] = None,
        level_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Return constrained centers, with a fixed zero in every subspace."""
        if self.product_residual_enabled:
            if subspace_position is None and level_position is None:
                return torch.stack(
                    [
                        torch.stack(
                            [
                                self.codebook(
                                    field_position,
                                    subspace_position=subspace,
                                    level_position=level,
                                )
                                for level in range(self.num_residual_levels)
                            ],
                            dim=0,
                        )
                        for subspace in range(self.num_subspaces)
                    ],
                    dim=0,
                )
            if subspace_position is None or level_position is None:
                raise ValueError(
                    "product_residual codebook requires both subspace and level"
                )
            if not 0 <= subspace_position < self.num_subspaces:
                raise ValueError("invalid zero-anchor subspace position")
            if not 0 <= level_position < self.num_residual_levels:
                raise ValueError("invalid zero-anchor residual level")
            directions = self.nonzero_directions[
                field_position, subspace_position, level_position
            ]
            radius_logits = self.nonzero_radius_logits[
                field_position, subspace_position, level_position
            ]
            zero = self.zero_codes[
                field_position, subspace_position, level_position
            ]
            level_scale = self._multi_codebook_scale(level_position)
            level_margin = self.margin * level_scale
        elif self.num_residual_levels > 1:
            if subspace_position not in (None, 0):
                raise ValueError("multi-level mode has no product subspaces")
            if level_position is None:
                return torch.stack(
                    [
                        self.codebook(
                            field_position,
                            level_position=level,
                        )
                        for level in range(self.num_residual_levels)
                    ],
                    dim=0,
                )
            if not 0 <= level_position < self.num_residual_levels:
                raise ValueError("invalid zero-anchor residual level")
            directions = self.nonzero_directions[
                field_position, level_position
            ]
            radius_logits = self.nonzero_radius_logits[
                field_position, level_position
            ]
            zero = self.zero_codes[field_position, level_position]
            level_scale = self._multi_codebook_scale(level_position)
            level_margin = self.margin * level_scale
        else:
            level_margin = self.margin
        if (
            not self.product_residual_enabled
            and self.num_subspaces > 1
            and subspace_position is None
        ):
            return torch.stack(
                [
                    self.codebook(field_position, subspace)
                    for subspace in range(self.num_subspaces)
                ],
                dim=0,
            )
        if self.product_residual_enabled:
            pass
        elif self.num_residual_levels > 1:
            pass
        elif self.num_subspaces > 1:
            if subspace_position is None or not (
                0 <= subspace_position < self.num_subspaces
            ):
                raise ValueError("invalid zero-anchor subspace position")
            directions = self.nonzero_directions[
                field_position, subspace_position
            ]
            radius_logits = self.nonzero_radius_logits[
                field_position, subspace_position
            ]
            zero = self.zero_codes[field_position, subspace_position]
        else:
            if subspace_position not in (None, 0):
                raise ValueError("single-codebook mode only has subspace zero")
            directions = self.nonzero_directions[field_position]
            radius_logits = self.nonzero_radius_logits[field_position]
            zero = self.zero_codes[field_position]
        active_nonzero_centers = (
            self.codebook_size_for_field(field_position) - 1
        )
        directions = directions[..., :active_nonzero_centers, :]
        radius_logits = radius_logits[..., :active_nonzero_centers]
        if self.codebook_transform_mode == "shared_linear_residual":
            delta = self.shared_codebook_transform_delta[field_position]
            transform = torch.eye(
                self.embedding_dim,
                dtype=delta.dtype,
                device=delta.device,
            ) + delta
            directions = directions @ transform
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        fallback = torch.zeros_like(directions)
        fallback[..., 0] = 1.0
        unit = torch.where(
            norms.gt(1e-12),
            directions / norms.clamp_min(1e-12),
            fallback,
        )
        radii = level_margin + F.softplus(radius_logits)
        nonzero = unit * radii.unsqueeze(-1)
        zero = zero.to(
            dtype=nonzero.dtype, device=nonzero.device
        )
        return torch.cat((zero, nonzero), dim=0)

    def _axis_coordinates(self, axis: int) -> tuple[Optional[int], Optional[int]]:
        if not 0 <= axis < self.assignment_axis_count:
            raise ValueError("invalid zero-anchor assignment axis")
        if self.product_residual_enabled:
            return (
                axis // self.num_residual_levels,
                axis % self.num_residual_levels,
            )
        if self.num_subspaces > 1:
            return axis, None
        if self.num_residual_levels > 1:
            return None, axis
        return None, None

    def _axis_label(self, axis: int) -> str:
        subspace, level = self._axis_coordinates(axis)
        if subspace is not None and level is not None:
            return f"subspace_{subspace}_level_{level}"
        if subspace is not None:
            return f"subspace_{subspace}"
        if level is not None:
            return (
                f"additive_{level}"
                if self.multi_codebook_mode == "parallel_additive"
                else f"level_{level}"
            )
        return "single"

    def quantize_residuals(
        self,
        residuals: torch.Tensor,
        field_position: int,
        *,
        hard_indices: Optional[torch.Tensor] = None,
        force_zero_mask: Optional[torch.Tensor] = None,
        force_nonzero_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Hard center in the forward pass, soft nearest-center in backward."""
        field_codebook_size = self.codebook_size_for_field(field_position)
        if residuals.ndim != 2 or residuals.size(1) != self.embedding_dim:
            raise ValueError(
                "zero-anchor residuals must have shape "
                f"(B,{self.embedding_dim}), got {tuple(residuals.shape)}"
            )
        if hard_indices is not None:
            if self.num_subspaces != 1 or self.num_residual_levels != 1:
                raise ValueError("forced hard assignments support M1 only")
            if hard_indices.shape != residuals.shape[:1]:
                raise ValueError(
                    "forced hard assignments must have shape "
                    f"({residuals.size(0)},), got {tuple(hard_indices.shape)}"
                )
            if hard_indices.dtype != torch.long:
                raise ValueError("forced hard assignments must be torch.long")
            if hard_indices.numel() and (
                int(hard_indices.min().item()) < 0
                or int(hard_indices.max().item()) >= field_codebook_size
            ):
                raise ValueError("forced hard assignment index out of range")
        if (force_zero_mask is None) != (force_nonzero_mask is None):
            raise ValueError(
                "forced zero/nonzero routing masks must be provided together"
            )
        if force_zero_mask is not None:
            if hard_indices is not None:
                raise ValueError(
                    "forced routing masks cannot be combined with hard indices"
                )
            if self.num_subspaces != 1 or self.num_residual_levels != 1:
                raise ValueError("forced zero/nonzero routing supports M1 only")
            for label, mask in (
                ("zero", force_zero_mask),
                ("nonzero", force_nonzero_mask),
            ):
                if mask.shape != residuals.shape[:1] or mask.dtype != torch.bool:
                    raise ValueError(
                        f"forced {label} routing mask must be bool with shape "
                        f"({residuals.size(0)},)"
                    )
            if (force_zero_mask & force_nonzero_mask).any():
                raise ValueError("forced zero/nonzero routing masks overlap")
        if self.product_residual_enabled:
            return self._quantize_product_residual_levels(
                residuals,
                field_position,
            )
        if self.num_residual_levels > 1:
            if self.multi_codebook_mode == "parallel_additive":
                return self._quantize_parallel_additive_residuals(
                    residuals,
                    field_position,
                )
            return self._quantize_multilevel_residuals(
                residuals,
                field_position,
            )
        if self.num_subspaces > 1:
            return self._quantize_product_residuals(
                residuals,
                field_position,
            )
        codebook = self.codebook(field_position)
        if self.compact_distance_outputs:
            return self._quantize_single_codebook_row_chunked(
                residuals,
                codebook,
                hard_indices=hard_indices,
                force_zero_mask=force_zero_mask,
                force_nonzero_mask=force_nonzero_mask,
            )
        if hard_indices is not None:
            hard = codebook.index_select(0, hard_indices)
            hard_for_task = (
                hard
                if self.task_codebook_gradient_mode
                == "legacy_hard_plus_soft"
                else hard.detach()
            )
            surrogate = residuals.to(dtype=hard.dtype)
            straight_through = hard_for_task + (
                surrogate - surrogate.detach()
            )
            probabilities = F.one_hot(
                hard_indices,
                num_classes=field_codebook_size,
            ).to(dtype=torch.float32)
            return {
                "value": straight_through.to(dtype=residuals.dtype),
                "hard_value": hard.to(dtype=residuals.dtype),
                "soft_value": surrogate.to(dtype=residuals.dtype),
                "indices": hard_indices,
                "probabilities": probabilities,
                "distances": None,
                "codebook": codebook,
            }
        residuals_float = residuals.float()
        codebook_float = codebook.float()
        task_codebook_float = (
            codebook_float.detach()
            if self.task_codebook_gradient_mode == "codebook_fit_only"
            else codebook_float
        )
        if self.distance_backend == "broadcast":
            distances = (
                residuals_float.unsqueeze(1)
                - task_codebook_float.unsqueeze(0)
            ).square().sum(dim=-1)
        else:
            # Avoid materializing B x K x D when screening larger codebooks.
            distances = (
                residuals_float.square().sum(dim=-1, keepdim=True)
                + task_codebook_float.square().sum(dim=-1).unsqueeze(0)
                - 2.0
                * residuals_float
                @ task_codebook_float.transpose(0, 1)
            )
        indices = distances.argmin(dim=-1)
        if force_zero_mask is not None:
            nearest_nonzero = distances[:, 1:].argmin(dim=-1) + 1
            indices = torch.where(force_nonzero_mask, nearest_nonzero, indices)
            indices = torch.where(
                force_zero_mask,
                torch.zeros_like(indices),
                indices,
            )
        hard = codebook.index_select(0, indices)
        probabilities = torch.softmax(
            -distances / self.current_temperature, dim=-1
        )
        task_codebook = task_codebook_float.to(dtype=codebook.dtype)
        soft = probabilities.to(dtype=codebook.dtype) @ task_codebook
        hard_for_task = (
            hard
            if self.task_codebook_gradient_mode == "legacy_hard_plus_soft"
            else hard.detach()
        )
        straight_through = hard_for_task + (soft - soft.detach())
        return {
            "value": straight_through.to(dtype=residuals.dtype),
            "hard_value": hard.to(dtype=residuals.dtype),
            "soft_value": soft.to(dtype=residuals.dtype),
            "indices": indices,
            "probabilities": probabilities,
            "distances": distances,
            "codebook": codebook,
        }

    def _quantize_compact_row_chunk(
        self,
        residuals: torch.Tensor,
        codebook: torch.Tensor,
        hard_indices: Optional[torch.Tensor] = None,
        force_zero_mask: Optional[torch.Tensor] = None,
        force_nonzero_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Exact dense soft assignment for one checkpointed row chunk."""
        if hard_indices is not None and hard_indices.numel() > 0:
            hard = codebook.index_select(0, hard_indices)
            hard_for_task = (
                hard
                if self.task_codebook_gradient_mode
                == "legacy_hard_plus_soft"
                else hard.detach()
            )
            surrogate = residuals.to(dtype=hard.dtype)
            value = hard_for_task + (surrogate - surrogate.detach())
            return (
                value.to(dtype=residuals.dtype),
                hard.to(dtype=residuals.dtype),
                surrogate.to(dtype=residuals.dtype),
                hard_indices,
                residuals.new_ones(residuals.size(0), dtype=torch.float32),
            )
        residuals_float = residuals.float()
        codebook_float = codebook.float()
        task_codebook_float = (
            codebook_float.detach()
            if self.task_codebook_gradient_mode == "codebook_fit_only"
            else codebook_float
        )
        distances = (
            residuals_float.square().sum(dim=-1, keepdim=True)
            + task_codebook_float.square().sum(dim=-1).unsqueeze(0)
            - 2.0
            * residuals_float
            @ task_codebook_float.transpose(0, 1)
        )
        indices = distances.argmin(dim=-1)
        if force_zero_mask is not None and force_zero_mask.numel() > 0:
            nearest_nonzero = distances[:, 1:].argmin(dim=-1) + 1
            indices = torch.where(force_nonzero_mask, nearest_nonzero, indices)
            indices = torch.where(
                force_zero_mask,
                torch.zeros_like(indices),
                indices,
            )
        hard = codebook.index_select(0, indices)
        probabilities = torch.softmax(
            -distances / self.current_temperature,
            dim=-1,
        )
        task_codebook = task_codebook_float.to(dtype=codebook.dtype)
        soft = probabilities.to(dtype=codebook.dtype) @ task_codebook
        hard_for_task = (
            hard
            if self.task_codebook_gradient_mode == "legacy_hard_plus_soft"
            else hard.detach()
        )
        value = hard_for_task + (soft - soft.detach())
        hard_probabilities = probabilities.gather(
            1, indices.unsqueeze(1)
        ).squeeze(1)
        return (
            value.to(dtype=residuals.dtype),
            hard.to(dtype=residuals.dtype),
            soft.to(dtype=residuals.dtype),
            indices,
            hard_probabilities,
        )

    def _quantize_single_codebook_row_chunked(
        self,
        residuals: torch.Tensor,
        codebook: torch.Tensor,
        *,
        hard_indices: Optional[torch.Tensor] = None,
        force_zero_mask: Optional[torch.Tensor] = None,
        force_nonzero_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | None]:
        """Preserve exact softmax semantics without retaining B x K outputs."""
        chunk_results = []
        for start in range(0, residuals.size(0), self.distance_row_chunk_size):
            current = residuals[
                start : start + self.distance_row_chunk_size
            ]
            current_hard_indices = (
                None
                if hard_indices is None
                else hard_indices[
                    start : start + self.distance_row_chunk_size
                ]
            )
            current_force_zero_mask = (
                None
                if force_zero_mask is None
                else force_zero_mask[
                    start : start + self.distance_row_chunk_size
                ]
            )
            current_force_nonzero_mask = (
                None
                if force_nonzero_mask is None
                else force_nonzero_mask[
                    start : start + self.distance_row_chunk_size
                ]
            )
            if torch.is_grad_enabled() and (
                current.requires_grad or codebook.requires_grad
            ):
                result = checkpoint(
                    self._quantize_compact_row_chunk,
                    current,
                    codebook,
                    current_hard_indices,
                    current_force_zero_mask,
                    current_force_nonzero_mask,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                result = self._quantize_compact_row_chunk(
                    current,
                    codebook,
                    current_hard_indices,
                    current_force_zero_mask,
                    current_force_nonzero_mask,
                )
            chunk_results.append(result)
        columns = list(zip(*chunk_results))
        return {
            "value": torch.cat(columns[0], dim=0),
            "hard_value": torch.cat(columns[1], dim=0),
            "soft_value": torch.cat(columns[2], dim=0),
            "indices": torch.cat(columns[3], dim=0),
            "hard_probabilities": torch.cat(columns[4], dim=0),
            "probabilities": None,
            "distances": None,
            "codebook": codebook,
        }

    def _quantize_against_codebook(
        self,
        residuals: torch.Tensor,
        codebook: torch.Tensor,
        *,
        temperature_multiplier: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        residuals_float = residuals.float()
        codebook_float = codebook.float()
        if self.distance_backend == "broadcast":
            distances = (
                residuals_float.unsqueeze(1) - codebook_float.unsqueeze(0)
            ).square().sum(dim=-1)
        else:
            distances = (
                residuals_float.square().sum(dim=-1, keepdim=True)
                + codebook_float.square().sum(dim=-1).unsqueeze(0)
                - 2.0 * residuals_float @ codebook_float.transpose(0, 1)
            )
        indices = distances.argmin(dim=-1)
        hard = codebook.index_select(0, indices)
        effective_temperature = (
            self.current_temperature * float(temperature_multiplier)
        )
        if effective_temperature <= 0.0:
            raise ValueError("effective zero-anchor temperature must be positive")
        probabilities = torch.softmax(-distances / effective_temperature, dim=-1)
        soft = probabilities.to(dtype=codebook.dtype) @ codebook
        value = hard + (soft - soft.detach())
        return {
            "value": value.to(dtype=residuals.dtype),
            "hard_value": hard.to(dtype=residuals.dtype),
            "soft_value": soft.to(dtype=residuals.dtype),
            "indices": indices,
            "probabilities": probabilities,
            "distances": distances,
            "codebook": codebook,
        }

    def _quantize_multilevel_residuals(
        self,
        residuals: torch.Tensor,
        field_position: int,
    ) -> dict[str, torch.Tensor]:
        """Serially quantize remaining error; never expose the final residual."""
        remaining = residuals
        level_results = []
        for level in range(self.num_residual_levels):
            level_input = remaining
            current = self._quantize_against_codebook(
                level_input,
                self.codebook(
                    field_position,
                    level_position=level,
                ),
                temperature_multiplier=(
                    self.residual_level_temperature_decay**level
                ),
            )
            level_results.append(
                {
                    **current,
                    "residual_input": level_input,
                }
            )
            remaining = remaining - current["value"]

        values = torch.stack(
            [item["value"] for item in level_results],
            dim=1,
        )
        hard_values = torch.stack(
            [item["hard_value"] for item in level_results],
            dim=1,
        )
        soft_values = torch.stack(
            [item["soft_value"] for item in level_results],
            dim=1,
        )
        return {
            "value": values.sum(dim=1),
            "hard_value": hard_values.sum(dim=1),
            "soft_value": soft_values.sum(dim=1),
            "indices": torch.stack(
                [item["indices"] for item in level_results],
                dim=1,
            ),
            "probabilities": torch.stack(
                [item["probabilities"] for item in level_results],
                dim=1,
            ),
            "distances": torch.stack(
                [item["distances"] for item in level_results],
                dim=1,
            ),
            "codebook": torch.stack(
                [item["codebook"] for item in level_results],
                dim=0,
            ),
            "level_values": values,
            "level_hard_values": hard_values,
            "level_soft_values": soft_values,
            "level_residual_inputs": torch.stack(
                [item["residual_input"] for item in level_results],
                dim=1,
            ),
            "remaining_residual": remaining,
        }

    def _quantize_parallel_additive_residuals(
        self,
        residuals: torch.Tensor,
        field_position: int,
    ) -> dict[str, torch.Tensor]:
        """Quantize one scaled full-dimensional target per additive codebook."""
        target = residuals / float(self.num_residual_levels)
        codebook_results = []
        for position in range(self.num_residual_levels):
            current = self._quantize_against_codebook(
                target,
                self.codebook(
                    field_position,
                    level_position=position,
                ),
            )
            codebook_results.append(
                {
                    **current,
                    "residual_input": target,
                }
            )

        values = torch.stack(
            [item["value"] for item in codebook_results],
            dim=1,
        )
        hard_values = torch.stack(
            [item["hard_value"] for item in codebook_results],
            dim=1,
        )
        soft_values = torch.stack(
            [item["soft_value"] for item in codebook_results],
            dim=1,
        )
        summed = values.sum(dim=1)
        return {
            "value": summed,
            "hard_value": hard_values.sum(dim=1),
            "soft_value": soft_values.sum(dim=1),
            "indices": torch.stack(
                [item["indices"] for item in codebook_results],
                dim=1,
            ),
            "probabilities": torch.stack(
                [item["probabilities"] for item in codebook_results],
                dim=1,
            ),
            "distances": torch.stack(
                [item["distances"] for item in codebook_results],
                dim=1,
            ),
            "codebook": torch.stack(
                [item["codebook"] for item in codebook_results],
                dim=0,
            ),
            "level_values": values,
            "level_hard_values": hard_values,
            "level_soft_values": soft_values,
            "level_residual_inputs": torch.stack(
                [item["residual_input"] for item in codebook_results],
                dim=1,
            ),
            "remaining_residual": residuals - summed,
        }

    def _quantize_product_residuals(
        self,
        residuals: torch.Tensor,
        field_position: int,
    ) -> dict[str, torch.Tensor]:
        """Quantize disjoint residual subspaces without a residual bypass."""
        batch_size = residuals.size(0)
        residuals_subspace = residuals.reshape(
            batch_size,
            self.num_subspaces,
            self.subspace_dim,
        )
        codebook = self.codebook(field_position)
        residuals_float = residuals_subspace.float()
        codebook_float = codebook.float()
        if self.distance_backend == "broadcast":
            distances = (
                residuals_float.unsqueeze(2) - codebook_float.unsqueeze(0)
            ).square().sum(dim=-1)
        else:
            distances = (
                residuals_float.square().sum(dim=-1, keepdim=True)
                + codebook_float.square().sum(dim=-1).unsqueeze(0)
                - 2.0
                * torch.einsum(
                    "bmd,mkd->bmk",
                    residuals_float,
                    codebook_float,
                )
            )
        indices = distances.argmin(dim=-1)
        subspaces = torch.arange(
            self.num_subspaces,
            device=indices.device,
        ).unsqueeze(0)
        hard = codebook[subspaces, indices]
        probabilities = torch.softmax(
            -distances / self.current_temperature,
            dim=-1,
        )
        soft = torch.einsum(
            "bmk,mkd->bmd",
            probabilities.to(dtype=codebook.dtype),
            codebook,
        )
        straight_through = hard + (soft - soft.detach())
        return {
            "value": straight_through.reshape(batch_size, self.embedding_dim).to(
                dtype=residuals.dtype
            ),
            "hard_value": hard.reshape(batch_size, self.embedding_dim).to(
                dtype=residuals.dtype
            ),
            "soft_value": soft.reshape(batch_size, self.embedding_dim).to(
                dtype=residuals.dtype
            ),
            "indices": indices,
            "probabilities": probabilities,
            "distances": distances,
            "codebook": codebook,
        }

    def _quantize_product_residual_levels(
        self,
        residuals: torch.Tensor,
        field_position: int,
    ) -> dict[str, torch.Tensor]:
        """Serially refine every product subspace without residual bypass."""
        batch_size = residuals.size(0)
        residuals_subspace = residuals.reshape(
            batch_size,
            self.num_subspaces,
            self.subspace_dim,
        )
        summed_values = []
        summed_hard_values = []
        summed_soft_values = []
        axis_results = []
        remaining_by_subspace = []
        for subspace in range(self.num_subspaces):
            remaining = residuals_subspace[:, subspace, :]
            subspace_results = []
            for level in range(self.num_residual_levels):
                level_input = remaining
                current = self._quantize_against_codebook(
                    level_input,
                    self.codebook(
                        field_position,
                        subspace_position=subspace,
                        level_position=level,
                    ),
                    temperature_multiplier=(
                        self.residual_level_temperature_decay**level
                    ),
                )
                current = {
                    **current,
                    "residual_input": level_input,
                }
                subspace_results.append(current)
                axis_results.append(current)
                remaining = remaining - current["value"]
            summed_values.append(
                torch.stack(
                    [item["value"] for item in subspace_results],
                    dim=1,
                ).sum(dim=1)
            )
            summed_hard_values.append(
                torch.stack(
                    [item["hard_value"] for item in subspace_results],
                    dim=1,
                ).sum(dim=1)
            )
            summed_soft_values.append(
                torch.stack(
                    [item["soft_value"] for item in subspace_results],
                    dim=1,
                ).sum(dim=1)
            )
            remaining_by_subspace.append(remaining)

        axis_values = torch.stack(
            [item["value"] for item in axis_results],
            dim=1,
        )
        axis_hard_values = torch.stack(
            [item["hard_value"] for item in axis_results],
            dim=1,
        )
        axis_soft_values = torch.stack(
            [item["soft_value"] for item in axis_results],
            dim=1,
        )
        value = torch.stack(summed_values, dim=1).reshape(
            batch_size, self.embedding_dim
        )
        hard_value = torch.stack(summed_hard_values, dim=1).reshape(
            batch_size, self.embedding_dim
        )
        soft_value = torch.stack(summed_soft_values, dim=1).reshape(
            batch_size, self.embedding_dim
        )
        return {
            "value": value,
            "hard_value": hard_value,
            "soft_value": soft_value,
            "indices": torch.stack(
                [item["indices"] for item in axis_results],
                dim=1,
            ),
            "probabilities": torch.stack(
                [item["probabilities"] for item in axis_results],
                dim=1,
            ),
            "distances": torch.stack(
                [item["distances"] for item in axis_results],
                dim=1,
            ),
            "codebook": torch.stack(
                [item["codebook"] for item in axis_results],
                dim=0,
            ),
            "level_values": axis_values,
            "level_hard_values": axis_hard_values,
            "level_soft_values": axis_soft_values,
            "level_residual_inputs": torch.stack(
                [item["residual_input"] for item in axis_results],
                dim=1,
            ),
            "remaining_residual": torch.stack(
                remaining_by_subspace,
                dim=1,
            ).reshape(batch_size, self.embedding_dim),
        }

    @staticmethod
    def _release_ramp_multiplier(
        progress: float,
        release: float,
        ramp: float,
    ) -> float:
        if progress < release:
            return 0.0
        if ramp == 0.0:
            return 1.0
        return min(max((progress - release) / ramp, 0.0), 1.0)

    def set_progress(self, current_step: int, total_steps: int) -> None:
        self._audit_current_step = int(current_step)
        denominator = max(int(total_steps) - 1, 1)
        progress = min(max(float(current_step) / denominator, 0.0), 1.0)
        self.current_progress = progress
        self.current_temperature = (
            self.temperature_start
            + progress * (self.temperature_end - self.temperature_start)
        )
        multiplier = self._release_ramp_multiplier(
            progress,
            self.regularization_release_fraction,
            self.regularization_ramp_fraction,
        )
        self.current_regularization_multiplier = multiplier
        self.current_commitment_multiplier = multiplier
        self.current_codebook_multiplier = self._release_ramp_multiplier(
            progress,
            self.codebook_release_fraction,
            self.codebook_ramp_fraction,
        )

    def _should_audit_assignments(self) -> bool:
        return bool(
            self.audit_enabled
            and self.training
            and self._audit_current_step % self.audit_assignment_every == 0
        )

    @torch.no_grad()
    def _freeze_audit_anchors(
        self,
        field_position: int,
        ids: torch.Tensor,
        subspace_position: int = 0,
    ) -> None:
        slot = (
            field_position * self.assignment_axis_count
            + subspace_position
        )
        if self._audit_anchor_frozen[slot]:
            return
        ids = torch.unique(ids.detach().reshape(-1).to(dtype=torch.long), sorted=True)
        count = min(ids.numel(), self.audit_anchor_observations)
        if count:
            self._audit_anchor_ids[slot, :count].copy_(ids[:count])
        self._audit_anchor_observation_count[slot] = int(ids.numel())
        self._audit_anchor_count[slot] = int(count)
        self._audit_anchor_frozen[slot] = True

    @torch.no_grad()
    def _record_assignment_audit(
        self,
        *,
        field_position: int,
        ids: torch.Tensor,
        indices: torch.Tensor,
        probabilities: torch.Tensor,
        soft_values: torch.Tensor,
        hard_values: torch.Tensor,
        codebook: torch.Tensor,
        level_soft_values: Optional[torch.Tensor] = None,
        level_hard_values: Optional[torch.Tensor] = None,
    ) -> None:
        if not self._should_audit_assignments():
            return
        if self.num_residual_levels > 1:
            if level_soft_values is None or level_hard_values is None:
                raise ValueError(
                    "multi-level assignment audit requires per-level values"
                )
            for axis in range(self.assignment_axis_count):
                self._record_assignment_audit_slot(
                    field_position=field_position,
                    subspace_position=axis,
                    ids=ids,
                    indices=indices[:, axis],
                    probabilities=probabilities[:, axis, :],
                    soft_values=level_soft_values[:, axis, :],
                    hard_values=level_hard_values[:, axis, :],
                    codebook=codebook[axis],
                )
            return
        if self.num_subspaces == 1:
            self._record_assignment_audit_slot(
                field_position=field_position,
                subspace_position=0,
                ids=ids,
                indices=indices,
                probabilities=probabilities,
                soft_values=soft_values,
                hard_values=hard_values,
                codebook=codebook,
            )
            return
        soft_subspaces = soft_values.reshape(
            soft_values.size(0),
            self.num_subspaces,
            self.subspace_dim,
        )
        hard_subspaces = hard_values.reshape_as(soft_subspaces)
        for subspace in range(self.num_subspaces):
            self._record_assignment_audit_slot(
                field_position=field_position,
                subspace_position=subspace,
                ids=ids,
                indices=indices[:, subspace],
                probabilities=probabilities[:, subspace, :],
                soft_values=soft_subspaces[:, subspace, :],
                hard_values=hard_subspaces[:, subspace, :],
                codebook=codebook[subspace],
            )

    @torch.no_grad()
    def _record_assignment_audit_slot(
        self,
        *,
        field_position: int,
        subspace_position: int,
        ids: torch.Tensor,
        indices: torch.Tensor,
        probabilities: torch.Tensor,
        soft_values: torch.Tensor,
        hard_values: torch.Tensor,
        codebook: torch.Tensor,
    ) -> None:
        slot = (
            field_position * self.assignment_axis_count
            + subspace_position
        )
        proxy_distances = (
            soft_values.detach().float().unsqueeze(1)
            - codebook.detach().float().unsqueeze(0)
        ).square().sum(dim=-1)
        proxy_indices = proxy_distances.argmin(dim=-1)
        proxy_agreement = proxy_indices.eq(indices).float().sum()
        hard_probability = probabilities.gather(
            1, indices.unsqueeze(1)
        ).sum()
        soft_hard_gap = torch.linalg.vector_norm(
            soft_values.detach().float() - hard_values.detach().float(),
            dim=-1,
        ).sum()
        metrics = torch.stack(
            (proxy_agreement, hard_probability, soft_hard_gap)
        ).to(dtype=torch.float64)
        self._audit_soft_metric_sums[slot].add_(metrics)
        self._audit_soft_metric_counts[slot].add_(indices.numel())

        self._freeze_audit_anchors(
            field_position,
            ids,
            subspace_position,
        )
        anchor_count = self._audit_anchor_count[slot]
        if anchor_count == 0:
            return

        order = torch.argsort(ids, stable=True)
        sorted_ids = ids[order]
        sorted_codes = indices[order]
        unique_ids, duplicate_counts = torch.unique_consecutive(
            sorted_ids, return_counts=True
        )
        unique_codes = sorted_codes[duplicate_counts.cumsum(dim=0) - 1]
        anchors = self._audit_anchor_ids[slot, :anchor_count]
        positions = torch.searchsorted(anchors, unique_ids)
        in_bounds = positions.lt(anchor_count)
        matched = torch.zeros_like(in_bounds)
        if in_bounds.any():
            matched[in_bounds] = anchors[positions[in_bounds]].eq(
                unique_ids[in_bounds]
            )
        if not matched.any():
            return
        positions = positions[matched]
        unique_codes = unique_codes[matched]
        previous = self._audit_anchor_last_codes[slot, positions]
        comparable = previous.ge(0)
        self._audit_churn_comparisons[slot].add_(comparable.sum())
        self._audit_churn_changes[slot].add_(
            (comparable & previous.ne(unique_codes)).sum()
        )
        self._audit_anchor_last_codes[slot, positions] = unique_codes

    def _field_losses(
        self,
        residuals: torch.Tensor,
        hard_values: torch.Tensor,
        indices: torch.Tensor,
        quantized: Optional[dict[str, torch.Tensor]] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        def weighted_mse(
            prediction: torch.Tensor,
            target: torch.Tensor,
            weights: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if weights is None:
                return F.mse_loss(prediction.float(), target.float())
            per_row = (prediction.float() - target.float()).square()
            per_row = per_row.reshape(per_row.size(0), -1).mean(dim=1)
            normalized = weights.to(device=per_row.device, dtype=per_row.dtype)
            return (per_row * normalized).sum() / normalized.sum()

        zero = residuals.float().sum() * 0.0
        if self.num_residual_levels > 1:
            if quantized is None:
                raise ValueError("multi-level losses require quantized level state")
            level_losses = []
            level_residuals = quantized["level_residual_inputs"]
            level_hard = quantized["level_hard_values"]
            for axis in range(self.assignment_axis_count):
                nonzero = indices[:, axis].ne(0)
                if nonzero.any():
                    level_losses.append(
                        F.mse_loss(
                            level_hard[nonzero, axis].float(),
                            level_residuals[
                                nonzero, axis
                            ].detach().float(),
                        )
                    )
                else:
                    level_losses.append(zero)
            codebook_fit = torch.stack(level_losses).mean()
        else:
            nonzero = indices.ne(0)
            if nonzero.any() and self.num_subspaces == 1:
                codebook_fit = weighted_mse(
                    hard_values[nonzero].float(),
                    residuals[nonzero].detach().float(),
                    sample_weights[nonzero] if sample_weights is not None else None,
                )
            elif nonzero.any():
                residuals_subspace = residuals.reshape(
                    residuals.size(0),
                    self.num_subspaces,
                    self.subspace_dim,
                )
                hard_subspace = hard_values.reshape_as(residuals_subspace)
                codebook_fit = F.mse_loss(
                    hard_subspace[nonzero].float(),
                    residuals_subspace[nonzero].detach().float(),
                )
            else:
                # d0 is a buffer and deliberately excluded from L_code.
                codebook_fit = zero
        commitment = weighted_mse(
            residuals.float(),
            hard_values.detach().float(),
            sample_weights,
        )
        zero_l2 = weighted_mse(
            residuals.float(),
            torch.zeros_like(residuals),
            sample_weights,
        )
        return {
            "codebook": codebook_fit,
            "commitment": commitment,
            "zero_l2": zero_l2,
        }

    @staticmethod
    def _gini(counts: torch.Tensor) -> torch.Tensor:
        counts = counts.float()
        total = counts.sum()
        if counts.numel() == 0 or total <= 0:
            return counts.new_zeros(())
        ordered = counts.sort().values
        n = ordered.numel()
        ranks = torch.arange(1, n + 1, device=counts.device, dtype=counts.dtype)
        return (
            (2.0 * (ranks * ordered).sum()) / (n * total)
            - (n + 1.0) / n
        )

    def _assignment_diagnostics(
        self,
        counts: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        counts_float = counts.float()
        total = counts_float.sum()
        if total <= 0:
            zero = counts_float.new_zeros(())
            return {
                f"{prefix}_effective_centers": zero,
                f"{prefix}_perplexity": zero,
                f"{prefix}_max_fraction": zero,
                f"{prefix}_zero_fraction": zero,
                f"{prefix}_gini": zero,
            }
        probabilities = counts_float / total
        positive = probabilities.gt(0)
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        return {
            f"{prefix}_effective_centers": positive.sum().float(),
            f"{prefix}_perplexity": entropy.exp(),
            f"{prefix}_max_fraction": probabilities.max(),
            f"{prefix}_zero_fraction": probabilities[0],
            f"{prefix}_gini": self._gini(counts_float),
        }

    @staticmethod
    def _assignment_diagnostics_from_indices(
        indices: torch.Tensor,
        *,
        codebook_size: int,
        prefix: str,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute exact batch diagnostics without allocating a length-K vector."""
        flat = indices.detach().reshape(-1)
        if flat.numel() == 0:
            zero = torch.zeros((), dtype=torch.float32, device=indices.device)
            return {
                f"{prefix}_effective_centers": zero,
                f"{prefix}_perplexity": zero,
                f"{prefix}_max_fraction": zero,
                f"{prefix}_zero_fraction": zero,
                f"{prefix}_gini": zero,
            }
        if sample_weights is None:
            unique, counts = torch.unique(flat, return_counts=True)
        else:
            weights = sample_weights.detach().reshape(-1)
            if weights.shape != flat.shape:
                raise ValueError("assignment diagnostic weights must match indices")
            unique, inverse = torch.unique(flat, return_inverse=True)
            counts = torch.zeros(
                unique.numel(),
                dtype=weights.dtype,
                device=weights.device,
            )
            counts.scatter_add_(0, inverse, weights)
        counts_float = counts.float()
        total = counts_float.sum()
        probabilities = counts_float / total
        entropy = -(probabilities * probabilities.log()).sum()
        zero_matches = unique.eq(0)
        zero_fraction = (
            probabilities[zero_matches].sum()
            if zero_matches.any()
            else probabilities.new_zeros(())
        )
        ordered = counts_float.sort().values
        positive_count = ordered.numel()
        ranks = torch.arange(
            codebook_size - positive_count + 1,
            codebook_size + 1,
            device=counts.device,
            dtype=counts_float.dtype,
        )
        gini = (
            (2.0 * (ranks * ordered).sum())
            / (float(codebook_size) * total)
            - (float(codebook_size) + 1.0) / float(codebook_size)
        )
        return {
            f"{prefix}_effective_centers": counts_float.new_tensor(
                float(positive_count)
            ),
            f"{prefix}_perplexity": entropy.exp(),
            f"{prefix}_max_fraction": probabilities.max(),
            f"{prefix}_zero_fraction": zero_fraction,
            f"{prefix}_gini": gini,
        }

    @staticmethod
    def _aggregate_product_diagnostics(
        subspace_diagnostics: list[dict[str, torch.Tensor]],
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Expose the worst subspace under the legacy field-level names."""
        if not subspace_diagnostics:
            raise ValueError("product diagnostics require at least one subspace")

        def values(suffix: str) -> torch.Tensor:
            return torch.stack(
                [
                    item[f"aggregate_{suffix}"]
                    for item in subspace_diagnostics
                ]
            )

        return {
            f"{prefix}_effective_centers": values("effective_centers").min(),
            f"{prefix}_perplexity": values("perplexity").min(),
            f"{prefix}_max_fraction": values("max_fraction").max(),
            f"{prefix}_zero_fraction": values("zero_fraction").max(),
            f"{prefix}_gini": values("gini").max(),
        }

    def forward(
        self,
        fields: torch.Tensor,
        sparse: torch.Tensor,
        row_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if fields.ndim != 3 or fields.size(1) != len(self.field_names):
            raise ValueError(
                "zero-anchor fields must have shape "
                f"(B,{len(self.field_names)},E), got {tuple(fields.shape)}"
            )
        if sparse.shape != fields.shape[:2]:
            raise ValueError(
                "zero-anchor sparse IDs must match field batch/width, "
                f"got sparse={tuple(sparse.shape)}, fields={tuple(fields.shape)}"
            )
        if row_weights is not None:
            if row_weights.ndim != 1 or row_weights.size(0) != fields.size(0):
                raise ValueError(
                    "zero-anchor row weights must have shape (B,), got "
                    f"{tuple(row_weights.shape)}"
                )
            if not bool(torch.isfinite(row_weights.float()).all().item()):
                raise ValueError("zero-anchor row weights must be finite")
            if not bool(row_weights.gt(0).all().item()):
                raise ValueError("zero-anchor row weights must be positive")
            if row_weights.dtype.is_floating_point and not bool(
                row_weights.eq(row_weights.round()).all().item()
            ):
                raise ValueError("zero-anchor row weights must be integer counts")
            if (
                len(self.identity_fields) != 1
                or self.num_subspaces != 1
                or self.num_residual_levels != 1
                or self.assignment_stability_mode != "free_nearest"
                or self.audit_enabled
                or self.code_logit_bias_enabled
                or not self.sparse_batch_diagnostics
            ):
                raise ValueError(
                    "weighted zero-anchor rows support only the audited M1 "
                    "single-field, single-codebook, stateless-nearest path"
                )

        output = fields.clone()
        field_losses: list[tuple[str, dict[str, torch.Tensor]]] = []
        diagnostics: dict[str, torch.Tensor] = {}
        logit_bias = fields.new_zeros(fields.size(0))
        padding_id = self.special_token_ids["padding"]
        oov_id = self.special_token_ids["oov"]
        missing_id = self.special_token_ids["missing"]
        reserved_id = self.special_token_ids["reserved_zero"]

        for position, field in enumerate(self.identity_fields):
            index = self.name_to_index[field]
            field_codebook_size = self.codebook_size_for_field(position)
            field_base_embedding_mode = self.base_embedding_modes_by_field[
                field
            ]
            ids = sparse[:, index]
            residuals = fields[:, index, :]
            regular_mask = ids.ge(self.first_private_id)
            hard_indices = None
            if (
                self.private_row_initialization_mode
                == "train_first_touch_frozen_deterministic_code"
            ):
                hard_indices = torch.zeros_like(ids, dtype=torch.long)
                hard_indices[regular_mask] = (
                    self.deterministic_private_code_indices(
                        ids[regular_mask],
                        field_position=position,
                    )
                )
            force_zero_mask, force_nonzero_mask = (
                self.hard_routing_control_masks(ids, position)
            )
            quantized = self.quantize_residuals(
                residuals,
                position,
                hard_indices=hard_indices,
                force_zero_mask=force_zero_mask,
                force_nonzero_mask=force_nonzero_mask,
            )
            assignment_switch_fraction = residuals.new_zeros(())
            assignment_considered_fraction = residuals.new_zeros(())
            if self.assignment_stability_mode != "free_nearest":
                (
                    quantized,
                    assignment_switch_fraction,
                    assignment_considered_fraction,
                ) = self._apply_stable_assignments(
                    field_position=position,
                    ids=ids,
                    regular_mask=regular_mask,
                    residuals=residuals,
                    quantized=quantized,
                )
            base = self.base_embeddings[position].to(dtype=fields.dtype)
            zero_state_base = self.zero_state_base_embeddings[position].to(
                dtype=fields.dtype
            )
            oov_base = self.oov_base_embeddings[position].to(
                dtype=fields.dtype
            )
            if field_base_embedding_mode == "learned":
                output_base = base.unsqueeze(0).expand(fields.size(0), -1)
            elif field_base_embedding_mode == "fixed_zero":
                output_base = base.mul(0.0).unsqueeze(0).expand(
                    fields.size(0), -1
                )
            else:
                has_nonzero_code = (
                    quantized["indices"]
                    .reshape(fields.size(0), -1)
                    .ne(0)
                    .any(dim=1)
                )
                if field_base_embedding_mode == "zero_center_gated":
                    output_base = (
                        base.unsqueeze(0)
                        * has_nonzero_code.to(dtype=fields.dtype).unsqueeze(-1)
                    )
                else:
                    output_base = torch.where(
                        has_nonzero_code.unsqueeze(-1),
                        base.unsqueeze(0),
                        zero_state_base.unsqueeze(0),
                    )
            field_output = output_base + quantized["value"]
            field_output = torch.where(
                ids.eq(padding_id).unsqueeze(-1),
                torch.zeros_like(field_output),
                field_output,
            )
            if field_base_embedding_mode == "split_oov_zero_base":
                field_output = torch.where(
                    ids.eq(oov_id).unsqueeze(-1),
                    oov_base.unsqueeze(0),
                    field_output,
                )
            field_output = torch.where(
                ids.eq(missing_id).unsqueeze(-1),
                self.missing_embeddings[position]
                .to(dtype=fields.dtype)
                .unsqueeze(0),
                field_output,
            )
            field_output = torch.where(
                ids.eq(reserved_id).unsqueeze(-1),
                self.reserved_zero_embeddings[position]
                .to(dtype=fields.dtype)
                .unsqueeze(0),
                field_output,
            )
            output[:, index, :] = field_output

            if self.code_logit_bias_enabled:
                if quantized["probabilities"] is None:
                    raise RuntimeError(
                        "code logit bias requires dense assignment probabilities"
                    )
                field_biases = self.code_logit_biases[
                    position, ..., :field_codebook_size
                ]
                if self.num_subspaces == 1:
                    probabilities = quantized["probabilities"].unsqueeze(1)
                    indices = quantized["indices"].unsqueeze(1)
                else:
                    probabilities = quantized["probabilities"]
                    indices = quantized["indices"]
                if self.code_logit_bias_mode == "all_codes":
                    soft_bias = (
                        probabilities.to(dtype=field_biases.dtype)
                        * field_biases
                    ).sum(dim=-1).sum(dim=-1)
                    hard_bias = field_biases.gather(
                        1, indices.transpose(0, 1)
                    ).sum(dim=0)
                    regular_bias = soft_bias + (
                        hard_bias - soft_bias
                    ).detach()
                    zero_bias = field_biases[:, 0].sum()
                    missing_bias = self.missing_logit_biases[position]
                    reserved_bias = self.reserved_zero_logit_biases[position]
                else:
                    soft_zero = probabilities[..., 0].prod(dim=-1)
                    hard_zero = indices.eq(0).all(dim=-1).to(
                        dtype=soft_zero.dtype
                    )
                    zero_gate = soft_zero + (hard_zero - soft_zero).detach()
                    zero_bias = self.zero_state_logit_biases[position]
                    regular_bias = zero_gate * zero_bias
                    missing_bias = zero_bias.new_zeros(())
                    reserved_bias = zero_bias.new_zeros(())
                current_bias = torch.where(
                    ids.eq(padding_id),
                    torch.zeros_like(regular_bias),
                    regular_bias,
                )
                current_bias = torch.where(
                    ids.eq(oov_id), zero_bias.expand_as(current_bias), current_bias
                )
                current_bias = torch.where(
                    ids.eq(missing_id),
                    missing_bias.expand_as(current_bias),
                    current_bias,
                )
                current_bias = torch.where(
                    ids.eq(reserved_id),
                    reserved_bias.expand_as(current_bias),
                    current_bias,
                )
                logit_bias = logit_bias + current_bias.to(dtype=fields.dtype)

            regular_indices = quantized["indices"][regular_mask]
            regular_weights = (
                row_weights[regular_mask] if row_weights is not None else None
            )
            if regular_mask.any():
                losses = self._field_losses(
                    residuals[regular_mask],
                    quantized["hard_value"][regular_mask],
                    regular_indices,
                    quantized=(
                        {
                            "level_residual_inputs": quantized[
                                "level_residual_inputs"
                            ][regular_mask],
                            "level_hard_values": quantized[
                                "level_hard_values"
                            ][regular_mask],
                        }
                        if self.num_residual_levels > 1
                        else None
                    ),
                    sample_weights=regular_weights,
                )
                field_losses.append((field, losses))
                if self.num_residual_levels > 1:
                    batch_counts = torch.stack(
                        [
                            torch.bincount(
                                regular_indices[:, axis].detach(),
                                minlength=field_codebook_size,
                            )
                            for axis in range(self.assignment_axis_count)
                        ],
                        dim=0,
                    )
                    if self.training:
                        with torch.no_grad():
                            self.assignment_counts[
                                position, ..., :field_codebook_size
                            ].add_(
                                batch_counts.to(self.assignment_counts.device)
                            )
                            if self.product_residual_enabled:
                                index_grid = regular_indices.reshape(
                                    regular_indices.size(0),
                                    self.num_subspaces,
                                    self.num_residual_levels,
                                )
                                for subspace in range(self.num_subspaces):
                                    encoded = torch.zeros_like(
                                        index_grid[:, subspace, 0],
                                        dtype=torch.long,
                                    )
                                    for level, buffer_name in enumerate(
                                        self._multilevel_prefix_buffer_names
                                    ):
                                        encoded = (
                                            encoded * self.codebook_size
                                            + index_grid[
                                                :, subspace, level
                                            ].detach()
                                        )
                                        prefix_counts = torch.bincount(
                                            encoded,
                                            minlength=(
                                                self.codebook_size ** (level + 1)
                                            ),
                                        )
                                        getattr(self, buffer_name)[
                                            position, subspace
                                        ].add_(
                                            prefix_counts.to(
                                                self.assignment_counts.device
                                            )
                                        )
                            else:
                                encoded = torch.zeros_like(
                                    regular_indices[:, 0],
                                    dtype=torch.long,
                                )
                                for level, buffer_name in enumerate(
                                    self._multilevel_prefix_buffer_names
                                ):
                                    encoded = (
                                        encoded * self.codebook_size
                                        + regular_indices[:, level].detach()
                                    )
                                    prefix_counts = torch.bincount(
                                        encoded,
                                        minlength=(
                                            self.codebook_size ** (level + 1)
                                        ),
                                    )
                                    getattr(self, buffer_name)[position].add_(
                                        prefix_counts.to(
                                            self.assignment_counts.device
                                        )
                                    )
                    axis_diagnostics = []
                    for axis in range(self.assignment_axis_count):
                        current = self._assignment_diagnostics(
                            batch_counts[axis],
                            prefix=(
                                f"za_{field}_{self._axis_label(axis)}_batch"
                            ),
                        )
                        diagnostics.update(current)
                        axis_diagnostics.append(
                            self._assignment_diagnostics(
                                batch_counts[axis],
                                prefix="aggregate",
                            )
                        )
                    diagnostics.update(
                        self._aggregate_product_diagnostics(
                            axis_diagnostics,
                            prefix=f"za_{field}_batch",
                        )
                    )
                elif self.num_subspaces == 1:
                    if self.sparse_batch_diagnostics:
                        if self.training:
                            with torch.no_grad():
                                self.assignment_counts[position].scatter_add_(
                                    0,
                                    regular_indices.detach().to(
                                        self.assignment_counts.device
                                    ),
                                    (
                                        regular_weights.detach().to(
                                            device=self.assignment_counts.device,
                                            dtype=self.assignment_counts.dtype,
                                        )
                                        if regular_weights is not None
                                        else torch.ones_like(
                                            regular_indices,
                                            device=self.assignment_counts.device,
                                            dtype=self.assignment_counts.dtype,
                                        )
                                    ),
                                )
                        diagnostics.update(
                            self._assignment_diagnostics_from_indices(
                                regular_indices,
                                codebook_size=field_codebook_size,
                                prefix=f"za_{field}_batch",
                                sample_weights=regular_weights,
                            )
                        )
                    else:
                        batch_counts = torch.bincount(
                            regular_indices.detach(),
                            minlength=field_codebook_size,
                        )
                        if self.training:
                            with torch.no_grad():
                                self.assignment_counts[
                                    position, :field_codebook_size
                                ].add_(
                                    batch_counts.to(
                                        self.assignment_counts.device
                                    )
                                )
                        diagnostics.update(
                            self._assignment_diagnostics(
                                batch_counts,
                                prefix=f"za_{field}_batch",
                            )
                        )
                else:
                    batch_counts = torch.stack(
                        [
                            torch.bincount(
                                regular_indices[:, subspace].detach(),
                                minlength=field_codebook_size,
                            )
                            for subspace in range(self.num_subspaces)
                        ],
                        dim=0,
                    )
                    if self.training:
                        with torch.no_grad():
                            self.assignment_counts[
                                position, ..., :field_codebook_size
                            ].add_(
                                batch_counts.to(self.assignment_counts.device)
                            )
                    subspace_diagnostics = []
                    for subspace in range(self.num_subspaces):
                        current = self._assignment_diagnostics(
                            batch_counts[subspace],
                            prefix=f"za_{field}_subspace_{subspace}_batch",
                        )
                        diagnostics.update(current)
                        subspace_diagnostics.append(
                            self._assignment_diagnostics(
                                batch_counts[subspace],
                                prefix="aggregate",
                            )
                        )
                    diagnostics.update(
                        self._aggregate_product_diagnostics(
                            subspace_diagnostics,
                            prefix=f"za_{field}_batch",
                        )
                    )
                error = torch.linalg.vector_norm(
                    residuals[regular_mask].detach().float()
                    - quantized["hard_value"][regular_mask].detach().float(),
                    dim=-1,
                )
                if regular_weights is None:
                    diagnostics[f"za_{field}_quantization_error_mean"] = error.mean()
                    diagnostics[f"za_{field}_residual_norm_mean"] = (
                        torch.linalg.vector_norm(
                            residuals[regular_mask].detach().float(), dim=-1
                        ).mean()
                    )
                else:
                    diagnostic_weights = regular_weights.to(
                        device=error.device, dtype=error.dtype
                    )
                    diagnostics[f"za_{field}_quantization_error_mean"] = (
                        error * diagnostic_weights
                    ).sum() / diagnostic_weights.sum()
                    residual_norm = torch.linalg.vector_norm(
                        residuals[regular_mask].detach().float(), dim=-1
                    )
                    diagnostics[f"za_{field}_residual_norm_mean"] = (
                        residual_norm * diagnostic_weights
                    ).sum() / diagnostic_weights.sum()
                if self.num_residual_levels > 1:
                    for axis in range(self.assignment_axis_count):
                        level_input = quantized["level_residual_inputs"][
                            regular_mask, axis
                        ]
                        level_hard = quantized["level_hard_values"][
                            regular_mask, axis
                        ]
                        axis_label = self._axis_label(axis)
                        diagnostics[
                            f"za_{field}_{axis_label}_input_norm_mean"
                        ] = torch.linalg.vector_norm(
                            level_input.detach().float(),
                            dim=-1,
                        ).mean()
                        diagnostics[
                            f"za_{field}_{axis_label}_remaining_norm_mean"
                        ] = torch.linalg.vector_norm(
                            (level_input - level_hard).detach().float(),
                            dim=-1,
                        ).mean()
                if quantized.get("hard_probabilities") is not None:
                    gathered_probability = quantized[
                        "hard_probabilities"
                    ][regular_mask]
                else:
                    gathered_probability = quantized["probabilities"][
                        regular_mask
                    ].gather(
                        -1,
                        regular_indices.unsqueeze(-1),
                    )
                if regular_weights is None:
                    diagnostics[f"za_{field}_hard_probability_mean"] = (
                        gathered_probability.mean()
                    )
                else:
                    probability_weights = regular_weights.to(
                        device=gathered_probability.device,
                        dtype=gathered_probability.dtype,
                    )
                    diagnostics[f"za_{field}_hard_probability_mean"] = (
                        gathered_probability * probability_weights
                    ).sum() / probability_weights.sum()
                if self.audit_enabled:
                    self._record_assignment_audit(
                        field_position=position,
                        ids=ids[regular_mask],
                        indices=regular_indices,
                        probabilities=quantized["probabilities"][regular_mask],
                        soft_values=quantized["soft_value"][regular_mask],
                        hard_values=quantized["hard_value"][regular_mask],
                        codebook=quantized["codebook"],
                        level_soft_values=(
                            quantized["level_soft_values"][regular_mask]
                            if self.num_residual_levels > 1
                            else None
                        ),
                        level_hard_values=(
                            quantized["level_hard_values"][regular_mask]
                            if self.num_residual_levels > 1
                            else None
                        ),
                    )
            else:
                zero = fields.new_zeros(())
                empty_counts = (
                    None
                    if self.sparse_batch_diagnostics
                    else torch.zeros(
                        field_codebook_size,
                        dtype=torch.long,
                        device=fields.device,
                    )
                )
                if self.num_residual_levels > 1:
                    axis_diagnostics = []
                    for axis in range(self.assignment_axis_count):
                        current = self._assignment_diagnostics(
                            empty_counts,
                            prefix=(
                                f"za_{field}_{self._axis_label(axis)}_batch"
                            ),
                        )
                        diagnostics.update(current)
                        axis_diagnostics.append(
                            self._assignment_diagnostics(
                                empty_counts,
                                prefix="aggregate",
                            )
                        )
                    diagnostics.update(
                        self._aggregate_product_diagnostics(
                            axis_diagnostics,
                            prefix=f"za_{field}_batch",
                        )
                    )
                elif self.num_subspaces == 1:
                    diagnostics.update(
                        self._assignment_diagnostics_from_indices(
                            ids.new_empty((0,)),
                            codebook_size=field_codebook_size,
                            prefix=f"za_{field}_batch",
                        )
                        if self.sparse_batch_diagnostics
                        else self._assignment_diagnostics(
                            empty_counts,
                            prefix=f"za_{field}_batch",
                        )
                    )
                else:
                    subspace_diagnostics = []
                    for subspace in range(self.num_subspaces):
                        current = self._assignment_diagnostics(
                            empty_counts,
                            prefix=f"za_{field}_subspace_{subspace}_batch",
                        )
                        diagnostics.update(current)
                        subspace_diagnostics.append(
                            self._assignment_diagnostics(
                                empty_counts,
                                prefix="aggregate",
                            )
                        )
                    diagnostics.update(
                        self._aggregate_product_diagnostics(
                            subspace_diagnostics,
                            prefix=f"za_{field}_batch",
                        )
                    )
                diagnostics[f"za_{field}_quantization_error_mean"] = zero
                diagnostics[f"za_{field}_residual_norm_mean"] = zero
                diagnostics[f"za_{field}_hard_probability_mean"] = zero

            if row_weights is None:
                diagnostics[f"za_{field}_regular_fraction"] = regular_mask.float().mean()
                diagnostics[f"za_{field}_oov_fraction"] = ids.eq(oov_id).float().mean()
                diagnostics[f"za_{field}_missing_fraction"] = (
                    ids.eq(missing_id).float().mean()
                )
                diagnostics[f"za_{field}_reserved_zero_fraction"] = (
                    ids.eq(reserved_id).float().mean()
                )
            else:
                diagnostic_weights = row_weights.to(
                    device=ids.device, dtype=fields.dtype
                )
                weight_total = diagnostic_weights.sum()
                diagnostics[f"za_{field}_regular_fraction"] = (
                    regular_mask.to(fields.dtype) * diagnostic_weights
                ).sum() / weight_total
                diagnostics[f"za_{field}_oov_fraction"] = (
                    ids.eq(oov_id).to(fields.dtype) * diagnostic_weights
                ).sum() / weight_total
                diagnostics[f"za_{field}_missing_fraction"] = (
                    ids.eq(missing_id).to(fields.dtype) * diagnostic_weights
                ).sum() / weight_total
                diagnostics[f"za_{field}_reserved_zero_fraction"] = (
                    ids.eq(reserved_id).to(fields.dtype) * diagnostic_weights
                ).sum() / weight_total
            diagnostics[f"za_{field}_assignment_switch_fraction"] = (
                assignment_switch_fraction.detach()
            )
            diagnostics[f"za_{field}_assignment_considered_fraction"] = (
                assignment_considered_fraction.detach()
            )
            if self.defer_cumulative_diagnostics:
                diagnostics[
                    f"za_{field}_cumulative_diagnostics_deferred"
                ] = fields.new_ones(())
            elif self.num_residual_levels > 1:
                cumulative_diagnostics = []
                for axis in range(self.assignment_axis_count):
                    current = self._assignment_diagnostics(
                        self.assignment_counts[
                            position, axis, :field_codebook_size
                        ],
                        prefix=(
                            f"za_{field}_{self._axis_label(axis)}_cumulative"
                        ),
                    )
                    diagnostics.update(current)
                    cumulative_diagnostics.append(
                        self._assignment_diagnostics(
                            self.assignment_counts[
                                position, axis, :field_codebook_size
                            ],
                            prefix="aggregate",
                        )
                    )
                diagnostics.update(
                    self._aggregate_product_diagnostics(
                        cumulative_diagnostics,
                        prefix=f"za_{field}_cumulative",
                    )
                )
            elif self.num_subspaces == 1:
                diagnostics.update(
                    self._assignment_diagnostics(
                        self.assignment_counts[
                            position, :field_codebook_size
                        ],
                        prefix=f"za_{field}_cumulative",
                    )
                )
            else:
                cumulative_diagnostics = []
                for subspace in range(self.num_subspaces):
                    current = self._assignment_diagnostics(
                        self.assignment_counts[
                            position, subspace, :field_codebook_size
                        ],
                        prefix=(
                            f"za_{field}_subspace_{subspace}_cumulative"
                        ),
                    )
                    diagnostics.update(current)
                    cumulative_diagnostics.append(
                        self._assignment_diagnostics(
                            self.assignment_counts[
                                position, subspace, :field_codebook_size
                            ],
                            prefix="aggregate",
                        )
                    )
                diagnostics.update(
                    self._aggregate_product_diagnostics(
                        cumulative_diagnostics,
                        prefix=f"za_{field}_cumulative",
                    )
                )

        if field_losses:
            component_means = {
                name: torch.stack(
                    [item[name] for _, item in field_losses]
                ).mean()
                for name in ("codebook", "commitment", "zero_l2")
            }
            weighted_commitment = torch.stack(
                [
                    losses["commitment"]
                    * self.commitment_weights_by_field[field]
                    for field, losses in field_losses
                ]
            ).mean()
        else:
            zero = fields.sum() * 0.0
            component_means = {
                "codebook": zero,
                "commitment": zero,
                "zero_l2": zero,
            }
            weighted_commitment = zero
        self._last_loss_components = component_means
        self._last_aux_loss = (
            self.codebook_loss_weight
            * self.current_codebook_multiplier
            * component_means["codebook"]
            + self.current_commitment_multiplier * weighted_commitment
            + self.zero_l2_weight
            * self.current_regularization_multiplier
            * component_means["zero_l2"]
            if self.training
            else None
        )
        diagnostics.update(
            {
                "za_codebook_loss": component_means["codebook"].detach(),
                "za_codebook_multiplier": fields.new_tensor(
                    self.current_codebook_multiplier
                ),
                "za_commitment_loss": component_means["commitment"].detach(),
                "za_weighted_commitment_loss": (
                    weighted_commitment.detach()
                ),
                "za_zero_l2_loss": component_means["zero_l2"].detach(),
                "za_aux_loss": (
                    self._last_aux_loss.detach()
                    if self._last_aux_loss is not None
                    else fields.new_zeros(())
                ),
                "za_nonzero_center_norm_min": torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.codebook(position)[..., 1:, :].float(),
                            dim=-1,
                        ).min()
                        for position in range(len(self.identity_fields))
                    ]
                ).min(),
                "za_temperature": fields.new_tensor(self.current_temperature),
                "za_commitment_multiplier": fields.new_tensor(
                    self.current_commitment_multiplier
                ),
                "za_regularization_multiplier": fields.new_tensor(
                    self.current_regularization_multiplier
                ),
            }
        )
        if self.codebook_transform_mode == "shared_linear_residual":
            with torch.no_grad():
                delta = self.shared_codebook_transform_delta.detach().float()
                identity = torch.eye(
                    self.embedding_dim,
                    dtype=delta.dtype,
                    device=delta.device,
                ).unsqueeze(0)
                transforms = identity + delta
                singular_values = torch.linalg.svdvals(transforms)
                singular_min_by_field = singular_values.min(dim=-1).values
                singular_max_by_field = singular_values.max(dim=-1).values
                singular_ratio_by_field = (
                    singular_min_by_field
                    / singular_max_by_field.clamp_min(1e-12)
                )
                diagnostics.update(
                    {
                        "za_shared_transform_delta_frobenius_mean": (
                            torch.linalg.matrix_norm(
                                delta,
                                ord="fro",
                                dim=(-2, -1),
                            ).mean()
                        ),
                        "za_shared_transform_singular_min": (
                            singular_values.min()
                        ),
                        "za_shared_transform_singular_max": (
                            singular_values.max()
                        ),
                        "za_shared_transform_condition_max": (
                            (
                                singular_max_by_field
                                / singular_min_by_field.clamp_min(1e-12)
                            ).max()
                        ),
                        "za_shared_transform_singular_ratio_min": (
                            singular_ratio_by_field.min()
                        ),
                        "za_shared_transform_tripwire_pass": (
                            singular_ratio_by_field.ge(
                                self.SHARED_TRANSFORM_MIN_SINGULAR_RATIO
                            ).all().to(dtype=fields.dtype)
                        ),
                    }
                )
        self._diagnostics = diagnostics
        self._last_logit_bias = (
            logit_bias if self.code_logit_bias_enabled else None
        )
        return output

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self._last_aux_loss

    def get_logit_bias(self) -> Optional[torch.Tensor]:
        return self._last_logit_bias

    def get_loss_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_loss_components)

    def get_diagnostics(self) -> dict[str, float]:
        return {
            name: float(value.detach().float().cpu().item())
            for name, value in self._diagnostics.items()
        }

    def get_metadata(self) -> dict:
        return json.loads(json.dumps(self._metadata))

    @torch.no_grad()
    def get_shared_codebook_transform_report(self) -> dict:
        """Return the fixed structural tripwire for the shared transform."""
        threshold = self.SHARED_TRANSFORM_MIN_SINGULAR_RATIO
        if self.codebook_transform_mode != "shared_linear_residual":
            return {
                "enabled": False,
                "mode": self.codebook_transform_mode,
                "minimum_singular_ratio_tripwire": threshold,
                "maximum_condition_number_tripwire": 1.0 / threshold,
                "fields": [],
                "all_checks_pass": True,
            }

        reports = []
        identity = torch.eye(
            self.embedding_dim,
            dtype=self.shared_codebook_transform_delta.dtype,
            device=self.shared_codebook_transform_delta.device,
        )
        for position, field in enumerate(self.identity_fields):
            delta = self.shared_codebook_transform_delta[position].detach()
            transform = identity + delta
            finite = bool(torch.isfinite(transform).all().item())
            singular_min = None
            singular_max = None
            singular_ratio = None
            condition_number = None
            if finite:
                try:
                    singular_values = torch.linalg.svdvals(transform.float())
                except RuntimeError:
                    finite = False
                else:
                    finite = bool(torch.isfinite(singular_values).all().item())
                    if finite:
                        singular_min = float(singular_values.min().item())
                        singular_max = float(singular_values.max().item())
                        if singular_max > 0.0:
                            singular_ratio = singular_min / singular_max
                        if singular_min > 0.0:
                            condition_number = singular_max / singular_min
            ratio_pass = bool(
                finite
                and singular_ratio is not None
                and singular_ratio >= threshold
            )
            reports.append(
                {
                    "field": field,
                    "delta_frobenius": float(
                        torch.linalg.matrix_norm(delta.float(), ord="fro").item()
                    ),
                    "singular_min": singular_min,
                    "singular_max": singular_max,
                    "singular_min_to_max_ratio": singular_ratio,
                    "condition_number": condition_number,
                    "finite": finite,
                    "singular_ratio_pass": ratio_pass,
                    "all_checks_pass": finite and ratio_pass,
                }
            )
        return {
            "enabled": True,
            "mode": self.codebook_transform_mode,
            "mechanism_scope": "shared_nonzero_direction_geometry_only",
            "radii_are_unchanged_by_transform": True,
            "effective_parameters_per_field": (
                self.embedding_dim * self.embedding_dim - 1
            ),
            "isotropic_scale_null_dimensions_per_field": 1,
            "minimum_singular_ratio_tripwire": threshold,
            "maximum_condition_number_tripwire": 1.0 / threshold,
            "fields": reports,
            "all_checks_pass": all(
                report["all_checks_pass"] for report in reports
            ),
        }

    @torch.no_grad()
    def get_center_displacement_by_assignment_frequency_report(self) -> dict:
        """Summarize selected-checkpoint center motion by usage rank."""
        if self.num_subspaces != 1 or self.num_residual_levels != 1:
            return {
                "applicable": False,
                "reason": "center displacement strata are registered for M1",
                "fields": [],
                "all_checks_pass": True,
            }

        def summarize_stratum(
            *,
            name: str,
            indices: torch.Tensor,
            counts: torch.Tensor,
            vector_displacement: torch.Tensor,
            direction_displacement: torch.Tensor,
            radius_displacement: torch.Tensor,
        ) -> dict:
            center_count = int(indices.numel())
            if center_count == 0:
                return {
                    "name": name,
                    "center_count": 0,
                    "assignment_count": 0,
                    "selection_count_min": None,
                    "selection_count_max": None,
                    "vector_l2_mean": None,
                    "vector_l2_standard_error": None,
                    "vector_l2_p90": None,
                    "direction_one_minus_cosine_mean": None,
                    "direction_one_minus_cosine_standard_error": None,
                    "direction_one_minus_cosine_p90": None,
                    "radius_absolute_change_mean": None,
                    "radius_absolute_change_standard_error": None,
                    "radius_absolute_change_p90": None,
                    "assignment_weighted_vector_l2_mean": None,
                    "all_values_finite": True,
                }
            stratum_counts = counts.index_select(0, indices)
            vector_values = vector_displacement.index_select(0, indices)
            direction_values = direction_displacement.index_select(0, indices)
            radius_values = radius_displacement.index_select(0, indices)
            assignment_count = int(stratum_counts.sum().item())
            weighted_mean = None
            if assignment_count > 0:
                weighted_mean = float(
                    (
                        vector_values
                        * stratum_counts.to(dtype=vector_values.dtype)
                    ).sum().item()
                    / assignment_count
                )
            finite = bool(
                torch.isfinite(vector_values).all().item()
                and torch.isfinite(direction_values).all().item()
                and torch.isfinite(radius_values).all().item()
            )
            return {
                "name": name,
                "center_count": center_count,
                "assignment_count": assignment_count,
                "selection_count_min": int(stratum_counts.min().item()),
                "selection_count_max": int(stratum_counts.max().item()),
                "vector_l2_mean": float(vector_values.mean().item()),
                "vector_l2_standard_error": float(
                    vector_values.std(unbiased=False).item()
                    / math.sqrt(center_count)
                ),
                "vector_l2_p90": float(
                    torch.quantile(vector_values, 0.9).item()
                ),
                "direction_one_minus_cosine_mean": float(
                    direction_values.mean().item()
                ),
                "direction_one_minus_cosine_standard_error": float(
                    direction_values.std(unbiased=False).item()
                    / math.sqrt(center_count)
                ),
                "direction_one_minus_cosine_p90": float(
                    torch.quantile(direction_values, 0.9).item()
                ),
                "radius_absolute_change_mean": float(
                    radius_values.mean().item()
                ),
                "radius_absolute_change_standard_error": float(
                    radius_values.std(unbiased=False).item()
                    / math.sqrt(center_count)
                ),
                "radius_absolute_change_p90": float(
                    torch.quantile(radius_values, 0.9).item()
                ),
                "assignment_weighted_vector_l2_mean": weighted_mean,
                "all_values_finite": finite,
            }

        fields = []
        for position, field in enumerate(self.identity_fields):
            field_codebook_size = self.codebook_size_for_field(position)
            counts = self.assignment_counts[
                position, 1:field_codebook_size
            ].detach().cpu()
            current = self.codebook(position)[1:].detach().float().cpu()
            initial = self._initial_m1_nonzero_centers[
                position, : field_codebook_size - 1
            ].detach().float().cpu()
            shape_matches = current.shape == initial.shape and (
                counts.numel() == current.size(0)
            )
            if not shape_matches:
                fields.append(
                    {
                        "field": field,
                        "shape_matches": False,
                        "strata": [],
                        "all_checks_pass": False,
                    }
                )
                continue

            vector_displacement = torch.linalg.vector_norm(
                current - initial, dim=-1
            )
            current_unit = F.normalize(current, dim=-1)
            initial_unit = F.normalize(initial, dim=-1)
            direction_displacement = 1.0 - (
                current_unit * initial_unit
            ).sum(dim=-1).clamp(-1.0, 1.0)
            radius_displacement = (
                torch.linalg.vector_norm(current, dim=-1)
                - torch.linalg.vector_norm(initial, dim=-1)
            ).abs()

            unselected = torch.nonzero(counts.eq(0), as_tuple=False).flatten()
            selected = torch.nonzero(counts.gt(0), as_tuple=False).flatten()
            selected_counts = counts.index_select(0, selected)
            order = torch.argsort(selected_counts, stable=True)
            sorted_selected = selected.index_select(0, order)
            selected_quartiles = torch.tensor_split(sorted_selected, 4)
            strata = [
                summarize_stratum(
                    name="unselected",
                    indices=unselected,
                    counts=counts,
                    vector_displacement=vector_displacement,
                    direction_displacement=direction_displacement,
                    radius_displacement=radius_displacement,
                )
            ]
            for label, indices in zip(
                (
                    "selected_q1_lowest_frequency",
                    "selected_q2",
                    "selected_q3",
                    "selected_q4_highest_frequency",
                ),
                selected_quartiles,
            ):
                strata.append(
                    summarize_stratum(
                        name=label,
                        indices=indices,
                        counts=counts,
                        vector_displacement=vector_displacement,
                        direction_displacement=direction_displacement,
                        radius_displacement=radius_displacement,
                    )
                )
            fields.append(
                {
                    "field": field,
                    "shape_matches": True,
                    "nonzero_centers": int(current.size(0)),
                    "selected_centers": int(selected.numel()),
                    "unselected_centers": int(unselected.numel()),
                    "strata": strata,
                    "all_checks_pass": all(
                        stratum["all_values_finite"] for stratum in strata
                    ),
                }
            )
        return {
            "applicable": True,
            "frequency_source": (
                "selected_checkpoint_restored_cumulative_training_"
                "assignment_counts"
            ),
            "uses_labels": False,
            "uses_offline_id_frequency": False,
            "reference": "deterministic_step0_nonzero_centers",
            "stratification": (
                "unselected_plus_equal_center_count_quartiles_ranked_by_"
                "positive_selection_count"
            ),
            "displacement_metrics": [
                "vector_l2",
                "direction_one_minus_cosine",
                "radius_absolute_change",
            ],
            "fields": fields,
            "all_checks_pass": all(
                field["all_checks_pass"] for field in fields
            ),
        }

    @torch.no_grad()
    def reset_assignment_counts(self) -> None:
        """Reset utilization counts before a frozen final-assignment pass."""
        self.assignment_counts.zero_()
        for name in self._multilevel_prefix_buffer_names:
            getattr(self, name).zero_()

    def get_multilevel_prefix_report(self) -> dict:
        """Return joint prefix-tuple utilization for residual levels."""
        if self.num_residual_levels == 1:
            return {
                "enabled": False,
                "num_residual_levels": 1,
                "fields": [],
            }
        fields = []
        for position, field in enumerate(self.identity_fields):
            subspaces = (
                range(self.num_subspaces)
                if self.product_residual_enabled
                else (None,)
            )
            for subspace in subspaces:
                for prefix_length, name in enumerate(
                    self._multilevel_prefix_buffer_names,
                    start=1,
                ):
                    stored = getattr(self, name)[position]
                    counts = (
                        stored[subspace]
                        if subspace is not None
                        else stored
                    ).detach().cpu()
                    total = int(counts.sum().item())
                    effective = int(counts.gt(0).sum().item())
                    if total:
                        probabilities = counts.float() / total
                        positive = probabilities.gt(0)
                        entropy = float(
                            -(
                                probabilities[positive]
                                * probabilities[positive].log()
                            ).sum().item()
                        )
                        perplexity = math.exp(entropy)
                        max_fraction = float(probabilities.max().item())
                        all_zero_fraction = float(probabilities[0].item())
                        gini = float(self._gini(counts).item())
                    else:
                        perplexity = 0.0
                        max_fraction = 0.0
                        all_zero_fraction = 0.0
                        gini = 0.0
                    fields.append(
                        {
                            "field": field,
                            "subspace": subspace,
                            "prefix_length": prefix_length,
                            "possible_tuples": self.codebook_size**prefix_length,
                            "assignments": total,
                            "effective_tuples": effective,
                            "perplexity": perplexity,
                            "max_tuple_fraction": max_fraction,
                            "all_zero_tuple_fraction": all_zero_fraction,
                            "gini": gini,
                        }
                    )
        return {
            "enabled": True,
            "num_residual_levels": self.num_residual_levels,
            "fields": fields,
        }

    def get_assignment_audit_report(self) -> dict:
        """Return bounded same-ID churn and soft-proxy diagnostics."""
        if not self.audit_enabled:
            return {
                "enabled": False,
                "assignment_every_steps": self.audit_assignment_every,
                "fields": [],
            }
        fields = []
        for position, field in enumerate(self.identity_fields):
            for axis in range(self.assignment_axis_count):
                slot = position * self.assignment_axis_count + axis
                samples = int(self._audit_soft_metric_counts[slot].item())
                comparisons = int(self._audit_churn_comparisons[slot].item())
                changes = int(self._audit_churn_changes[slot].item())
                sums = self._audit_soft_metric_sums[slot]
                subspace, level = self._axis_coordinates(axis)
                fields.append(
                    {
                        "field": field,
                        "subspace": subspace,
                        "level": level,
                        "sampled_occurrences": samples,
                        "anchor_observations": (
                            self._audit_anchor_observation_count[slot]
                        ),
                        "unique_frozen_anchors": self._audit_anchor_count[slot],
                        "anchors_with_assignment": int(
                            self._audit_anchor_last_codes[slot]
                            .ge(0)
                            .sum()
                            .item()
                        ),
                        "assignment_churn": {
                            "comparisons": comparisons,
                            "changes": changes,
                            "fraction": (
                                changes / comparisons if comparisons else None
                            ),
                        },
                        "soft_proxy_hard_agreement": (
                            float(sums[0].item() / samples)
                            if samples
                            else None
                        ),
                        "hard_code_probability_mean": (
                            float(sums[1].item() / samples)
                            if samples
                            else None
                        ),
                        "soft_hard_center_gap_mean": (
                            float(sums[2].item() / samples)
                            if samples
                            else None
                        ),
                    }
                )
        return {
            "enabled": True,
            "assignment_every_steps": self.audit_assignment_every,
            "anchor_capacity_per_field": self.audit_anchor_observations,
            "anchor_selection": (
                "sorted unique private IDs from the first audited batch"
            ),
            "assignment_weighting": "sampled_training_occurrences",
            "fields": fields,
        }

    def get_health_report(
        self,
        *,
        min_effective_nonzero_centers: int,
        min_perplexity: Optional[float] = None,
        max_zero_fraction: float,
        max_center_fraction: float,
        max_assignment_churn: Optional[float] = None,
        min_soft_proxy_hard_agreement: Optional[float] = None,
    ) -> dict:
        """Evaluate pre-registered utilization gates on training assignments."""
        if min_effective_nonzero_centers < 1:
            raise ValueError("min_effective_nonzero_centers must be positive")
        if min_perplexity is not None and min_perplexity <= 1.0:
            raise ValueError("min_perplexity must be greater than 1")
        if not 0.0 < max_zero_fraction <= 1.0:
            raise ValueError("max_zero_fraction must be in (0,1]")
        if not 0.0 < max_center_fraction <= 1.0:
            raise ValueError("max_center_fraction must be in (0,1]")
        if max_assignment_churn is not None and not (
            0.0 <= max_assignment_churn <= 1.0
        ):
            raise ValueError("max_assignment_churn must be in [0,1]")
        if min_soft_proxy_hard_agreement is not None and not (
            0.0 <= min_soft_proxy_hard_agreement <= 1.0
        ):
            raise ValueError(
                "min_soft_proxy_hard_agreement must be in [0,1]"
            )

        audit_report = self.get_assignment_audit_report()
        fields = []
        for position, field in enumerate(self.identity_fields):
            field_codebook_size = self.codebook_size_for_field(position)
            if self.assignment_axis_count == 1:
                entries = [
                    (
                        None,
                        self.assignment_counts[
                            position, :field_codebook_size
                        ].detach().cpu(),
                        (
                            audit_report["fields"][position]
                            if audit_report["enabled"]
                            else None
                        ),
                    )
                ]
            else:
                entries = [
                    (
                        axis,
                        self.assignment_counts[
                            position, axis, :field_codebook_size
                        ].detach().cpu(),
                        (
                            audit_report["fields"][
                                position * self.assignment_axis_count + axis
                            ]
                            if audit_report["enabled"]
                            else None
                        ),
                    )
                    for axis in range(self.assignment_axis_count)
                ]
            for axis, counts, audit_field in entries:
                axis_index = 0 if axis is None else axis
                subspace, level = self._axis_coordinates(axis_index)
                total = int(counts.sum().item())
                effective_nonzero = int(counts[1:].gt(0).sum().item())
                if total:
                    zero_fraction = float(counts[0].item() / total)
                    max_fraction = float(counts.max().item() / total)
                    probabilities = counts.float() / total
                    positive = probabilities.gt(0)
                    entropy = float(
                        -(
                            probabilities[positive]
                            * probabilities[positive].log()
                        ).sum().item()
                    )
                    perplexity = math.exp(entropy)
                else:
                    zero_fraction = 0.0
                    max_fraction = 0.0
                    perplexity = 0.0
                checks = {
                    "enough_nonzero_centers": (
                        effective_nonzero >= min_effective_nonzero_centers
                    ),
                    "zero_center_not_monopolized": (
                        total > 0 and zero_fraction <= max_zero_fraction
                    ),
                    "largest_center_bounded": (
                        total > 0 and max_fraction <= max_center_fraction
                    ),
                }
                if min_perplexity is not None:
                    checks["perplexity_sufficient"] = (
                        perplexity >= min_perplexity
                    )
                if max_assignment_churn is not None:
                    churn = (
                        audit_field["assignment_churn"]["fraction"]
                        if audit_field is not None
                        else None
                    )
                    checks["assignment_churn_bounded"] = (
                        churn is not None and churn <= max_assignment_churn
                    )
                if min_soft_proxy_hard_agreement is not None:
                    agreement = (
                        audit_field["soft_proxy_hard_agreement"]
                        if audit_field is not None
                        else None
                    )
                    checks["soft_proxy_hard_agreement_sufficient"] = (
                        agreement is not None
                        and agreement >= min_soft_proxy_hard_agreement
                    )
                fields.append(
                    {
                        "field": field,
                        "subspace": subspace,
                        "level": level,
                        "assignments": total,
                        "effective_nonzero_centers": effective_nonzero,
                        "perplexity": perplexity,
                        "zero_fraction": zero_fraction,
                        "max_center_fraction": max_fraction,
                        "assignment_audit": audit_field,
                        "checks": checks,
                        "all_checks_pass": all(checks.values()),
                    }
                )
        prefix_report = self.get_multilevel_prefix_report()
        prefix_fields = []
        for entry in prefix_report["fields"]:
            checks = {
                "enough_effective_tuples": (
                    entry["effective_tuples"]
                    >= min_effective_nonzero_centers
                ),
                "tuple_perplexity_sufficient": (
                    min_perplexity is None
                    or entry["perplexity"] >= min_perplexity
                ),
                "all_zero_tuple_not_monopolized": (
                    entry["assignments"] > 0
                    and entry["all_zero_tuple_fraction"]
                    <= max_zero_fraction
                ),
                "largest_tuple_bounded": (
                    entry["assignments"] > 0
                    and entry["max_tuple_fraction"]
                    <= max_center_fraction
                ),
            }
            prefix_fields.append(
                {
                    **entry,
                    "checks": checks,
                    "all_checks_pass": all(checks.values()),
                }
            )
        return {
            "thresholds": {
                "min_effective_nonzero_centers": min_effective_nonzero_centers,
                "min_perplexity": min_perplexity,
                "max_zero_fraction": max_zero_fraction,
                "max_center_fraction": max_center_fraction,
                "max_assignment_churn": max_assignment_churn,
                "min_soft_proxy_hard_agreement": (
                    min_soft_proxy_hard_agreement
                ),
            },
            "fields": fields,
            "prefix_tuples": {
                **prefix_report,
                "fields": prefix_fields,
                "all_checks_pass": all(
                    entry["all_checks_pass"] for entry in prefix_fields
                ),
            },
            "all_checks_pass": all(
                field["all_checks_pass"] for field in fields
            )
            and all(entry["all_checks_pass"] for entry in prefix_fields),
        }
