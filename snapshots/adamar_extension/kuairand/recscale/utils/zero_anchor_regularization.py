"""Explicit whole-table L2 for high-cardinality identity embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


IDENTITY_OUTPUT_TABLE_TARGET = "identity_output_table"
PREQUANTIZATION_ASSIGNMENT_TABLE_TARGET = (
    "prequantization_assignment_table"
)
QUANTIZED_OUTPUT_CODEBOOK_TARGET = "quantized_output_codebook"


def _raw_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def get_identity_embedding_weights(
    model: nn.Module,
    *,
    identity_fields: list[str] | tuple[str, ...],
    field_names: list[str] | tuple[str, ...],
) -> list[tuple[str, nn.Parameter]]:
    """Return named identity tables without requiring a quantizer."""
    raw_model = _raw_model(model)
    encoder = getattr(raw_model, "encoder", None)
    embeddings = getattr(
        getattr(encoder, "sparse_arch", None),
        "embeddings",
        None,
    )
    if embeddings is None:
        raise ValueError(
            "whole-table identity L2 requires encoder.sparse_arch.embeddings"
        )
    name_to_index = {
        str(field): index for index, field in enumerate(field_names)
    }
    missing = [field for field in identity_fields if field not in name_to_index]
    if missing:
        raise ValueError(
            f"whole-table identity L2 fields absent from sparse fields: {missing}"
        )
    return [
        (str(field), embeddings[name_to_index[str(field)]].weight)
        for field in identity_fields
    ]


def get_zero_anchor_identity_embedding_weights(
    model: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """Return the selected identity tables in quantizer field order."""
    raw_model = _raw_model(model)
    encoder = getattr(raw_model, "encoder", None)
    quantizer = getattr(encoder, "zero_anchor_identity_quantizer", None)
    embeddings = getattr(
        getattr(encoder, "sparse_arch", None),
        "embeddings",
        None,
    )
    if quantizer is None or embeddings is None:
        raise ValueError(
            "whole-table zero-anchor L2 requires an enabled identity quantizer "
            "and encoder.sparse_arch.embeddings"
        )

    metadata = quantizer.get_metadata()
    fields = metadata["identity_fields"]
    indices = metadata["field_indices"]
    if len(fields) != len(indices):
        raise RuntimeError("zero-anchor identity field metadata is inconsistent")
    return [
        (str(field), embeddings[int(index)].weight)
        for field, index in zip(fields, indices)
    ]


def _add_full_table_l2_gradients(
    weights: list[tuple[str, nn.Parameter]],
    *,
    regularization_target: str,
    coefficient: float,
    multiplier: float,
    first_regularized_row: int = 0,
) -> dict:
    """Add explicit L2 gradients to the registered rows of selected tables."""
    coefficient = float(coefficient)
    multiplier = float(multiplier)
    first_regularized_row = int(first_regularized_row)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("whole-table L2 coefficient must be finite and non-negative")
    if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise ValueError("whole-table L2 multiplier must be finite and in [0,1]")
    if first_regularized_row < 0:
        raise ValueError("first_regularized_row must be non-negative")

    scale = 2.0 * coefficient * multiplier
    fields = []
    for field, parameter in weights:
        if first_regularized_row >= int(parameter.size(0)):
            raise ValueError(
                f"first_regularized_row exceeds table cardinality for {field}"
            )
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if parameter.grad.is_sparse:
            raise ValueError("whole-table identity L2 requires dense embedding gradients")
        if scale:
            parameter.grad[first_regularized_row:].add_(
                parameter.detach()[first_regularized_row:],
                alpha=scale,
            )
        fields.append(
            {
                "field": field,
                "rows": int(parameter.size(0)),
                "regularized_rows": (
                    int(parameter.size(0)) - first_regularized_row
                ),
                "embedding_dim": int(parameter.size(1)),
            }
        )
    return {
        "regularization_target": regularization_target,
        "scope": "registered_private_rows_of_selected_identity_tables",
        "gradient_formula": "grad += 2 * coefficient * multiplier * weight",
        "coefficient": coefficient,
        "multiplier": multiplier,
        "scale": scale,
        "first_regularized_row": first_regularized_row,
        "identity_fields": [item["field"] for item in fields],
        "fields": fields,
    }


def add_identity_full_table_l2_gradients(
    model: nn.Module,
    *,
    identity_fields: list[str] | tuple[str, ...],
    field_names: list[str] | tuple[str, ...],
    coefficient: float,
    multiplier: float = 1.0,
    first_regularized_row: int = 0,
) -> dict:
    """Apply full-table L2 to continuous identity output tables."""
    return _add_full_table_l2_gradients(
        get_identity_embedding_weights(
            model,
            identity_fields=identity_fields,
            field_names=field_names,
        ),
        regularization_target=IDENTITY_OUTPUT_TABLE_TARGET,
        coefficient=coefficient,
        multiplier=multiplier,
        first_regularized_row=first_regularized_row,
    )


def add_zero_anchor_full_table_l2_gradients(
    model: nn.Module,
    *,
    coefficient: float,
    multiplier: float,
    first_regularized_row: int = 0,
) -> dict:
    """Add L2 to the prequantization coordinates used for code assignment.

    The pass happens after backward and AMP unscale. Its exact position
    relative to gradient clipping is registered by ``Trainer``; the SHRED
    causal protocol clips pure task gradients first and adds this L2 gradient
    afterward. It implements an explicit L2 objective under the configured
    optimizer, including registered private rows absent from the current
    batch. This target is not the quantized representation sent downstream:
    shrinking it changes routing toward the fixed zero code. Exact-zero rows
    receive an exact-zero L2 gradient. Callers may exclude leading special
    token rows.
    """
    return _add_full_table_l2_gradients(
        get_zero_anchor_identity_embedding_weights(model),
        regularization_target=PREQUANTIZATION_ASSIGNMENT_TABLE_TARGET,
        coefficient=coefficient,
        multiplier=multiplier,
        first_regularized_row=first_regularized_row,
    )


def add_zero_anchor_output_codebook_l2_gradients(
    model: nn.Module,
    *,
    coefficient: float,
    multiplier: float,
) -> dict:
    """Add L2 gradients to the nonzero centers consumed downstream.

    This target is intentionally distinct from the per-ID prequantization
    tables. It regularizes the constrained M1 codebook values emitted by the
    quantizer, so it cannot pull assignment coordinates toward the fixed zero
    route. The zero center is a buffer and is excluded by construction.
    """
    coefficient = float(coefficient)
    multiplier = float(multiplier)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("output-codebook L2 coefficient must be finite and non-negative")
    if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise ValueError("output-codebook L2 multiplier must be finite and in [0,1]")

    raw_model = _raw_model(model)
    quantizer = getattr(
        getattr(raw_model, "encoder", None),
        "zero_anchor_identity_quantizer",
        None,
    )
    if quantizer is None:
        raise ValueError("output-codebook L2 requires an enabled identity quantizer")
    if quantizer.num_subspaces != 1 or quantizer.num_residual_levels != 1:
        raise ValueError("output-codebook L2 currently supports Product-M1 only")

    parameters = [
        quantizer.nonzero_directions,
        quantizer.nonzero_radius_logits,
    ]
    penalty = None
    fields = []
    for position, field in enumerate(quantizer.identity_fields):
        nonzero = quantizer.codebook(position)[1:]
        current = nonzero.float().square().sum()
        penalty = current if penalty is None else penalty + current
        fields.append(
            {
                "field": str(field),
                "centers": int(nonzero.size(0)),
                "embedding_dim": int(nonzero.size(1)),
            }
        )
    if penalty is None:
        raise RuntimeError("output-codebook L2 found no identity codebooks")

    scale = coefficient * multiplier
    if scale:
        gradients = torch.autograd.grad(
            penalty,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        for parameter, gradient in zip(parameters, gradients):
            scaled = gradient.detach().mul(scale)
            if parameter.grad is None:
                parameter.grad = scaled
            else:
                parameter.grad.add_(scaled)

    return {
        "regularization_target": QUANTIZED_OUTPUT_CODEBOOK_TARGET,
        "scope": "all_nonzero_constrained_m1_codebook_centers",
        "gradient_formula": (
            "grad += coefficient * multiplier * "
            "d(sum(nonzero_center ** 2))/d(parameter)"
        ),
        "coefficient": coefficient,
        "multiplier": multiplier,
        "scale": scale,
        "first_regularized_code": 1,
        "identity_fields": [item["field"] for item in fields],
        "fields": fields,
    }
