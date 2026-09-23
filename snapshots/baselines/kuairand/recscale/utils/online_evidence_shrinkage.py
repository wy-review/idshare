"""Online evidence-adaptive lazy shrinkage for identity embeddings.

This is a decoupled shrinkage mechanism, not an exact reimplementation of
explicit L2 under Adam. It requires no precomputed frequency table: each row
stores only its online touch count and the last step at which its decay was
materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class OnlineEvidenceState:
    """Per-row state that can be colocated with an embedding row in a PS."""

    touch_counts: torch.Tensor
    last_touch_steps: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_rows: int,
        *,
        device: torch.device | str = "cpu",
    ) -> "OnlineEvidenceState":
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        return cls(
            touch_counts=torch.zeros(
                num_rows, dtype=torch.int64, device=device
            ),
            last_touch_steps=torch.full(
                (num_rows,), -1, dtype=torch.int64, device=device
            ),
        )


def evidence_adaptive_coefficient(
    touch_counts: torch.Tensor,
    *,
    max_coefficient: float,
    transition_count: float,
    power: float,
) -> torch.Tensor:
    """Return lambda(n)=lambda_max*(1+n/tau)^(-power)."""
    if not math.isfinite(max_coefficient) or max_coefficient < 0.0:
        raise ValueError("max_coefficient must be finite and non-negative")
    if not math.isfinite(transition_count) or transition_count <= 0.0:
        raise ValueError("transition_count must be finite and positive")
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and positive")
    counts = touch_counts.to(dtype=torch.float64)
    return max_coefficient * torch.pow(
        1.0 + counts / transition_count,
        -power,
    )


@torch.no_grad()
def apply_online_evidence_shrinkage_(
    weight: torch.Tensor,
    row_ids: torch.Tensor,
    state: OnlineEvidenceState,
    *,
    global_step: int,
    learning_rate: float,
    max_coefficient: float,
    transition_count: float,
    power: float = 1.0,
    first_private_id: int = 4,
    update_evidence: bool = True,
) -> dict:
    """Materialize lazy decoupled decay for private rows used by this batch.

    Duplicate IDs contribute their full occurrence count to online evidence
    when ``update_evidence`` is true, while decay is materialized once per
    unique row and optimizer step.
    Special rows below ``first_private_id`` are excluded. A row's first touch
    initializes its clock without decay; later touches apply the accumulated
    gap using the evidence count known before the current batch.
    """
    if weight.ndim != 2:
        raise ValueError("weight must have shape (num_rows, embedding_dim)")
    if row_ids.ndim != 1:
        raise ValueError("row_ids must be one-dimensional")
    if state.touch_counts.shape != (weight.size(0),):
        raise ValueError("touch_counts shape does not match embedding rows")
    if state.last_touch_steps.shape != (weight.size(0),):
        raise ValueError("last_touch_steps shape does not match embedding rows")
    if state.touch_counts.device != weight.device:
        raise ValueError("state and weight must be on the same device")
    if row_ids.device != weight.device:
        raise ValueError("row_ids and weight must be on the same device")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError("learning_rate must be finite and non-negative")

    private = row_ids[row_ids.ge(int(first_private_id))]
    if private.numel() == 0:
        return {
            "unique_private_rows": 0,
            "private_occurrences": 0,
            "mean_multiplier": 1.0,
            "min_multiplier": 1.0,
            "mean_coefficient": 0.0,
        }
    if int(private.max()) >= weight.size(0):
        raise IndexError("row_ids contain an out-of-range private ID")

    unique_rows, occurrences = torch.unique(
        private, sorted=True, return_counts=True
    )
    old_counts = state.touch_counts.index_select(0, unique_rows)
    old_steps = state.last_touch_steps.index_select(0, unique_rows)
    elapsed = torch.where(
        old_steps.ge(0),
        torch.full_like(old_steps, int(global_step)) - old_steps,
        torch.zeros_like(old_steps),
    )
    if torch.any(elapsed.lt(0)):
        raise ValueError("global_step moved backwards for at least one row")

    coefficients = evidence_adaptive_coefficient(
        old_counts,
        max_coefficient=max_coefficient,
        transition_count=transition_count,
        power=power,
    )
    exponents = (
        -2.0
        * float(learning_rate)
        * coefficients
        * elapsed.to(dtype=torch.float64)
    )
    multipliers = torch.exp(exponents).to(dtype=weight.dtype)
    selected = weight.index_select(0, unique_rows)
    weight.index_copy_(0, unique_rows, selected * multipliers.unsqueeze(-1))

    if update_evidence:
        state.touch_counts.index_add_(
            0,
            unique_rows,
            occurrences.to(dtype=state.touch_counts.dtype),
        )
    state.last_touch_steps.index_fill_(0, unique_rows, int(global_step))
    return {
        "unique_private_rows": int(unique_rows.numel()),
        "private_occurrences": int(occurrences.sum()),
        "mean_multiplier": float(multipliers.mean()),
        "min_multiplier": float(multipliers.min()),
        "mean_coefficient": float(coefficients.mean()),
    }
