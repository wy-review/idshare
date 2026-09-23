"""Per-token weight initialization helpers (init_protocol v2).

SwiGLU gate/up weights are stored as ``(..., fan_in, fan_out)`` for einsum
``btd,tdf`` (input dim D, output dim F).  PyTorch's global ``kaiming_uniform_``
on a 3D+ tensor uses fan_in = shape[1] * prod(shape[2:]), not the per-slice D.

These helpers initialize each independent ``(..., fan_in, fan_out)`` matrix
using the caller's intended fan_in (typically ``weight.shape[-2]``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _kaiming_uniform_bound(fan_in: int, a: float = 0.0) -> float:
    gain = nn.init.calculate_gain("relu", a)
    std = gain / math.sqrt(fan_in)
    return math.sqrt(3.0) * std


def _xavier_uniform_bound(fan_in: int, fan_out: int, gain: float = 1.0) -> float:
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return math.sqrt(3.0) * std


def init_per_token_kaiming_uniform_(
    weight: torch.Tensor,
    a: float = 0.0,
    fan_in: int | None = None,
) -> None:
    """Initialize each ``(..., fan_in, fan_out)`` slice with kaiming_uniform."""
    if weight.dim() < 2:
        raise ValueError(f"weight must be at least 2D, got shape {tuple(weight.shape)}")
    if fan_in is None:
        fan_in = weight.shape[-2]
    bound = _kaiming_uniform_bound(fan_in, a=a)
    flat = weight.view(-1, weight.shape[-2], weight.shape[-1])
    with torch.no_grad():
        for i in range(flat.shape[0]):
            flat[i].uniform_(-bound, bound)


def init_per_token_xavier_uniform_(
    weight: torch.Tensor,
    gain: float = 1.0,
    fan_in: int | None = None,
    fan_out: int | None = None,
) -> None:
    """Initialize each ``(..., fan_in, fan_out)`` slice with xavier_uniform."""
    if weight.dim() < 2:
        raise ValueError(f"weight must be at least 2D, got shape {tuple(weight.shape)}")
    if fan_in is None:
        fan_in = weight.shape[-2]
    if fan_out is None:
        fan_out = weight.shape[-1]
    bound = _xavier_uniform_bound(fan_in, fan_out, gain=gain)
    flat = weight.view(-1, weight.shape[-2], weight.shape[-1])
    with torch.no_grad():
        for i in range(flat.shape[0]):
            flat[i].uniform_(-bound, bound)


def kaiming_global_bound(weight: torch.Tensor, a: float = 0.0) -> float:
    """Bound PyTorch uses for a full-tensor ``kaiming_uniform_(weight)`` call."""
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    return _kaiming_uniform_bound(fan_in, a=a)


def expected_per_token_kaiming_bound(fan_in: int, a: float = 0.0) -> float:
    """Expected max-abs scale for v2 per-token kaiming with intended fan_in."""
    return _kaiming_uniform_bound(fan_in, a=a)
