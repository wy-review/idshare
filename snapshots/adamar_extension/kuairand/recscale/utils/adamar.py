"""Dense, row-wise AdamAR reference adapter, not author-code reproduction.

Source: https://arxiv.org/html/2511.06374v2, Section 3.1 / Algorithm 1.
Only the last-valid-step timestamp is lazy. Untouched rows still receive
moment updates and shrinkage on every step. Sparse/on-touch Adam is not used.
The Trainer adapter retains PyTorch Adam's denominator and adds the paper's
AR rule. The literal paper denominator is available only for unit diagnostics.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .zero_anchor_regularization import (
    get_identity_embedding_weights,
    get_zero_anchor_identity_quantizer,
)


EPS_PLACEMENTS = ("paper_inside_sqrt", "pytorch_outside_sqrt")
SEMANTICS = "dense_row_lvs_v1"
MATCHED_PROFILE = "identity_row_ar_pytorch_adam_v1"
FORMULA_PROFILE = "identity_row_ar_paper_denominator_diagnostic_v1"


class RowwiseAdamAR(Optimizer):
    """Apply AR to explicitly selected embedding groups, ordinary Adam elsewhere.

    Selected groups require ``adamar=True`` and ``first_regularized_row``.
    Leading special rows use ordinary Adam, not AR. ``grad=None`` on a selected
    table is an error; a dense zero gradient is valid and advances its clock.
    No timestamp or optimizer state is changed by inference.
    """

    def __init__(
        self, params, *, lr, alpha, eps_placement,
        betas=(0.9, 0.999), eps=1e-8, row_chunk_size=262144,
    ):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and non-negative")
        if not math.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha must be finite and in [0, 1)")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if eps_placement not in EPS_PLACEMENTS:
            raise ValueError(f"eps_placement must be one of {EPS_PLACEMENTS}")
        if type(row_chunk_size) is not int or row_chunk_size < 1:
            raise ValueError("row_chunk_size must be a positive integer")
        super().__init__(params, dict(
            lr=lr, betas=betas, eps=eps, alpha=alpha,
            eps_placement=eps_placement, row_chunk_size=row_chunk_size,
            adamar=False, first_regularized_row=0, semantics=SEMANTICS,
            implementation_profile=(MATCHED_PROFILE if eps_placement == "pytorch_outside_sqrt"
                                    else FORMULA_PROFILE),
        ))
        selected = False
        for group in self.param_groups:
            if group.get("weight_decay", 0) != 0:
                raise ValueError("AdamAR must not be combined with weight decay")
            for name in ("alpha", "eps_placement", "betas", "eps", "semantics", "implementation_profile"):
                if group[name] != self.defaults[name]:
                    raise ValueError(f"per-group override of {name} is unsupported")
            if not group["adamar"]:
                continue
            selected = True
            first = group["first_regularized_row"]
            if type(first) is not int or first < 0:
                raise ValueError("first_regularized_row must be a non-negative integer")
            for p in group["params"]:
                if p.ndim != 2 or first >= p.shape[0]:
                    raise ValueError("AR requires a rank-2 table with real rows")
        if not selected:
            raise ValueError("at least one explicit adamar parameter group is required")
        for group in self.param_groups:
            for p in group["params"]:
                if p.dtype not in (torch.float32, torch.float64):
                    raise ValueError("reference AdamAR supports only FP32/FP64 parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reject invalid inputs before updating any parameter or timestamp.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    if group["adamar"]:
                        raise RuntimeError("selected identity table has grad=None")
                    continue
                if p.grad.layout != torch.strided:
                    raise RuntimeError("AdamAR reference requires dense gradients")
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError("AdamAR received a non-finite gradient")

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["adamar"]:
                        state["last_valid_step"] = torch.zeros(
                            p.shape[0] - group["first_regularized_row"],
                            dtype=torch.int64, device=p.device,
                        )
                state["step"] += 1
                step = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                if group["adamar"]:
                    first = group["first_regularized_row"]
                    last = state["last_valid_step"]
                    for start in range(first, p.shape[0], group["row_chunk_size"]):
                        end = min(start + group["row_chunk_size"], p.shape[0])
                        clock = last[start - first:end - first]
                        interval = step - clock - 1
                        decay = (interval.to(p.dtype) * group["alpha"]).clamp_(max=1)
                        p[start:end].mul_((1 - decay).unsqueeze(1))
                        active = g[start:end].ne(0).any(dim=1)
                        clock.masked_fill_(active, step)

                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                # Non-identity groups always retain PyTorch Adam's denominator.
                denom = v.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                if group["adamar"] and group["eps_placement"] == "paper_inside_sqrt":
                    first = group["first_regularized_row"]
                    denom[first:] = (v[first:] / (1 - beta2 ** step) + group["eps"]).sqrt()
                p.addcdiv_(m, denom, value=-group["lr"] / (1 - beta1 ** step))
        return loss

    def load_state_dict(self, state_dict):
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("AdamAR checkpoint parameter groups differ")
        for saved, current in zip(saved_groups, self.param_groups):
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("AdamAR checkpoint parameter counts differ")
            if saved.get("identity_fields") != current.get("identity_fields"):
                raise ValueError("AdamAR checkpoint changes identity_fields")
            for name in ("semantics", "adamar", "first_regularized_row", "alpha",
                         "eps_placement", "eps", "betas", "implementation_profile"):
                if saved.get(name) != current[name]:
                    raise ValueError(f"AdamAR checkpoint changes {name}")
        clocks = {}
        clean = dict(state_dict, state={})
        for saved_id, state in state_dict["state"].items():
            clean["state"][saved_id] = dict(state)
            if "last_valid_step" in state:
                clock = state["last_valid_step"]
                if clock.dtype != torch.int64:
                    raise ValueError("AdamAR timestamp must be int64")
                clocks[saved_id] = clock
                del clean["state"][saved_id]["last_valid_step"]
        for saved, current in zip(saved_groups, self.param_groups):
            for saved_id, p in zip(saved["params"], current["params"]):
                state = clean["state"].get(saved_id)
                if not state:
                    continue
                if type(state.get("step")) is not int or state["step"] < 1:
                    raise ValueError("AdamAR checkpoint step must be a positive integer")
                for key in ("exp_avg", "exp_avg_sq"):
                    if key not in state or state[key].shape != p.shape:
                        raise ValueError("AdamAR checkpoint moment shape mismatch")
                if current["adamar"]:
                    clock = clocks.get(saved_id)
                    if clock is None or clock.shape != (p.shape[0] - current["first_regularized_row"],):
                        raise ValueError("AdamAR checkpoint timestamp shape mismatch")
                    if (clock < 0).any() or (clock > state["step"]).any():
                        raise ValueError("AdamAR checkpoint timestamp is outside its step range")
        # Optimizer.load_state_dict normally casts state tensors to parameter
        # dtype. Keep integer clocks out of that cast to avoid precision loss.
        super().load_state_dict(clean)
        for saved, current in zip(saved_groups, self.param_groups):
            for saved_id, p in zip(saved["params"], current["params"]):
                if not current["adamar"] or not self.state.get(p):
                    continue
                self.state[p]["last_valid_step"] = clocks[saved_id].to(p.device).clone()


def build_identity_adamar(model, config, *, world_size=1):
    """Matched Adam + row-wise AR; data protocols must be frozen by a runner.

    Not an author-code reproduction. The paper's inside-sqrt denominator is
    intentionally excluded here because it changes the alpha=0 comparator.
    """
    tc = config["training"]
    cfg = tc.get("adamar")
    if not isinstance(cfg, dict) or cfg.get("semantics") != SEMANTICS:
        raise ValueError(f"training.adamar must explicitly bind semantics={SEMANTICS}")
    required = {"alpha", "identity_fields", "first_regularized_row", "eps_placement", "semantics"}
    if required - cfg.keys() or cfg.keys() - required - {"row_chunk_size"}:
        raise ValueError("training.adamar has missing or unknown settings")
    if cfg["eps_placement"] != "pytorch_outside_sqrt":
        raise ValueError("matched AdamAR requires pytorch_outside_sqrt; paper denominator is diagnostic-only")
    if world_size != 1 or tc.get("use_amp", False):
        raise ValueError("AdamAR reference adapter requires single-process, no AMP")
    l2 = tc.get("zero_anchor_full_table_l2") or {}
    if (tc.get("weight_decay", 0) != 0 or tc.get("embedding_weight_decay") is not None
            or l2.get("enabled", False) or l2.get("coefficient", 0) != 0):
        raise ValueError("AdamAR must not be combined with L2 or weight decay")
    if tc.get("embedding_weight_decay_field_indices") is not None:
        raise ValueError("embedding weight decay field selection is unsupported with AdamAR")
    if tc.get("optimizer_foreach") not in (None, False):
        raise ValueError("AdamAR reference does not support foreach=True")
    if get_zero_anchor_identity_quantizer(model) is not None:
        raise ValueError("AdamAR baseline must use Continuous, not IDShare")
    fields = cfg["identity_fields"]
    if not isinstance(fields, (list, tuple)) or len(fields) != 1:
        raise ValueError("AdamAR baseline requires one explicit identity field")
    weights = get_identity_embedding_weights(
        model, identity_fields=fields,
        field_names=config["dataset"].get("sparse_cols", []),
    )
    selected_ids = {id(p) for _, p in weights}
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not selected_ids <= {id(p) for p in trainable}:
        raise ValueError("selected identity weight must be a registered trainable parameter")
    groups = [dict(
        params=[p for _, p in weights], adamar=True,
        identity_fields=list(fields), first_regularized_row=cfg["first_regularized_row"],
    ), dict(params=[p for p in trainable if id(p) not in selected_ids])]
    return RowwiseAdamAR(
        groups, lr=float(tc["lr"]), alpha=float(cfg["alpha"]),
        eps_placement=cfg["eps_placement"], row_chunk_size=cfg.get("row_chunk_size", 262144),
    )
