"""
recscale.models.tokenmixer_large_aux — TokenMixer-Large + Inter-layer Aux Loss

Reference paper §3.3.4: **Inter-layer Aux Loss**. For each mixer block except
(optionally) the last, feed its output hidden through a lightweight predictor
and compute an auxiliary CTR loss. Final objective:

    L_total = L_main + lambda_aux * (1/L_aux) * sum_i L_aux_i

Rationale (from paper): lower layers otherwise learn "filler" features rather
than predictive ones. Forcing them to make predictions on their own hidden
creates a gradient signal that teaches them predictive features directly.

This is **NOT** Switch Transformer's load-balance aux loss.

This is a sandbox file under explore/tokenmixer-large track. If S2 proves
beneficial the logic will be folded into the final tokenmixer_large_v3.py
as an optional switch. Until then tokenmixer_large_v2.py remains the
unchanged shipping artifact.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .tokenmixer_large_v2 import _TokenMixerLargeV2Backbone


class _AuxPredictor(nn.Module):
    """Lightweight per-layer predictor: flatten(T*D) -> Linear -> scalar logit.

    Intentionally small — the goal is to drive gradient into lower layers,
    not to build a separate full classifier.
    """

    def __init__(self, num_tokens: int, token_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tokens * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B,) logit"""
        flat = x.flatten(start_dim=1)  # (B, T*D)
        return self.net(flat).squeeze(-1)


@register_model
class TokenMixerLargeAuxModel(_TokenMixerLargeV2Backbone):
    """
    TokenMixer-Large V2 with inter-layer aux loss (pointwise CTR).

    Config additions:
      aux_loss_weight: float = 0.3        # lambda_aux
      aux_pred_hidden_dim: int = 64       # predictor MLP hidden dim
      aux_exclude_last_layer: bool = True # common choice: skip last-layer aux

    Uses the same FeatureFieldPooling / SparseArch / MixingAndRevertingBlock
    stack as TokenMixerLargeV2Model. Aux-loss-specific additions:
      - Per-layer _AuxPredictor (not aux_exclude_last_layer'd)
      - forward() returns (main_logit, aux_loss_scalar) tuple; the trainer's
        existing (logits, aux_loss) handling path (trainer.py:184-198) picks
        this up and adds aux_loss to main loss automatically.
    """
    model_name = "tokenmixer_large_aux"

    def __init__(self, config: dict):
        super().__init__(config, extra_tokens=0)
        mc = config["model"]
        self.aux_loss_weight = float(mc.get("aux_loss_weight", 0.3))
        self.aux_pred_hidden_dim = int(mc.get("aux_pred_hidden_dim", 64))
        self.aux_exclude_last_layer = bool(mc.get("aux_exclude_last_layer", True))

        n_layers = len(self.mixer_blocks)
        # Number of aux predictors = L if including last, L-1 otherwise
        self.num_aux = n_layers - 1 if self.aux_exclude_last_layer else n_layers
        if self.num_aux > 0:
            self.aux_predictors = nn.ModuleList([
                _AuxPredictor(self.num_tokens, self.emb_dim, self.aux_pred_hidden_dim)
                for _ in range(self.num_aux)
            ])
        print(
            f"[TokenMixerLargeAux] aux_loss_weight={self.aux_loss_weight}, "
            f"num_aux_predictors={self.num_aux} (exclude_last={self.aux_exclude_last_layer})"
        )

    def _run_blocks_with_aux(self, x: torch.Tensor, label: Optional[torch.Tensor]):
        """
        Same as _run_blocks but also collects per-layer hidden states for aux
        predictors and computes the aggregate aux loss.

        Returns: (final_x, aux_loss_scalar). aux_loss_scalar is None in eval
        mode or when label is not provided.
        """
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

        # Compute aux loss only during training when we have a label
        if not self.training or label is None or self.num_aux == 0 or self.aux_loss_weight <= 0.0:
            return x, None

        aux_loss = x.new_zeros(())  # scalar tensor on correct device
        for i, predictor in enumerate(self.aux_predictors):
            aux_logit = predictor(hidden_states[i])  # (B,)
            aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                aux_logit, label.float()
            )
        aux_loss = aux_loss / self.num_aux  # mean across layers
        aux_loss = self.aux_loss_weight * aux_loss

        return x, aux_loss

    def forward(self, batch: dict):
        x = self._sparse_tokens(batch)
        label = batch.get("label")
        x, aux_loss = self._run_blocks_with_aux(x, label)
        main_logit = self._output(x)
        if aux_loss is None:
            return main_logit
        return main_logit, aux_loss
