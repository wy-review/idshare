"""Offline frequency-oracle tail-ID collapse for mechanism experiments.

This module intentionally depends on a frequency cache computed before model
training. It is an ablation for testing whether rare IDs benefit from sharing a
single trainable key, not a production-ready online-frequency solution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _days_key(days) -> str:
    return ",".join(str(int(day)) for day in days)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfflineFrequencyTailIdCollapse(nn.Module):
    """Route offline-known rare IDs to one trainable parameter per field."""

    CONFIG_KEY = "offline_frequency_tail_collapse"

    def __init__(
        self,
        config: dict,
        *,
        cardinalities: list[int],
        field_names: list[str],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        collapse_cfg = model_cfg.get(self.CONFIG_KEY, {})
        if not isinstance(collapse_cfg, dict):
            raise ValueError(f"model.{self.CONFIG_KEY} must be a mapping")
        if not bool(collapse_cfg.get("enabled", False)):
            raise ValueError(
                "OfflineFrequencyTailIdCollapse requires "
                f"model.{self.CONFIG_KEY}.enabled=true"
            )

        raw_indices = collapse_cfg.get("field_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError(
                f"model.{self.CONFIG_KEY}.field_indices must be a non-empty list"
            )
        field_indices = [int(index) for index in raw_indices]
        if len(set(field_indices)) != len(field_indices):
            raise ValueError(
                f"model.{self.CONFIG_KEY}.field_indices must not contain duplicates"
            )
        for index in field_indices:
            if index < 0 or index >= len(cardinalities):
                raise ValueError(
                    f"tail-collapse field index {index} must be within "
                    f"[0, {len(cardinalities)})"
                )

        self.max_count = int(collapse_cfg.get("max_count", 5))
        if self.max_count < 0:
            raise ValueError("tail-collapse max_count must be non-negative")
        self.include_unseen = bool(collapse_cfg.get("include_unseen", True))

        cache_value = collapse_cfg.get("frequency_cache_path")
        if not cache_value:
            raise ValueError(
                f"model.{self.CONFIG_KEY}.frequency_cache_path is required"
            )
        cache_path = Path(str(cache_value)).expanduser()
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"tail-collapse frequency cache does not exist: {cache_path}"
            )

        train_days = config.get("dataset", {}).get("train_days")
        if not isinstance(train_days, list) or not train_days:
            raise ValueError(
                "dataset.train_days is required to validate the tail-collapse cache"
            )
        expected_days = _days_key(train_days)

        mask_names: list[str] = []
        field_metadata: list[dict] = []
        with np.load(cache_path, allow_pickle=False) as data:
            cached_days = str(data["days"].item())
            if cached_days != expected_days:
                raise ValueError(
                    "tail-collapse cache train days mismatch: "
                    f"cache={cached_days!r}, expected={expected_days!r}"
                )
            cached_fields = set(data["fields"].astype(np.int64).tolist())
            for position, field_index in enumerate(field_indices):
                if field_index not in cached_fields:
                    raise ValueError(
                        f"tail-collapse field {field_index} is absent from cache fields"
                    )
                key = f"counts_{field_index}"
                if key not in data:
                    raise ValueError(f"tail-collapse cache is missing {key}")
                counts = data[key].astype(np.uint32)
                original_cardinality = int(cardinalities[field_index])
                if counts.shape != (original_cardinality,):
                    raise ValueError(
                        f"tail-collapse {key} length {len(counts)} must match "
                        f"cardinality {original_cardinality}"
                    )

                if self.include_unseen:
                    collapse_mask = counts <= self.max_count
                else:
                    collapse_mask = (counts > 0) & (counts <= self.max_count)
                # ID 0 remains the dedicated missing/padding value. The shared
                # tail key is a new trainable row at the end of the table.
                collapse_mask[0] = False
                tail_key_virtual_id = original_cardinality

                mask_name = f"collapse_mask_{position}"
                mask_tensor = torch.frombuffer(
                    memoryview(collapse_mask), dtype=torch.bool
                ).clone()
                self.register_buffer(
                    mask_name,
                    mask_tensor,
                    persistent=False,
                )
                mask_names.append(mask_name)
                field_name = (
                    str(field_names[field_index])
                    if field_index < len(field_names)
                    else f"field_{field_index}"
                )
                field_metadata.append(
                    {
                        "index": field_index,
                        "name": field_name,
                        "original_cardinality": original_cardinality,
                        "private_embedding_rows": original_cardinality,
                        "tail_key_virtual_id": tail_key_virtual_id,
                        "tail_key_storage": "dedicated_zero_initialized_parameter",
                        "collapsed_id_count": int(collapse_mask.sum()),
                        "collapsed_observed_id_count": int(
                            ((counts > 0) & collapse_mask).sum()
                        ),
                        "collapsed_unseen_id_count": int(
                            ((counts == 0) & collapse_mask).sum()
                        ),
                    }
                )

        self.field_indices = tuple(field_indices)
        self._mask_names = tuple(mask_names)
        self._field_metadata = tuple(field_metadata)
        self.shared_tail_embeddings = nn.Parameter(
            torch.zeros(len(field_indices), int(embedding_dim))
        )
        self.cache_path = str(cache_path)
        self.cache_sha256 = _sha256(cache_path)
        self.train_days = tuple(int(day) for day in train_days)
        self._last_collapse_fraction = torch.tensor(0.0)

    def forward(
        self, sparse: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if sparse.ndim != 2:
            raise ValueError(
                f"tail-collapse sparse input must be rank 2, got {tuple(sparse.shape)}"
            )
        mapped = sparse.clone()
        collapsed = sparse.new_zeros((), dtype=torch.long)
        total = sparse.new_zeros((), dtype=torch.long)
        collapse_masks: list[torch.Tensor] = []
        for position, field_index in enumerate(self.field_indices):
            ids = sparse[:, field_index]
            mask = getattr(self, self._mask_names[position]).index_select(0, ids)
            collapse_masks.append(mask)
            # Use padding only as an intermediate lookup. The resulting zero
            # vector is replaced by the dedicated shared tail parameter below.
            mapped[:, field_index] = torch.where(
                mask,
                torch.zeros_like(ids),
                ids,
            )
            collapsed = collapsed + mask.sum()
            total = total + mask.numel()
        self._last_collapse_fraction = (
            collapsed.detach().float() / total.clamp_min(1).detach().float()
        )
        return mapped, tuple(collapse_masks)

    def replace_collapsed_embeddings(
        self,
        fields: torch.Tensor,
        collapse_masks: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(collapse_masks) != len(self.field_indices):
            raise ValueError("tail-collapse mask count must match selected fields")
        mask_by_field = {
            field_index: collapse_masks[position]
            for position, field_index in enumerate(self.field_indices)
        }
        position_by_field = {
            field_index: position
            for position, field_index in enumerate(self.field_indices)
        }
        output_fields = []
        for field_index in range(fields.size(1)):
            field = fields[:, field_index, :]
            if field_index in mask_by_field:
                position = position_by_field[field_index]
                field = torch.where(
                    mask_by_field[field_index].unsqueeze(1),
                    self.shared_tail_embeddings[position].unsqueeze(0),
                    field,
                )
            output_fields.append(field)
        return torch.stack(output_fields, dim=1)

    def get_diagnostics(self) -> dict[str, float]:
        return {
            "offline_tail_collapse_fraction": float(
                self._last_collapse_fraction.detach().cpu().item()
            )
        }

    def get_metadata(self) -> dict:
        return {
            "enabled": True,
            "experimental_scope": "offline_frequency_oracle_ablation",
            "production_ready": False,
            "max_count": self.max_count,
            "include_unseen": self.include_unseen,
            "missing_id_collapsed": False,
            "frequency_cache_path": self.cache_path,
            "frequency_cache_sha256": self.cache_sha256,
            "train_days": list(self.train_days),
            "fields": [dict(value) for value in self._field_metadata],
        }
