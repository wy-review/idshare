"""KuaiRand K1 high-cardinality identity routing.

The router keeps the model shape identical across the four K1 arms. Every arm
owns one trainable tail vector per registered identity field; configuration
only decides whether side identities are visible and which fields route
train-count 1..cutoff IDs through those shared vectors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class K1IdentityRouter(nn.Module):
    """Route low-evidence identities without allocating test-only private rows."""

    CONFIG_KEY = "k1_identity_routing"
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
        count_arrays: Optional[dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        cfg = config["model"].get(self.CONFIG_KEY, {})
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            raise ValueError("K1IdentityRouter requires model.k1_identity_routing.enabled=true")

        self.field_names = list(field_names)
        self.name_to_index = {name: index for index, name in enumerate(self.field_names)}
        self.identity_fields = tuple(
            cfg.get("identity_fields", ["video_id", "author_id", "music_id"])
        )
        if len(set(self.identity_fields)) != len(self.identity_fields):
            raise ValueError("k1_identity_routing.identity_fields must be unique")
        missing_fields = [name for name in self.identity_fields if name not in self.name_to_index]
        if missing_fields:
            raise ValueError(f"K1 identity fields absent from dataset.sparse_cols: {missing_fields}")

        self.side_fields = tuple(cfg.get("side_fields", ["author_id", "music_id"]))
        if any(name not in self.identity_fields for name in self.side_fields):
            raise ValueError("k1_identity_routing.side_fields must be identity fields")
        self.side_enabled = bool(cfg.get("side_enabled", True))
        self.tail_share_fields = frozenset(cfg.get("tail_share_fields", []))
        unknown_tail_fields = self.tail_share_fields.difference(self.identity_fields)
        if unknown_tail_fields:
            raise ValueError(f"tail_share_fields are not identity fields: {sorted(unknown_tail_fields)}")

        self.max_count = int(cfg.get("max_count", 5))
        if self.max_count < 1:
            raise ValueError(f"k1_identity_routing.max_count must be >=1, got {self.max_count}")

        special_ids = dict(self.DEFAULT_SPECIAL_IDS)
        special_ids.update(cfg.get("special_token_ids", {}))
        if sorted(special_ids.values()) != list(range(len(special_ids))):
            raise ValueError(
                "K1 special token IDs must be distinct contiguous values starting at zero"
            )
        self.special_token_ids = special_ids
        self.register_buffer(
            "_special_ids",
            torch.tensor(sorted(special_ids.values()), dtype=torch.long),
            persistent=False,
        )

        if count_arrays is None:
            cache_path = Path(str(cfg.get("frequency_cache_path", "")))
            if not cache_path.is_file():
                raise FileNotFoundError(f"K1 frequency cache does not exist: {cache_path}")
            count_arrays, cache_metadata = self._load_count_cache(cache_path)
            self.frequency_cache_path = str(cache_path)
            self.frequency_cache_metadata = cache_metadata
        else:
            self.frequency_cache_path = "<in-memory>"
            self.frequency_cache_metadata = {"source": "in-memory"}

        self._count_buffer_names: dict[str, str] = {}
        field_metadata = []
        for field in self.identity_fields:
            field_index = self.name_to_index[field]
            counts = np.asarray(count_arrays[field], dtype=np.int64)
            expected = int(cardinalities[field_index])
            if counts.ndim != 1 or counts.shape[0] != expected:
                raise ValueError(
                    f"K1 count array for {field} must have shape ({expected},), "
                    f"got {counts.shape}"
                )
            if np.any(counts < 0):
                raise ValueError(f"K1 count array for {field} contains negative values")
            for token_id in self.special_token_ids.values():
                if counts[token_id] != 0:
                    raise ValueError(
                        f"K1 count array for {field} must keep special token {token_id} at count 0"
                    )
            buffer_name = f"_counts_{field}"
            count_tensor = torch.frombuffer(
                memoryview(np.ascontiguousarray(counts)), dtype=torch.int64
            ).clone()
            self.register_buffer(buffer_name, count_tensor, persistent=False)
            self._count_buffer_names[field] = buffer_name
            tail_count = int(((counts > 0) & (counts <= self.max_count)).sum())
            field_metadata.append(
                {
                    "field": field,
                    "index": field_index,
                    "cardinality": expected,
                    "tail_private_rows": tail_count,
                    "tail_share_active": field in self.tail_share_fields,
                    "side_disabled": field in self.side_fields and not self.side_enabled,
                }
            )

        self.tail_embeddings = nn.ParameterDict(
            {
                field: nn.Parameter(torch.zeros(int(embedding_dim)))
                for field in self.identity_fields
            }
        )
        self._metadata = {
            "enabled": True,
            "identity_fields": list(self.identity_fields),
            "side_fields": list(self.side_fields),
            "side_enabled": self.side_enabled,
            "tail_share_fields": sorted(self.tail_share_fields),
            "max_count": self.max_count,
            "special_token_ids": dict(self.special_token_ids),
            "frequency_cache_path": self.frequency_cache_path,
            "frequency_cache_metadata": self.frequency_cache_metadata,
            "fields": field_metadata,
            "production_ready": False,
            "experimental_scope": "offline_train_frequency_oracle_ablation",
        }
        self._diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _load_count_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {
                name[: -len("_counts")]: np.asarray(payload[name], dtype=np.int64)
                for name in payload.files
                if name.endswith("_counts")
            }
            metadata = {}
            if "metadata_json" in payload.files:
                metadata = json.loads(str(payload["metadata_json"].item()))
        return arrays, metadata

    def forward(
        self, sparse: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if sparse.ndim != 2 or sparse.size(1) != len(self.field_names):
            raise ValueError(
                "K1 identity router expects sparse shape "
                f"(B,{len(self.field_names)}), got {tuple(sparse.shape)}"
            )
        routed = sparse.clone()
        masks: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, torch.Tensor] = {}

        for field in self.identity_fields:
            index = self.name_to_index[field]
            ids = sparse[:, index]
            counts = getattr(self, self._count_buffer_names[field])
            if ids.numel() and (ids.min() < 0 or ids.max() >= counts.numel()):
                raise IndexError(
                    f"K1 sparse IDs for {field} exceed [0,{counts.numel() - 1}]"
                )

            side_disabled = field in self.side_fields and not self.side_enabled
            if side_disabled:
                mask = torch.zeros_like(ids, dtype=torch.bool)
                routed[:, index] = self.special_token_ids["padding"]
            elif field in self.tail_share_fields:
                observed_counts = counts.index_select(0, ids)
                mask = observed_counts.gt(0) & observed_counts.le(self.max_count)
                routed[:, index] = torch.where(
                    mask,
                    torch.full_like(ids, self.special_token_ids["padding"]),
                    ids,
                )
            else:
                mask = torch.zeros_like(ids, dtype=torch.bool)

            masks[field] = mask
            diagnostics[f"k1_{field}_tail_fraction"] = mask.float().mean()
            diagnostics[f"k1_{field}_disabled_fraction"] = torch.tensor(
                float(side_disabled), device=sparse.device
            )

        self._diagnostics = diagnostics
        return routed, masks

    def replace_routed_embeddings(
        self,
        fields: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        output = fields.clone()
        for field in self.identity_fields:
            index = self.name_to_index[field]
            tail = self.tail_embeddings[field].to(dtype=output.dtype)
            mask = masks[field]
            if mask.any():
                output[:, index, :] = torch.where(
                    mask.unsqueeze(-1),
                    tail.view(1, -1),
                    output[:, index, :],
                )
            # Keep all three tail parameters in every arm's graph while preserving
            # exactly zero contribution for inactive routes.
            output[:, index, :] = output[:, index, :] + tail.view(1, -1) * 0.0
        return output

    def get_diagnostics(self) -> dict[str, float]:
        return {
            name: float(value.detach().float().cpu().item())
            for name, value in self._diagnostics.items()
        }

    def get_metadata(self) -> dict:
        return json.loads(json.dumps(self._metadata))
