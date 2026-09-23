"""TAAC2025 temporal reader wrapper.

This reader expects an already prepared directory containing train/val/test
samples.parquet directories and shared TAAC assets. It avoids importing pyarrow
until construction time so the registry remains available on machines without
TAAC dependencies installed.
"""

from __future__ import annotations

from recscale.datasets import register_dataset
from recscale.datasets.base import BaseDataset


@register_dataset("taac2025_time_temporal")
class TAAC2025TimeTemporalDataset(BaseDataset):
    """Delegating wrapper around recscale.datasets.taac2025_time.TAAC2025TimeDataset."""

    def __init__(self, config: dict, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"[TAAC2025TimeTemporal] split must be train/val/test, got {split}")
        from recscale.datasets.taac2025_time import TAAC2025TimeDataset

        self._inner = TAAC2025TimeDataset(config, split=split)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]

    @staticmethod
    def collate_fn(samples):
        from recscale.datasets.taac2025_time import TAAC2025TimeDataset

        if hasattr(TAAC2025TimeDataset, "collate_fn"):
            return TAAC2025TimeDataset.collate_fn(samples)
        return BaseDataset.collate_fn(samples)
