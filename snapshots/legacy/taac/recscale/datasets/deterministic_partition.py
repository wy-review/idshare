"""Deterministic, disjoint index partitions for frozen map-style datasets."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from torch.utils.data import Dataset


class ModuloPartitionDataset(Dataset):
    """Expose one ``index % modulus`` partition without copying samples."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        modulus: int,
        remainder: int,
        base_split_sha256: str,
        partition_name: str,
    ):
        modulus = int(modulus)
        remainder = int(remainder)
        if modulus < 2:
            raise ValueError("modulus must be at least 2")
        if not 0 <= remainder < modulus:
            raise ValueError("remainder must be in [0, modulus)")
        if not base_split_sha256:
            raise ValueError("base_split_sha256 must be non-empty")
        if not partition_name:
            raise ValueError("partition_name must be non-empty")
        self.dataset = dataset
        self.modulus = modulus
        self.remainder = remainder
        self.base_split_sha256 = str(base_split_sha256)
        self.partition_name = str(partition_name)
        self.collate_fn = getattr(dataset, "collate_fn", None)
        self.base_rows = len(dataset)
        self._length = max(
            0,
            (self.base_rows - self.remainder + self.modulus - 1)
            // self.modulus,
        )
        self.membership_sha256 = self.compute_membership_sha256(
            base_rows=self.base_rows,
            modulus=self.modulus,
            remainder=self.remainder,
            base_split_sha256=self.base_split_sha256,
            partition_name=self.partition_name,
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return self.dataset[self.remainder + index * self.modulus]

    @staticmethod
    def compute_membership_sha256(
        *,
        base_rows: int,
        modulus: int,
        remainder: int,
        base_split_sha256: str,
        partition_name: str,
        chunk_rows: int = 1_000_000,
    ) -> str:
        """Hash the canonical partition spec and every selected base index."""
        base_rows = int(base_rows)
        modulus = int(modulus)
        remainder = int(remainder)
        chunk_rows = int(chunk_rows)
        if base_rows < 0 or chunk_rows <= 0:
            raise ValueError("invalid membership digest dimensions")
        header = json.dumps(
            {
                "algorithm": "base_index_modulo_v1",
                "base_rows": base_rows,
                "base_split_sha256": str(base_split_sha256),
                "modulus": modulus,
                "partition_name": str(partition_name),
                "remainder": remainder,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hashlib.sha256()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        first = remainder
        for start in range(first, base_rows, modulus * chunk_rows):
            stop = min(base_rows, start + modulus * chunk_rows)
            indices = np.arange(start, stop, modulus, dtype="<u8")
            digest.update(indices.tobytes())
        return digest.hexdigest()

    def report(self) -> dict:
        return {
            "algorithm": "base_index_modulo_v1",
            "partition_name": self.partition_name,
            "base_rows": self.base_rows,
            "rows": self._length,
            "modulus": self.modulus,
            "remainder": self.remainder,
            "base_split_sha256": self.base_split_sha256,
            "membership_sha256": self.membership_sha256,
        }
