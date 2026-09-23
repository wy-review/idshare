"""TAAC loader with the frozen shared target/history identity boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .taac_dataloader import TaacCTRDataset


class TaacSharedIdentityDataset(TaacCTRDataset):
    _mask_key = None
    _mask = None

    def __init__(
        self,
        feature_map,
        manifest_path,
        *,
        train_input_seen_mask_path,
        train_input_seen_mask_sha256,
        shared_item_table_cardinality,
        expected_raw_data_root,
        expected_samples_path_by_split,
        split="train",
        **kwargs,
    ):
        self.split = str(split)
        if self.split not in {"train", "valid"}:
            raise ValueError("IDShare bridge permits train/validation only")
        split_key = "train" if self.split == "train" else "validation"
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_samples = expected_samples_path_by_split.get(split_key)
        manifest_checks = {
            "raw_data_root": manifest.get("raw_data_root")
            == str(expected_raw_data_root),
            "samples_path": manifest.get("samples_path")
            == str(expected_samples),
            "split": manifest.get("split") == split_key,
            "direct_split": manifest.get("index_array") in (None, ""),
            "test_or_holdout": False,
        }
        if not all(
            manifest_checks[key] is True
            for key in ("raw_data_root", "samples_path", "split", "direct_split")
        ):
            raise ValueError(f"TAAC frozen manifest contract failed: {manifest_checks}")
        self.manifest_contract = manifest_checks
        self.shared_item_table_cardinality = int(shared_item_table_cardinality)
        mask_path = Path(train_input_seen_mask_path)
        expected_sha = str(train_input_seen_mask_sha256)
        key = (str(mask_path.resolve()), expected_sha)
        if self.__class__._mask_key != key:
            digest = hashlib.sha256()
            with mask_path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_sha:
                raise ValueError("train-input seen mask SHA256 mismatch")
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if (
                mask.ndim != 1
                or mask.dtype != np.bool_
                or bool(mask[0])
                or mask.size + 3 != self.shared_item_table_cardinality
            ):
                raise ValueError("invalid train-input seen mask schema")
            self.__class__._mask = mask
            self.__class__._mask_key = key
        self.seen_mask = self.__class__._mask
        super().__init__(feature_map, str(manifest_path), **kwargs)

    def _remap(self, values):
        values = np.asarray(values, dtype=np.int64)
        remapped = np.zeros_like(values)
        positive = values > 0
        in_range = positive & (values < self.seen_mask.size)
        seen = np.zeros_like(positive)
        seen[in_range] = self.seen_mask[values[in_range]]
        remapped[positive] = 1
        remapped[seen] = values[seen] + 3
        if self.split == "train" and np.any(positive & ~seen):
            raise RuntimeError(
                "training input contains an identity outside the frozen seen mask"
            )
        return remapped

    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["target_item_id"] = np.int64(
            self._remap(np.asarray([row["target_item_id"]]))[0]
        )
        row["item_seq"] = self._remap(row["item_seq"])
        return row


class TaacSharedIdentityDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        feature_map,
        data_path,
        split="train",
        batch_size=2048,
        shuffle=False,
        num_workers=0,
        **kwargs,
    ):
        dataset = TaacSharedIdentityDataset(
            feature_map,
            data_path,
            split=split,
            maxlen=kwargs.get("maxlen", 100),
            user_array_maxlen=kwargs.get("user_array_maxlen", 10),
            max_samples=kwargs.get("max_samples"),
            sequence_side_fields=kwargs.get("sequence_side_fields"),
            emit_sequence_timestamp=False,
            emit_sequence_action_type=False,
            train_input_seen_mask_path=kwargs["train_input_seen_mask_path"],
            train_input_seen_mask_sha256=kwargs[
                "train_input_seen_mask_sha256"
            ],
            shared_item_table_cardinality=kwargs[
                "shared_item_table_cardinality"
            ],
            expected_raw_data_root=kwargs["expected_raw_data_root"],
            expected_samples_path_by_split=kwargs[
                "expected_samples_path_by_split"
            ],
        )
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=torch.utils.data.dataloader.default_collate,
            drop_last=False,
        )
        self.num_samples = len(dataset)
        self.num_blocks = 1
        self.num_batches = int(math.ceil(self.num_samples / batch_size))

    def __len__(self):
        return self.num_batches
