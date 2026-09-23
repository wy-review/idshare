"""Memory-mapped pointwise KuaiRand-27K dataset for the K1 experiment.

The processed format stores user and video features once and keeps interaction
shards compact. Train shards are already deterministically shuffled, so the
Trainer must use ``training.shuffle=false`` to preserve the frozen order.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import register_dataset
from .base import BaseDataset


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@register_dataset("kuairand27k_k1_mmap")
class KuaiRand27KK1MMapDataset(BaseDataset):
    """Map-style reader over frozen interaction shards and static side tables."""

    def __init__(self, config: dict, split: str = "train"):
        dataset_cfg = config["dataset"]
        manifest_path = Path(str(dataset_cfg["processed_manifest"])).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"K1 processed manifest does not exist: {manifest_path}")
        self.root = manifest_path.parent
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "kuairand27k_k1_mmap_v1":
            raise ValueError(
                "unsupported K1 processed format: "
                f"{self.manifest.get('format')!r}"
            )
        if split not in self.manifest["splits"]:
            raise ValueError(f"split {split!r} absent from K1 manifest")
        self.split = split
        self.field_names = list(self.manifest["field_names"])
        configured_fields = list(dataset_cfg.get("sparse_cols") or self.field_names)
        if configured_fields != self.field_names:
            raise ValueError(
                "dataset.sparse_cols must exactly match the frozen K1 manifest field order"
            )
        configured_cards = list(dataset_cfg.get("cardinalities") or [])
        manifest_cards = list(self.manifest["cardinalities"])
        if configured_cards and configured_cards != manifest_cards:
            raise ValueError("dataset.cardinalities differ from K1 manifest")

        self.user_field_count = int(self.manifest["tables"]["user_sparse"]["shape"][1])
        self.video_field_count = int(self.manifest["tables"]["video_sparse"]["shape"][1])
        if self.user_field_count + self.video_field_count != len(self.field_names):
            raise ValueError("K1 static table widths do not close to field_names")

        self.user_sparse = self._open_table("user_sparse")
        self.video_sparse = self._open_table("video_sparse")
        self.video_eval_meta = self._open_table("video_eval_meta")
        if self.video_eval_meta.shape[1] != 2:
            raise ValueError("video_eval_meta must contain [train_band, side_bits]")

        split_meta = self.manifest["splits"][split]
        self.shards = list(split_meta["shards"])
        self.shard_ends = []
        total = 0
        for shard in self.shards:
            total += int(shard["rows"])
            self.shard_ends.append(total)
        if total != int(split_meta["rows"]):
            raise ValueError(f"K1 {split} shard rows do not close: {total}")
        self._length = total
        self._opened_shards: dict[int, np.ndarray] = {}

        if split == "train" and not bool(split_meta.get("fixed_order", False)):
            raise ValueError("K1 train split must declare fixed_order=true")
        if split == "train" and bool(config["training"].get("shuffle", True)):
            raise ValueError(
                "K1 train shards are physically shuffled; set training.shuffle=false"
            )
        self.sha256_verification = {
            "enabled": False,
            "splits": [],
            "artifact_count": 0,
        }
        if bool(dataset_cfg.get("verify_processed_sha256", False)):
            configured_splits = dataset_cfg.get(
                "verify_processed_sha256_splits"
            )
            if configured_splits is None:
                verification_splits = tuple(self.manifest["splits"])
            else:
                if not isinstance(configured_splits, (list, tuple)):
                    raise ValueError(
                        "verify_processed_sha256_splits must be a list or tuple"
                    )
                verification_splits = tuple(
                    str(value) for value in configured_splits
                )
                if not verification_splits:
                    raise ValueError(
                        "verify_processed_sha256_splits must not be empty"
                    )
                if len(set(verification_splits)) != len(
                    verification_splits
                ):
                    raise ValueError(
                        "verify_processed_sha256_splits contains duplicates"
                    )
                unknown = sorted(
                    set(verification_splits) - set(self.manifest["splits"])
                )
                if unknown:
                    raise ValueError(
                        "verify_processed_sha256_splits contains unknown "
                        f"splits: {unknown}"
                    )
            self.sha256_verification = self.verify_sha256(
                splits=verification_splits
            )

    def _resolve(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError(f"K1 artifact escapes processed root: {relative_path}")
        return path

    def _open_table(self, name: str) -> np.ndarray:
        meta = self.manifest["tables"][name]
        path = self._resolve(meta["path"])
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != list(meta["shape"]) or str(array.dtype) != meta["dtype"]:
            raise ValueError(f"K1 table metadata mismatch for {name}")
        return array

    def verify_sha256(
        self,
        *,
        splits: list[str] | tuple[str, ...],
    ) -> dict:
        """Verify static/auxiliary artifacts plus the selected split shards."""
        selected_splits = tuple(str(value) for value in splits)
        artifacts = []
        artifacts.extend(self.manifest["tables"].values())
        for split in selected_splits:
            if split not in self.manifest["splits"]:
                raise ValueError(f"unknown K1 verification split {split!r}")
            split_meta = self.manifest["splits"][split]
            artifacts.extend(split_meta["shards"])
        artifacts.extend(self.manifest.get("auxiliary_artifacts", []))
        for artifact in artifacts:
            expected = artifact.get("sha256")
            if not expected:
                raise ValueError(f"missing SHA256 for K1 artifact {artifact['path']}")
            actual = _sha256_file(self._resolve(artifact["path"]))
            if actual != expected:
                raise ValueError(
                    f"K1 artifact SHA256 mismatch for {artifact['path']}: "
                    f"{actual} != {expected}"
                )
        return {
            "enabled": True,
            "splits": list(selected_splits),
            "artifact_count": len(artifacts),
        }

    def verify_all_sha256(self) -> None:
        """Backward-compatible verification of every frozen split."""
        self.sha256_verification = self.verify_sha256(
            splits=tuple(self.manifest["splits"])
        )

    def __len__(self) -> int:
        return self._length

    def _open_shard(self, shard_index: int) -> np.ndarray:
        if shard_index not in self._opened_shards:
            meta = self.shards[shard_index]
            shard = np.load(
                self._resolve(meta["path"]), mmap_mode="r", allow_pickle=False
            )
            required = {"user_index", "video_index", "label"}
            if shard.dtype.names is None or not required.issubset(shard.dtype.names):
                raise ValueError(f"K1 interaction shard has invalid dtype: {meta['path']}")
            if len(shard) != int(meta["rows"]):
                raise ValueError(f"K1 interaction shard row mismatch: {meta['path']}")
            self._opened_shards[shard_index] = shard
        return self._opened_shards[shard_index]

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.shard_ends, index)
        shard_start = 0 if shard_index == 0 else self.shard_ends[shard_index - 1]
        row = self._open_shard(shard_index)[index - shard_start]
        user_index = int(row["user_index"])
        video_index = int(row["video_index"])
        sparse = np.concatenate(
            (
                np.asarray(self.user_sparse[user_index], dtype=np.int64),
                np.asarray(self.video_sparse[video_index], dtype=np.int64),
            )
        )
        eval_meta = self.video_eval_meta[video_index]
        return {
            "sparse": sparse,
            "label": np.float32(row["label"]),
            "k1_video_train_band": np.uint8(eval_meta[0]),
            "k1_side_bits": np.uint8(eval_meta[1]),
        }

    @staticmethod
    def _tensor_from_array(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        contiguous = np.ascontiguousarray(array)
        tensor = torch.frombuffer(memoryview(contiguous), dtype=dtype)
        return tensor.reshape(contiguous.shape).clone()

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        """Fixed-shape collate without relying on PyTorch's NumPy C bridge."""
        sparse = np.stack([sample["sparse"] for sample in samples]).astype(
            np.int64, copy=False
        )
        labels = np.asarray([sample["label"] for sample in samples], dtype=np.float32)
        bands = np.asarray(
            [sample["k1_video_train_band"] for sample in samples], dtype=np.uint8
        )
        side_bits = np.asarray(
            [sample["k1_side_bits"] for sample in samples], dtype=np.uint8
        )
        return {
            "sparse": KuaiRand27KK1MMapDataset._tensor_from_array(
                sparse, torch.int64
            ),
            "label": KuaiRand27KK1MMapDataset._tensor_from_array(
                labels, torch.float32
            ),
            "k1_video_train_band": KuaiRand27KK1MMapDataset._tensor_from_array(
                bands, torch.uint8
            ),
            "k1_side_bits": KuaiRand27KK1MMapDataset._tensor_from_array(
                side_bits, torch.uint8
            ),
        }

    @property
    def num_samples(self) -> int:
        return self._length
