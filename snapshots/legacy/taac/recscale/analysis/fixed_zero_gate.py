"""Deterministic train-frequency gates for zero-anchor mechanism controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FIRST_PRIVATE_ID = 4
FREQUENCY_BUCKET_BOUNDS = {
    "count_1": (1, 1),
    "count_2": (2, 2),
    "count_3": (3, 3),
    "count_4": (4, 4),
    "count_5": (5, 5),
    "count_6_10": (6, 10),
    "count_11_20": (11, 20),
    "count_21_50": (21, 50),
    "count_51_100": (51, 100),
    "count_101_500": (101, 500),
    "count_gt_500": (501, None),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bucket_mask(counts: np.ndarray, bucket: str) -> np.ndarray:
    if bucket not in FREQUENCY_BUCKET_BOUNDS:
        raise KeyError(f"unknown frequency bucket {bucket!r}")
    lower, upper = FREQUENCY_BUCKET_BOUNDS[bucket]
    if upper is None:
        return counts >= lower
    return (counts >= lower) & (counts <= upper)


def _stable_seed(base_seed: int, field: str, bucket: str) -> np.uint64:
    payload = f"{int(base_seed)}:{field}:{bucket}".encode("utf-8")
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def _splitmix64(values: np.ndarray, seed: np.uint64) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = values + seed + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        return mixed ^ (mixed >> np.uint64(31))


def _select_exact_by_hash(
    candidate_ids: np.ndarray,
    *,
    target_count: int,
    base_seed: int,
    field: str,
    bucket: str,
) -> np.ndarray:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    target_count = int(target_count)
    if not 0 <= target_count <= candidate_ids.size:
        raise ValueError(
            f"invalid fixed-gate target for {field}/{bucket}: "
            f"{target_count} of {candidate_ids.size}"
        )
    if target_count == 0:
        return np.empty(0, dtype=np.int64)
    if target_count == candidate_ids.size:
        return candidate_ids
    hashes = _splitmix64(
        candidate_ids,
        _stable_seed(base_seed, field, bucket),
    )
    cutoff = np.partition(hashes, target_count - 1)[target_count - 1]
    selected = candidate_ids[hashes < cutoff]
    remaining = target_count - selected.size
    if remaining:
        ties = np.sort(candidate_ids[hashes == cutoff])
        selected = np.concatenate((selected, ties[:remaining]))
    if selected.size != target_count:
        raise RuntimeError("exact fixed-gate hash selection did not close")
    return selected


def build_fixed_zero_gate(
    counts: np.ndarray,
    *,
    field: str,
    policy: dict,
    first_private_id: int = FIRST_PRIVATE_ID,
) -> tuple[np.ndarray, dict]:
    """Build an exact fixed gate without validation data or labels."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size <= first_private_id:
        raise ValueError("frequency counts must be a nonempty rank-1 array")
    if np.any(counts[:first_private_id] != 0):
        raise ValueError("special rows must have zero train frequency")
    if np.any(counts[first_private_id:] <= 0):
        raise ValueError("every private train-vocabulary row must be observed")
    mode = str(policy["mode"])
    hash_seed = int(policy["hash_seed"])
    private_counts = counts[first_private_id:]
    gate = np.zeros(counts.size, dtype=np.bool_)
    bucket_report = {}

    if mode == "oracle_bucket_matched":
        targets = {
            str(key): int(value)
            for key, value in policy["target_zero_ids_by_bucket"].items()
        }
        if set(targets) != set(FREQUENCY_BUCKET_BOUNDS):
            raise ValueError("oracle gate must define every frequency bucket")
        for bucket in FREQUENCY_BUCKET_BOUNDS:
            offsets = np.flatnonzero(_bucket_mask(private_counts, bucket))
            candidate_ids = offsets + first_private_id
            selected = _select_exact_by_hash(
                candidate_ids,
                target_count=targets[bucket],
                base_seed=hash_seed,
                field=field,
                bucket=bucket,
            )
            gate[selected] = True
            bucket_report[bucket] = {
                "private_ids": int(candidate_ids.size),
                "gated_private_ids": int(selected.size),
            }
    elif mode == "frequency_threshold_matched":
        full_gate_max_count = int(policy["full_gate_max_count"])
        boundary_bucket = str(policy["boundary_bucket"])
        boundary_target = int(policy["boundary_target_zero_ids"])
        if boundary_bucket not in FREQUENCY_BUCKET_BOUNDS:
            raise ValueError("unknown threshold boundary bucket")
        gate[first_private_id:] = private_counts <= full_gate_max_count
        boundary_offsets = np.flatnonzero(
            _bucket_mask(private_counts, boundary_bucket)
        )
        if np.any(gate[boundary_offsets + first_private_id]):
            raise ValueError("threshold boundary overlaps fully gated counts")
        selected = _select_exact_by_hash(
            boundary_offsets + first_private_id,
            target_count=boundary_target,
            base_seed=hash_seed,
            field=field,
            bucket=boundary_bucket,
        )
        gate[selected] = True
        for bucket in FREQUENCY_BUCKET_BOUNDS:
            ids = np.flatnonzero(_bucket_mask(private_counts, bucket))
            bucket_gate = gate[ids + first_private_id]
            bucket_report[bucket] = {
                "private_ids": int(ids.size),
                "gated_private_ids": int(bucket_gate.sum()),
            }
    else:
        raise ValueError(f"unsupported fixed zero gate mode {mode!r}")

    gated = int(gate[first_private_id:].sum())
    expected = int(policy["expected_total_zero_ids"])
    if gated != expected:
        raise RuntimeError(
            f"fixed gate total differs from frozen target: {gated} != {expected}"
        )
    report = {
        "mode": mode,
        "selection_algorithm": "splitmix64_rank_exact_v1",
        "hash_seed": hash_seed,
        "field": field,
        "frequency_source": "train_only",
        "uses_validation_data": False,
        "uses_validation_labels": False,
        "private_ids": int(private_counts.size),
        "gated_private_ids": gated,
        "gated_private_id_fraction": gated / float(private_counts.size),
        "buckets": bucket_report,
    }
    return gate, report


def build_fixed_zero_gate_from_cache(
    cache_path: str | Path,
    *,
    expected_cache_sha256: str,
    expected_cardinality: int,
    field: str,
    policy: dict,
) -> tuple[np.ndarray, dict]:
    cache_path = Path(cache_path).resolve()
    actual_sha256 = sha256_file(cache_path)
    if actual_sha256 != expected_cache_sha256:
        raise RuntimeError("fixed-gate frequency cache SHA256 mismatch")
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("train_only") is not True:
            raise ValueError("fixed-gate frequency cache must be train-only")
        counts = np.asarray(payload[f"{field}_counts"], dtype=np.int64)
    if counts.size != int(expected_cardinality):
        raise RuntimeError("fixed-gate frequency cardinality mismatch")
    gate, report = build_fixed_zero_gate(
        counts,
        field=field,
        policy=policy,
    )
    report.update(
        {
            "frequency_cache_sha256": actual_sha256,
            "frequency_cache_metadata_train_only": True,
        }
    )
    return gate, report
