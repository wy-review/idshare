"""Train-frequency-stratified late-training audit for identity quantization.

The audit is deliberately read-only. It samples a deterministic set of
train-vocabulary rows per frequency bucket, then snapshots their raw norms and
hard assignments at pre-registered late-training progress points. This makes
assignment churn measurable even for count-one IDs, which cannot be audited by
waiting for the same ID to appear in another mini-batch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..utils.zero_anchor_regularization import (
    get_identity_embedding_weights,
    get_zero_anchor_identity_embedding_weights,
)


FIRST_PRIVATE_ID = 4
FREQUENCY_BUCKETS = (
    "unseen",
    "count_1",
    "count_2",
    "count_3",
    "count_4",
    "count_5",
    "count_6_10",
    "count_11_20",
    "count_21_50",
    "count_51_100",
    "count_101_500",
    "count_gt_500",
)
VALIDATION_BUCKETS = (
    "padding",
    "oov_unseen",
    "missing",
    "reserved_zero",
) + FREQUENCY_BUCKETS[1:]
VALIDATION_AGGREGATES = {
    "seen_rare_1_5": (
        "count_1",
        "count_2",
        "count_3",
        "count_4",
        "count_5",
    ),
    "seen_6_50": (
        "count_6_10",
        "count_11_20",
        "count_21_50",
    ),
    "seen_51_500": ("count_51_100", "count_101_500"),
    "all_seen_private": FREQUENCY_BUCKETS[1:],
}


def _bucket_mask(counts: np.ndarray, label: str) -> np.ndarray:
    if label == "count_1":
        return counts == 1
    if label == "count_2":
        return counts == 2
    if label == "count_3":
        return counts == 3
    if label == "count_4":
        return counts == 4
    if label == "count_5":
        return counts == 5
    if label == "count_6_10":
        return (counts >= 6) & (counts <= 10)
    if label == "count_11_20":
        return (counts >= 11) & (counts <= 20)
    if label == "count_21_50":
        return (counts >= 21) & (counts <= 50)
    if label == "count_51_100":
        return (counts >= 51) & (counts <= 100)
    if label == "count_101_500":
        return (counts >= 101) & (counts <= 500)
    if label == "count_gt_500":
        return counts > 500
    raise KeyError(f"unknown frequency bucket {label!r}")


def _stable_seed(base_seed: int, field: str, bucket: str) -> int:
    payload = f"{int(base_seed)}:{field}:{bucket}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sha256_int64(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.int64).tobytes()
    ).hexdigest()


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    ordered = np.sort(values)
    indices = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        np.sum((2.0 * indices - ordered.size - 1.0) * ordered)
        / (ordered.size * ordered.sum())
    )


def _distribution(values: np.ndarray, *, zero_index: int = 0) -> dict:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return {
            "assignments": 0,
            "effective_centers": 0,
            "effective_nonzero_centers": 0,
            "perplexity": 0.0,
            "max_center_fraction": 0.0,
            "zero_center_fraction": 0.0,
            "gini": 0.0,
        }
    size = max(int(values.max()) + 1, zero_index + 1)
    counts = np.bincount(values, minlength=size).astype(np.float64)
    return _distribution_from_counts(counts, zero_index=zero_index)


def _distribution_from_counts(
    counts: np.ndarray, *, zero_index: int = 0
) -> dict:
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts.size <= zero_index:
        counts = np.pad(counts, (0, zero_index + 1 - counts.size))
    total = float(counts.sum())
    if total <= 0:
        return {
            "assignments": 0,
            "effective_centers": 0,
            "effective_nonzero_centers": 0,
            "perplexity": 0.0,
            "max_center_fraction": 0.0,
            "zero_center_fraction": 0.0,
            "gini": 0.0,
        }
    probabilities = counts / total
    positive = probabilities > 0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    return {
        "assignments": int(total),
        "effective_centers": int(positive.sum()),
        "effective_nonzero_centers": int(
            np.count_nonzero(counts[1:] > 0)
        ),
        "perplexity": float(np.exp(entropy)),
        "max_center_fraction": float(probabilities.max()),
        "zero_center_fraction": float(probabilities[zero_index]),
        "gini": _gini(counts),
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _all_numeric_values_finite(value) -> bool:
    if isinstance(value, dict):
        return all(
            _all_numeric_values_finite(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _positive_frequency_bins(counts: torch.Tensor) -> torch.Tensor:
    if counts.ndim != 1 or counts.dtype != torch.long:
        raise ValueError("frequency counts must be a rank-1 int64 tensor")
    if counts.numel() and counts.min().item() <= 0:
        raise ValueError("private frequency rows must be positive")
    bins = torch.full_like(counts, 10)
    bins[counts <= 500] = 9
    bins[counts <= 100] = 8
    bins[counts <= 50] = 7
    bins[counts <= 20] = 6
    bins[counts <= 10] = 5
    bins[counts <= 5] = counts[counts <= 5] - 1
    return bins


def _validation_frequency_bins(
    encoded_ids: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    encoded_ids = np.asarray(encoded_ids, dtype=np.int64).reshape(-1)
    if encoded_ids.size and (
        encoded_ids.min() < 0 or encoded_ids.max() >= counts.size
    ):
        raise ValueError("encoded identity ID exceeds frequency cache")
    result = np.empty(encoded_ids.size, dtype=np.int8)
    special = encoded_ids < FIRST_PRIVATE_ID
    result[special] = encoded_ids[special].astype(np.int8, copy=False)
    private = ~special
    private_counts = counts[encoded_ids[private]]
    if private_counts.size and np.any(private_counts <= 0):
        raise ValueError("private validation ID has a non-positive train count")
    private_bins = np.full(private_counts.size, 10, dtype=np.int8)
    private_bins[private_counts <= 500] = 9
    private_bins[private_counts <= 100] = 8
    private_bins[private_counts <= 50] = 7
    private_bins[private_counts <= 20] = 6
    private_bins[private_counts <= 10] = 5
    low = private_counts <= 5
    private_bins[low] = private_counts[low] - 1
    result[private] = private_bins + 4
    return result


def _absolute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("AUC labels/scores must be nonempty and aligned")
    if np.unique(labels).size < 2:
        raise ValueError("AUC requires both positive and negative labels")
    return float(roc_auc_score(labels, scores))


def _validation_auc_report(
    *,
    score_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
    total_rows: int,
    distinct_encoded_rows: int,
    raw_distinct_identity_count_available: bool,
) -> dict:
    scores = (
        np.concatenate(score_parts)
        if score_parts
        else np.empty(0, dtype=np.float32)
    )
    labels = (
        np.concatenate(label_parts)
        if label_parts
        else np.empty(0, dtype=np.uint8)
    )
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    auc = (
        _absolute_auc(labels, scores)
        if positives > 0 and negatives > 0
        else None
    )
    return {
        "rows": int(labels.size),
        "row_fraction": (
            float(labels.size / total_rows) if total_rows else 0.0
        ),
        "positives": positives,
        "negatives": negatives,
        "absolute_auc": float(auc) if auc is not None else None,
        "distinct_encoded_rows_touched": int(distinct_encoded_rows),
        "raw_distinct_identity_count_available": bool(
            raw_distinct_identity_count_available
        ),
    }


@torch.no_grad()
def evaluate_validation_frequency_auc(
    model: nn.Module,
    validation_dataset,
    *,
    frequency_cache_path: str | Path,
    identity_fields: list[str] | tuple[str, ...],
    field_names: list[str] | tuple[str, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Evaluate absolute validation AUC by each field's train frequency."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid validation loader parameters")
    identity_fields = tuple(str(field) for field in identity_fields)
    field_names = tuple(str(field) for field in field_names)
    field_indices = {
        field: field_names.index(field) for field in identity_fields
    }
    cache_path = Path(frequency_cache_path).resolve()
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("train_only") is not True:
            raise ValueError("validation frequency cache must be train-only")
        counts = {
            field: np.asarray(
                payload[f"{field}_counts"], dtype=np.int64
            )
            for field in identity_fields
        }
    chunks = {
        field: {
            label: {"scores": [], "labels": []}
            for label in VALIDATION_BUCKETS
        }
        for field in identity_fields
    }
    visited = {
        field: np.zeros(values.size, dtype=np.bool_)
        for field, values in counts.items()
    }
    all_scores = []
    all_labels = []
    loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=validation_dataset.collate_fn,
        pin_memory=True,
    )
    was_training = model.training
    model.eval()
    for batch in loader:
        sparse_cpu = batch["sparse"]
        frequency_sparse_cpu = batch.get("frequency_sparse", sparse_cpu)
        labels = (
            batch["label"].detach().cpu().numpy().reshape(-1).astype(
                np.uint8, copy=False
            )
        )
        device_batch = {
            key: (
                value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
            if key != "frequency_sparse"
        }
        output = model(device_batch)
        logits = output[0] if isinstance(output, tuple) else output
        scores = (
            torch.sigmoid(logits)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        all_scores.append(scores)
        all_labels.append(labels)
        sparse = frequency_sparse_cpu.numpy()
        for field in identity_fields:
            encoded = np.asarray(
                sparse[:, field_indices[field]], dtype=np.int64
            )
            visited[field][encoded] = True
            bucket_ids = _validation_frequency_bins(
                encoded, counts[field]
            )
            for index, label in enumerate(VALIDATION_BUCKETS):
                selected = bucket_ids == index
                if selected.any():
                    chunks[field][label]["scores"].append(scores[selected])
                    chunks[field][label]["labels"].append(labels[selected])
    if was_training:
        model.train()
    scores_all = np.concatenate(all_scores)
    labels_all = np.concatenate(all_labels)
    overall_auc = _absolute_auc(labels_all, scores_all)
    fields = {}
    total_rows = int(labels_all.size)
    for field in identity_fields:
        bucket_reports = {}
        encoded_rows = np.flatnonzero(visited[field])
        encoded_buckets = _validation_frequency_bins(
            encoded_rows, counts[field]
        )
        for index, label in enumerate(VALIDATION_BUCKETS):
            score_parts = chunks[field][label]["scores"]
            label_parts = chunks[field][label]["labels"]
            distinct_encoded = int(
                np.count_nonzero(encoded_buckets == index)
            )
            bucket_reports[label] = _validation_auc_report(
                score_parts=score_parts,
                label_parts=label_parts,
                total_rows=total_rows,
                distinct_encoded_rows=distinct_encoded,
                raw_distinct_identity_count_available=(
                    label != "oov_unseen"
                ),
            )
        aggregate_reports = {}
        for aggregate, labels in VALIDATION_AGGREGATES.items():
            score_parts = [
                part
                for label in labels
                for part in chunks[field][label]["scores"]
            ]
            label_parts = [
                part
                for label in labels
                for part in chunks[field][label]["labels"]
            ]
            distinct_encoded = sum(
                bucket_reports[label]["distinct_encoded_rows_touched"]
                for label in labels
            )
            aggregate_reports[aggregate] = {
                **_validation_auc_report(
                    score_parts=score_parts,
                    label_parts=label_parts,
                    total_rows=total_rows,
                    distinct_encoded_rows=distinct_encoded,
                    raw_distinct_identity_count_available=True,
                ),
                "source_buckets": list(labels),
            }
        field_checks = {
            "rows_close": (
                sum(item["rows"] for item in bucket_reports.values())
                == total_rows
            ),
            "positives_close": (
                sum(
                    item["positives"]
                    for item in bucket_reports.values()
                )
                == int(labels_all.sum())
            ),
            "negatives_close": (
                sum(
                    item["negatives"]
                    for item in bucket_reports.values()
                )
                == total_rows - int(labels_all.sum())
            ),
        }
        fields[field] = {
            "buckets": bucket_reports,
            "aggregates": aggregate_reports,
            "checks": field_checks,
            "all_checks_pass": all(field_checks.values()),
        }
    checks = {
        "nonempty_validation": total_rows > 0,
        "all_fields_close": all(
            result["all_checks_pass"] for result in fields.values()
        ),
        "overall_auc_finite": math.isfinite(overall_auc),
    }
    return {
        "split": "validation",
        "uses_test_dataset": False,
        "uses_test_labels": False,
        "frequency_source": "train_only",
        "frequency_cache_path": str(cache_path),
        "frequency_cache_sha256": (
            ZeroAnchorFrequencyDynamicsAudit._sha256_file(cache_path)
        ),
        "validation_buckets": list(VALIDATION_BUCKETS),
        "overall": {
            "rows": total_rows,
            "positives": int(labels_all.sum()),
            "negatives": total_rows - int(labels_all.sum()),
            "absolute_auc": overall_auc,
        },
        "identity_fields": fields,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "conclusion_boundary": {
            "oov_unseen_is_shared_encoded_row": True,
            "raw_unseen_distinct_counts_require_external_audit": True,
            "frequency_buckets_are_report_only": True,
        },
    }


@torch.no_grad()
def evaluate_validation_tailmask_auc(
    model: nn.Module,
    validation_dataset,
    *,
    frequency_cache_path: str | Path,
    identity_fields: list[str] | tuple[str, ...],
    field_names: list[str] | tuple[str, ...],
    cutoffs: list[int] | tuple[int, ...],
    include_unseen: bool,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Evaluate validation AUC after cumulative identity masking to OOV."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid validation loader parameters")
    cutoffs = tuple(int(value) for value in cutoffs)
    if (
        not cutoffs
        or any(value <= 0 for value in cutoffs)
        or tuple(sorted(set(cutoffs))) != cutoffs
    ):
        raise ValueError("cutoffs must be unique, positive, and increasing")
    identity_fields = tuple(str(field) for field in identity_fields)
    field_names = tuple(str(field) for field in field_names)
    field_indices = {
        field: field_names.index(field) for field in identity_fields
    }
    cache_path = Path(frequency_cache_path).resolve()
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("train_only") is not True:
            raise ValueError("validation Tailmask cache must be train-only")
        counts = {
            field: np.asarray(
                payload[f"{field}_counts"], dtype=np.int64
            )
            for field in identity_fields
        }

    score_parts = {"clean": []}
    score_parts.update({str(cutoff): [] for cutoff in cutoffs})
    label_parts: list[np.ndarray] = []
    coverage = {
        str(cutoff): {
            "selected_rows": 0,
            "changed_rows": 0,
            "already_oov_rows": 0,
            "fields": {
                field: {
                    "selected_rows": 0,
                    "changed_rows": 0,
                    "already_oov_rows": 0,
                    "distinct_seen_rare_ids": set(),
                }
                for field in identity_fields
            },
        }
        for cutoff in cutoffs
    }
    nested_masks_valid = True
    loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=validation_dataset.collate_fn,
        pin_memory=True,
    )
    was_training = model.training
    model.eval()
    for batch in loader:
        sparse_cpu = batch["sparse"]
        frequency_sparse_cpu = batch.get("frequency_sparse", sparse_cpu)
        labels = (
            batch["label"].detach().cpu().numpy().reshape(-1).astype(
                np.uint8, copy=False
            )
        )
        label_parts.append(labels)
        clean_batch = {
            key: (
                value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
            if key != "frequency_sparse"
        }
        clean_output = model(clean_batch)
        clean_logits = (
            clean_output[0]
            if isinstance(clean_output, tuple)
            else clean_output
        )
        score_parts["clean"].append(
            torch.sigmoid(clean_logits)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=False)
        )

        sparse_numpy = frequency_sparse_cpu.detach().cpu().numpy()
        previous_selected_mask = np.zeros(labels.size, dtype=np.bool_)
        previous_changed_mask = np.zeros(labels.size, dtype=np.bool_)
        for cutoff in cutoffs:
            masked_sparse = sparse_cpu.clone()
            selected_union = np.zeros(labels.size, dtype=np.bool_)
            changed_union = np.zeros(labels.size, dtype=np.bool_)
            oov_union = np.zeros(labels.size, dtype=np.bool_)
            cutoff_report = coverage[str(cutoff)]
            for field in identity_fields:
                column = field_indices[field]
                encoded = np.asarray(
                    sparse_numpy[:, column], dtype=np.int64
                )
                if encoded.size and (
                    encoded.min() < 0
                    or encoded.max() >= counts[field].size
                ):
                    raise ValueError(
                        f"encoded identity ID exceeds cache for {field}"
                    )
                already_oov = encoded == 1
                private = encoded >= FIRST_PRIVATE_ID
                private_counts = counts[field][encoded[private]]
                if private_counts.size and np.any(private_counts <= 0):
                    raise ValueError(
                        f"private validation ID has invalid count for {field}"
                    )
                changed = np.zeros(encoded.size, dtype=np.bool_)
                changed[private] = private_counts <= cutoff
                selected = changed | (
                    already_oov if include_unseen else False
                )
                if changed.any():
                    masked_sparse[torch.from_numpy(changed), column] = 1
                selected_union |= selected
                changed_union |= changed
                oov_union |= already_oov
                field_report = cutoff_report["fields"][field]
                field_report["selected_rows"] += int(selected.sum())
                field_report["changed_rows"] += int(changed.sum())
                field_report["already_oov_rows"] += int(
                    already_oov.sum()
                )
                field_report["distinct_seen_rare_ids"].update(
                    int(value) for value in np.unique(encoded[changed])
                )
            nested_masks_valid &= bool(
                np.all(~previous_selected_mask | selected_union)
                and np.all(~previous_changed_mask | changed_union)
            )
            previous_selected_mask = selected_union
            previous_changed_mask = changed_union
            cutoff_report["selected_rows"] += int(selected_union.sum())
            cutoff_report["changed_rows"] += int(changed_union.sum())
            cutoff_report["already_oov_rows"] += int(oov_union.sum())
            masked_batch = {
                key: (
                    masked_sparse.to(device, non_blocking=True)
                    if key == "sparse"
                    else (
                        value.to(device, non_blocking=True)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                )
                for key, value in batch.items()
                if key != "frequency_sparse"
            }
            masked_output = model(masked_batch)
            masked_logits = (
                masked_output[0]
                if isinstance(masked_output, tuple)
                else masked_output
            )
            score_parts[str(cutoff)].append(
                torch.sigmoid(masked_logits)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32, copy=False)
            )
    if was_training:
        model.train()

    labels_all = np.concatenate(label_parts)
    positives = int(labels_all.sum())
    negatives = int(labels_all.size - positives)
    clean_scores = np.concatenate(score_parts["clean"])
    clean_auc = _absolute_auc(labels_all, clean_scores)
    results = {}
    previous_selected = -1
    previous_changed = -1
    nested_coverage_valid = True
    for cutoff in cutoffs:
        key = str(cutoff)
        masked_scores = np.concatenate(score_parts[key])
        masked_auc = _absolute_auc(labels_all, masked_scores)
        cutoff_coverage = coverage[key]
        selected_rows = int(cutoff_coverage["selected_rows"])
        changed_rows = int(cutoff_coverage["changed_rows"])
        nested_coverage_valid &= (
            selected_rows >= previous_selected
            and changed_rows >= previous_changed
        )
        previous_selected = selected_rows
        previous_changed = changed_rows
        fields = {}
        for field, field_report in cutoff_coverage["fields"].items():
            distinct_ids = field_report.pop("distinct_seen_rare_ids")
            fields[field] = {
                **field_report,
                "selected_row_fraction": (
                    field_report["selected_rows"] / labels_all.size
                ),
                "changed_row_fraction": (
                    field_report["changed_rows"] / labels_all.size
                ),
                "distinct_seen_rare_ids": len(distinct_ids),
            }
        results[key] = {
            "cutoff": cutoff,
            "masked_absolute_auc": masked_auc,
            "auc_drop_from_clean": clean_auc - masked_auc,
            "selected_rows": selected_rows,
            "selected_row_fraction": selected_rows / labels_all.size,
            "changed_rows": changed_rows,
            "changed_row_fraction": changed_rows / labels_all.size,
            "already_oov_rows": int(
                cutoff_coverage["already_oov_rows"]
            ),
            "fields": fields,
        }
    checks = {
        "nonempty_validation": labels_all.size > 0,
        "both_labels_present": positives > 0 and negatives > 0,
        "all_auc_values_finite": (
            math.isfinite(clean_auc)
            and all(
                math.isfinite(item["masked_absolute_auc"])
                for item in results.values()
            )
        ),
        "nested_coverage_valid": (
            nested_coverage_valid and nested_masks_valid
        ),
        "masking_only_changes_seen_private_rows": all(
            item["changed_rows"] <= item["selected_rows"]
            for item in results.values()
        ),
    }
    return {
        "analysis": "validation_cumulative_identity_tailmask",
        "split": "validation",
        "uses_test_dataset": False,
        "uses_test_labels": False,
        "frequency_source": "train_only",
        "frequency_cache_path": str(cache_path),
        "frequency_cache_sha256": (
            ZeroAnchorFrequencyDynamicsAudit._sha256_file(cache_path)
        ),
        "identity_fields": list(identity_fields),
        "intervention": "mask_to_shared_oov",
        "simultaneous_fields": True,
        "include_unseen": bool(include_unseen),
        "cutoffs": list(cutoffs),
        "clean": {
            "rows": int(labels_all.size),
            "positives": positives,
            "negatives": negatives,
            "absolute_auc": clean_auc,
        },
        "results": results,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "conclusion_boundary": {
            "directly_measures_validation_identity_damage": True,
            "already_unseen_ids_are_oov_before_intervention": True,
            "selected_rows_include_already_oov_when_requested": bool(
                include_unseen
            ),
            "changed_rows_isolate_seen_rare_replacements": True,
            "report_only_no_hyperparameter_selection": True,
        },
    }


def _torch_distribution(values: torch.Tensor, *, zero_index: int = 0) -> dict:
    return _distribution_from_counts(
        values.detach().cpu().numpy(),
        zero_index=zero_index,
    )


def _deterministic_cpu_bincount(
    indices: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    minlength: int,
) -> torch.Tensor:
    """Run bincount on CPU because CUDA has no deterministic implementation."""
    cpu_indices = indices.detach().to(device="cpu", dtype=torch.int64)
    cpu_weights = (
        None
        if weights is None
        else weights.detach().to(device="cpu", dtype=torch.float64)
    )
    return torch.bincount(
        cpu_indices,
        weights=cpu_weights,
        minlength=minlength,
    ).double()


def _nearest_codebook_indices_chunked(
    residuals: torch.Tensor,
    codebook: torch.Tensor,
    *,
    codebook_chunk_size: int,
) -> torch.Tensor:
    """Find exact nearest centers without materializing rows x full-K."""
    if residuals.ndim != 2 or codebook.ndim != 2:
        raise ValueError("chunked nearest-center inputs must be two-dimensional")
    if residuals.size(1) != codebook.size(1):
        raise ValueError("residual and codebook dimensions differ")
    if codebook_chunk_size <= 0:
        raise ValueError("codebook_chunk_size must be positive")
    residual_norms = residuals.square().sum(dim=-1, keepdim=True)
    best_distances = torch.full(
        (residuals.size(0),),
        float("inf"),
        dtype=torch.float32,
        device=residuals.device,
    )
    best_indices = torch.zeros(
        residuals.size(0),
        dtype=torch.long,
        device=residuals.device,
    )
    for start in range(0, codebook.size(0), codebook_chunk_size):
        current = codebook[start : start + codebook_chunk_size]
        distances = (
            residual_norms
            + current.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * residuals @ current.transpose(0, 1)
        )
        local_distances, local_indices = distances.min(dim=-1)
        improved = local_distances.lt(best_distances)
        best_distances = torch.where(
            improved,
            local_distances,
            best_distances,
        )
        best_indices = torch.where(
            improved,
            local_indices + start,
            best_indices,
        )
    return best_indices


@torch.no_grad()
def analyze_exact_frequency_geometry(
    model: nn.Module,
    *,
    frequency_cache_path: str | Path,
    identity_fields: list[str] | tuple[str, ...],
    field_names: list[str] | tuple[str, ...],
    chunk_rows: int = 65_536,
    codebook_chunk_size: int = 4_096,
    maximum_distance_elements: int = 16_777_216,
) -> dict:
    """Scan all train-vocabulary rows for exact mean geometry and utilization."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if codebook_chunk_size <= 0:
        raise ValueError("codebook_chunk_size must be positive")
    if maximum_distance_elements <= 0:
        raise ValueError("maximum_distance_elements must be positive")
    raw_model = model.module if hasattr(model, "module") else model
    quantizer = raw_model.encoder.zero_anchor_identity_quantizer
    quantized = quantizer is not None
    identity_fields = tuple(str(field) for field in identity_fields)
    field_names = tuple(str(field) for field in field_names)
    if quantized:
        weights = dict(get_zero_anchor_identity_embedding_weights(raw_model))
        if (
            quantizer.num_subspaces not in {1, 2}
            or quantizer.num_residual_levels != 1
        ):
            raise ValueError("exact core audit expects Product M1 or M2")
    else:
        weights = dict(
            get_identity_embedding_weights(
                raw_model,
                identity_fields=identity_fields,
                field_names=field_names,
            )
        )
    cache_path = Path(frequency_cache_path).resolve()
    results = {}
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("train_only") is not True:
            raise ValueError("exact geometry cache must be train-only")
        for field_position, field in enumerate(identity_fields):
            weight = weights[field]
            field_codebook_size = (
                quantizer.codebook_size_for_field(field_position)
                if quantized
                else None
            )
            counts = np.asarray(
                payload[f"{field}_counts"], dtype=np.int64
            )
            if counts.size != weight.size(0):
                raise ValueError(f"frequency cardinality differs for {field}")
            if np.any(counts[:FIRST_PRIVATE_ID] != 0):
                raise ValueError(f"special rows have train counts for {field}")
            private_counts = counts[FIRST_PRIVATE_ID:]
            if np.any(private_counts <= 0):
                raise ValueError(f"zero-count private row found for {field}")
            device = weight.device
            bucket_count = len(FREQUENCY_BUCKETS) - 1
            key_counts = torch.zeros(
                bucket_count, dtype=torch.float64, device="cpu"
            )
            occurrence_counts = torch.zeros_like(key_counts)
            norm_sums = torch.zeros_like(key_counts)
            norm_square_sums = torch.zeros_like(key_counts)
            if quantized:
                stable_assignment = bool(
                    quantizer.assignment_stability_mode != "free_nearest"
                )
                hash_retention_applicable = bool(
                    quantizer.num_subspaces == 1
                    and quantizer.private_row_initialization_mode
                    in {
                        "train_first_touch_deterministic_code",
                        "train_first_touch_frozen_deterministic_code",
                    }
                )
                frozen_hash_assignment = bool(
                    quantizer.private_row_initialization_mode
                    == "train_first_touch_frozen_deterministic_code"
                )
                hash_retained_key_counts = torch.zeros_like(key_counts)
                hash_retained_occurrence_counts = torch.zeros_like(key_counts)
                error_sums = torch.zeros_like(key_counts)
                zero_tuple_counts = torch.zeros_like(key_counts)
                center_counts = torch.zeros(
                    bucket_count,
                    quantizer.num_subspaces,
                    field_codebook_size,
                    dtype=torch.float64,
                    device="cpu",
                )
                codebooks = quantizer.codebook(field_position).float()
                tuple_count = (
                    field_codebook_size**quantizer.num_subspaces
                )
                joint_counts = torch.zeros(
                    bucket_count,
                    tuple_count,
                    dtype=torch.float64,
                    device="cpu",
                )
                use_legacy_product_path = bool(
                    quantizer.num_subspaces == 2
                    and field_codebook_size <= 1_024
                )
                if use_legacy_product_path:
                    codebook_norms = codebooks.square().sum(dim=-1)
                effective_chunk_rows = min(
                    chunk_rows,
                    max(
                        1,
                        maximum_distance_elements
                        // min(
                            field_codebook_size,
                            codebook_chunk_size,
                        ),
                    ),
                )

            for start in range(
                FIRST_PRIVATE_ID,
                weight.size(0),
                (
                    effective_chunk_rows
                    if quantized
                    else chunk_rows
                ),
            ):
                end = min(
                    start
                    + (
                        effective_chunk_rows
                        if quantized
                        else chunk_rows
                    ),
                    weight.size(0),
                )
                residuals = weight[start:end].detach().float()
                count_array = np.ascontiguousarray(
                    counts[start:end], dtype=np.int64
                )
                count_tensor = (
                    torch.frombuffer(
                        memoryview(count_array), dtype=torch.int64
                    )
                    .clone()
                    .to(device=device)
                )
                buckets = _positive_frequency_bins(count_tensor)
                occurrences = count_tensor.double()
                norms = torch.linalg.vector_norm(residuals, dim=-1).double()
                key_counts.add_(
                    _deterministic_cpu_bincount(
                        buckets, minlength=bucket_count
                    ).double()
                )
                occurrence_counts.add_(
                    _deterministic_cpu_bincount(
                        buckets,
                        weights=occurrences,
                        minlength=bucket_count,
                    ).double()
                )
                norm_sums.add_(
                    _deterministic_cpu_bincount(
                        buckets,
                        weights=norms,
                        minlength=bucket_count,
                    ).double()
                )
                norm_square_sums.add_(
                    _deterministic_cpu_bincount(
                        buckets,
                        weights=norms.square(),
                        minlength=bucket_count,
                    ).double()
                )
                if not quantized:
                    continue

                subspaces = residuals.reshape(
                    -1,
                    quantizer.num_subspaces,
                    quantizer.subspace_dim,
                )
                private_ids = torch.arange(
                    start,
                    end,
                    dtype=torch.long,
                    device=device,
                )
                if frozen_hash_assignment:
                    indices = quantizer.deterministic_private_code_indices(
                        private_ids,
                        field_position=field_position,
                    ).unsqueeze(1)
                elif stable_assignment:
                    stored = getattr(
                        quantizer,
                        quantizer._stable_assignment_buffer_names[
                            field_position
                        ],
                    ).index_select(0, private_ids).to(dtype=torch.long)
                    if stored.lt(0).any():
                        raise RuntimeError(
                            "stable assignment state is incomplete for a "
                            "train-vocabulary row"
                        )
                    indices = stored.unsqueeze(1)
                elif use_legacy_product_path:
                    distances = (
                        subspaces.square().sum(dim=-1, keepdim=True)
                        + codebook_norms.unsqueeze(0)
                        - 2.0
                        * torch.einsum(
                            "bmd,mkd->bmk", subspaces, codebooks
                        )
                    )
                    indices = distances.argmin(dim=-1)
                else:
                    indices = torch.stack(
                        [
                            _nearest_codebook_indices_chunked(
                                subspaces[:, subspace],
                                (
                                    codebooks[subspace]
                                    if quantizer.num_subspaces > 1
                                    else codebooks
                                ),
                                codebook_chunk_size=codebook_chunk_size,
                            )
                            for subspace in range(
                                quantizer.num_subspaces
                            )
                        ],
                        dim=1,
                    )
                force_zero, force_nonzero = (
                    quantizer.hard_routing_control_masks(
                        private_ids,
                        field_position,
                    )
                )
                if force_zero is not None:
                    indices[force_zero, 0] = 0
                    if force_nonzero.any():
                        indices[force_nonzero, 0] = (
                            _nearest_codebook_indices_chunked(
                                residuals[force_nonzero],
                                codebooks[1:],
                                codebook_chunk_size=codebook_chunk_size,
                            )
                            + 1
                        )
                if hash_retention_applicable:
                    initial_hash_indices = (
                        quantizer.deterministic_private_code_indices(
                            private_ids,
                            field_position=field_position,
                        )
                    )
                    retained = indices[:, 0].eq(initial_hash_indices).double()
                    hash_retained_key_counts.add_(
                        _deterministic_cpu_bincount(
                            buckets,
                            weights=retained,
                            minlength=bucket_count,
                        ).double()
                    )
                    hash_retained_occurrence_counts.add_(
                        _deterministic_cpu_bincount(
                            buckets,
                            weights=retained * occurrences,
                            minlength=bucket_count,
                        ).double()
                    )
                subspace_ids = torch.arange(
                    quantizer.num_subspaces, device=device
                ).unsqueeze(0)
                if quantizer.num_subspaces > 1:
                    hard = codebooks[subspace_ids, indices].reshape_as(
                        residuals
                    )
                else:
                    hard = codebooks.index_select(
                        0, indices[:, 0]
                    ).reshape_as(residuals)
                errors = torch.linalg.vector_norm(
                    residuals - hard, dim=-1
                ).double()
                error_sums.add_(
                    _deterministic_cpu_bincount(
                        buckets,
                        weights=errors,
                        minlength=bucket_count,
                    ).double()
                )
                zero_tuple_counts.add_(
                    _deterministic_cpu_bincount(
                        buckets,
                        weights=indices.eq(0).all(dim=1).double(),
                        minlength=bucket_count,
                    ).double()
                )
                for subspace in range(quantizer.num_subspaces):
                    flat = (
                        buckets * field_codebook_size
                        + indices[:, subspace]
                    )
                    center_counts[:, subspace].add_(
                        _deterministic_cpu_bincount(
                            flat,
                            minlength=(
                                bucket_count * field_codebook_size
                            ),
                        )
                        .reshape(bucket_count, field_codebook_size)
                        .double()
                    )
                tuple_ids = torch.zeros_like(indices[:, 0])
                for subspace in range(quantizer.num_subspaces):
                    tuple_ids = (
                        tuple_ids * field_codebook_size
                        + indices[:, subspace]
                    )
                flat_tuples = (
                    buckets
                    * (field_codebook_size**quantizer.num_subspaces)
                    + tuple_ids
                )
                joint_counts.add_(
                    _deterministic_cpu_bincount(
                        flat_tuples,
                        minlength=(
                            bucket_count
                            * field_codebook_size
                            ** quantizer.num_subspaces
                        ),
                    )
                    .reshape(bucket_count, -1)
                    .double()
                )

            buckets_report = {}
            for index, label in enumerate(FREQUENCY_BUCKETS[1:]):
                keys = float(key_counts[index].item())
                occurrences = float(occurrence_counts[index].item())
                if keys <= 0 or occurrences <= 0:
                    raise RuntimeError(f"empty exact bucket {field}/{label}")
                entry = {
                    "unique_private_ids": int(keys),
                    "private_id_fraction": (
                        keys / float(private_counts.size)
                    ),
                    "train_occurrences": int(occurrences),
                    "train_occurrence_fraction": (
                        occurrences / float(private_counts.sum())
                    ),
                    "raw_norm_mean": float(
                        norm_sums[index].item() / keys
                    ),
                    "raw_norm_rms": float(
                        math.sqrt(norm_square_sums[index].item() / keys)
                    ),
                }
                if quantized:
                    entry.update(
                        {
                            "quantization_error_mean": float(
                                error_sums[index].item() / keys
                            ),
                            "all_zero_tuple_fraction": float(
                                zero_tuple_counts[index].item() / keys
                            ),
                            "subspaces": [
                                _torch_distribution(
                                    center_counts[index, subspace]
                                )
                                for subspace in range(
                                    quantizer.num_subspaces
                                )
                            ],
                            "joint_tuple": _torch_distribution(
                                joint_counts[index]
                            ),
                        }
                    )
                    if hash_retention_applicable:
                        entry["initial_hash_assignment_retention"] = {
                            "retained_unique_private_ids": int(
                                hash_retained_key_counts[index].item()
                            ),
                            "unique_private_id_fraction": float(
                                hash_retained_key_counts[index].item() / keys
                            ),
                            "retained_train_occurrences": int(
                                hash_retained_occurrence_counts[index].item()
                            ),
                            "train_occurrence_fraction": float(
                                hash_retained_occurrence_counts[index].item()
                                / occurrences
                            ),
                        }
                buckets_report[label] = entry

            oov_residual = weight[1:2].detach().float()
            oov = {
                "shared_rows": 1,
                "raw_norm": float(
                    torch.linalg.vector_norm(oov_residual).item()
                ),
            }
            if quantized:
                oov_indices = quantizer.quantize_residuals(
                    oov_residual, field_position
                )["indices"].reshape(1, -1)
                oov["indices"] = [
                    int(value) for value in oov_indices[0].tolist()
                ]
                oov["all_zero_tuple"] = bool(
                    oov_indices.eq(0).all().item()
                )
            field_checks = {
                "private_ids_close": (
                    int(key_counts.sum().item())
                    == int(private_counts.size)
                ),
                "train_occurrences_close": (
                    int(occurrence_counts.sum().item())
                    == int(private_counts.sum())
                ),
                "quantized_oov_raw_row_is_zero": (
                    not quantized or oov["raw_norm"] == 0.0
                ),
                "oov_uses_all_zero_tuple": (
                    not quantized or oov["all_zero_tuple"]
                ),
            }
            if quantized and hash_retention_applicable:
                field_checks["initial_hash_retention_counts_close"] = bool(
                    hash_retained_key_counts.le(key_counts).all().item()
                    and hash_retained_occurrence_counts.le(
                        occurrence_counts
                    ).all().item()
                )
                field_checks["frozen_hash_assignment_is_exact"] = bool(
                    not frozen_hash_assignment
                    or (
                        torch.equal(hash_retained_key_counts, key_counts)
                        and torch.equal(
                            hash_retained_occurrence_counts,
                            occurrence_counts,
                        )
                    )
                )
            field_result = {
                "private_train_ids": int(private_counts.size),
                "train_occurrences": int(private_counts.sum()),
                "shared_oov": oov,
                "buckets": buckets_report,
                "checks": field_checks,
                "all_checks_pass": all(field_checks.values()),
            }
            if quantized:
                field_result["hard_assignment_route"] = (
                    "deterministic_hash"
                    if frozen_hash_assignment
                    else (
                        quantizer.assignment_stability_mode
                        if stable_assignment
                        else "nearest_center"
                    )
                )
                if hash_retention_applicable:
                    field_result["initial_hash_assignment_retention"] = {
                        "retained_unique_private_ids": int(
                            hash_retained_key_counts.sum().item()
                        ),
                        "unique_private_id_fraction": float(
                            hash_retained_key_counts.sum().item()
                            / float(private_counts.size)
                        ),
                        "retained_train_occurrences": int(
                            hash_retained_occurrence_counts.sum().item()
                        ),
                        "train_occurrence_fraction": float(
                            hash_retained_occurrence_counts.sum().item()
                            / float(private_counts.sum())
                        ),
                    }
            results[field] = field_result
    report_checks = {
        "all_fields_close": all(
            result["all_checks_pass"] for result in results.values()
        ),
        "all_numeric_values_finite": all(
            _all_numeric_values_finite(result)
            for result in results.values()
        ),
    }
    return {
        "quantized": quantized,
        "aggregation_device": "cpu",
        "deterministic_bincount": True,
        "hard_assignment_backend": (
            "deterministic_hash_for_frozen_hash_otherwise_"
            "legacy_product_gemm_or_exact_chunked_gemm"
        ),
        "nearest_center_backend": (
            "legacy_product_gemm_or_exact_chunked_gemm"
        ),
        "codebook_chunk_size": int(codebook_chunk_size),
        "maximum_distance_elements": int(maximum_distance_elements),
        "frequency_source": "train_only",
        "frequency_cache_path": str(cache_path),
        "frequency_cache_sha256": (
            ZeroAnchorFrequencyDynamicsAudit._sha256_file(cache_path)
        ),
        "frequency_buckets": list(FREQUENCY_BUCKETS),
        "identity_fields": results,
        "checks": report_checks,
        "all_checks_pass": all(report_checks.values()),
    }


class ZeroAnchorFrequencyDynamicsAudit:
    """Capture deterministic late-training identity geometry snapshots."""

    def __init__(
        self,
        model: nn.Module,
        *,
        frequency_cache_path: str | Path,
        identity_fields: list[str] | tuple[str, ...],
        field_names: list[str] | tuple[str, ...],
        snapshot_progresses: tuple[float, ...] = (0.80, 0.90, 1.00),
        sample_ids_per_bucket: int = 4096,
        sample_seed: int = 20260729,
        allow_partial_snapshots: bool = False,
        minimum_partial_snapshots: int = 2,
    ) -> None:
        if sample_ids_per_bucket <= 0:
            raise ValueError("sample_ids_per_bucket must be positive")
        progresses = tuple(float(value) for value in snapshot_progresses)
        if (
            not progresses
            or any(not 0.0 < value <= 1.0 for value in progresses)
            or tuple(sorted(set(progresses))) != progresses
        ):
            raise ValueError(
                "snapshot_progresses must be unique, increasing, and in (0,1]"
            )
        self.identity_fields = tuple(str(field) for field in identity_fields)
        self.field_names = tuple(str(field) for field in field_names)
        self.snapshot_progresses = progresses
        self.sample_ids_per_bucket = int(sample_ids_per_bucket)
        self.sample_seed = int(sample_seed)
        self.allow_partial_snapshots = bool(allow_partial_snapshots)
        self.minimum_partial_snapshots = int(minimum_partial_snapshots)
        if self.minimum_partial_snapshots < 1:
            raise ValueError("minimum_partial_snapshots must be positive")
        if (
            self.allow_partial_snapshots
            and self.minimum_partial_snapshots > len(progresses)
        ):
            raise ValueError(
                "minimum_partial_snapshots cannot exceed snapshot count"
            )
        self._snapshots: list[dict] = []
        self._next_snapshot = 0

        raw_model = model.module if hasattr(model, "module") else model
        quantizer = raw_model.encoder.zero_anchor_identity_quantizer
        self.quantized = quantizer is not None
        if self.quantized:
            weights = get_zero_anchor_identity_embedding_weights(raw_model)
            if tuple(field for field, _ in weights) != self.identity_fields:
                raise ValueError(
                    "quantizer identity field order differs from audit order"
                )
            if (
                quantizer.num_subspaces not in {1, 2}
                or quantizer.num_residual_levels != 1
            ):
                raise ValueError(
                    "core dynamics audit expects Product M1 or M2 "
                    "with one level"
                )
        else:
            weights = get_identity_embedding_weights(
                raw_model,
                identity_fields=self.identity_fields,
                field_names=self.field_names,
            )
        self._row_counts: dict[str, int] = {
            field: int(weight.size(0)) for field, weight in weights
        }

        cache_path = Path(frequency_cache_path).resolve()
        with np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata.get("train_only") is not True:
                raise ValueError("frequency dynamics cache must be train-only")
            self._samples = {
                field: self._select_samples(
                    np.asarray(payload[f"{field}_counts"], dtype=np.int64),
                    field=field,
                )
                for field in self.identity_fields
            }
        for field in self.identity_fields:
            if self._samples[field]["cardinality"] != self._row_counts[field]:
                raise ValueError(
                    f"frequency cache cardinality differs for {field}"
                )
        self.frequency_cache_path = str(cache_path)
        self.frequency_cache_sha256 = self._sha256_file(cache_path)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _select_samples(self, counts: np.ndarray, *, field: str) -> dict:
        if counts.ndim != 1 or counts.size <= FIRST_PRIVATE_ID:
            raise ValueError(f"invalid frequency counts for {field}")
        if np.any(counts[:FIRST_PRIVATE_ID] != 0):
            raise ValueError(f"special rows must have zero counts for {field}")
        if np.any(counts[FIRST_PRIVATE_ID:] <= 0):
            raise ValueError(
                f"private train-vocabulary rows must be observed for {field}"
            )
        buckets = {
            "unseen": {
                "ids": np.asarray([1], dtype=np.int64),
                "population": None,
                "selection": "shared_oov_row_only",
            }
        }
        private_counts = counts[FIRST_PRIVATE_ID:]
        private_total = int(private_counts.size)
        for label in FREQUENCY_BUCKETS[1:]:
            candidate_offsets = np.flatnonzero(
                _bucket_mask(private_counts, label)
            )
            population = int(candidate_offsets.size)
            if population == 0:
                raise ValueError(f"empty train frequency bucket {field}/{label}")
            ids = candidate_offsets + FIRST_PRIVATE_ID
            if population > self.sample_ids_per_bucket:
                rng = np.random.default_rng(
                    _stable_seed(self.sample_seed, field, label)
                )
                ids = np.sort(
                    rng.choice(
                        ids,
                        size=self.sample_ids_per_bucket,
                        replace=False,
                    )
                )
            ids = np.asarray(ids, dtype=np.int64)
            buckets[label] = {
                "ids": ids,
                "population": population,
                "population_fraction": population / private_total,
                "selection": "deterministic_uniform_without_replacement",
            }
        return {
            "cardinality": int(counts.size),
            "private_train_ids": private_total,
            "buckets": buckets,
        }

    @staticmethod
    def _model_weights(raw_model: nn.Module, quantized: bool, fields, names):
        if quantized:
            return dict(get_zero_anchor_identity_embedding_weights(raw_model))
        return dict(
            get_identity_embedding_weights(
                raw_model,
                identity_fields=fields,
                field_names=names,
            )
        )

    @torch.no_grad()
    def __call__(
        self,
        raw_model: nn.Module,
        *,
        completed_step: int,
        total_steps: int,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        progress = min(max(completed_step / total_steps, 0.0), 1.0)
        while (
            self._next_snapshot < len(self.snapshot_progresses)
            and progress + 1e-15
            >= self.snapshot_progresses[self._next_snapshot]
        ):
            target = self.snapshot_progresses[self._next_snapshot]
            self._snapshots.append(
                self._capture(
                    raw_model,
                    target_progress=target,
                    actual_progress=progress,
                    completed_step=completed_step,
                    total_steps=total_steps,
                )
            )
            self._next_snapshot += 1

    @torch.no_grad()
    def _capture(
        self,
        raw_model: nn.Module,
        *,
        target_progress: float,
        actual_progress: float,
        completed_step: int,
        total_steps: int,
    ) -> dict:
        weights = self._model_weights(
            raw_model,
            self.quantized,
            self.identity_fields,
            self.field_names,
        )
        quantizer = raw_model.encoder.zero_anchor_identity_quantizer
        fields = {}
        for field_position, field in enumerate(self.identity_fields):
            weight = weights[field]
            field_buckets = {}
            for label in FREQUENCY_BUCKETS:
                ids_np = self._samples[field]["buckets"][label]["ids"]
                ids = torch.from_numpy(ids_np).to(
                    device=weight.device,
                    dtype=torch.long,
                )
                residuals = weight.index_select(0, ids)
                entry = {
                    "ids": ids_np,
                    "norms": torch.linalg.vector_norm(
                        residuals.float(), dim=-1
                    )
                    .cpu()
                    .numpy(),
                }
                if self.quantized:
                    hard_indices = None
                    if (
                        quantizer.private_row_initialization_mode
                        == "train_first_touch_frozen_deterministic_code"
                    ):
                        hard_indices = (
                            quantizer.deterministic_private_code_indices(
                                ids,
                                field_position=field_position,
                            )
                        )
                    elif quantizer.assignment_stability_mode != "free_nearest":
                        stored = getattr(
                            quantizer,
                            quantizer._stable_assignment_buffer_names[
                                field_position
                            ],
                        ).index_select(0, ids).to(dtype=torch.long)
                        hard_indices = stored.clamp_min(0)
                    force_zero, force_nonzero = (
                        quantizer.hard_routing_control_masks(
                            ids,
                            field_position,
                        )
                    )
                    quantized = quantizer.quantize_residuals(
                        residuals,
                        field_position,
                        hard_indices=hard_indices,
                        force_zero_mask=force_zero,
                        force_nonzero_mask=force_nonzero,
                    )
                    indices = (
                        quantized["indices"]
                        .reshape(-1, quantizer.num_subspaces)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    hard = quantized["hard_value"].detach().float()
                    errors = torch.linalg.vector_norm(
                        residuals.detach().float() - hard,
                        dim=-1,
                    )
                    entry.update(
                        {
                            "indices": indices,
                            "errors": errors.cpu().numpy(),
                            "soft_proxy_agreement": (
                                self._soft_proxy_agreement(
                                    quantized,
                                    quantizer,
                                    field_position,
                                )
                            ),
                        }
                    )
                field_buckets[label] = entry
            fields[field] = field_buckets
        return {
            "target_progress": target_progress,
            "actual_progress": actual_progress,
            "completed_step": int(completed_step),
            "total_steps": int(total_steps),
            "temperature": (
                float(quantizer.current_temperature)
                if quantizer is not None
                else None
            ),
            "fields": fields,
        }

    @staticmethod
    @torch.no_grad()
    def _soft_proxy_agreement(quantized, quantizer, field_position) -> np.ndarray:
        indices = quantized["indices"].reshape(
            -1,
            quantizer.num_subspaces,
        )
        soft = quantized["soft_value"].float().reshape(
            -1,
            quantizer.num_subspaces,
            quantizer.subspace_dim,
        )
        codebooks = quantizer.codebook(field_position).float()
        agreements = []
        for subspace in range(quantizer.num_subspaces):
            codebook = (
                codebooks[subspace]
                if quantizer.num_subspaces > 1
                else codebooks
            )
            proxy = _nearest_codebook_indices_chunked(
                soft[:, subspace],
                codebook,
                codebook_chunk_size=min(4_096, codebook.size(0)),
            )
            agreements.append(proxy.eq(indices[:, subspace]))
        return torch.stack(agreements, dim=1).cpu().numpy()

    def _selection_report(self) -> dict:
        report = {}
        for field in self.identity_fields:
            source = self._samples[field]
            buckets = {}
            for label in FREQUENCY_BUCKETS:
                item = source["buckets"][label]
                ids = item["ids"]
                buckets[label] = {
                    "sampled_rows": int(ids.size),
                    "sample_ids_sha256": _sha256_int64(ids),
                    "population_private_ids": item.get("population"),
                    "population_fraction": item.get("population_fraction"),
                    "selection": item["selection"],
                }
            report[field] = {
                "cardinality": source["cardinality"],
                "private_train_ids": source["private_train_ids"],
                "buckets": buckets,
            }
        return report

    @staticmethod
    def _snapshot_report(snapshot: dict, *, quantized: bool) -> dict:
        fields = {}
        for field, buckets in snapshot["fields"].items():
            field_report = {}
            for label, item in buckets.items():
                entry = {"raw_norm": _summary(item["norms"])}
                if quantized:
                    indices = np.asarray(item["indices"], dtype=np.int64)
                    entry.update(
                        {
                            "quantization_error": _summary(item["errors"]),
                            "all_zero_tuple_fraction": float(
                                np.all(indices == 0, axis=1).mean()
                            ),
                            "subspaces": [
                                {
                                    "assignment": _distribution(
                                        indices[:, subspace]
                                    ),
                                    "soft_proxy_hard_agreement": float(
                                        np.asarray(
                                            item["soft_proxy_agreement"]
                                        )[:, subspace].mean()
                                    ),
                                }
                                for subspace in range(indices.shape[1])
                            ],
                        }
                    )
                field_report[label] = entry
            fields[field] = field_report
        return {
            key: snapshot[key]
            for key in (
                "target_progress",
                "actual_progress",
                "completed_step",
                "total_steps",
                "temperature",
            )
        } | {"fields": fields}

    def _churn_report(self) -> dict:
        if not self.quantized or len(self._snapshots) < 2:
            return {"applicable": self.quantized, "intervals": [], "fields": {}}
        fields = {}
        for field in self.identity_fields:
            buckets = {}
            for label in FREQUENCY_BUCKETS:
                per_subspace = []
                joint_changes = 0
                joint_comparisons = 0
                subspace_count = np.asarray(
                    self._snapshots[0]["fields"][field][label]["indices"]
                ).shape[1]
                for subspace in range(subspace_count):
                    changes = 0
                    comparisons = 0
                    for previous, current in zip(
                        self._snapshots[:-1], self._snapshots[1:]
                    ):
                        before = np.asarray(
                            previous["fields"][field][label]["indices"]
                        )[:, subspace]
                        after = np.asarray(
                            current["fields"][field][label]["indices"]
                        )[:, subspace]
                        comparisons += int(before.size)
                        changes += int(np.count_nonzero(before != after))
                    per_subspace.append(
                        {
                            "comparisons": comparisons,
                            "changes": changes,
                            "fraction": (
                                changes / comparisons if comparisons else None
                            ),
                        }
                    )
                for previous, current in zip(
                    self._snapshots[:-1], self._snapshots[1:]
                ):
                    before = np.asarray(
                        previous["fields"][field][label]["indices"]
                    )
                    after = np.asarray(
                        current["fields"][field][label]["indices"]
                    )
                    joint_comparisons += int(before.shape[0])
                    joint_changes += int(
                        np.count_nonzero(np.any(before != after, axis=1))
                    )
                buckets[label] = {
                    "subspaces": per_subspace,
                    "joint_tuple": {
                        "comparisons": joint_comparisons,
                        "changes": joint_changes,
                        "fraction": (
                            joint_changes / joint_comparisons
                            if joint_comparisons
                            else None
                        ),
                    },
                }
            fields[field] = buckets
        return {
            "applicable": True,
            "snapshot_intervals": [
                {
                    "from": previous["target_progress"],
                    "to": current["target_progress"],
                }
                for previous, current in zip(
                    self._snapshots[:-1], self._snapshots[1:]
                )
            ],
            "weighting": "deterministic_uniform_sample_of_train_vocabulary_ids",
            "fields": fields,
        }

    def report(self) -> dict:
        snapshots = [
            self._snapshot_report(snapshot, quantized=self.quantized)
            for snapshot in self._snapshots
        ]
        captured_targets = tuple(
            item["target_progress"] for item in self._snapshots
        )
        all_snapshots_captured = (
            len(self._snapshots) == len(self.snapshot_progresses)
        )
        snapshot_prefix_matches = (
            captured_targets
            == self.snapshot_progresses[: len(captured_targets)]
        )
        capture_policy_satisfied = (
            all_snapshots_captured
            or (
                self.allow_partial_snapshots
                and len(self._snapshots) >= self.minimum_partial_snapshots
                and snapshot_prefix_matches
            )
        )
        checks = {
            "all_snapshots_captured": all_snapshots_captured,
            "snapshot_prefix_matches": snapshot_prefix_matches,
            "capture_policy_satisfied": capture_policy_satisfied,
            "all_summary_values_finite": all(
                _all_numeric_values_finite(snapshot)
                for snapshot in snapshots
            ),
        }
        return {
            "enabled": True,
            "quantized": self.quantized,
            "frequency_cache_path": self.frequency_cache_path,
            "frequency_cache_sha256": self.frequency_cache_sha256,
            "frequency_source": "train_only",
            "first_private_id": FIRST_PRIVATE_ID,
            "frequency_buckets": list(FREQUENCY_BUCKETS),
            "snapshot_progresses": list(self.snapshot_progresses),
            "sample_ids_per_bucket": self.sample_ids_per_bucket,
            "sample_seed": self.sample_seed,
            "allow_partial_snapshots": self.allow_partial_snapshots,
            "minimum_partial_snapshots": self.minimum_partial_snapshots,
            "selection": self._selection_report(),
            "snapshots": snapshots,
            "late_training_assignment_churn": self._churn_report(),
            "checks": checks,
            "all_checks_pass": (
                checks["capture_policy_satisfied"]
                and checks["snapshot_prefix_matches"]
                and checks["all_summary_values_finite"]
            ),
            "conclusion_boundary": {
                "norm_quantiles_and_churn_are_deterministic_samples": True,
                "mean_full_table_geometry_requires_final_exact_audit": True,
                "unseen_is_one_shared_oov_row_offline": True,
            },
        }
