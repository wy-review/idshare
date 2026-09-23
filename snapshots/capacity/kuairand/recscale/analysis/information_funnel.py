"""Direct sparse-embedding to prediction transmission diagnostics for S2DRec."""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


METRIC_NAMES = (
    "embedding_rms",
    "token_delta_rms",
    "token_context_rms",
    "token_local_relative",
    "backbone_input_local_relative",
    "ln_relative_gain",
    "backbone_output_relative",
    "representation_relative",
    "logit_abs_delta",
    "prob_abs_delta",
)

TAIL_SWEEP_METRIC_NAMES = (
    "masked_fields_per_sample",
    "field_embedding_relative",
    "tokenizer_relative",
    "backbone_input_relative",
    "backbone_output_relative",
    "representation_relative",
    "logit_abs_delta",
    "prob_abs_delta",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_ctr_metrics(
    labels: list[float] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> dict:
    labels_arr = np.asarray(labels, dtype=np.float64)
    scores_arr = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    positives = labels_arr > 0.5
    num_pos = int(positives.sum())
    num_neg = int(len(labels_arr) - num_pos)

    if num_pos == 0 or num_neg == 0:
        auc = 0.0
    else:
        order = np.argsort(scores_arr, kind="mergesort")
        sorted_scores = scores_arr[order]
        ranks = np.empty(len(scores_arr), dtype=np.float64)
        start = 0
        while start < len(sorted_scores):
            end = start + 1
            while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            average_rank = 0.5 * ((start + 1) + end)
            ranks[order[start:end]] = average_rank
            start = end
        positive_rank_sum = float(ranks[positives].sum())
        auc = (
            positive_rank_sum - num_pos * (num_pos + 1) / 2.0
        ) / (num_pos * num_neg)

    logloss = -float(
        np.mean(
            labels_arr * np.log(scores_arr)
            + (1.0 - labels_arr) * np.log(1.0 - scores_arr)
        )
    )
    return {
        "auc": float(auc),
        "logloss": logloss,
        "num_samples": int(len(labels_arr)),
        "num_pos": num_pos,
        "pos_rate": float(num_pos / len(labels_arr)) if len(labels_arr) else 0.0,
    }


@dataclass
class _RunningMetrics:
    count: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(len(METRIC_NAMES), dtype=np.float64))
    sum_squares: np.ndarray = field(
        default_factory=lambda: np.zeros(len(METRIC_NAMES), dtype=np.float64)
    )

    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        selected = values[mask]
        if selected.numel() == 0:
            return
        self.count += int(selected.shape[0])
        self.sums += np.asarray(
            selected.sum(dim=0).double().cpu().tolist(), dtype=np.float64
        )
        self.sum_squares += np.asarray(
            selected.square().sum(dim=0).double().cpu().tolist(), dtype=np.float64
        )

    def result(self) -> dict:
        if self.count == 0:
            return {"count": 0, "metrics": {}}
        means = self.sums / self.count
        variances = np.maximum(self.sum_squares / self.count - means * means, 0.0)
        return {
            "count": self.count,
            "metrics": {
                name: {
                    "mean": float(means[idx]),
                    "std": float(math.sqrt(variances[idx])),
                }
                for idx, name in enumerate(METRIC_NAMES)
            },
        }


@dataclass
class _TailSweepRunningMetrics:
    count: int = 0
    sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(TAIL_SWEEP_METRIC_NAMES), dtype=np.float64)
    )
    sum_squares: np.ndarray = field(
        default_factory=lambda: np.zeros(len(TAIL_SWEEP_METRIC_NAMES), dtype=np.float64)
    )

    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        selected = values[mask]
        if selected.numel() == 0:
            return
        self.count += int(selected.shape[0])
        self.sums += np.asarray(
            selected.sum(dim=0).double().cpu().tolist(), dtype=np.float64
        )
        self.sum_squares += np.asarray(
            selected.square().sum(dim=0).double().cpu().tolist(), dtype=np.float64
        )

    def result(self) -> dict:
        if self.count == 0:
            return {"count": 0, "metrics": {}}
        means = self.sums / self.count
        variances = np.maximum(self.sum_squares / self.count - means * means, 0.0)
        return {
            "count": self.count,
            "metrics": {
                name: {
                    "mean": float(means[idx]),
                    "std": float(math.sqrt(variances[idx])),
                }
                for idx, name in enumerate(TAIL_SWEEP_METRIC_NAMES)
            },
        }


@dataclass
class _ConditionalPredictions:
    labels: list[np.ndarray] = field(default_factory=list)
    clean_scores: list[np.ndarray] = field(default_factory=list)
    masked_scores: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        *,
        labels: torch.Tensor,
        clean_logits: torch.Tensor,
        masked_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not bool(mask.any()):
            return
        self.labels.append(
            np.asarray(labels[mask].detach().float().cpu().tolist(), dtype=np.float32)
        )
        self.clean_scores.append(
            np.asarray(
                torch.sigmoid(clean_logits[mask]).detach().float().cpu().tolist(),
                dtype=np.float32,
            )
        )
        self.masked_scores.append(
            np.asarray(
                torch.sigmoid(masked_logits[mask]).detach().float().cpu().tolist(),
                dtype=np.float32,
            )
        )

    def result(
        self,
        *,
        min_samples: int = 0,
        min_positives: int = 1,
        min_negatives: int = 1,
    ) -> dict:
        if not self.labels:
            return {
                "num_samples": 0,
                "clean": {},
                "masked": {},
                "auc_drop": None,
                "logloss_increase": None,
                "auc_valid": False,
                "auc_invalid_reason": "no touched samples",
            }
        labels = np.concatenate(self.labels)
        clean_scores = np.concatenate(self.clean_scores)
        masked_scores = np.concatenate(self.masked_scores)
        clean = _compute_ctr_metrics(labels, clean_scores)
        masked = _compute_ctr_metrics(labels, masked_scores)
        num_neg = clean["num_samples"] - clean["num_pos"]
        invalid_reasons = []
        if clean["num_samples"] < min_samples:
            invalid_reasons.append(
                f"num_samples={clean['num_samples']} < {min_samples}"
            )
        if clean["num_pos"] < min_positives:
            invalid_reasons.append(f"num_pos={clean['num_pos']} < {min_positives}")
        if num_neg < min_negatives:
            invalid_reasons.append(f"num_neg={num_neg} < {min_negatives}")
        return {
            "num_samples": int(labels.size),
            "clean": clean,
            "masked": masked,
            "auc_drop": float(clean["auc"] - masked["auc"]),
            "logloss_increase": float(masked["logloss"] - clean["logloss"]),
            "auc_valid": not invalid_reasons,
            "auc_invalid_reason": "; ".join(invalid_reasons) or None,
        }


def _validate_frequency_boundaries(
    tail_max_count: int,
    head_min_count: int,
) -> None:
    if tail_max_count < 1:
        raise ValueError("tail_max_count must be >= 1")
    if head_min_count <= tail_max_count + 1:
        raise ValueError("head_min_count must leave a non-empty mid-frequency range")


def _frequency_bucket_masks(
    counts: np.ndarray,
    ids: torch.Tensor,
    *,
    tail_max_count: int = 5,
    head_min_count: int = 100,
) -> dict[str, torch.Tensor]:
    _validate_frequency_boundaries(tail_max_count, head_min_count)
    ids_np = np.asarray(ids.detach().cpu().tolist(), dtype=np.int64)
    safe_ids = np.clip(ids_np, 0, len(counts) - 1)
    observed = counts[safe_ids]
    nonzero = ids_np != 0
    return {
        "unseen": torch.tensor((nonzero & (observed == 0)).tolist(), dtype=torch.bool),
        "tail": torch.tensor(
            (nonzero & (observed >= 1) & (observed <= tail_max_count)).tolist(),
            dtype=torch.bool,
        ),
        "mid": torch.tensor(
            (
                nonzero
                & (observed >= tail_max_count + 1)
                & (observed <= head_min_count - 1)
            ).tolist(),
            dtype=torch.bool,
        ),
        "head": torch.tensor(
            (nonzero & (observed >= head_min_count)).tolist(), dtype=torch.bool
        ),
    }


def _id_population(
    counts: np.ndarray,
    *,
    tail_max_count: int = 5,
    head_min_count: int = 100,
) -> dict[str, dict[str, float | int]]:
    """Summarize unique usable ID slots by training-frequency bucket."""
    _validate_frequency_boundaries(tail_max_count, head_min_count)
    usable = counts[1:]
    total = int(len(usable))
    bucket_counts = {
        "unseen": int((usable == 0).sum()),
        "tail": int(((usable >= 1) & (usable <= tail_max_count)).sum()),
        "mid": int(
            ((usable >= tail_max_count + 1) & (usable <= head_min_count - 1)).sum()
        ),
        "head": int((usable >= head_min_count).sum()),
    }
    return {
        bucket: {
            "unique_ids": count,
            "id_share": float(count / total) if total else 0.0,
        }
        for bucket, count in bucket_counts.items()
    }


def resolve_field_specs(config: dict, analysis_cfg: dict) -> list[dict]:
    """Resolve an explicit field panel or the model's complete field list."""
    raw_specs = analysis_cfg.get("fields")
    all_fields = analysis_cfg.get("field_scope") == "all" or raw_specs == "all"
    field_names = list(config.get("dataset", {}).get("sparse_cols") or [])

    if all_fields:
        if not field_names:
            raise ValueError(
                "field_scope=all requires dataset.sparse_cols to be populated"
            )
        specs = []
        for index, name in enumerate(field_names):
            field_type = (
                "numerical_bucket"
                if str(name).startswith("I") or str(name).endswith("_bucket")
                else "categorical_id"
            )
            specs.append(
                {
                    "index": index,
                    "name": str(name),
                    "group": field_type,
                    "field_type": field_type,
                }
            )
    else:
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("analysis.information_funnel.fields must be a non-empty list or 'all'")
        specs = []
        for raw_spec in raw_specs:
            spec = dict(raw_spec)
            index = int(spec["index"])
            name = str(spec["name"])
            field_type = spec.get("field_type")
            if field_type is None:
                field_type = (
                    "numerical_bucket"
                    if name.startswith("I") or name.endswith("_bucket") or index >= 26
                    else "categorical_id"
                )
            spec.update(
                {
                    "index": index,
                    "name": name,
                    "group": str(spec.get("group", field_type)),
                    "field_type": str(field_type),
                }
            )
            specs.append(spec)

    indices = [int(spec["index"]) for spec in specs]
    names = [str(spec["name"]) for spec in specs]
    if len(indices) != len(set(indices)) or len(names) != len(set(names)):
        raise ValueError("information-funnel field indices and names must be unique")
    if field_names and any(index < 0 or index >= len(field_names) for index in indices):
        raise ValueError(f"field index outside model field range [0, {len(field_names)})")
    return specs


def _field_token_membership(model: torch.nn.Module) -> dict[int, dict]:
    """Describe deterministic field-to-token membership for the current tokenizer."""
    raw_model = model.module if hasattr(model, "module") else model
    tokenizer = raw_model.tokenizer
    field_names = list(raw_model.config.get("dataset", {}).get("sparse_cols") or [])
    groups = None

    if hasattr(tokenizer, "group_indices") and hasattr(tokenizer, "group_mask"):
        indices = tokenizer.group_indices.detach().cpu()
        mask = tokenizer.group_mask.detach().cpu()
        groups = [indices[token_idx][mask[token_idx]].tolist() for token_idx in range(indices.shape[0])]
    elif (
        getattr(raw_model, "tokenizer_type", None) == "uniform_proj"
        and getattr(tokenizer, "proj_mode", None) == "split"
    ):
        if tokenizer.field_perm is None:
            permutation = list(range(tokenizer.num_fields))
        else:
            permutation = tokenizer.field_perm.detach().cpu().tolist()
        group_size = tokenizer.num_fields // tokenizer.num_tokens
        groups = [
            permutation[start : start + group_size]
            for start in range(0, tokenizer.num_fields, group_size)
        ]
    elif (
        getattr(raw_model, "tokenizer_type", None)
        in ("per_field_proj", "bounded_residual", "contextual_fallback")
        and tokenizer.num_tokens == tokenizer.num_fields
    ):
        groups = [[index] for index in range(tokenizer.num_fields)]

    if groups is None:
        return {}

    membership = {}
    for token_index, member_indices in enumerate(groups):
        member_names = [
            field_names[index] if index < len(field_names) else f"field_{index}"
            for index in member_indices
        ]
        for field_index in member_indices:
            membership[int(field_index)] = {
                "token_index": token_index,
                "token_group_size": len(member_indices),
                "token_group_indices": [int(index) for index in member_indices],
                "token_group_members": member_names,
            }
    return membership


def _local_stage_metrics(clean: torch.Tensor, masked: torch.Tensor) -> tuple[torch.Tensor, ...]:
    eps = torch.finfo(clean.dtype).eps
    delta = clean - masked
    changed = delta.square().sum(dim=-1) > eps
    changed_f = changed.to(dtype=clean.dtype)
    delta_sq = (delta.square().sum(dim=-1) * changed_f).sum(dim=-1)
    clean_sq = (clean.square().sum(dim=-1) * changed_f).sum(dim=-1)
    local_dims = changed_f.sum(dim=-1).clamp_min(1.0) * clean.shape[-1]
    delta_rms = torch.sqrt(delta_sq / local_dims)
    context_rms = torch.sqrt(clean_sq / local_dims)
    relative = torch.sqrt(delta_sq / clean_sq.clamp_min(eps))
    return delta_rms, context_rms, relative


def _relative_delta(clean: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(clean.dtype).eps
    reduce_dims = tuple(range(1, clean.ndim))
    delta_sq = (clean - masked).square().sum(dim=reduce_dims)
    clean_sq = clean.square().sum(dim=reduce_dims)
    return torch.sqrt(delta_sq / clean_sq.clamp_min(eps))


def _paired_metrics(
    clean: dict[str, torch.Tensor | None],
    masked: dict[str, torch.Tensor | None],
    field_idx: int,
) -> torch.Tensor:
    field_embedding = clean["field_embedding"][:, field_idx, :]
    embedding_rms = field_embedding.square().mean(dim=-1).sqrt()

    token_delta_rms, token_context_rms, token_relative = _local_stage_metrics(
        clean["tokenizer_output"], masked["tokenizer_output"]
    )
    _, _, backbone_input_relative = _local_stage_metrics(
        clean["backbone_input"], masked["backbone_input"]
    )
    eps = torch.finfo(token_relative.dtype).eps
    ln_relative_gain = backbone_input_relative / token_relative.clamp_min(eps)
    backbone_output_relative = _relative_delta(
        clean["backbone_output"], masked["backbone_output"]
    )
    representation_relative = _relative_delta(
        clean["representation"], masked["representation"]
    )
    logit_abs_delta = (clean["logits"] - masked["logits"]).abs()
    prob_abs_delta = (
        torch.sigmoid(clean["logits"]) - torch.sigmoid(masked["logits"])
    ).abs()
    return torch.stack(
        (
            embedding_rms,
            token_delta_rms,
            token_context_rms,
            token_relative,
            backbone_input_relative,
            ln_relative_gain,
            backbone_output_relative,
            representation_relative,
            logit_abs_delta,
            prob_abs_delta,
        ),
        dim=1,
    )


def _tail_mask_for_cutoff(
    counts: np.ndarray,
    ids: torch.Tensor,
    cutoff: int,
    include_unseen: bool = False,
) -> torch.Tensor:
    eligible, _, _ = _tail_frequency_segment_masks(
        counts=counts,
        ids=ids,
        cutoff=cutoff,
        include_unseen=include_unseen,
    )
    return eligible


def _tail_frequency_segment_masks(
    *,
    counts: np.ndarray,
    ids: torch.Tensor,
    cutoff: int,
    include_unseen: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_np = np.asarray(ids.detach().cpu().tolist(), dtype=np.int64)
    safe_ids = np.clip(ids_np, 0, len(counts) - 1)
    observed = counts[safe_ids]
    non_padding = ids_np != 0
    unseen = non_padding & (observed == 0)
    seen_rare = non_padding & (observed >= 1) & (observed <= cutoff)
    eligible = seen_rare | unseen if include_unseen else seen_rare
    return tuple(
        torch.tensor(mask_values.tolist(), dtype=torch.bool)
        for mask_values in (eligible, seen_rare, unseen & eligible)
    )


def _tail_random_replacements(
    *,
    counts: np.ndarray,
    ids: torch.Tensor,
    cutoff: int,
    cutoffs: list[int],
    candidates_by_band: dict[int, np.ndarray],
    replacement_seed: int,
    field_idx: int,
    include_unseen: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_np = np.asarray(ids.detach().cpu().tolist(), dtype=np.int64)
    safe_ids = np.clip(ids_np, 0, len(counts) - 1)
    observed = counts[safe_ids]
    eligible = (ids_np != 0) & (observed <= cutoff)
    if not include_unseen:
        eligible &= observed >= 1
    replacements = ids_np.copy()

    def replace_band(
        band_mask: np.ndarray,
        candidates: np.ndarray,
        band_idx: int,
    ) -> None:
        if not bool(band_mask.any()) or len(candidates) <= 1:
            return
        band_ids = ids_np[band_mask]
        hashed = (
            band_ids * 1103515245
            + int(replacement_seed) * 12345
            + int(field_idx) * 2654435761
            + band_idx * 97531
        )
        candidate_indices = np.mod(hashed, len(candidates)).astype(np.int64)
        band_replacements = candidates[candidate_indices]
        same = band_replacements == band_ids
        if bool(same.any()):
            candidate_indices[same] = (candidate_indices[same] + 1) % len(candidates)
            band_replacements = candidates[candidate_indices]
        replacements[band_mask] = band_replacements

    if include_unseen:
        replace_band(eligible & (observed == 0), candidates_by_band[0], -1)

    lower = 1
    for band_idx, upper in enumerate(cutoffs):
        if lower > cutoff:
            break
        band_mask = eligible & (observed >= lower) & (observed <= upper)
        candidates = candidates_by_band[upper]
        replace_band(band_mask, candidates, band_idx)
        lower = upper + 1
    changed = eligible & (replacements != ids_np)
    return (
        torch.tensor(changed.tolist(), dtype=torch.bool),
        torch.tensor(replacements.tolist(), dtype=torch.long),
    )


def _tail_sweep_metrics(
    clean: dict[str, torch.Tensor | None],
    masked: dict[str, torch.Tensor | None],
    masked_fields: torch.Tensor,
) -> torch.Tensor:
    logit_abs_delta = (clean["logits"] - masked["logits"]).abs()
    prob_abs_delta = (
        torch.sigmoid(clean["logits"]) - torch.sigmoid(masked["logits"])
    ).abs()
    return torch.stack(
        (
            masked_fields,
            _relative_delta(clean["field_embedding"], masked["field_embedding"]),
            _relative_delta(clean["tokenizer_output"], masked["tokenizer_output"]),
            _relative_delta(clean["backbone_input"], masked["backbone_input"]),
            _relative_delta(clean["backbone_output"], masked["backbone_output"]),
            _relative_delta(clean["representation"], masked["representation"]),
            logit_abs_delta,
            prob_abs_delta,
        ),
        dim=1,
    )


def measure_cumulative_tail_cutoff_sweep(
    *,
    model: torch.nn.Module,
    data_loader,
    field_counts: dict[int, np.ndarray],
    field_specs: list[dict],
    device: torch.device,
    max_batches: int,
    cutoffs: list[int] | tuple[int, ...],
    intervention: str = "mask",
    replacement_seed: int = 2021,
    conditional_min_samples: int = 0,
    conditional_min_positives: int = 1,
    conditional_min_negatives: int = 1,
    include_unseen: bool = False,
) -> dict:
    raw_model = model.module if hasattr(model, "module") else model
    if not hasattr(raw_model, "forward_stages"):
        raise TypeError("tail cutoff sweep requires model.forward_stages(batch)")

    ordered_cutoffs = sorted({int(cutoff) for cutoff in cutoffs})
    if not ordered_cutoffs or ordered_cutoffs[0] < 1:
        raise ValueError("tail cutoffs must contain positive integers")
    if intervention not in {"mask", "tail_random"}:
        raise ValueError("intervention must be 'mask' or 'tail_random'")
    categorical_specs = [
        spec
        for spec in field_specs
        if str(spec.get("field_type", spec.get("group"))) == "categorical_id"
    ]
    if not categorical_specs:
        raise ValueError("tail cutoff sweep requires categorical_id fields")

    global_metrics = {
        cutoff: _TailSweepRunningMetrics() for cutoff in ordered_cutoffs
    }
    touched_metrics = {
        cutoff: _TailSweepRunningMetrics() for cutoff in ordered_cutoffs
    }
    global_predictions = {
        cutoff: _ConditionalPredictions() for cutoff in ordered_cutoffs
    }
    touched_predictions = {
        cutoff: _ConditionalPredictions() for cutoff in ordered_cutoffs
    }
    field_occurrences = {
        cutoff: {
            str(spec["name"]): {
                "count": 0,
                "num_pos": 0,
                "seen_rare_count": 0,
                "seen_rare_num_pos": 0,
                "unseen_count": 0,
                "unseen_num_pos": 0,
            }
            for spec in categorical_specs
        }
        for cutoff in ordered_cutoffs
    }
    touched_samples = {cutoff: 0 for cutoff in ordered_cutoffs}
    eligible_ids = {
        cutoff: {
            str(spec["name"]): int(
                (
                    field_counts[int(spec["index"])][1:] <= cutoff
                    if include_unseen
                    else (
                        (field_counts[int(spec["index"])][1:] >= 1)
                        & (field_counts[int(spec["index"])][1:] <= cutoff)
                    )
                ).sum()
            )
            for spec in categorical_specs
        }
        for cutoff in ordered_cutoffs
    }
    eligible_id_segments = {
        cutoff: {
            str(spec["name"]): {
                "seen_rare": int(
                    (
                        (field_counts[int(spec["index"])][1:] >= 1)
                        & (field_counts[int(spec["index"])][1:] <= cutoff)
                    ).sum()
                ),
                "unseen": int(
                    (field_counts[int(spec["index"])][1:] == 0).sum()
                )
                if include_unseen
                else 0,
            }
            for spec in categorical_specs
        }
        for cutoff in ordered_cutoffs
    }
    replacement_candidates = {}
    if intervention == "tail_random":
        for spec in categorical_specs:
            field_idx = int(spec["index"])
            counts = field_counts[field_idx]
            replacement_candidates[field_idx] = {}
            if include_unseen:
                unseen_candidates = np.flatnonzero(counts == 0).astype(np.int64)
                replacement_candidates[field_idx][0] = unseen_candidates[
                    unseen_candidates != 0
                ]
            lower = 1
            for upper in ordered_cutoffs:
                replacement_candidates[field_idx][upper] = np.flatnonzero(
                    (counts >= lower) & (counts <= upper)
                ).astype(np.int64)
                lower = upper + 1

    was_training = raw_model.training
    raw_model.eval()
    batches_seen = 0
    samples_seen = 0
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(data_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            sparse_cpu = cpu_batch["sparse"]
            labels_cpu = cpu_batch["label"].detach().cpu() > 0.5
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in cpu_batch.items()
            }
            clean = raw_model.forward_stages(batch)
            batch_size = int(sparse_cpu.shape[0])
            all_samples = torch.ones(batch_size, dtype=torch.bool, device=device)
            batches_seen += 1
            samples_seen += batch_size

            for cutoff in ordered_cutoffs:
                masked_sparse = batch["sparse"].clone()
                touched_cpu = torch.zeros(batch_size, dtype=torch.bool)
                masked_field_count_cpu = torch.zeros(batch_size, dtype=torch.float32)
                for spec in categorical_specs:
                    field_idx = int(spec["index"])
                    field_name = str(spec["name"])
                    if intervention == "tail_random":
                        field_mask_cpu, replacements_cpu = _tail_random_replacements(
                            counts=field_counts[field_idx],
                            ids=sparse_cpu[:, field_idx],
                            cutoff=cutoff,
                            cutoffs=ordered_cutoffs,
                            candidates_by_band=replacement_candidates[field_idx],
                            replacement_seed=replacement_seed,
                            field_idx=field_idx,
                            include_unseen=include_unseen,
                        )
                        _, seen_rare_mask_cpu, unseen_mask_cpu = (
                            _tail_frequency_segment_masks(
                                counts=field_counts[field_idx],
                                ids=sparse_cpu[:, field_idx],
                                cutoff=cutoff,
                                include_unseen=include_unseen,
                            )
                        )
                    else:
                        (
                            field_mask_cpu,
                            seen_rare_mask_cpu,
                            unseen_mask_cpu,
                        ) = _tail_frequency_segment_masks(
                            counts=field_counts[field_idx],
                            ids=sparse_cpu[:, field_idx],
                            cutoff=cutoff,
                            include_unseen=include_unseen,
                        )
                        replacements_cpu = None
                    count = int(field_mask_cpu.sum().item())
                    field_occurrences[cutoff][field_name]["count"] += count
                    field_occurrences[cutoff][field_name]["num_pos"] += int(
                        (labels_cpu & field_mask_cpu).sum().item()
                    )
                    field_occurrences[cutoff][field_name][
                        "seen_rare_count"
                    ] += int(seen_rare_mask_cpu.sum().item())
                    field_occurrences[cutoff][field_name][
                        "seen_rare_num_pos"
                    ] += int((labels_cpu & seen_rare_mask_cpu).sum().item())
                    field_occurrences[cutoff][field_name]["unseen_count"] += int(
                        unseen_mask_cpu.sum().item()
                    )
                    field_occurrences[cutoff][field_name][
                        "unseen_num_pos"
                    ] += int((labels_cpu & unseen_mask_cpu).sum().item())
                    if count == 0:
                        continue
                    touched_cpu |= field_mask_cpu
                    masked_field_count_cpu += field_mask_cpu.to(dtype=torch.float32)
                    field_mask = field_mask_cpu.to(device=device)
                    if replacements_cpu is None:
                        masked_sparse[field_mask, field_idx] = 0
                    else:
                        masked_sparse[field_mask, field_idx] = replacements_cpu[
                            field_mask_cpu
                        ].to(device=device)

                touched = touched_cpu.to(device=device)
                touched_samples[cutoff] += int(touched_cpu.sum().item())
                masked_batch = dict(batch)
                masked_batch["sparse"] = masked_sparse
                masked = raw_model.forward_stages(masked_batch)
                metrics = _tail_sweep_metrics(
                    clean,
                    masked,
                    masked_field_count_cpu.to(
                        device=device, dtype=clean["logits"].dtype
                    ),
                )
                global_metrics[cutoff].update(metrics, all_samples)
                touched_metrics[cutoff].update(metrics, touched)
                global_predictions[cutoff].update(
                    labels=batch["label"],
                    clean_logits=clean["logits"],
                    masked_logits=masked["logits"],
                    mask=all_samples,
                )
                touched_predictions[cutoff].update(
                    labels=batch["label"],
                    clean_logits=clean["logits"],
                    masked_logits=masked["logits"],
                    mask=touched,
                )

    if was_training:
        raw_model.train()

    results = {}
    previous_eligible = 0
    previous_occurrences = 0
    previous_touched = 0
    total_possible_occurrences = samples_seen * len(categorical_specs)
    for cutoff in ordered_cutoffs:
        fields = {}
        total_occurrences = 0
        total_segment_occurrences = {"seen_rare": 0, "unseen": 0}
        total_segment_positives = {"seen_rare": 0, "unseen": 0}
        total_segment_eligible_ids = {"seen_rare": 0, "unseen": 0}
        for spec in categorical_specs:
            field_name = str(spec["name"])
            occurrence = dict(field_occurrences[cutoff][field_name])
            total_occurrences += occurrence["count"]
            field_frequency_segments = {}
            for segment in ("seen_rare", "unseen"):
                segment_occurrences = occurrence[f"{segment}_count"]
                segment_positives = occurrence[f"{segment}_num_pos"]
                segment_eligible_ids = eligible_id_segments[cutoff][field_name][
                    segment
                ]
                total_segment_occurrences[segment] += segment_occurrences
                total_segment_positives[segment] += segment_positives
                total_segment_eligible_ids[segment] += segment_eligible_ids
                field_frequency_segments[segment] = {
                    "eligible_unique_ids": segment_eligible_ids,
                    "eval_occurrences": segment_occurrences,
                    "eval_positives": segment_positives,
                    "sample_coverage_all": (
                        float(segment_occurrences / samples_seen)
                        if samples_seen
                        else 0.0
                    ),
                }
            fields[field_name] = {
                "index": int(spec["index"]),
                "eligible_unique_ids": eligible_ids[cutoff][field_name],
                "eval_occurrences": occurrence["count"],
                "eval_positives": occurrence["num_pos"],
                "sample_coverage_all": (
                    float(occurrence["count"] / samples_seen)
                    if samples_seen
                    else 0.0
                ),
                "frequency_segments": field_frequency_segments,
            }
        total_eligible = sum(eligible_ids[cutoff].values())
        global_performance = global_predictions[cutoff].result(
            min_samples=0,
            min_positives=1,
            min_negatives=1,
        )
        touched_performance = touched_predictions[cutoff].result(
            min_samples=conditional_min_samples,
            min_positives=conditional_min_positives,
            min_negatives=conditional_min_negatives,
        )
        results[str(cutoff)] = {
            "cutoff": cutoff,
            "eligible_unique_ids": total_eligible,
            "incremental_eligible_unique_ids": total_eligible - previous_eligible,
            "masked_field_occurrences": total_occurrences,
            "incremental_masked_field_occurrences": (
                total_occurrences - previous_occurrences
            ),
            "field_occurrence_coverage_all": (
                float(total_occurrences / total_possible_occurrences)
                if total_possible_occurrences
                else 0.0
            ),
            "touched_samples": touched_samples[cutoff],
            "incremental_touched_samples": touched_samples[cutoff] - previous_touched,
            "sample_union_coverage_all": (
                float(touched_samples[cutoff] / samples_seen)
                if samples_seen
                else 0.0
            ),
            "frequency_segments": {
                segment: {
                    "eligible_unique_ids": total_segment_eligible_ids[segment],
                    "eval_occurrences": total_segment_occurrences[segment],
                    "eval_positives": total_segment_positives[segment],
                    "field_occurrence_coverage_all": (
                        float(
                            total_segment_occurrences[segment]
                            / total_possible_occurrences
                        )
                        if total_possible_occurrences
                        else 0.0
                    ),
                }
                for segment in ("seen_rare", "unseen")
            },
            "frequency_segments_valid": (
                total_segment_occurrences["seen_rare"]
                + total_segment_occurrences["unseen"]
                == total_occurrences
                and total_segment_eligible_ids["seen_rare"]
                + total_segment_eligible_ids["unseen"]
                == total_eligible
            ),
            "global_performance": global_performance,
            "touched_performance": touched_performance,
            "global_sensitivity": global_metrics[cutoff].result(),
            "touched_sensitivity": touched_metrics[cutoff].result(),
            "fields": fields,
        }
        previous_eligible = total_eligible
        previous_occurrences = total_occurrences
        previous_touched = touched_samples[cutoff]

    return {
        "batches_seen": batches_seen,
        "samples_seen": samples_seen,
        "num_categorical_fields": len(categorical_specs),
        "intervention": intervention,
        "include_unseen": bool(include_unseen),
        "tail_frequency_range": [0 if include_unseen else 1, ordered_cutoffs[-1]],
        "replacement_seed": replacement_seed if intervention == "tail_random" else None,
        "cutoffs": ordered_cutoffs,
        "nested_coverage_valid": all(
            results[str(current)]["touched_samples"]
            <= results[str(next_cutoff)]["touched_samples"]
            for current, next_cutoff in zip(ordered_cutoffs, ordered_cutoffs[1:])
        ),
        "results": results,
    }


def measure_stage_influence(
    *,
    model: torch.nn.Module,
    data_loader,
    field_counts: dict[int, np.ndarray],
    field_specs: list[dict],
    device: torch.device,
    max_batches: int,
    include_group_interventions: bool = True,
    conditional_min_samples: int = 0,
    conditional_min_positives: int = 1,
    conditional_min_negatives: int = 1,
    tail_max_count: int = 5,
    head_min_count: int = 100,
) -> dict:
    raw_model = model.module if hasattr(model, "module") else model
    if not hasattr(raw_model, "forward_stages"):
        raise TypeError("stage influence requires model.forward_stages(batch)")

    buckets = ("unseen", "tail", "mid", "head")
    field_acc = {
        (str(spec["name"]), bucket): _RunningMetrics()
        for spec in field_specs
        for bucket in buckets
    }
    group_names = sorted({str(spec["group"]) for spec in field_specs})
    group_acc = {
        (group_name, bucket): _RunningMetrics()
        for group_name in group_names
        for bucket in buckets
    }
    field_predictions = {
        (str(spec["name"]), bucket): _ConditionalPredictions()
        for spec in field_specs
        for bucket in buckets
    }
    group_predictions = (
        {
            (group_name, bucket): _ConditionalPredictions()
            for group_name in group_names
            for bucket in buckets
        }
        if include_group_interventions
        else {}
    )
    token_membership = _field_token_membership(raw_model)

    was_training = raw_model.training
    raw_model.eval()
    batches_seen = 0
    samples_seen = 0
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(data_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            sparse_cpu = cpu_batch["sparse"]
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in cpu_batch.items()
            }
            clean = raw_model.forward_stages(batch)
            batches_seen += 1
            samples_seen += int(sparse_cpu.shape[0])
            batch_bucket_masks = {}

            for spec in field_specs:
                field_idx = int(spec["index"])
                ids = sparse_cpu[:, field_idx]
                bucket_masks = _frequency_bucket_masks(
                    field_counts[field_idx],
                    ids,
                    tail_max_count=tail_max_count,
                    head_min_count=head_min_count,
                )
                if include_group_interventions:
                    batch_bucket_masks[field_idx] = bucket_masks

                masked_batch = dict(batch)
                masked_sparse = batch["sparse"].clone()
                masked_sparse[:, field_idx] = 0
                masked_batch["sparse"] = masked_sparse
                masked = raw_model.forward_stages(masked_batch)
                metrics = _paired_metrics(clean, masked, field_idx)

                for bucket, cpu_mask in bucket_masks.items():
                    device_mask = cpu_mask.to(device=device)
                    field_acc[(str(spec["name"]), bucket)].update(metrics, device_mask)
                    group_acc[(str(spec["group"]), bucket)].update(metrics, device_mask)
                    field_predictions[(str(spec["name"]), bucket)].update(
                        labels=batch["label"],
                        clean_logits=clean["logits"],
                        masked_logits=masked["logits"],
                        mask=device_mask,
                    )

            if include_group_interventions:
                # Group-level conditional evaluation masks every matching field
                # in the same sample, mirroring the original IF-W3 intervention.
                for group_name in group_names:
                    group_specs = [
                        spec for spec in field_specs if str(spec["group"]) == group_name
                    ]
                    for bucket in buckets:
                        touched_cpu = torch.zeros(sparse_cpu.shape[0], dtype=torch.bool)
                        masked_sparse = batch["sparse"].clone()
                        for spec in group_specs:
                            field_idx = int(spec["index"])
                            field_mask_cpu = batch_bucket_masks[field_idx][bucket]
                            touched_cpu |= field_mask_cpu
                            field_mask = field_mask_cpu.to(device=device)
                            masked_sparse[field_mask, field_idx] = 0
                        touched = touched_cpu.to(device=device)
                        if not bool(touched.any()):
                            continue
                        masked_batch = dict(batch)
                        masked_batch["sparse"] = masked_sparse
                        group_masked = raw_model.forward_stages(masked_batch)
                        group_predictions[(group_name, bucket)].update(
                            labels=batch["label"],
                            clean_logits=clean["logits"],
                            masked_logits=group_masked["logits"],
                            mask=touched,
                        )

    if was_training:
        raw_model.train()

    fields = {}
    for spec in field_specs:
        name = str(spec["name"])
        bucket_results = {
            bucket: field_acc[(name, bucket)].result() for bucket in buckets
        }
        nonzero_occurrences = sum(result["count"] for result in bucket_results.values())
        for result in bucket_results.values():
            result["sample_coverage_all"] = (
                float(result["count"] / samples_seen) if samples_seen else 0.0
            )
            result["sample_coverage_nonzero"] = (
                float(result["count"] / nonzero_occurrences)
                if nonzero_occurrences
                else 0.0
            )
        for bucket, result in bucket_results.items():
            result["conditional_performance"] = field_predictions[(name, bucket)].result(
                min_samples=conditional_min_samples,
                min_positives=conditional_min_positives,
                min_negatives=conditional_min_negatives,
            )
        field_idx = int(spec["index"])
        field_result = {
            "index": field_idx,
            "group": str(spec["group"]),
            "field_type": str(spec.get("field_type", spec["group"])),
            "nonzero_eval_occurrences": nonzero_occurrences,
            "id_population": _id_population(
                field_counts[field_idx],
                tail_max_count=tail_max_count,
                head_min_count=head_min_count,
            ),
            "buckets": bucket_results,
        }
        field_result.update(token_membership.get(field_idx, {}))
        fields[name] = field_result
    groups = {}
    for group_name in group_names:
        num_group_fields = sum(
            1 for spec in field_specs if str(spec["group"]) == group_name
        )
        total_group_occurrences = samples_seen * num_group_fields
        bucket_results = {
            bucket: group_acc[(group_name, bucket)].result() for bucket in buckets
        }
        nonzero_occurrences = sum(result["count"] for result in bucket_results.values())
        for result in bucket_results.values():
            result["field_occurrence_coverage_all"] = (
                float(result["count"] / total_group_occurrences)
                if total_group_occurrences
                else 0.0
            )
        for bucket, result in bucket_results.items():
            if include_group_interventions:
                conditional = group_predictions[(group_name, bucket)].result(
                    min_samples=conditional_min_samples,
                    min_positives=conditional_min_positives,
                    min_negatives=conditional_min_negatives,
                )
                conditional["sample_union_coverage_all"] = (
                    float(conditional["num_samples"] / samples_seen)
                    if samples_seen
                    else 0.0
                )
                conditional["collected"] = True
            else:
                conditional = {
                    "collected": False,
                    "num_samples": 0,
                    "clean": {},
                    "masked": {},
                    "auc_drop": None,
                    "logloss_increase": None,
                    "auc_valid": False,
                    "auc_invalid_reason": "group interventions disabled",
                    "sample_union_coverage_all": None,
                }
            result["conditional_performance"] = conditional
            result["field_occurrence_coverage_nonzero"] = (
                float(result["count"] / nonzero_occurrences)
                if nonzero_occurrences
                else 0.0
            )
        groups[group_name] = {
            "num_fields": num_group_fields,
            "nonzero_eval_occurrences": nonzero_occurrences,
            "buckets": bucket_results,
        }
    return {
        "batches_seen": batches_seen,
        "samples_seen": samples_seen,
        "coverage_semantics": {
            "field_sample_coverage_all": "bucket count / evaluated samples",
            "field_sample_coverage_nonzero": "bucket count / nonzero occurrences of that field",
            "group_field_occurrence_coverage_all": (
                "bucket field-occurrences / (evaluated samples * fields in group)"
            ),
            "group_sample_union_coverage_all": (
                "samples touching at least one bucket ID in the group / evaluated samples"
            ),
            "conditional_auc_drop": "clean subset AUC - masked subset AUC",
            "conditional_logloss_increase": "masked subset LogLoss - clean subset LogLoss",
            "conditional_auc_valid": (
                "true only when configured sample, positive, and negative thresholds are met"
            ),
        },
        "analysis_options": {
            "include_group_interventions": include_group_interventions,
            "conditional_min_samples": conditional_min_samples,
            "conditional_min_positives": conditional_min_positives,
            "conditional_min_negatives": conditional_min_negatives,
            "tail_max_count": tail_max_count,
            "head_min_count": head_min_count,
        },
        "fields": fields,
        "groups": groups,
    }


def measure_frequency_coverage(
    *,
    data_loader,
    field_counts: dict[int, np.ndarray],
    field_specs: list[dict],
    max_batches: int,
    tail_max_count: int = 5,
    head_min_count: int = 100,
) -> dict:
    """Measure per-field bucket prevalence without loading a model checkpoint."""
    buckets = ("unseen", "tail", "mid", "head")
    counts = {
        (str(spec["name"]), bucket): {"count": 0, "num_pos": 0}
        for spec in field_specs
        for bucket in buckets
    }
    samples_seen = 0
    batches_seen = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        sparse = batch["sparse"]
        labels = batch["label"].detach().cpu() > 0.5
        samples_seen += int(sparse.shape[0])
        batches_seen += 1
        for spec in field_specs:
            field_idx = int(spec["index"])
            bucket_masks = _frequency_bucket_masks(
                field_counts[field_idx],
                sparse[:, field_idx],
                tail_max_count=tail_max_count,
                head_min_count=head_min_count,
            )
            for bucket, mask in bucket_masks.items():
                result = counts[(str(spec["name"]), bucket)]
                result["count"] += int(mask.sum().item())
                result["num_pos"] += int((labels & mask).sum().item())

    fields = {}
    for spec in field_specs:
        name = str(spec["name"])
        bucket_results = {}
        nonzero_occurrences = sum(
            counts[(name, bucket)]["count"] for bucket in buckets
        )
        for bucket in buckets:
            result = dict(counts[(name, bucket)])
            result["num_neg"] = result["count"] - result["num_pos"]
            result["pos_rate"] = (
                float(result["num_pos"] / result["count"])
                if result["count"]
                else 0.0
            )
            result["sample_coverage_all"] = (
                float(result["count"] / samples_seen) if samples_seen else 0.0
            )
            result["sample_coverage_nonzero"] = (
                float(result["count"] / nonzero_occurrences)
                if nonzero_occurrences
                else 0.0
            )
            bucket_results[bucket] = result
        field_idx = int(spec["index"])
        fields[name] = {
            "index": field_idx,
            "group": str(spec["group"]),
            "field_type": str(spec.get("field_type", spec["group"])),
            "nonzero_eval_occurrences": nonzero_occurrences,
            "id_population": _id_population(
                field_counts[field_idx],
                tail_max_count=tail_max_count,
                head_min_count=head_min_count,
            ),
            "buckets": bucket_results,
        }

    return {
        "batches_seen": batches_seen,
        "samples_seen": samples_seen,
        "fields": fields,
    }


def _load_analysis_field_counts(
    *,
    dataset,
    config: dict,
    analysis_cfg: dict,
    field_specs: list[dict],
) -> dict[int, np.ndarray]:
    fields = [int(spec["index"]) for spec in field_specs]
    train_days = analysis_cfg.get("train_days", config["dataset"]["train_days"])
    if isinstance(train_days, (int, str)):
        train_days = [int(train_days)]
    return dataset.get_field_frequency_counts(
        fields=fields,
        train_days=train_days,
        cache_path=analysis_cfg["frequency_cache_path"],
        max_rows=int(analysis_cfg.get("frequency_max_rows", 0)),
    )


def _sample_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "max": float(values.max().item()),
    }


def _code_assignment_summary(
    counts_per_code: torch.Tensor,
) -> dict[str, float | int | None]:
    counts = counts_per_code.detach().to(dtype=torch.float64, device="cpu")
    total = float(counts.sum().item())
    if total <= 0.0:
        return {
            "assigned_unique_codes": 0,
            "active_codebook_fraction": 0.0,
            "top_code_fraction": None,
            "assignment_entropy_nats": None,
            "normalized_assignment_entropy": None,
            "assignment_perplexity": None,
            "effective_code_fraction": None,
            "assignment_gini": None,
        }
    probabilities = counts / total
    positive = probabilities.gt(0)
    entropy = float(
        -(probabilities[positive] * probabilities[positive].log()).sum().item()
    )
    perplexity = math.exp(entropy)
    codebook_size = counts.numel()
    sorted_counts = counts.sort().values
    indices = torch.arange(1, codebook_size + 1, dtype=torch.float64)
    gini = float(
        ((2.0 * indices - codebook_size - 1.0) * sorted_counts).sum().item()
        / (codebook_size * total)
    )
    assigned_unique = int(positive.sum().item())
    return {
        "assigned_unique_codes": assigned_unique,
        "active_codebook_fraction": assigned_unique / codebook_size,
        "top_code_fraction": float(probabilities.max().item()),
        "assignment_entropy_nats": entropy,
        "normalized_assignment_entropy": entropy / math.log(codebook_size),
        "assignment_perplexity": perplexity,
        "effective_code_fraction": perplexity / codebook_size,
        "assignment_gini": gini,
    }


@torch.no_grad()
def measure_embedding_frequency_profile(
    *,
    model: torch.nn.Module,
    field_counts: dict[int, np.ndarray],
    field_specs: list[dict],
    max_ids_per_bucket: int = 1024,
) -> dict:
    """Profile raw embeddings and discrete assignments by training frequency."""
    if max_ids_per_bucket <= 0:
        raise ValueError("max_ids_per_bucket must be positive")

    raw_model = model.module if hasattr(model, "module") else model
    embeddings = getattr(
        getattr(getattr(raw_model, "encoder", None), "sparse_arch", None),
        "embeddings",
        None,
    )
    if embeddings is None:
        raise ValueError("model.encoder.sparse_arch.embeddings was not found")

    tokenizer = getattr(raw_model, "tokenizer", None)
    selected_fields = tuple(getattr(tokenizer, "selected_field_indices", ()))
    codebooks = getattr(tokenizer, "codebooks", None)
    distance_chunk_size = int(getattr(tokenizer, "distance_chunk_size", 2048))
    bucket_ranges = (
        ("unseen", 0, 0),
        ("count_1_5", 1, 5),
        ("count_6_10", 6, 10),
        ("count_11_20", 11, 20),
        ("count_21_50", 21, 50),
        ("head_gt_50", 51, None),
    )

    report = {
        "sampling": {
            "strategy": "deterministic_even_spacing",
            "max_ids_per_bucket": int(max_ids_per_bucket),
            "padding_id_excluded": True,
        },
        "fields": {},
    }
    for spec in field_specs:
        field_idx = int(spec["index"])
        if field_idx < 0 or field_idx >= len(embeddings):
            raise ValueError(
                f"analysis field index {field_idx} is outside [0, {len(embeddings)})"
            )
        counts = np.asarray(field_counts[field_idx])
        embedding_weight = embeddings[field_idx].weight
        population_size = min(len(counts), embedding_weight.size(0))
        valid_ids = np.arange(1, population_size, dtype=np.int64)
        valid_counts = counts[1:population_size]

        selected_pos = None
        if field_idx in selected_fields and codebooks is not None:
            selected_pos = selected_fields.index(field_idx)
        codebook = (
            codebooks[selected_pos].detach().float()
            if selected_pos is not None
            else None
        )
        field_report = {
            "field_index": field_idx,
            "field_name": str(spec.get("name", field_idx)),
            "embedding_rows": int(embedding_weight.size(0)),
            "count_rows": int(len(counts)),
            "discrete": codebook is not None,
            "codebook_size": int(codebook.size(0)) if codebook is not None else None,
            "buckets": {},
        }

        for bucket_name, lower, upper in bucket_ranges:
            mask = valid_counts >= lower
            if upper is not None:
                mask &= valid_counts <= upper
            population_ids = valid_ids[mask]
            sample_size = min(population_ids.size, int(max_ids_per_bucket))
            if sample_size:
                positions = np.linspace(
                    0,
                    population_ids.size - 1,
                    num=sample_size,
                    dtype=np.int64,
                )
                sampled_ids = population_ids[positions]
                index = torch.as_tensor(
                    sampled_ids,
                    dtype=torch.long,
                    device=embedding_weight.device,
                )
                sampled_embeddings = embedding_weight.index_select(0, index).float()
                raw_norm = torch.linalg.vector_norm(sampled_embeddings, dim=-1)
            else:
                sampled_embeddings = embedding_weight.new_empty(
                    (0, embedding_weight.size(1)),
                    dtype=torch.float32,
                )
                raw_norm = embedding_weight.new_empty((0,), dtype=torch.float32)

            bucket_report = {
                "frequency_range": [int(lower), int(upper) if upper is not None else None],
                "unique_ids": int(population_ids.size),
                "sampled_ids": int(sample_size),
                "raw_embedding_norm": _sample_summary(raw_norm),
                "quantization_error": None,
                "assigned_unique_codes": None,
                "active_codebook_fraction": None,
                "top_code_fraction": None,
                "assignment_entropy_nats": None,
                "normalized_assignment_entropy": None,
                "assignment_perplexity": None,
                "effective_code_fraction": None,
                "assignment_gini": None,
            }
            if codebook is not None and sample_size:
                best_distances = sampled_embeddings.new_full(
                    (sample_size,),
                    float("inf"),
                )
                best_codes = torch.zeros(
                    sample_size,
                    dtype=torch.long,
                    device=sampled_embeddings.device,
                )
                for start in range(0, codebook.size(0), distance_chunk_size):
                    end = min(start + distance_chunk_size, codebook.size(0))
                    distances = (
                        sampled_embeddings.square().sum(dim=-1, keepdim=True)
                        + codebook[start:end].square().sum(dim=-1).unsqueeze(0)
                        - 2.0 * sampled_embeddings @ codebook[start:end].transpose(0, 1)
                    ).clamp_min(0.0)
                    chunk_distances, chunk_codes = distances.min(dim=1)
                    improved = chunk_distances < best_distances
                    best_distances[improved] = chunk_distances[improved]
                    best_codes[improved] = chunk_codes[improved] + start
                counts_per_code = torch.bincount(
                    best_codes,
                    minlength=codebook.size(0),
                )
                bucket_report.update(
                    {
                        "quantization_error": _sample_summary(best_distances.sqrt()),
                        **_code_assignment_summary(counts_per_code),
                    }
                )
            field_report["buckets"][bucket_name] = bucket_report
        report["fields"][field_report["field_name"]] = field_report
    return report


def run_information_funnel_coverage_census(
    *,
    data_loader,
    dataset,
    config: dict,
    output_dir: str,
) -> dict:
    analysis_cfg = config["analysis"]["information_funnel"]
    field_specs = resolve_field_specs(config, analysis_cfg)
    field_counts = _load_analysis_field_counts(
        dataset=dataset,
        config=config,
        analysis_cfg=analysis_cfg,
        field_specs=field_specs,
    )
    tail_max_count = int(analysis_cfg.get("tail_max_count", 5))
    head_min_count = int(analysis_cfg.get("head_min_count", 100))
    coverage = measure_frequency_coverage(
        data_loader=data_loader,
        field_counts=field_counts,
        field_specs=field_specs,
        max_batches=int(analysis_cfg.get("max_batches", 16)),
        tail_max_count=tail_max_count,
        head_min_count=head_min_count,
    )
    report = {
        "schema_version": 2,
        "analysis": "information_funnel_frequency_coverage_census",
        "frequency_buckets": {
            "unseen": [0, 0],
            "tail": [1, tail_max_count],
            "mid": [tail_max_count + 1, head_min_count - 1],
            "head": [head_min_count, None],
        },
        "coverage": coverage,
    }
    output_path = Path(output_dir) / "training_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[IF-W5 coverage] fields={len(field_specs)} "
        f"samples={coverage['samples_seen']} training_report={output_path}"
    )
    return report


def run_information_funnel_tail_cutoff_sweep(
    *,
    model: torch.nn.Module,
    data_loader,
    dataset,
    config: dict,
    device: torch.device,
    output_dir: str,
    best_auc: float,
) -> dict:
    analysis_cfg = config["analysis"]["information_funnel"]
    field_specs = resolve_field_specs(config, analysis_cfg)
    field_counts = _load_analysis_field_counts(
        dataset=dataset,
        config=config,
        analysis_cfg=analysis_cfg,
        field_specs=field_specs,
    )
    raw_cutoffs = analysis_cfg.get("tail_cutoffs", [5, 10, 20, 50])
    if isinstance(raw_cutoffs, str):
        raw_cutoffs = [value.strip() for value in raw_cutoffs.split(",") if value.strip()]
    cutoffs = [int(value) for value in raw_cutoffs]
    intervention = str(analysis_cfg.get("intervention", "mask"))
    replacement_seed = int(analysis_cfg.get("replacement_seed", 2021))
    include_unseen = bool(analysis_cfg.get("include_unseen", False))
    frequency_cache_path = Path(analysis_cfg["frequency_cache_path"])
    if not frequency_cache_path.is_file():
        raise FileNotFoundError(
            f"information-funnel frequency cache not found: {frequency_cache_path}"
        )
    frequency_cache_sha256 = _sha256_file(frequency_cache_path)
    expected_frequency_cache_sha256 = os.environ.get(
        "RECSCALE_FREQUENCY_CACHE_SHA256"
    )
    if (
        expected_frequency_cache_sha256
        and frequency_cache_sha256 != expected_frequency_cache_sha256
    ):
        raise ValueError(
            "information-funnel frequency cache hash differs from frozen runtime "
            "archive"
        )
    sweep = measure_cumulative_tail_cutoff_sweep(
        model=model,
        data_loader=data_loader,
        field_counts=field_counts,
        field_specs=field_specs,
        device=device,
        max_batches=int(analysis_cfg.get("max_batches", 0)),
        cutoffs=cutoffs,
        intervention=intervention,
        replacement_seed=replacement_seed,
        include_unseen=include_unseen,
        conditional_min_samples=int(analysis_cfg.get("conditional_min_samples", 0)),
        conditional_min_positives=int(
            analysis_cfg.get("conditional_min_positives", 1)
        ),
        conditional_min_negatives=int(
            analysis_cfg.get("conditional_min_negatives", 1)
        ),
    )
    embedding_profile_cfg = analysis_cfg.get("embedding_frequency_profile", {})
    embedding_profile = None
    if bool(embedding_profile_cfg.get("enabled", False)):
        embedding_profile = measure_embedding_frequency_profile(
            model=model,
            field_counts=field_counts,
            field_specs=field_specs,
            max_ids_per_bucket=int(
                embedding_profile_cfg.get("max_ids_per_bucket", 1024)
            ),
        )
    raw_model = model.module if hasattr(model, "module") else model
    discrete_training_audit = None
    if hasattr(raw_model, "get_discrete_training_audit"):
        discrete_training_audit = raw_model.get_discrete_training_audit()
    offline_tail_id_collapse = None
    if hasattr(raw_model, "get_offline_tail_id_collapse_metadata"):
        offline_tail_id_collapse = (
            raw_model.get_offline_tail_id_collapse_metadata()
        )
    report = {
        "schema_version": 3,
        "analysis": "information_funnel_cumulative_tail_cutoff_sweep",
        "model_path": str(analysis_cfg.get("path_name", config["model"]["tokenizer_type"])),
        "seed": int(config.get("seed", 42)),
        "tokenizer_seed": int(config["model"].get("tokenizer_seed", config.get("seed", 42))),
        "data_seed": int(
            config.get("training", {}).get("data_seed", config.get("seed", 42))
        ),
        "deterministic_training": bool(
            config.get("training", {}).get("deterministic", False)
        ),
        "optimizer_foreach": config.get("training", {}).get("optimizer_foreach"),
        "source_manifest_sha256": os.environ.get("RECSCALE_SOURCE_MANIFEST_SHA256"),
        "best_auc": float(best_auc),
        "tail_cutoffs": sweep["cutoffs"],
        "intervention": intervention,
        "include_unseen": include_unseen,
        "replacement_seed": replacement_seed if intervention == "tail_random" else None,
        "frequency_cache": {
            "path": str(frequency_cache_path),
            "sha256": frequency_cache_sha256,
            "size_bytes": frequency_cache_path.stat().st_size,
            "expected_sha256": expected_frequency_cache_sha256,
            "archive_path": os.environ.get(
                "RECSCALE_FREQUENCY_CACHE_ARCHIVE_PATH"
            ),
        },
        "sweep": sweep,
    }
    if embedding_profile is not None:
        report["embedding_frequency_profile"] = embedding_profile
    if discrete_training_audit is not None:
        report["discrete_training_audit"] = discrete_training_audit
    if offline_tail_id_collapse is not None:
        report["offline_tail_id_collapse"] = offline_tail_id_collapse
    output_path = Path(output_dir) / "training_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[Information Funnel tail sweep] path={report['model_path']} "
        f"seed={report['seed']} intervention={intervention} "
        f"include_unseen={include_unseen} "
        f"samples={sweep['samples_seen']} "
        f"cutoffs={sweep['cutoffs']} training_report={output_path}"
    )
    for cutoff in sweep["cutoffs"]:
        result = sweep["results"][str(cutoff)]
        global_performance = result["global_performance"]
        print(
            f"[Information Funnel tail sweep] cutoff={cutoff} "
            f"coverage={result['sample_union_coverage_all']:.6f} "
            f"touched={result['touched_samples']} "
            f"auc_drop={global_performance['auc_drop']:.8f} "
            f"logloss_inc={global_performance['logloss_increase']:.8f}"
        )
    return report


def run_information_funnel_analysis(
    *,
    model: torch.nn.Module,
    data_loader,
    dataset,
    config: dict,
    device: torch.device,
    output_dir: str,
    best_auc: float,
) -> dict:
    analysis_cfg = config["analysis"]["information_funnel"]
    field_specs = resolve_field_specs(config, analysis_cfg)
    field_counts = _load_analysis_field_counts(
        dataset=dataset,
        config=config,
        analysis_cfg=analysis_cfg,
        field_specs=field_specs,
    )
    tail_max_count = int(analysis_cfg.get("tail_max_count", 5))
    head_min_count = int(analysis_cfg.get("head_min_count", 100))
    influence = measure_stage_influence(
        model=model,
        data_loader=data_loader,
        field_counts=field_counts,
        field_specs=field_specs,
        device=device,
        max_batches=int(analysis_cfg.get("max_batches", 16)),
        include_group_interventions=bool(
            analysis_cfg.get("include_group_interventions", True)
        ),
        conditional_min_samples=int(analysis_cfg.get("conditional_min_samples", 0)),
        conditional_min_positives=int(
            analysis_cfg.get("conditional_min_positives", 1)
        ),
        conditional_min_negatives=int(
            analysis_cfg.get("conditional_min_negatives", 1)
        ),
        tail_max_count=tail_max_count,
        head_min_count=head_min_count,
    )
    report = {
        "schema_version": 2,
        "analysis": "information_funnel_stage_influence",
        "model_path": str(analysis_cfg.get("path_name", config["model"]["tokenizer_type"])),
        "seed": int(config.get("seed", 42)),
        "tokenizer_seed": int(config["model"].get("tokenizer_seed", config.get("seed", 42))),
        "best_auc": float(best_auc),
        "frequency_buckets": {
            "unseen": [0, 0],
            "tail": [1, tail_max_count],
            "mid": [tail_max_count + 1, head_min_count - 1],
            "head": [head_min_count, None],
        },
        "influence": influence,
    }

    output_path = Path(output_dir) / "training_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Information Funnel] training_report={output_path}")
    for group_name, group_result in influence["groups"].items():
        for bucket_name, result in group_result["buckets"].items():
            metrics = result["metrics"]
            if not metrics:
                print(
                    f"[Information Funnel] group={group_name} bucket={bucket_name} "
                    "count=0 coverage_all=0.000000"
                )
                continue
            conditional = result["conditional_performance"]
            sample_union = conditional["sample_union_coverage_all"]
            sample_union_text = (
                f"{sample_union:.6f}" if sample_union is not None else "not_collected"
            )
            print(
                "[Information Funnel] "
                f"group={group_name} bucket={bucket_name} count={result['count']} "
                f"coverage_all={result['field_occurrence_coverage_all']:.6f} "
                f"coverage_nonzero={result['field_occurrence_coverage_nonzero']:.6f} "
                f"sample_union={sample_union_text} "
                f"auc_drop={conditional['auc_drop']} "
                f"logloss_inc={conditional['logloss_increase']} "
                f"tok_rel={metrics['token_local_relative']['mean']:.6f} "
                f"postln_rel={metrics['backbone_input_local_relative']['mean']:.6f} "
                f"logit_delta={metrics['logit_abs_delta']['mean']:.6f}"
            )
    return report
