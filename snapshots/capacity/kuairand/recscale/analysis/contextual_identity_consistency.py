"""Measure whether self-excluded context can identify tailrand corruption."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recscale.analysis.information_funnel import (
    _compute_ctr_metrics,
    _load_analysis_field_counts,
    _tail_random_replacements,
    resolve_field_specs,
)
from recscale.datasets import create_dataset
from recscale.models import create_model
from recscale.utils.config import get_config


def _rank_auc(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float | None:
    if len(negative_scores) == 0 or len(positive_scores) == 0:
        return None
    scores = np.concatenate([negative_scores, positive_scores]).astype(np.float64)
    labels = np.concatenate(
        [
            np.zeros(len(negative_scores), dtype=np.bool_),
            np.ones(len(positive_scores), dtype=np.bool_),
        ]
    )
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    num_pos = int(labels.sum())
    num_neg = len(labels) - num_pos
    return float(
        (ranks[labels].sum() - num_pos * (num_pos + 1) / 2.0)
        / (num_pos * num_neg)
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) != len(x):
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(
        np.sqrt(np.square(x_centered).sum() * np.square(y_centered).sum())
    )
    if denominator == 0.0:
        return None
    return float((x_centered * y_centered).sum() / denominator)


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return np.asarray(value.detach().float().cpu().tolist(), dtype=np.float64)


@dataclass
class _FieldAccumulator:
    count: int = 0
    clean_sum: float = 0.0
    random_sum: float = 0.0
    increase_count: int = 0

    def update(self, clean: torch.Tensor, random_scores: torch.Tensor) -> None:
        clean_cpu = clean.detach().float().cpu()
        random_cpu = random_scores.detach().float().cpu()
        self.count += int(clean_cpu.numel())
        self.clean_sum += float(clean_cpu.sum().item())
        self.random_sum += float(random_cpu.sum().item())
        self.increase_count += int((random_cpu > clean_cpu).sum().item())

    def result(self) -> dict:
        if self.count == 0:
            return {
                "changed_occurrences": 0,
                "clean_score_mean": None,
                "random_score_mean": None,
                "score_delta_mean": None,
                "paired_positive_fraction": None,
            }
        clean_mean = self.clean_sum / self.count
        random_mean = self.random_sum / self.count
        return {
            "changed_occurrences": self.count,
            "clean_score_mean": clean_mean,
            "random_score_mean": random_mean,
            "score_delta_mean": random_mean - clean_mean,
            "paired_positive_fraction": self.increase_count / self.count,
        }


@dataclass
class _ConditionAccumulator:
    field_names: list[str]
    field_indices: dict[str, int]
    clean_scores: list[np.ndarray] = field(default_factory=list)
    random_scores: list[np.ndarray] = field(default_factory=list)
    sample_score_deltas: list[np.ndarray] = field(default_factory=list)
    sample_logit_deltas: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    clean_probabilities: list[np.ndarray] = field(default_factory=list)
    random_probabilities: list[np.ndarray] = field(default_factory=list)
    fields: dict[str, _FieldAccumulator] = field(init=False)
    changed_occurrences: int = 0
    touched_samples: int = 0
    samples_seen: int = 0

    def __post_init__(self) -> None:
        self.fields = {name: _FieldAccumulator() for name in self.field_names}

    def update(
        self,
        *,
        changed_mask: torch.Tensor,
        clean_score: torch.Tensor,
        random_score: torch.Tensor,
        labels: torch.Tensor,
        clean_logits: torch.Tensor,
        random_logits: torch.Tensor,
    ) -> None:
        changed_mask = changed_mask.to(device=clean_score.device, dtype=torch.bool)
        touched = changed_mask.any(dim=1)
        changed_count = int(changed_mask.sum().item())
        touched_count = int(touched.sum().item())
        self.changed_occurrences += changed_count
        self.touched_samples += touched_count
        self.samples_seen += int(changed_mask.size(0))

        selected_clean = _tensor_to_numpy(clean_score[changed_mask])
        selected_random = _tensor_to_numpy(random_score[changed_mask])
        self.clean_scores.append(selected_clean)
        self.random_scores.append(selected_random)

        changed_float = changed_mask.to(dtype=clean_score.dtype)
        per_sample_count = changed_float.sum(dim=1).clamp_min(1.0)
        clean_mean = (clean_score * changed_float).sum(dim=1) / per_sample_count
        random_mean = (random_score * changed_float).sum(dim=1) / per_sample_count
        self.sample_score_deltas.append(
            _tensor_to_numpy(random_mean[touched] - clean_mean[touched])
        )
        self.sample_logit_deltas.append(
            _tensor_to_numpy((random_logits[touched] - clean_logits[touched]).abs())
        )
        self.labels.append(_tensor_to_numpy(labels))
        self.clean_probabilities.append(_tensor_to_numpy(torch.sigmoid(clean_logits)))
        self.random_probabilities.append(_tensor_to_numpy(torch.sigmoid(random_logits)))
        for name in self.field_names:
            field_index = self.field_indices[name]
            field_mask = changed_mask[:, field_index]
            if bool(field_mask.any()):
                self.fields[name].update(
                    clean_score[field_mask, field_index],
                    random_score[field_mask, field_index],
                )

    def result(self) -> dict:
        clean_scores = np.concatenate(self.clean_scores) if self.clean_scores else np.array([])
        random_scores = (
            np.concatenate(self.random_scores) if self.random_scores else np.array([])
        )
        score_deltas = random_scores - clean_scores
        sample_score_deltas = (
            np.concatenate(self.sample_score_deltas)
            if self.sample_score_deltas
            else np.array([])
        )
        sample_logit_deltas = (
            np.concatenate(self.sample_logit_deltas)
            if self.sample_logit_deltas
            else np.array([])
        )
        labels = np.concatenate(self.labels) if self.labels else np.array([])
        clean_probabilities = (
            np.concatenate(self.clean_probabilities)
            if self.clean_probabilities
            else np.array([])
        )
        random_probabilities = (
            np.concatenate(self.random_probabilities)
            if self.random_probabilities
            else np.array([])
        )
        clean_ctr = _compute_ctr_metrics(labels, clean_probabilities)
        random_ctr = _compute_ctr_metrics(labels, random_probabilities)
        field_results = {name: stats.result() for name, stats in self.fields.items()}
        ranked_fields = sorted(
            self.field_names,
            key=lambda name: (
                field_results[name]["score_delta_mean"]
                if field_results[name]["score_delta_mean"] is not None
                else float("-inf")
            ),
            reverse=True,
        )
        return {
            "samples_seen": self.samples_seen,
            "changed_occurrences": self.changed_occurrences,
            "touched_samples": self.touched_samples,
            "sample_coverage": (
                self.touched_samples / self.samples_seen if self.samples_seen else 0.0
            ),
            "score_name": "contextual_smooth_l1",
            "token_detector_auc": _rank_auc(clean_scores, random_scores),
            "clean_score_mean": (
                float(clean_scores.mean()) if len(clean_scores) else None
            ),
            "random_score_mean": (
                float(random_scores.mean()) if len(random_scores) else None
            ),
            "score_delta_mean": (
                float(score_deltas.mean()) if len(score_deltas) else None
            ),
            "score_delta_median": (
                float(np.median(score_deltas)) if len(score_deltas) else None
            ),
            "paired_positive_fraction": (
                float((score_deltas > 0).mean()) if len(score_deltas) else None
            ),
            "score_damage_pearson": _pearson(
                sample_score_deltas, sample_logit_deltas
            ),
            "clean": clean_ctr,
            "tailrand": random_ctr,
            "auc_drop": clean_ctr["auc"] - random_ctr["auc"],
            "logloss_increase": random_ctr["logloss"] - clean_ctr["logloss"],
            "fields": field_results,
            "ranking_by_score_delta": ranked_fields,
        }


def _replacement_candidates(
    field_counts: dict[int, np.ndarray],
    field_specs: list[dict],
    cutoffs: list[int],
    include_unseen: bool,
) -> dict[int, dict[int, np.ndarray]]:
    candidates: dict[int, dict[int, np.ndarray]] = {}
    for spec in field_specs:
        field_idx = int(spec["index"])
        counts = field_counts[field_idx]
        candidates[field_idx] = {}
        if include_unseen:
            unseen = np.flatnonzero(counts == 0).astype(np.int64)
            candidates[field_idx][0] = unseen[unseen != 0]
        lower = 1
        for upper in cutoffs:
            candidates[field_idx][upper] = np.flatnonzero(
                (counts >= lower) & (counts <= upper)
            ).astype(np.int64)
            lower = upper + 1
    return candidates


def measure_contextual_identity_consistency(
    *,
    model: torch.nn.Module,
    data_loader,
    field_counts: dict[int, np.ndarray],
    field_specs: list[dict],
    device: torch.device,
    cutoffs: list[int],
    replacement_seeds: list[int],
    include_unseen: bool,
    max_batches: int = 0,
) -> dict:
    raw_model = model.module if hasattr(model, "module") else model
    tokenizer = getattr(raw_model, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "contextual_consistency"):
        raise TypeError("contextual identity analysis requires contextual_fallback tokenizer")
    ordered_cutoffs = sorted({int(value) for value in cutoffs})
    ordered_seeds = list(dict.fromkeys(int(value) for value in replacement_seeds))
    categorical_specs = [
        spec
        for spec in field_specs
        if str(spec.get("field_type", spec.get("group"))) == "categorical_id"
    ]
    field_names = [str(spec["name"]) for spec in categorical_specs]
    field_indices = {
        str(spec["name"]): int(spec["index"]) for spec in categorical_specs
    }
    candidates = _replacement_candidates(
        field_counts, categorical_specs, ordered_cutoffs, include_unseen
    )
    conditions = {
        seed: {
            cutoff: _ConditionAccumulator(field_names, field_indices)
            for cutoff in ordered_cutoffs
        }
        for seed in ordered_seeds
    }

    was_training = raw_model.training
    raw_model.eval()
    batches_seen = 0
    samples_seen = 0
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(data_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in cpu_batch.items()
            }
            clean_stages = raw_model.forward_stages(batch)
            clean_consistency = tokenizer.contextual_consistency(
                clean_stages["field_embedding"], batch["sparse"].eq(0)
            )["smooth_l1"]
            batches_seen += 1
            samples_seen += int(batch["sparse"].size(0))

            for replacement_seed in ordered_seeds:
                for cutoff in ordered_cutoffs:
                    randomized_sparse = batch["sparse"].clone()
                    changed = torch.zeros_like(randomized_sparse, dtype=torch.bool)
                    for spec in categorical_specs:
                        field_idx = int(spec["index"])
                        changed_cpu, replacements_cpu = _tail_random_replacements(
                            counts=field_counts[field_idx],
                            ids=cpu_batch["sparse"][:, field_idx],
                            cutoff=cutoff,
                            cutoffs=ordered_cutoffs,
                            candidates_by_band=candidates[field_idx],
                            replacement_seed=replacement_seed,
                            field_idx=field_idx,
                            include_unseen=include_unseen,
                        )
                        if not bool(changed_cpu.any()):
                            continue
                        field_changed = changed_cpu.to(device=device)
                        randomized_sparse[field_changed, field_idx] = replacements_cpu[
                            changed_cpu
                        ].to(device=device)
                        changed[:, field_idx] = field_changed

                    randomized_batch = dict(batch)
                    randomized_batch["sparse"] = randomized_sparse
                    random_stages = raw_model.forward_stages(randomized_batch)
                    random_consistency = tokenizer.contextual_consistency(
                        random_stages["field_embedding"], randomized_sparse.eq(0)
                    )["smooth_l1"]
                    conditions[replacement_seed][cutoff].update(
                        changed_mask=changed,
                        clean_score=clean_consistency,
                        random_score=random_consistency,
                        labels=batch["label"],
                        clean_logits=clean_stages["logits"],
                        random_logits=random_stages["logits"],
                    )
    if was_training:
        raw_model.train()

    results = {
        str(seed): {
            str(cutoff): conditions[seed][cutoff].result()
            for cutoff in ordered_cutoffs
        }
        for seed in ordered_seeds
    }
    return {
        "batches_seen": batches_seen,
        "samples_seen": samples_seen,
        "cutoffs": ordered_cutoffs,
        "replacement_seeds": ordered_seeds,
        "include_unseen": include_unseen,
        "field_names": field_names,
        "results": results,
    }


def _parse_int_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cutoffs", default="5,10,20,50")
    parser.add_argument("--replacement-seeds", default="2021,42")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args, overrides = parser.parse_known_args()

    config = get_config(args.config, overrides)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = create_dataset(config, split="train")
    test_dataset = create_dataset(config, split="valid")
    model = create_model(config).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(
        f"[Identity consistency] checkpoint={checkpoint_path} seed={seed} "
        f"step={checkpoint.get('step')} enabled={model.tokenizer.enabled}"
    )

    analysis_cfg = config["analysis"]["information_funnel"]
    field_specs = resolve_field_specs(config, analysis_cfg)
    field_counts = _load_analysis_field_counts(
        dataset=test_dataset,
        config=config,
        analysis_cfg=analysis_cfg,
        field_specs=field_specs,
    )
    data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=getattr(train_dataset, "collate_fn", None),
    )
    analysis = measure_contextual_identity_consistency(
        model=model,
        data_loader=data_loader,
        field_counts=field_counts,
        field_specs=field_specs,
        device=device,
        cutoffs=_parse_int_list(args.cutoffs),
        replacement_seeds=_parse_int_list(args.replacement_seeds),
        include_unseen=bool(analysis_cfg.get("include_unseen", False)),
        max_batches=args.max_batches,
    )
    report = {
        "schema_version": 1,
        "analysis": "contextual_identity_consistency_tailrand",
        "seed": seed,
        "tokenizer_seed": int(config["model"].get("tokenizer_seed", seed)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "analysis_result": analysis,
    }
    output_dir = Path(args.output_dir or os.environ.get("JOB_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    compact = {
        replacement_seed: {
            cutoff: analysis["results"][str(replacement_seed)][str(cutoff)][
                "token_detector_auc"
            ]
            for cutoff in analysis["cutoffs"]
        }
        for replacement_seed in analysis["replacement_seeds"]
    }
    print(
        f"[Identity consistency] samples={analysis['samples_seen']} "
        f"detector_auc={compact} training_report={output_path}"
    )


if __name__ == "__main__":
    main()
