"""Evaluate contextual fallback only on fields without natural ID=0 support."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recscale.analysis.information_funnel import (
    _ConditionalPredictions,
    _load_analysis_field_counts,
    measure_cumulative_tail_cutoff_sweep,
    resolve_field_specs,
)
from recscale.analysis.missing_id_support import summarize_missing_id_support
from recscale.datasets import create_dataset
from recscale.models import create_model
from recscale.utils.config import get_config


class SchemaGatedContextualFallbackModel(nn.Module):
    """Inject a fallback mask derived only from field identity and ID=0."""

    def __init__(self, model: nn.Module, eligible_sparse_indices: list[int]) -> None:
        super().__init__()
        self.model = model
        self.eligible_sparse_indices = tuple(sorted(set(eligible_sparse_indices)))

        raw_model = model.module if hasattr(model, "module") else model
        if getattr(raw_model, "tokenizer_type", None) != "contextual_fallback":
            raise TypeError("schema gate requires tokenizer_type=contextual_fallback")
        if not bool(getattr(raw_model.tokenizer, "enabled", False)):
            raise ValueError("schema gate requires contextual fallback enabled")

    def replacement_mask(self, sparse: torch.Tensor) -> torch.Tensor:
        if sparse.ndim != 2:
            raise ValueError(f"sparse IDs must be rank 2, got shape={tuple(sparse.shape)}")
        eligible = torch.zeros(
            sparse.size(1), dtype=torch.bool, device=sparse.device
        )
        for field_idx in self.eligible_sparse_indices:
            if field_idx < 0 or field_idx >= sparse.size(1):
                raise ValueError(
                    f"eligible sparse field {field_idx} is outside width {sparse.size(1)}"
                )
            eligible[field_idx] = True
        return sparse.eq(0) & eligible.unsqueeze(0)

    def forward_stages(self, batch: dict) -> dict[str, torch.Tensor | None]:
        gated_batch = dict(batch)
        gated_batch["contextual_fallback_mask"] = self.replacement_mask(batch["sparse"])
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        return raw_model.forward_stages(gated_batch)


def _paired_clean_result(predictions: _ConditionalPredictions) -> dict:
    result = predictions.result(min_samples=0, min_positives=1, min_negatives=1)
    return {
        "num_samples": result["num_samples"],
        "all_off": result["clean"],
        "schema_gated": result["masked"],
        "auc_delta_schema_minus_all_off": (
            -result["auc_drop"] if result["auc_drop"] is not None else None
        ),
        "logloss_delta_schema_minus_all_off": result["logloss_increase"],
        "auc_valid": result["auc_valid"],
        "auc_invalid_reason": result["auc_invalid_reason"],
    }


def measure_clean_invariance(
    *,
    model: nn.Module,
    gated_model: SchemaGatedContextualFallbackModel,
    data_loader,
    field_names: list[str],
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    """Compare schema-gated clean inference with fallback fully disabled."""
    raw_model = model.module if hasattr(model, "module") else model
    predictions = _ConditionalPredictions()
    field_activations = {name: 0 for name in field_names}
    activation_occurrences = 0
    activation_samples = 0
    samples_seen = 0
    batches_seen = 0
    max_abs_logit_delta = 0.0

    was_training = raw_model.training
    raw_model.eval()
    gated_model.eval()
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
            sparse = batch["sparse"]
            replacement_mask = gated_model.replacement_mask(sparse)
            all_off_batch = dict(batch)
            all_off_batch["contextual_fallback_mask"] = torch.zeros_like(
                replacement_mask
            )
            all_off = raw_model.forward_stages(all_off_batch)
            schema_gated = gated_model.forward_stages(batch)
            all_samples = torch.ones(
                sparse.size(0), dtype=torch.bool, device=device
            )
            predictions.update(
                labels=batch["label"],
                clean_logits=all_off["logits"],
                masked_logits=schema_gated["logits"],
                mask=all_samples,
            )

            activation_occurrences += int(replacement_mask.sum().item())
            activation_samples += int(replacement_mask.any(dim=1).sum().item())
            for field_idx, name in enumerate(field_names):
                field_activations[name] += int(
                    replacement_mask[:, field_idx].sum().item()
                )
            if sparse.size(0) > 0:
                max_abs_logit_delta = max(
                    max_abs_logit_delta,
                    float(
                        (all_off["logits"] - schema_gated["logits"])
                        .abs()
                        .max()
                        .item()
                    ),
                )
            samples_seen += int(sparse.size(0))
            batches_seen += 1

    if was_training:
        raw_model.train()

    return {
        "batches_seen": batches_seen,
        "samples_seen": samples_seen,
        "activation_occurrences": activation_occurrences,
        "activation_samples": activation_samples,
        "activation_sample_coverage": (
            activation_samples / samples_seen if samples_seen else 0.0
        ),
        "max_abs_logit_delta": max_abs_logit_delta,
        "field_activations": field_activations,
        "performance": _paired_clean_result(predictions),
    }


def _parse_cutoffs(raw: object) -> list[int]:
    if isinstance(raw, str):
        raw = [value.strip() for value in raw.split(",") if value.strip()]
    return [int(value) for value in raw]


def compare_tailrand_sweeps(schema_sweep: dict, all_off_sweep: dict) -> dict:
    if schema_sweep["cutoffs"] != all_off_sweep["cutoffs"]:
        raise ValueError("schema-gated and all-off sweeps must use identical cutoffs")
    results = {}
    for cutoff in schema_sweep["cutoffs"]:
        key = str(cutoff)
        schema = schema_sweep["results"][key]["global_performance"]
        all_off = all_off_sweep["results"][key]["global_performance"]
        results[key] = {
            "schema_tailrand": schema["masked"],
            "all_off_tailrand": all_off["masked"],
            "auc_delta_schema_minus_all_off": (
                schema["masked"]["auc"] - all_off["masked"]["auc"]
            ),
            "logloss_delta_schema_minus_all_off": (
                schema["masked"]["logloss"] - all_off["masked"]["logloss"]
            ),
            "auc_drop_delta_schema_minus_all_off": (
                schema["auc_drop"] - all_off["auc_drop"]
            ),
        }
    return {
        "cutoffs": schema_sweep["cutoffs"],
        "results": results,
        "all_auc_deltas_zero": all(
            result["auc_delta_schema_minus_all_off"] == 0.0
            for result in results.values()
        ),
        "all_logloss_deltas_zero": all(
            result["logloss_delta_schema_minus_all_off"] == 0.0
            for result in results.values()
        ),
        "all_auc_drop_deltas_zero": all(
            result["auc_drop_delta_schema_minus_all_off"] == 0.0
            for result in results.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--intervention", choices=("mask", "tail_random"), default="mask"
    )
    parser.add_argument("--replacement-seed", type=int, default=2021)
    args, overrides = parser.parse_known_args()

    config = get_config(args.config, overrides)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    create_dataset(config, split="train")
    test_dataset = create_dataset(config, split="valid")
    collate_fn = getattr(test_dataset, "collate_fn", None)
    data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=collate_fn,
    )

    model = create_model(config).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    analysis_cfg = config["analysis"]["information_funnel"]
    field_specs = resolve_field_specs(config, analysis_cfg)
    categorical_specs = [
        spec
        for spec in field_specs
        if str(spec.get("field_type", spec.get("group"))) == "categorical_id"
    ]
    categorical_indices = [int(spec["index"]) for spec in categorical_specs]
    categorical_names = [str(spec["name"]) for spec in categorical_specs]
    missing_support = summarize_missing_id_support(
        analysis_cfg["frequency_cache_path"], target_fields=categorical_indices
    )
    eligible_indices = [
        int(field["field_index"])
        for field in missing_support["fields"]
        if int(field["missing_id_count"]) == 0
    ]
    excluded_indices = sorted(set(categorical_indices) - set(eligible_indices))
    name_by_index = {
        int(spec["index"]): str(spec["name"]) for spec in categorical_specs
    }
    eligible_fields = [name_by_index[index] for index in eligible_indices]
    excluded_fields = [name_by_index[index] for index in excluded_indices]
    if not eligible_indices or not excluded_indices:
        raise ValueError(
            "schema gate requires both zero-missing and naturally-missing categorical fields"
        )

    gated_model = SchemaGatedContextualFallbackModel(model, eligible_indices).to(device)
    max_batches = (
        int(args.max_batches)
        if args.max_batches is not None
        else int(analysis_cfg.get("max_batches", 0))
    )
    clean_invariance = measure_clean_invariance(
        model=model,
        gated_model=gated_model,
        data_loader=data_loader,
        field_names=categorical_names,
        device=device,
        max_batches=max_batches,
    )

    field_counts = _load_analysis_field_counts(
        dataset=test_dataset,
        config=config,
        analysis_cfg=analysis_cfg,
        field_specs=field_specs,
    )
    cutoffs = _parse_cutoffs(analysis_cfg.get("tail_cutoffs", [5, 10, 20, 50]))
    sweep = measure_cumulative_tail_cutoff_sweep(
        model=gated_model,
        data_loader=data_loader,
        field_counts=field_counts,
        field_specs=field_specs,
        device=device,
        max_batches=max_batches,
        cutoffs=cutoffs,
        intervention=args.intervention,
        replacement_seed=int(args.replacement_seed),
        include_unseen=bool(analysis_cfg.get("include_unseen", False)),
        conditional_min_samples=int(analysis_cfg.get("conditional_min_samples", 0)),
        conditional_min_positives=int(
            analysis_cfg.get("conditional_min_positives", 1)
        ),
        conditional_min_negatives=int(
            analysis_cfg.get("conditional_min_negatives", 1)
        ),
    )
    for cutoff in sweep["cutoffs"]:
        result = sweep["results"][str(cutoff)]
        result["fallback_activation_occurrences"] = (
            sum(
                result["fields"][name]["eval_occurrences"]
                for name in eligible_fields
            )
            if args.intervention == "mask"
            else 0
        )
        result["excluded_masked_occurrences"] = sum(
            result["fields"][name]["eval_occurrences"] for name in excluded_fields
        )

    tailrand_all_off_comparison = None
    if args.intervention == "tail_random":
        all_off_model = SchemaGatedContextualFallbackModel(model, []).to(device)
        all_off_sweep = measure_cumulative_tail_cutoff_sweep(
            model=all_off_model,
            data_loader=data_loader,
            field_counts=field_counts,
            field_specs=field_specs,
            device=device,
            max_batches=max_batches,
            cutoffs=cutoffs,
            intervention="tail_random",
            replacement_seed=int(args.replacement_seed),
            include_unseen=bool(analysis_cfg.get("include_unseen", False)),
            conditional_min_samples=int(
                analysis_cfg.get("conditional_min_samples", 0)
            ),
            conditional_min_positives=int(
                analysis_cfg.get("conditional_min_positives", 1)
            ),
            conditional_min_negatives=int(
                analysis_cfg.get("conditional_min_negatives", 1)
            ),
        )
        tailrand_all_off_comparison = compare_tailrand_sweeps(
            sweep, all_off_sweep
        )

    report = {
        "schema_version": 1,
        "analysis": (
            "schema_gated_contextual_fallback_tailmask"
            if args.intervention == "mask"
            else "schema_gated_contextual_fallback_tailrand"
        ),
        "seed": seed,
        "tokenizer_seed": int(config["model"].get("tokenizer_seed", seed)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_metrics": checkpoint.get("metrics") or {},
        "eligibility_rule": "full_train_missing_id_count_eq_zero",
        "frequency_is_model_input": False,
        "intervention": args.intervention,
        "replacement_seed": (
            int(args.replacement_seed)
            if args.intervention == "tail_random"
            else None
        ),
        "eligible_fields": eligible_fields,
        "excluded_natural_missing_fields": excluded_fields,
        "missing_support": missing_support,
        "clean_invariance": clean_invariance,
        "sweep": sweep,
        "tailrand_all_off_comparison": tailrand_all_off_comparison,
    }
    output_dir = Path(args.output_dir or os.environ.get("JOB_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[R5-S] checkpoint={checkpoint_path} seed={seed} "
        f"intervention={args.intervention} "
        f"eligible={eligible_fields} excluded={excluded_fields}"
    )
    print(
        f"[R5-S] clean samples={clean_invariance['samples_seen']} "
        f"activations={clean_invariance['activation_occurrences']} "
        f"max_logit_delta={clean_invariance['max_abs_logit_delta']:.9g}"
    )
    for cutoff in sweep["cutoffs"]:
        result = sweep["results"][str(cutoff)]
        performance = result["global_performance"]
        print(
            f"[R5-S] cutoff={cutoff} "
            f"coverage={result['sample_union_coverage_all']:.6f} "
            f"fallback_occurrences={result['fallback_activation_occurrences']} "
            f"auc_drop={performance['auc_drop']:.8f} "
            f"logloss_inc={performance['logloss_increase']:.8f}"
        )
    if tailrand_all_off_comparison is not None:
        print(
            "[R5-S] tailrand all-off invariance "
            f"auc={tailrand_all_off_comparison['all_auc_deltas_zero']} "
            f"logloss={tailrand_all_off_comparison['all_logloss_deltas_zero']} "
            f"auc_drop={tailrand_all_off_comparison['all_auc_drop_deltas_zero']}"
        )
    print(f"[R5-S] training_report={output_path}")


if __name__ == "__main__":
    main()
