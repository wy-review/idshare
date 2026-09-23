"""Counterfactual clean attribution for contextual missing-ID fallback."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recscale.analysis.information_funnel import _ConditionalPredictions
from recscale.datasets import create_dataset
from recscale.models import create_model
from recscale.utils.config import get_config


DEFAULT_FIELDS = ("C1", "C10", "C11", "C14", "C16", "C17", "C20", "C21", "C22", "C23")


def _paired_result(
    predictions: _ConditionalPredictions,
    *,
    baseline_name: str,
    counterfactual_name: str,
) -> dict:
    result = predictions.result(min_samples=0, min_positives=1, min_negatives=1)
    return {
        "num_samples": result["num_samples"],
        baseline_name: result["clean"],
        counterfactual_name: result["masked"],
        f"auc_delta_{counterfactual_name}_minus_{baseline_name}": (
            -result["auc_drop"] if result["auc_drop"] is not None else None
        ),
        f"logloss_delta_{counterfactual_name}_minus_{baseline_name}": result[
            "logloss_increase"
        ],
        "auc_valid": result["auc_valid"],
        "auc_invalid_reason": result["auc_invalid_reason"],
    }


def measure_contextual_fallback_clean_attribution(
    *,
    model: torch.nn.Module,
    data_loader,
    field_names: list[str],
    target_fields: list[str] | tuple[str, ...],
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    """Measure clean impact of disabling fallback globally or one field at a time."""
    raw_model = model.module if hasattr(model, "module") else model
    if getattr(raw_model, "tokenizer_type", None) != "contextual_fallback":
        raise TypeError("contextual fallback attribution requires tokenizer_type=contextual_fallback")
    if not bool(getattr(raw_model.tokenizer, "enabled", False)):
        raise ValueError("contextual fallback attribution requires fallback enabled")

    field_to_index = {name: index for index, name in enumerate(field_names)}
    unknown = [name for name in target_fields if name not in field_to_index]
    if unknown:
        raise ValueError(f"unknown attribution fields: {unknown}")
    target_fields = list(dict.fromkeys(target_fields))

    all_off_global = _ConditionalPredictions()
    all_off_any_missing = _ConditionalPredictions()
    all_off_no_missing = _ConditionalPredictions()
    field_global = {name: _ConditionalPredictions() for name in target_fields}
    field_conditional = {name: _ConditionalPredictions() for name in target_fields}
    field_missing_occurrences = {name: 0 for name in target_fields}
    samples_seen = 0
    batches_seen = 0
    no_missing_max_logit_delta = 0.0

    was_training = raw_model.training
    raw_model.eval()
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
            natural_missing = sparse.eq(0)
            any_missing = natural_missing.any(dim=1)
            no_missing = ~any_missing
            all_samples = torch.ones(sparse.size(0), dtype=torch.bool, device=device)

            all_on_logits = raw_model.forward_stages(batch)["logits"]
            all_off_batch = dict(batch)
            all_off_batch["contextual_fallback_mask"] = torch.zeros_like(
                natural_missing
            )
            all_off_logits = raw_model.forward_stages(all_off_batch)["logits"]

            all_off_global.update(
                labels=batch["label"],
                clean_logits=all_on_logits,
                masked_logits=all_off_logits,
                mask=all_samples,
            )
            all_off_any_missing.update(
                labels=batch["label"],
                clean_logits=all_on_logits,
                masked_logits=all_off_logits,
                mask=any_missing,
            )
            all_off_no_missing.update(
                labels=batch["label"],
                clean_logits=all_on_logits,
                masked_logits=all_off_logits,
                mask=no_missing,
            )
            if bool(no_missing.any()):
                no_missing_max_logit_delta = max(
                    no_missing_max_logit_delta,
                    float(
                        (all_on_logits[no_missing] - all_off_logits[no_missing])
                        .abs()
                        .max()
                        .item()
                    ),
                )

            for name in target_fields:
                field_idx = field_to_index[name]
                field_missing = natural_missing[:, field_idx]
                field_missing_occurrences[name] += int(field_missing.sum().item())
                leave_one_out_mask = natural_missing.clone()
                leave_one_out_mask[:, field_idx] = False
                leave_one_out_batch = dict(batch)
                leave_one_out_batch["contextual_fallback_mask"] = leave_one_out_mask
                leave_one_out_logits = raw_model.forward_stages(leave_one_out_batch)[
                    "logits"
                ]
                field_global[name].update(
                    labels=batch["label"],
                    clean_logits=all_on_logits,
                    masked_logits=leave_one_out_logits,
                    mask=all_samples,
                )
                field_conditional[name].update(
                    labels=batch["label"],
                    clean_logits=all_on_logits,
                    masked_logits=leave_one_out_logits,
                    mask=field_missing,
                )

            samples_seen += int(sparse.size(0))
            batches_seen += 1

    if was_training:
        raw_model.train()

    fields = {}
    for name in target_fields:
        fields[name] = {
            "index": field_to_index[name],
            "missing_occurrences": field_missing_occurrences[name],
            "sample_coverage": (
                field_missing_occurrences[name] / samples_seen if samples_seen else 0.0
            ),
            "global": _paired_result(
                field_global[name],
                baseline_name="all_on",
                counterfactual_name="leave_one_out",
            ),
            "conditional_on_field_missing": _paired_result(
                field_conditional[name],
                baseline_name="all_on",
                counterfactual_name="leave_one_out",
            ),
        }

    ranking = sorted(
        target_fields,
        key=lambda name: (
            fields[name]["global"]["auc_delta_leave_one_out_minus_all_on"]
            if fields[name]["global"]["auc_delta_leave_one_out_minus_all_on"]
            is not None
            else float("-inf")
        ),
        reverse=True,
    )
    return {
        "batches_seen": batches_seen,
        "samples_seen": samples_seen,
        "target_fields": target_fields,
        "global_all_off": _paired_result(
            all_off_global,
            baseline_name="all_on",
            counterfactual_name="all_off",
        ),
        "any_natural_missing": _paired_result(
            all_off_any_missing,
            baseline_name="all_on",
            counterfactual_name="all_off",
        ),
        "no_natural_missing": _paired_result(
            all_off_no_missing,
            baseline_name="all_on",
            counterfactual_name="all_off",
        ),
        "no_missing_max_logit_delta": no_missing_max_logit_delta,
        "fields": fields,
        "ranking_by_clean_auc_recovery": ranking,
    }


def _parse_fields(raw: str) -> list[str]:
    fields = [value.strip() for value in raw.split(",") if value.strip()]
    if not fields:
        raise ValueError("--fields must contain at least one field")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
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
    test_split = "test" if "test" in config["dataset"].get("split", {}) else "valid"
    test_dataset = create_dataset(config, split=test_split)
    model = create_model(config).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"R3 checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(
        f"[R3-D] loaded checkpoint={checkpoint_path} seed={seed} "
        f"checkpoint_step={checkpoint.get('step')}"
    )

    batch_size = int(config["training"]["batch_size"]) * 2
    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=getattr(train_dataset, "collate_fn", None),
    )
    field_names = list(config["dataset"].get("sparse_cols") or [])
    attribution = measure_contextual_fallback_clean_attribution(
        model=model,
        data_loader=data_loader,
        field_names=field_names,
        target_fields=_parse_fields(args.fields),
        device=device,
        max_batches=args.max_batches,
    )
    report = {
        "schema_version": 1,
        "analysis": "contextual_fallback_clean_attribution",
        "seed": seed,
        "tokenizer_seed": int(config["model"].get("tokenizer_seed", seed)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "attribution": attribution,
    }
    output_dir = Path(args.output_dir or os.environ.get("JOB_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[R3-D] samples={attribution['samples_seen']} "
        f"ranking={attribution['ranking_by_clean_auc_recovery']} "
        f"training_report={output_path}"
    )


if __name__ == "__main__":
    main()
