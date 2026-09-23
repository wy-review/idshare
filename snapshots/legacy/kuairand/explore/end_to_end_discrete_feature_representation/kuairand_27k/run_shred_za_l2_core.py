#!/usr/bin/env python3
"""Run one frozen SHRED full-table-L2 causal validation job."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recscale.analysis.zero_anchor_frequency_dynamics import (
    ZeroAnchorFrequencyDynamicsAudit,
    analyze_exact_frequency_geometry,
    evaluate_validation_frequency_auc,
    evaluate_validation_tailmask_auc,
)
from recscale.datasets.deterministic_partition import ModuloPartitionDataset
from recscale.datasets.kuairand27k_k1 import KuaiRand27KK1MMapDataset
from recscale.models.s2drec import S2DRecModel
from recscale.trainer import Trainer

try:
    from .build_shred_za_l2_core_matrix import (
        IDENTITY_FIELDS,
        PROTOCOL,
        assert_core_review_gate,
    )
except ImportError:
    from build_shred_za_l2_core_matrix import (
        IDENTITY_FIELDS,
        PROTOCOL,
        assert_core_review_gate,
    )


HEALTH_THRESHOLDS = {
    "min_effective_nonzero_centers": 2,
    "min_perplexity": 2.0,
    "max_zero_fraction": 0.95,
    "max_center_fraction": 0.95,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_order_fingerprint(
    dataset: KuaiRand27KK1MMapDataset,
    *,
    rows: int,
) -> str:
    """Fingerprint the first rows consumed by the frozen data pipeline."""
    digest = hashlib.sha256()
    for index in range(min(int(rows), len(dataset))):
        sample = dataset[index]
        digest.update(np.ascontiguousarray(sample["sparse"]).tobytes())
        digest.update(np.float32(sample["label"]).tobytes())
    return digest.hexdigest()


def _configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_validation_partitions(
    matrix: dict,
    validation_dataset,
) -> tuple[object, object, dict, dict]:
    """Resolve an optional frozen selection/report split."""
    spec = matrix.get("validation_partition")
    if spec is None:
        report = {
            "enabled": False,
            "selection": {
                "partition_name": "validation_full",
                "rows": len(validation_dataset),
            },
            "report": {
                "partition_name": "validation_full",
                "rows": len(validation_dataset),
            },
        }
        return validation_dataset, validation_dataset, report, {
            "full_validation_development_mode": True,
        }

    if spec.get("algorithm") != "base_index_modulo_v1":
        raise RuntimeError("unsupported frozen validation partition algorithm")
    modulus = int(spec["modulus"])
    selection_spec = spec["selection"]
    report_spec = spec["report"]
    selection = ModuloPartitionDataset(
        validation_dataset,
        modulus=modulus,
        remainder=int(selection_spec["remainder"]),
        base_split_sha256=matrix["expected_split_sha256"],
        partition_name=str(selection_spec["partition_name"]),
    )
    report = ModuloPartitionDataset(
        validation_dataset,
        modulus=modulus,
        remainder=int(report_spec["remainder"]),
        base_split_sha256=matrix["expected_split_sha256"],
        partition_name=str(report_spec["partition_name"]),
    )
    selection_report = selection.report()
    report_report = report.report()
    checks = {
        "selection_membership_sha256_matches": (
            selection.membership_sha256
            == selection_spec["expected_membership_sha256"]
        ),
        "report_membership_sha256_matches": (
            report.membership_sha256
            == report_spec["expected_membership_sha256"]
        ),
        "selection_rows_match": (
            len(selection) == int(selection_spec["expected_rows"])
        ),
        "report_rows_match": (
            len(report) == int(report_spec["expected_rows"])
        ),
        "partition_rows_close": (
            len(selection) + len(report) == len(validation_dataset)
        ),
        "partition_remainders_are_disjoint": (
            selection.remainder != report.remainder
        ),
        "partition_base_split_matches": (
            selection.base_split_sha256
            == report.base_split_sha256
            == matrix["expected_split_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"frozen validation partition checks failed: {checks}"
        )
    return selection, report, {
        "enabled": True,
        "selection_role": "early_stop_and_checkpoint_selection_only",
        "report_role": "formal_effect_readout_only",
        "selection": selection_report,
        "report": report_report,
    }, checks


class TrainingExecutionAudit:
    """Compose the dynamics observer with fail-closed schedule snapshots."""

    def __init__(self, trainer: Trainer, dynamics, *, steps_per_epoch: int):
        self.trainer = trainer
        self.dynamics = dynamics
        self.steps_per_epoch = int(steps_per_epoch)
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        raw_model = (
            trainer.model.module
            if hasattr(trainer.model, "module")
            else trainer.model
        )
        self.trace: list[dict] = [self._snapshot(raw_model, completed_step=0)]

    def _snapshot(self, raw_model, *, completed_step: int) -> dict:
        quantizer = raw_model.encoder.zero_anchor_identity_quantizer
        shared_transform = (
            {
                "enabled": False,
                "mode": "not_applicable",
                "fields": [],
                "all_checks_pass": True,
            }
            if quantizer is None
            or not hasattr(
                quantizer, "get_shared_codebook_transform_report"
            )
            else quantizer.get_shared_codebook_transform_report()
        )
        return {
            "completed_step": int(completed_step),
            "epoch_boundary": (
                completed_step > 0
                and completed_step % self.steps_per_epoch == 0
            ),
            "full_table_l2_multiplier": float(
                self.trainer.zero_anchor_full_table_l2_current_multiplier
            ),
            "full_table_l2_gradient_report_present": (
                self.trainer.zero_anchor_full_table_l2_last_gradient_report
                is not None
            ),
            "quantizer_regularization_multiplier": (
                None
                if quantizer is None
                else float(quantizer.current_regularization_multiplier)
            ),
            "quantizer_codebook_multiplier": (
                None
                if quantizer is None
                else float(quantizer.current_codebook_multiplier)
            ),
            "temperature": (
                None
                if quantizer is None
                else float(quantizer.current_temperature)
            ),
            "shared_codebook_transform": shared_transform,
        }

    @torch.no_grad()
    def __call__(
        self,
        raw_model,
        *,
        completed_step: int,
        total_steps: int,
    ) -> None:
        self.dynamics(
            raw_model,
            completed_step=completed_step,
            total_steps=total_steps,
        )
        if completed_step != 1 and completed_step % self.steps_per_epoch != 0:
            return
        snapshot = self._snapshot(raw_model, completed_step=completed_step)
        self.trace.append(snapshot)
        transform = snapshot["shared_codebook_transform"]
        if (
            snapshot["epoch_boundary"]
            and transform["enabled"]
            and not transform["all_checks_pass"]
        ):
            raise RuntimeError(
                "shared codebook transform singular-ratio tripwire failed "
                f"at completed_step={completed_step}: {transform}"
            )

    def report(self) -> dict:
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "trace": list(self.trace),
        }


def resolve_validation_tailmask_spec(matrix: dict) -> dict:
    interventions = matrix["analysis_only_interventions"]
    supported_keys = (
        "phase_a_validation_tailmask",
        "validation_tailmask",
    )
    present = [
        key for key in supported_keys if key in interventions
    ]
    if not present:
        raise KeyError(
            "analysis_only_interventions must define one of "
            f"{supported_keys}"
        )
    if len(present) == 2 and (
        interventions[present[0]] != interventions[present[1]]
    ):
        raise RuntimeError(
            "conflicting validation Tailmask specifications: "
            f"{present[0]} != {present[1]}"
        )
    return interventions[present[0]]


def resolve_expected_parameter_count(matrix: dict, arm: dict) -> int:
    sources: list[tuple[str, object]] = []
    if "expected_parameter_count" in arm:
        sources.append(
            ("arm.expected_parameter_count", arm["expected_parameter_count"])
        )

    fixed_model = matrix.get("fixed_model", {})
    family_counts = fixed_model.get("expected_parameter_count_by_family")
    if family_counts is not None:
        if not isinstance(family_counts, dict):
            raise TypeError(
                "fixed_model.expected_parameter_count_by_family must be a "
                "mapping"
            )
        family = arm["family"]
        if family not in family_counts:
            raise KeyError(
                "expected_parameter_count_by_family does not define family "
                f"{family!r}"
            )
        sources.append(
            (
                f"fixed_model.expected_parameter_count_by_family[{family!r}]",
                family_counts[family],
            )
        )

    if not sources:
        raise KeyError(
            "expected parameter count must be defined by either the arm or "
            "fixed_model.expected_parameter_count_by_family"
        )

    resolved = []
    for source, value in sources:
        count = int(value)
        if count <= 0:
            raise ValueError(f"{source} must be positive, got {count}")
        resolved.append((source, count))
    if len({count for _, count in resolved}) != 1:
        raise RuntimeError(
            "conflicting expected parameter counts: "
            + ", ".join(
                f"{source}={count}" for source, count in resolved
            )
        )
    return resolved[0][1]


def _canonical_tensor_fingerprint(payload: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        tensor = payload[name].detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        byte_count = tensor.numel() * tensor.element_size()
        digest.update(byte_count.to_bytes(8, "big"))
        flat = tensor.reshape(-1)
        chunk_elements = max(1, (4 * 1024 * 1024) // tensor.element_size())
        for start in range(0, flat.numel(), chunk_elements):
            raw = (
                flat[start : start + chunk_elements]
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            digest.update(raw)
    return digest.hexdigest()


def selected_initialization_fingerprint(model: S2DRecModel) -> str:
    """Fingerprint all mechanism parameters plus sparse-table row samples."""
    payload = {}
    for name, tensor in model.state_dict().items():
        if "sparse_arch.embeddings" not in name:
            payload[name] = tensor
    for field_index, embedding in enumerate(
        model.encoder.sparse_arch.embeddings
    ):
        payload[f"_embedding_sample_{field_index}"] = embedding.weight[
            : min(8, embedding.weight.size(0))
        ]
    return _canonical_tensor_fingerprint(payload)


def identity_initialization_fingerprint(
    model: S2DRecModel,
    *,
    identity_fields: tuple[str, ...] = IDENTITY_FIELDS,
) -> str:
    """Stream a full-table fingerprint for every selected identity field."""
    field_names = tuple(
        str(name)
        for name in model.config.get("dataset", {}).get("sparse_cols", [])
    )
    payload = {
        f"identity_embedding_full_{field}": (
            model.encoder.sparse_arch.embeddings[
                field_names.index(field)
            ].weight
        )
        for field in identity_fields
    }
    return _canonical_tensor_fingerprint(payload)


def common_initialization_fingerprint(
    model: S2DRecModel,
    *,
    identity_fields: tuple[str, ...] = IDENTITY_FIELDS,
) -> str:
    """Fingerprint only parameters that must match across model families.

    Product deliberately zero-initializes the selected raw identity tables,
    while Continuous uses the configured sparse initialization. Those
    mechanism-specific tables and the Product-only quantizer are excluded.
    Every other sparse table and every common downstream parameter must match.
    """
    field_names = tuple(
        str(name)
        for name in model.config.get("dataset", {}).get("sparse_cols", [])
    )
    identity_indices = {
        field_names.index(field)
        for field in identity_fields
        if field in field_names
    }
    payload = {}
    for name, tensor in model.state_dict().items():
        if "zero_anchor_identity_quantizer" in name:
            continue
        if name.endswith("assignment_counts"):
            continue
        if "sparse_arch.embeddings" not in name:
            payload[name] = tensor.detach().cpu()
    for field_index, embedding in enumerate(
        model.encoder.sparse_arch.embeddings
    ):
        if field_index in identity_indices:
            continue
        payload[f"_embedding_full_{field_index}"] = (
            embedding.weight.detach().cpu()
        )
    return _canonical_tensor_fingerprint(payload)


def _resolve_run(matrix: dict, run_key: str) -> tuple[dict, dict]:
    runs = {run["run_key"]: run for run in matrix["runs"]}
    if run_key not in runs:
        raise KeyError(f"unknown SHRED L2 core run {run_key!r}")
    run = runs[run_key]
    return run, matrix["arms"][run["arm_key"]]


def build_core_config(
    manifest_path: Path,
    *,
    run: dict,
    arm: dict,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    coefficient = float(arm["full_table_l2_coefficient"])
    config = {
        "seed": int(run["seed"]),
        "dataset": {
            "name": "kuairand27k_shred_l2_core_validation",
            "type": "kuairand27k_k1_mmap",
            "processed_manifest": str(manifest_path),
            "sparse_cols": list(manifest["field_names"]),
            "cardinalities": list(manifest["cardinalities"]),
            "dense_cols": [],
            "verify_processed_sha256": True,
            "verify_processed_sha256_splits": ["train", "val"],
            "split": {"train": "train", "validation": "val"},
        },
        "model": {
            "name": "s2drec",
            "embedding_dim": 16,
            "embedding_init": "uniform",
            "tokenizer_type": "per_field_proj",
            "per_field_proj_mode": "split",
            "num_tokens": 37,
            "d_model": 74,
            "backbone_type": "tokenmixer_v3",
            "num_mixer_layers": 2,
            "ffn_dim": 256,
            "dropout": 0.0,
            "head_hidden_units": [512, 256],
            "head_dropout": 0.0,
            "tokenizer_seed": 2021,
            "sparse_embedding_zero_init_rows_by_field": {
                field: [1] for field in IDENTITY_FIELDS
            },
        },
        "training": {
            "epochs": int(arm.get("max_epochs", 1)),
            "batch_size": 4096,
            "shuffle": False,
            "lr": 0.002,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "data_seed": 20260724,
            "deterministic": True,
            "optimizer_foreach": False,
            "grad_clip": 1.0,
            "use_amp": False,
            "num_workers": 4,
            "log_every": 2000,
            "eval_every": int(arm.get("eval_every", 0)),
            "save_best_checkpoint": bool(
                arm.get("save_best_checkpoint", False)
            ),
            "early_stop_patience": int(
                arm.get("early_stop_patience", 0)
            ),
            "early_stop_min_epochs": int(
                arm.get("early_stop_min_epochs", 1)
            ),
            "early_stop_min_delta": float(
                arm.get("early_stop_min_delta", 0.0)
            ),
            "checkpoint_selection_tolerance": float(
                arm.get("checkpoint_selection_tolerance", 0.0)
            ),
            "temperature_anneal_epochs": int(
                arm.get("temperature_anneal_epochs", 1)
            ),
            "suppress_effect_metric_logs": bool(
                arm.get("suppress_effect_metric_logs", False)
            ),
            "save_dir": (
                f"/tmp/kuairand27k_shred_l2_core_{run['run_key']}"
            ),
            "zero_anchor_full_table_l2": {
                "enabled": coefficient > 0.0,
                "coefficient": coefficient,
                "first_regularized_row": int(
                    arm["first_regularized_row"]
                ),
                "application_order": "after_global_clip",
            },
        },
        "distributed": {"enabled": False, "backend": "nccl"},
    }
    if coefficient > 0.0:
        config["training"]["zero_anchor_full_table_l2"].update(
            {
                "identity_fields": list(IDENTITY_FIELDS),
                "release_fraction": float(arm["l2_release_fraction"]),
                "ramp_fraction": float(arm["l2_ramp_fraction"]),
            }
        )
        if "full_table_l2_target" in arm:
            config["training"]["zero_anchor_full_table_l2"]["target"] = str(
                arm["full_table_l2_target"]
            )
    if arm["zero_anchor"]:
        config["model"]["zero_anchor_identity_quantization"] = {
            "enabled": True,
            "identity_fields": list(IDENTITY_FIELDS),
            "codebook_size": int(arm["codebook_size"]),
            "num_subspaces": int(arm["num_subspaces"]),
            "num_residual_levels": int(arm["num_residual_levels"]),
            "margin": float(arm.get("margin", 0.1)),
            "temperature_start": float(arm.get("temperature_start", 1.0)),
            "temperature_end": float(arm.get("temperature_end", 0.3)),
            "code_init_radius": float(arm.get("code_init_radius", 0.2)),
            "distance_backend": "gemm",
            "distance_row_chunk_size": int(
                arm.get("distance_row_chunk_size") or 0
            ),
            "compact_distance_outputs": bool(
                arm.get("compact_distance_outputs", False)
            ),
            "sparse_batch_diagnostics": bool(
                arm.get("compact_distance_outputs", False)
            ),
            "defer_cumulative_diagnostics": bool(
                arm.get("compact_distance_outputs", False)
            ),
            "codebook_loss_weight": float(
                arm.get("codebook_loss_weight", 1.0)
            ),
            "codebook_transform_mode": str(
                arm.get("codebook_transform_mode", "none")
            ),
            "task_codebook_gradient_mode": str(
                arm.get(
                    "task_codebook_gradient_mode",
                    "legacy_hard_plus_soft",
                )
            ),
            "private_row_initialization_mode": str(
                arm.get("private_row_initialization_mode", "zero")
            ),
            "assignment_stability_mode": str(
                arm.get("assignment_stability_mode", "free_nearest")
            ),
            "assignment_freeze_fraction": float(
                arm.get("assignment_freeze_fraction", 0.0)
            ),
            "assignment_switch_relative_improvement": float(
                arm.get(
                    "assignment_switch_relative_improvement",
                    0.0,
                )
            ),
            "codebook_release_fraction": float(
                arm.get("codebook_release_fraction", 0.0)
            ),
            "codebook_ramp_fraction": float(
                arm.get("codebook_ramp_fraction", 0.0)
            ),
            "commitment_weight": 0.0,
            "zero_l2_weight": 0.0,
            "regularization_release_fraction": float(
                arm.get(
                    "quantizer_regularization_release_fraction",
                    0.50,
                )
            ),
            "regularization_ramp_fraction": float(
                arm.get(
                    "quantizer_regularization_ramp_fraction",
                    0.40,
                )
            ),
            "audit_enabled": False,
            "base_embedding_mode": "split_zero_base",
            "continuous_residual": False,
            "isolate_initialization_rng": True,
            "initialization_seed": int(
                config["model"]["tokenizer_seed"]
            ),
        }
    elif arm.get("tokenizer_type") == "selective_single_level_discrete":
        config["model"].update(
            {
                "tokenizer_type": "selective_single_level_discrete",
                "selective_discrete_fields": list(IDENTITY_FIELDS),
                "selective_discrete_codebook_size": int(
                    arm["codebook_size"]
                ),
                "selective_discrete_candidate_top_m": int(
                    arm.get("candidate_top_m", 32)
                ),
                "selective_discrete_distance_chunk_size": int(
                    arm.get("distance_chunk_size", 2048)
                ),
                "selective_discrete_temperature": float(
                    arm.get("temperature_end", 0.3)
                ),
                "selective_discrete_temperature_start": float(
                    arm.get("temperature_start", 1.0)
                ),
                "selective_discrete_code_init_scale": float(
                    arm.get("code_init_scale", 0.05)
                ),
                "selective_discrete_quantization_loss_weight": float(
                    arm.get("codebook_loss_weight", 1.0)
                ),
                "selective_discrete_commitment_weight": float(
                    arm.get("commitment_weight", 0.0)
                ),
                "selective_discrete_warmup_fraction": float(
                    arm["warmup_fraction"]
                ),
                "selective_discrete_transition_fraction": float(
                    arm["transition_fraction"]
                ),
                "selective_discrete_codebook_init": str(
                    arm["codebook_init"]
                ),
                "selective_discrete_proj_mode": "split",
                "selective_discrete_assignment_mode": "nearest_ste",
                "selective_discrete_audit_enabled": False,
            }
        )
    return config


def _l2_schedule_report(trainer: Trainer, arm: dict) -> dict:
    coefficient = float(arm["full_table_l2_coefficient"])
    if coefficient == 0.0:
        return {
            "enabled": False,
            "coefficient": 0.0,
            "application_order": (
                trainer.zero_anchor_full_table_l2_application_order
            ),
            "trainer_schedule_configured": False,
            "trace": [],
        }
    release = float(arm["l2_release_fraction"])
    ramp = float(arm["l2_ramp_fraction"])
    trace = [
        {
            "progress": progress,
            "multiplier": Trainer._release_ramp_multiplier(
                round(progress * 1_000_000),
                1_000_001,
                release_fraction=release,
                ramp_fraction=ramp,
            ),
        }
        for progress in (0.0, 0.25, 0.5, 0.7, 0.9, 1.0)
    ]
    return {
        "enabled": True,
        "coefficient": coefficient,
        "regularization_target": (
            getattr(trainer, "zero_anchor_full_table_l2_target", None)
        ),
        "regularization_target_is_explicit": (
            getattr(
                trainer,
                "zero_anchor_full_table_l2_target_is_explicit",
                False,
            )
        ),
        "release_fraction": release,
        "ramp_fraction": ramp,
        "trainer_schedule_configured": (
            trainer.zero_anchor_full_table_l2_schedule_is_explicit
        ),
        "trace": trace,
        "final_multiplier": (
            trainer.zero_anchor_full_table_l2_current_multiplier
        ),
        "last_gradient_report": _json_safe(
            trainer.zero_anchor_full_table_l2_last_gradient_report
        ),
        "first_regularized_row": (
            trainer.zero_anchor_full_table_l2_first_regularized_row
        ),
        "application_order": (
            trainer.zero_anchor_full_table_l2_application_order
        ),
    }


@torch.no_grad()
def _quantizer_base_report(quantizer) -> dict:
    if quantizer is None:
        return {"applicable": False, "fields": {}}
    fields = {}
    for position, field in enumerate(quantizer.identity_fields):
        fields[field] = {
            "zero_state_base_norm": float(
                torch.linalg.vector_norm(
                    quantizer.zero_state_base_embeddings[position].float()
                ).item()
            ),
            "nonzero_state_base_norm": float(
                torch.linalg.vector_norm(
                    quantizer.base_embeddings[position].float()
                ).item()
            ),
            "unused_independent_oov_base_norm": float(
                torch.linalg.vector_norm(
                    quantizer.oov_base_embeddings[position].float()
                ).item()
            ),
        }
    return {
        "applicable": True,
        "initial_zero_state_base": "exact_zero",
        "fields": fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    matrix_path = Path(args.matrix).resolve()
    source_manifest_path = Path(args.source_manifest).resolve()
    matrix = json.loads(matrix_path.read_text())
    assert_core_review_gate(matrix)
    run, arm = _resolve_run(matrix, args.run_key)
    manifest_path = Path(matrix["processed_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    cache_path = manifest_path.parent / "identity_frequency_cache.npz"
    initial_checks = {
        "protocol": matrix.get("protocol") == PROTOCOL,
        "processed_manifest_sha256": (
            sha256_file(manifest_path)
            == matrix["expected_processed_manifest_sha256"]
        ),
        "frequency_cache_sha256": (
            sha256_file(cache_path)
            == matrix["expected_frequency_cache_sha256"]
        ),
        "split_sha256": (
            manifest["protocol"]["split_sha256"]
            == matrix["expected_split_sha256"]
        ),
        "train_rows": (
            manifest["splits"]["train"]["rows"]
            == matrix["expected_train_rows"]
        ),
        "validation_rows": (
            manifest["splits"]["val"]["rows"]
            == matrix["expected_validation_rows"]
        ),
        "matrix_forbids_test": (
            matrix["uses_test_dataset"] is False
            and matrix["uses_test_labels"] is False
        ),
    }
    if not all(initial_checks.values()):
        raise RuntimeError(f"SHRED L2 core frozen checks failed: {initial_checks}")

    seed = int(run["seed"])
    _configure_reproducibility(seed)
    config = build_core_config(manifest_path, run=run, arm=arm)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("SHRED L2 core formal job requires CUDA")
    instantiated_dataset_splits = []

    def instantiate_dataset(dataset_config: dict, split: str):
        if split not in {"train", "val"}:
            raise RuntimeError(
                f"SHRED L2 core forbids dataset split {split!r}"
            )
        instantiated_dataset_splits.append(split)
        return KuaiRand27KK1MMapDataset(dataset_config, split)

    train_dataset = instantiate_dataset(config, "train")
    validation_config = copy.deepcopy(config)
    validation_config["dataset"]["verify_processed_sha256"] = False
    validation_dataset = instantiate_dataset(validation_config, "val")
    (
        validation_selection_dataset,
        validation_report_dataset,
        validation_partition_report,
        validation_partition_checks,
    ) = resolve_validation_partitions(matrix, validation_dataset)
    processed_artifact_verification = dict(
        train_dataset.sha256_verification
    )
    model = S2DRecModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initialization_fingerprint = selected_initialization_fingerprint(model)
    identity_initialization = identity_initialization_fingerprint(model)
    common_initialization = common_initialization_fingerprint(model)
    model = model.to(device)
    order_fingerprint = data_order_fingerprint(
        train_dataset, rows=4096
    )
    quantizer = model.encoder.zero_anchor_identity_quantizer
    route_metadata = (
        quantizer.get_metadata() if quantizer is not None else None
    )
    route_checks = {
        "tokenizer_type_matches_arm": (
            model.tokenizer_type
            == str(arm.get("tokenizer_type", "per_field_proj"))
        ),
        "family_matches_quantizer": (
            (quantizer is not None) == bool(arm["zero_anchor"])
        ),
        "configured_no_continuous_residual_matches_metadata": (
            quantizer is None
            or (
                config["model"]["zero_anchor_identity_quantization"][
                    "continuous_residual"
                ]
                is False
                and route_metadata["continuous_residual"] is False
                and route_metadata[
                    "final_remaining_residual_enters_main_network"
                ]
                is False
            )
        ),
        "no_independent_oov_override": (
            quantizer is None
            or all(
                route_metadata[
                    "oov_uses_quantized_zero_path_by_field"
                ].values()
            )
        ),
        "split_zero_base": (
            quantizer is None
            or all(
                mode == "split_zero_base"
                for mode in route_metadata[
                    "base_embedding_mode_by_field"
                ].values()
            )
        ),
        "quantizer_initialization_rng_isolated": (
            quantizer is None
            or (
                route_metadata["initialization_rng_isolated"] is True
                and route_metadata["initialization_seed"]
                == int(matrix["tokenizer_seed"])
            )
        ),
        "code_init_radius_matches_arm": (
            quantizer is None
            or config["model"]["zero_anchor_identity_quantization"][
                "code_init_radius"
            ]
            == float(arm.get("code_init_radius", 0.2))
        ),
        "margin_matches_arm": (
            quantizer is None
            or config["model"]["zero_anchor_identity_quantization"]["margin"]
            == float(arm.get("margin", 0.1))
        ),
        "temperature_schedule_matches_arm": (
            quantizer is None
            or (
                route_metadata["temperature_start"]
                == float(arm.get("temperature_start", 1.0))
                and route_metadata["temperature_end"]
                == float(arm.get("temperature_end", 0.3))
            )
        ),
        "codebook_schedule_matches_arm": (
            quantizer is None
            or (
                route_metadata["codebook_loss_weight"]
                == float(arm.get("codebook_loss_weight", 1.0))
                and route_metadata["codebook_schedule"][
                    "release_fraction"
                ]
                == float(arm.get("codebook_release_fraction", 0.0))
                and route_metadata["codebook_schedule"]["ramp_fraction"]
                == float(arm.get("codebook_ramp_fraction", 0.0))
            )
        ),
        "private_row_initialization_matches_arm": (
            quantizer is None
            or route_metadata["private_row_initialization"]["mode"]
            == str(arm.get("private_row_initialization_mode", "zero"))
        ),
        "assignment_stability_matches_arm": (
            quantizer is None
            or (
                route_metadata["assignment_stability"]["mode"]
                == str(arm.get("assignment_stability_mode", "free_nearest"))
                and route_metadata["assignment_stability"]["freeze_fraction"]
                == float(arm.get("assignment_freeze_fraction", 0.0))
                and route_metadata["assignment_stability"]
                ["switch_relative_improvement"]
                == float(
                    arm.get(
                        "assignment_switch_relative_improvement",
                        0.0,
                    )
                )
            )
        ),
        "quantizer_regularization_schedule_matches_arm": (
            quantizer is None
            or (
                route_metadata["regularization_schedule"][
                    "release_fraction"
                ]
                == float(
                    arm.get(
                        "quantizer_regularization_release_fraction",
                        0.50,
                    )
                )
                and route_metadata["regularization_schedule"][
                    "ramp_fraction"
                ]
                == float(
                    arm.get(
                        "quantizer_regularization_ramp_fraction",
                        0.40,
                    )
                )
            )
        ),
        "task_codebook_gradient_mode_matches_arm": (
            quantizer is None
            or route_metadata["task_codebook_gradient_mode"]
            == str(
                arm.get(
                    "task_codebook_gradient_mode",
                    "legacy_hard_plus_soft",
                )
            )
        ),
        "codebook_transform_mode_matches_arm": (
            quantizer is None
            or route_metadata["codebook_transform"]["mode"]
            == str(arm.get("codebook_transform_mode", "none"))
        ),
        "shared_transform_scope_is_direction_only": (
            quantizer is None
            or str(arm.get("codebook_transform_mode", "none"))
            != "shared_linear_residual"
            or (
                route_metadata["codebook_transform"]["applies_to"]
                == ["nonzero_center_directions_only"]
                and route_metadata["codebook_transform"][
                    "cannot_change_radii"
                ]
                is True
            )
        ),
        "shared_transform_tripwire_is_fixed": (
            quantizer is None
            or str(arm.get("codebook_transform_mode", "none"))
            != "shared_linear_residual"
            or route_metadata["codebook_transform"][
                "minimum_singular_ratio_tripwire"
            ]
            == quantizer.SHARED_TRANSFORM_MIN_SINGULAR_RATIO
        ),
        "identity_oov_raw_rows_zero_at_initialization": all(
            torch.count_nonzero(
                model.encoder.sparse_arch.embeddings[
                    manifest["field_names"].index(field)
                ].weight[1]
            ).item()
            == 0
            for field in IDENTITY_FIELDS
        ),
    }
    if not all(route_checks.values()):
        raise RuntimeError(f"SHRED L2 core route checks failed: {route_checks}")

    dynamics = ZeroAnchorFrequencyDynamicsAudit(
        model,
        frequency_cache_path=cache_path,
        identity_fields=IDENTITY_FIELDS,
        field_names=manifest["field_names"],
        snapshot_progresses=tuple(
            matrix["frequency_audit"]["late_snapshot_progresses"]
        ),
        sample_ids_per_bucket=int(
            matrix["frequency_audit"]["sample_ids_per_bucket"]
        ),
        sample_seed=int(matrix["frequency_audit"]["sample_seed"]),
        allow_partial_snapshots=bool(
            matrix["frequency_audit"].get(
                "allow_early_stop_snapshot_prefix", False
            )
        ),
        minimum_partial_snapshots=int(
            matrix["frequency_audit"].get(
                "minimum_early_stop_snapshots", 2
            )
        ),
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=validation_selection_dataset,
        config=config,
        device=device,
        rank=0,
        world_size=1,
    )
    training_execution_audit = TrainingExecutionAudit(
        trainer,
        dynamics,
        steps_per_epoch=len(trainer.train_loader),
    )
    trainer.set_training_step_observer(training_execution_audit)
    best_validation_auc = trainer.train()
    selected_metrics = trainer.best_metrics or trainer.last_metrics
    if not selected_metrics:
        raise RuntimeError("SHRED L2 core training produced no validation metrics")
    checkpoint_required = bool(config["training"]["save_best_checkpoint"])
    checkpoint_restored = trainer.restore_best_model()
    if checkpoint_required and not checkpoint_restored:
        raise RuntimeError("best checkpoint was required but could not be restored")
    restored_selection_metrics = trainer.evaluate_dataset(
        validation_selection_dataset
    )
    if validation_partition_report["enabled"]:
        formal_report_metrics = trainer.evaluate_dataset(validation_report_dataset)
    else:
        formal_report_metrics = restored_selection_metrics
    if not restored_selection_metrics or not formal_report_metrics:
        raise RuntimeError("post-training validation evaluation was incomplete")
    dynamics_report = dynamics.report()
    schedule_trace = training_execution_audit.report()
    exact_geometry = analyze_exact_frequency_geometry(
        model,
        frequency_cache_path=cache_path,
        identity_fields=IDENTITY_FIELDS,
        field_names=manifest["field_names"],
    )
    validation_frequency = evaluate_validation_frequency_auc(
        model,
        validation_report_dataset,
        frequency_cache_path=cache_path,
        identity_fields=IDENTITY_FIELDS,
        field_names=manifest["field_names"],
        device=device,
        batch_size=config["training"]["batch_size"] * 2,
        num_workers=config["training"]["num_workers"],
    )
    tailmask_spec = resolve_validation_tailmask_spec(matrix)
    validation_tailmask = evaluate_validation_tailmask_auc(
        model,
        validation_report_dataset,
        frequency_cache_path=cache_path,
        identity_fields=IDENTITY_FIELDS,
        field_names=manifest["field_names"],
        cutoffs=tuple(tailmask_spec["cutoffs"]),
        include_unseen=bool(tailmask_spec["include_unseen"]),
        device=device,
        batch_size=config["training"]["batch_size"] * 2,
        num_workers=config["training"]["num_workers"],
    )
    l2_schedule = _l2_schedule_report(trainer, arm)
    quantizer_health = (
        quantizer.get_health_report(**HEALTH_THRESHOLDS)
        if quantizer is not None
        else {"applicable": False, "all_checks_pass": True}
    )
    quantizer_base = _quantizer_base_report(quantizer)
    shared_transform_report = (
        {
            "enabled": False,
            "mode": "not_applicable",
            "fields": [],
            "all_checks_pass": True,
        }
        if quantizer is None
        else quantizer.get_shared_codebook_transform_report()
    )
    center_displacement_report = (
        {
            "applicable": False,
            "reason": "continuous arm has no Product-M1 centers",
            "fields": [],
            "all_checks_pass": True,
        }
        if quantizer is None
        else quantizer.get_center_displacement_by_assignment_frequency_report()
    )
    coefficient = float(arm["full_table_l2_coefficient"])
    schedule_snapshots = schedule_trace["trace"]
    post_step_schedule_snapshots = [
        item for item in schedule_snapshots if item["completed_step"] > 0
    ]
    epoch_boundary_snapshots = [
        item for item in post_step_schedule_snapshots if item["epoch_boundary"]
    ]
    quantizer_release = float(
        arm.get("quantizer_regularization_release_fraction", 0.50)
    )
    quantizer_ramp = float(
        arm.get("quantizer_regularization_ramp_fraction", 0.40)
    )
    l2_checks = {
        "applied_exactly_when_registered": (
            (
                trainer.zero_anchor_full_table_l2_last_gradient_report
                is not None
            )
            == (coefficient > 0.0)
        ),
        "disabled_arm_has_no_trainer_schedule": (
            coefficient > 0.0
            or (
                trainer.zero_anchor_full_table_l2_enabled is False
                and trainer.zero_anchor_full_table_l2_schedule_is_explicit
                is False
            )
        ),
        "enabled_arm_uses_frozen_schedule": (
            coefficient == 0.0
            or (
                trainer.zero_anchor_full_table_l2_release_fraction
                == float(arm["l2_release_fraction"])
                and trainer.zero_anchor_full_table_l2_ramp_fraction
                == float(arm["l2_ramp_fraction"])
                and trainer.zero_anchor_full_table_l2_current_multiplier
                == 1.0
            )
        ),
        "l2_multiplier_matches_at_step_zero_and_epoch_boundaries": (
            coefficient == 0.0
            or all(
                item["full_table_l2_multiplier"] == 1.0
                for item in schedule_snapshots
                if item["completed_step"] == 0 or item["epoch_boundary"]
            )
        ),
        "enabled_l2_gradient_present_after_each_audited_step": (
            coefficient == 0.0
            or all(
                item["full_table_l2_gradient_report_present"]
                for item in post_step_schedule_snapshots
            )
        ),
        "disabled_l2_gradient_contribution_is_zero": (
            coefficient > 0.0
            or (
                trainer.zero_anchor_full_table_l2_enabled is False
                and trainer.zero_anchor_full_table_l2_last_gradient_report
                is None
                and all(
                    not item["full_table_l2_gradient_report_present"]
                    for item in schedule_snapshots
                )
            )
        ),
        "all_completed_epochs_have_schedule_snapshots": (
            len(epoch_boundary_snapshots) == trainer.epochs_ran
        ),
        "quantizer_constant_regularization_schedule_is_one": (
            quantizer is None
            or quantizer_release != 0.0
            or quantizer_ramp != 0.0
            or all(
                item["quantizer_regularization_multiplier"] == 1.0
                for item in schedule_snapshots
            )
        ),
        "quantizer_constant_codebook_schedule_is_one": (
            quantizer is None
            or float(arm.get("codebook_release_fraction", 0.0)) != 0.0
            or float(arm.get("codebook_ramp_fraction", 0.0)) != 0.0
            or all(
                item["quantizer_codebook_multiplier"] == 1.0
                for item in schedule_snapshots
            )
        ),
        "private_id_rows_only": (
            trainer.zero_anchor_full_table_l2_first_regularized_row == 4
        ),
        "l2_added_after_global_task_gradient_clip": (
            trainer.zero_anchor_full_table_l2_application_order
            == "after_global_clip"
        ),
        "shared_transform_epoch_boundary_tripwire_passed": all(
            item["shared_codebook_transform"]["all_checks_pass"]
            for item in schedule_snapshots
            if item["completed_step"] == 0 or item["epoch_boundary"]
        ),
        "selected_checkpoint_shared_transform_tripwire_passed": (
            shared_transform_report["all_checks_pass"]
        ),
        "center_displacement_frequency_strata_complete": (
            center_displacement_report["all_checks_pass"]
        ),
    }
    final_oov_rows_zero = all(
        torch.count_nonzero(
            model.encoder.sparse_arch.embeddings[
                manifest["field_names"].index(field)
            ].weight[1]
        ).item()
        == 0
        for field in IDENTITY_FIELDS
    )
    validation_auc = float(validation_frequency["overall"]["absolute_auc"])
    validation_report_rows = len(validation_report_dataset)
    checkpoint_auc_tolerance = float(
        config["training"]["checkpoint_selection_tolerance"]
    )
    checkpoint_sha256 = (
        sha256_file(trainer.best_checkpoint_path)
        if trainer.best_checkpoint_path is not None
        else None
    )
    required_segment_aucs = [
        validation_frequency["identity_fields"][field]["buckets"][
            label
        ]["absolute_auc"]
        for field in IDENTITY_FIELDS
        for label in ("oov_unseen", "count_gt_500")
    ] + [
        validation_frequency["identity_fields"][field]["aggregates"][
            "seen_rare_1_5"
        ]["absolute_auc"]
        for field in IDENTITY_FIELDS
    ]
    execution_checks = {
        **initial_checks,
        **validation_partition_checks,
        **route_checks,
        **l2_checks,
        "finite_validation_auc": math.isfinite(validation_auc),
        "parameter_count_matches_frozen_family": (
            parameter_count
            == resolve_expected_parameter_count(matrix, arm)
        ),
        "required_frequency_segment_aucs_finite": all(
            value is not None and math.isfinite(float(value))
            for value in required_segment_aucs
        ),
        "selected_checkpoint_auc_reproduced_on_selection_partition": (
            abs(
                float(selected_metrics["auc"])
                - float(restored_selection_metrics["auc"])
            )
            <= checkpoint_auc_tolerance
        ),
        "formal_report_auc_reproduced_by_frequency_reader": (
            abs(float(formal_report_metrics["auc"]) - validation_auc)
            <= 1e-12
        ),
        "best_auc_matches_selected_metrics": (
            abs(float(best_validation_auc) - float(selected_metrics["auc"]))
            <= checkpoint_auc_tolerance
        ),
        "checkpoint_restore_matches_requirement": (
            checkpoint_restored == checkpoint_required
        ),
        "checkpoint_metadata_complete": (
            not checkpoint_required
            or (
                trainer.best_checkpoint_epoch is not None
                and trainer.best_checkpoint_step is not None
                and checkpoint_sha256 is not None
            )
        ),
        "early_stop_min_epochs_enforced": (
            trainer.epochs_ran >= int(config["training"]["early_stop_min_epochs"])
        ),
        "mid_epoch_eval_disabled_for_early_stop": (
            int(config["training"]["early_stop_patience"]) == 0
            or int(config["training"]["eval_every"]) == 0
        ),
        "single_process_early_stop_fail_closed": trainer.world_size == 1,
        "validation_positive_counts_close": (
            (
                int(restored_selection_metrics["num_pos"])
                + int(formal_report_metrics["num_pos"])
                == int(matrix["expected_validation_positives"])
            )
            if validation_partition_report["enabled"]
            else (
                int(restored_selection_metrics["num_pos"])
                == int(formal_report_metrics["num_pos"])
                == int(matrix["expected_validation_positives"])
            )
        ),
        "validation_rows_close": (
            validation_frequency["overall"]["rows"]
            == validation_report_rows
        ),
        "validation_positives_close": (
            validation_frequency["overall"]["positives"]
            == int(formal_report_metrics["num_pos"])
        ),
        "frequency_dynamics_complete": dynamics_report["all_checks_pass"],
        "exact_geometry_complete": exact_geometry["all_checks_pass"],
        "validation_frequency_complete": (
            validation_frequency["all_checks_pass"]
        ),
        "validation_tailmask_complete": (
            validation_tailmask["all_checks_pass"]
        ),
        "validation_tailmask_clean_auc_reproduced": (
            abs(
                validation_tailmask["clean"]["absolute_auc"]
                - validation_auc
            )
            <= 1e-12
        ),
        "validation_tailmask_rows_close": (
            validation_tailmask["clean"]["rows"]
            == validation_report_rows
        ),
        "validation_tailmask_uses_no_test_data": (
            validation_tailmask["uses_test_dataset"] is False
            and validation_tailmask["uses_test_labels"] is False
        ),
        "processed_artifacts_verified_train_validation_only": (
            processed_artifact_verification["enabled"] is True
            and processed_artifact_verification["splits"]
            == ["train", "val"]
            and processed_artifact_verification["artifact_count"] > 0
        ),
        "only_train_and_validation_datasets_instantiated": (
            instantiated_dataset_splits == ["train", "val"]
        ),
        "test_dataset_not_instantiated": (
            "test" not in instantiated_dataset_splits
        ),
        "test_labels_not_loaded": (
            "test" not in instantiated_dataset_splits
        ),
        "identity_oov_raw_rows_remain_zero_after_training": (
            final_oov_rows_zero
        ),
        "determinism_environment_frozen": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            == matrix["fixed_model"]["runtime_environment"][
                "CUBLAS_WORKSPACE_CONFIG"
            ]
            and os.environ.get("PYTHONHASHSEED")
            == matrix["fixed_model"]["runtime_environment"][
                "PYTHONHASHSEED"
            ]
        ),
    }
    uses_train_dataset = "train" in instantiated_dataset_splits
    uses_validation_dataset = "val" in instantiated_dataset_splits
    uses_test_dataset = "test" in instantiated_dataset_splits
    report = {
        "protocol": PROTOCOL,
        "status": "completed",
        "review_gate": matrix["review_gate"],
        "run_key": run["run_key"],
        "arm_key": run["arm_key"],
        "display_name": arm["display_name"],
        "family": arm["family"],
        "seed": seed,
        "model_seed": seed,
        "tokenizer_seed": int(matrix["tokenizer_seed"]),
        "data_seed": int(matrix["data_seed"]),
        "deterministic_training": True,
        "optimizer_foreach": False,
        "uses_train_labels": uses_train_dataset,
        "uses_validation_dataset": uses_validation_dataset,
        "uses_validation_labels": uses_validation_dataset,
        "uses_test_dataset": uses_test_dataset,
        "uses_test_labels": uses_test_dataset,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "matrix_sha256": sha256_file(matrix_path),
        "processed_manifest_sha256": sha256_file(manifest_path),
        "frequency_cache_sha256": sha256_file(cache_path),
        "split_sha256": manifest["protocol"]["split_sha256"],
        "parameter_count": parameter_count,
        "selected_initialization_fingerprint": initialization_fingerprint,
        "identity_initialization_fingerprint": identity_initialization,
        "common_initialization_fingerprint": common_initialization,
        "initialization_fingerprint_format": (
            "sha256_sorted_name_dtype_shape_raw_tensor_bytes_v1"
        ),
        "data_order_fingerprint": order_fingerprint,
        "instantiated_dataset_splits": instantiated_dataset_splits,
        "processed_artifact_verification": (
            processed_artifact_verification
        ),
        "arm": arm,
        "route_metadata": _json_safe(route_metadata),
        "resolved_config": config,
        "best_validation_auc": float(best_validation_auc),
        "checkpoint_selection_metrics": _json_safe(selected_metrics),
        "restored_selection_metrics": _json_safe(
            restored_selection_metrics
        ),
        "validation_metrics": _json_safe(formal_report_metrics),
        "validation_partition": validation_partition_report,
        "training_control": {
            "epochs_ran": int(trainer.epochs_ran),
            "stop_reason": trainer.stop_reason,
            "epoch_history": _json_safe(trainer.epoch_history),
            "best_checkpoint_epoch": trainer.best_checkpoint_epoch,
            "best_checkpoint_step": trainer.best_checkpoint_step,
            "best_checkpoint_sha256": checkpoint_sha256,
            "checkpoint_restored": checkpoint_restored,
            "budget_censored": (
                trainer.stop_reason == "max_epochs"
                and trainer.best_checkpoint_epoch is not None
                and trainer.best_checkpoint_epoch
                > int(
                    matrix.get("early_stopping", {}).get(
                        "budget_uncensored_selected_epoch_max", 7
                    )
                )
            ),
        },
        "l2_schedule": l2_schedule,
        "runtime_schedule_trace": schedule_trace,
        "frequency_dynamics": dynamics_report,
        "exact_frequency_geometry": exact_geometry,
        "validation_frequency_auc": validation_frequency,
        "validation_tailmask_auc": validation_tailmask,
        "quantizer_health": _json_safe(quantizer_health),
        "quantizer_base_embeddings": _json_safe(quantizer_base),
        "shared_codebook_transform": _json_safe(
            shared_transform_report
        ),
        "center_displacement_by_assignment_frequency": _json_safe(
            center_displacement_report
        ),
        "quantizer_health_scope": (
            "cumulative_training_occurrences_diagnostic_only"
            if quantizer is not None
            else "not_applicable"
        ),
        "execution_checks": execution_checks,
        "all_execution_checks_pass": all(execution_checks.values()),
        "scientific_effect_gate_evaluated": False,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "conclusion_boundary": {
            "primary_metric": (
                "validation_report_absolute_auc"
                if validation_partition_report["enabled"]
                else "validation_full_absolute_auc"
            ),
            "validation_selection_is_checkpoint_only": (
                validation_partition_report["enabled"]
            ),
            "full_validation_used_for_development": (
                not validation_partition_report["enabled"]
            ),
            "frequency_geometry_is_train_only": True,
            "no_test_generalization_claim": True,
            "offline_unseen_is_one_shared_oov_row": True,
            "full_causal_claim_requires_h1_h2_h3_h4": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["all_execution_checks_pass"]:
        raise RuntimeError(
            f"SHRED L2 core execution checks failed: {execution_checks}"
        )
    final_log = {
        "status": "ok",
        "run_key": run["run_key"],
        "quantizer_health": quantizer_health.get("all_checks_pass"),
        "output": str(output),
    }
    if not config["training"]["suppress_effect_metric_logs"]:
        final_log["validation_auc"] = validation_auc
    print(json.dumps(final_log, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
