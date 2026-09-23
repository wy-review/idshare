#!/usr/bin/env python3
"""Run the frozen TAAC UnifiedMixer full L2 response grid."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fuxictr.features import FeatureMap  # noqa: E402
from fuxictr.pytorch.dataloaders import RankDataLoader  # noqa: E402
from fuxictr.pytorch.torch_utils import seed_everything  # noqa: E402
from fuxictr.utils import load_config, set_logger  # noqa: E402
import src  # noqa: E402


PROTOCOL = "taac_unifiedmixer_l2_response_r02"
DOSES = (1e-11, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8)
SOURCE_MANIFEST_NAME = "IDSHARE_UB_SOURCE_MANIFEST.sha256"
TARGET_FIELD = "target_item_id"
SEQUENCE_FIELD = "item_seq"
SIDE_SEQUENCE_FIELDS = (
    "item_seq__feat_102",
    "item_seq__feat_115",
    "item_seq__feat_119",
    "item_seq__feat_120",
)
EXECUTION_CONTRACT_KEYS = (
    "model",
    "dataset_id",
    "data_loader_class",
    "optimizer",
    "learning_rate",
    "embedding_regularizer",
    "net_regularizer",
    "batch_size",
    "embedding_dim",
    "d_model",
    "num_layers",
    "ffn_mult",
    "ffn_type",
    "block_norm",
    "num_ns_tokens",
    "seq_tokenizer",
    "num_seq_tokens",
    "recent_k",
    "concat_mode",
    "pooling",
    "seq_pooling",
    "mlp_dims",
    "dropout",
    "norm_type",
    "epochs",
    "shuffle",
    "seed",
    "maxlen",
    "user_array_maxlen",
    "idshare_enabled",
    "idshare_quantizer_config",
    "shared_item_table_cardinality",
    "train_input_seen_mask_path",
    "train_input_seen_mask_sha256",
    "expected_raw_data_root",
    "expected_samples_path_by_split",
    "full_table_l2_coefficient",
    "full_table_l2_target",
    "full_table_l2_first_regularized_row",
    "full_table_l2_application_order",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_report(payload: dict) -> Path:
    output = Path(os.environ.get("JOB_OUTPUT_DIR", ".")) / "training_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def binding_report() -> dict:
    return {
        "matrix_sha256": os.environ.get("IDSHARE_UB_MATRIX_SHA256"),
        "package_sha256": os.environ.get("IDSHARE_UB_PACKAGE_SHA256"),
        "source_manifest_sha256": os.environ.get(
            "IDSHARE_UB_SOURCE_MANIFEST_SHA256"
        ),
        "freeze_sha256": os.environ.get("IDSHARE_UB_FREEZE_SHA256"),
    }


def verify_source_manifest() -> dict:
    manifest_path = REPO / SOURCE_MANIFEST_NAME
    expected_sha = os.environ.get("IDSHARE_UB_SOURCE_MANIFEST_SHA256")
    if not expected_sha or not manifest_path.is_file():
        raise RuntimeError("source manifest binding is absent")
    if sha256_file(manifest_path) != expected_sha:
        raise RuntimeError("source manifest SHA256 mismatch")
    checked = 0
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative = raw_line.split("  ", 1)
        path = REPO / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"packaged source drift: {relative}")
        checked += 1
    if checked < 1:
        raise RuntimeError("source manifest is empty")
    return {
        "source_manifest_verified": True,
        "source_manifest_files_checked": checked,
    }


def execution_contract(params: dict) -> dict:
    return {key: copy.deepcopy(params.get(key)) for key in EXECUTION_CONTRACT_KEYS}


def validate_matrix(matrix: dict, matrix_path: Path) -> None:
    expected_sha = os.environ.get("IDSHARE_UB_MATRIX_SHA256")
    if expected_sha and sha256_file(matrix_path) != expected_sha:
        raise RuntimeError("matrix SHA256 mismatch")
    checks = {
        "protocol": matrix.get("protocol") == PROTOCOL,
        "frozen": matrix.get("status") == "frozen_before_results",
        "thirty_six_cells": len(matrix.get("runs", {})) == 36,
        "no_test": matrix.get("uses_test_dataset") is False
        and matrix.get("uses_test_labels") is False,
        "doses": matrix.get("l2_coefficients") == list(DOSES),
        "seeds": matrix.get("seeds") == [2021, 42, 2024],
        "carriers": matrix.get("carriers") == ["continuous", "idshare"],
        "capacity": matrix.get("idshare_capacity") == 20000,
    }
    if not all(checks.values()):
        raise RuntimeError(f"matrix contract failed: {checks}")
    seen = set()
    for key, cell in matrix["runs"].items():
        carrier = cell.get("carrier")
        seed = cell.get("seed")
        coefficient = float(cell.get("l2_coefficient", -1.0))
        target = cell.get("l2_target")
        expected_target = (
            "prequantization_assignment_table"
            if carrier == "idshare"
            else "identity_output_table"
        )
        if (
            carrier not in {"continuous", "idshare"}
            or seed not in {2021, 42, 2024}
            or coefficient not in DOSES
            or target != expected_target
            or cell.get("run_key") != key
        ):
            raise RuntimeError(f"invalid frozen cell: {key}")
        seen.add((carrier, coefficient, seed))
    expected = {
        (carrier, coefficient, seed)
        for carrier in ("continuous", "idshare")
        for coefficient in DOSES
        for seed in (2021, 42, 2024)
    }
    if seen != expected:
        raise RuntimeError("frozen L2 response grid is incomplete")


def _synthetic_feature_map(cardinality=64):
    features = OrderedDict()
    features[TARGET_FIELD] = {
        "type": "categorical",
        "source": "item",
        "vocab_size": cardinality,
        "padding_idx": 0,
    }
    for fid in (100, 101, 102, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122):
        features[f"feat_{fid}"] = {
            "type": "categorical",
            "source": "item",
            "vocab_size": 32,
            "padding_idx": 0,
        }
    for fid in (103, 104, 105, 109):
        features[f"feat_{fid}"] = {
            "type": "categorical",
            "source": "user",
            "vocab_size": 32,
            "padding_idx": 0,
        }
    for fid in (106, 107, 108, 110):
        features[f"feat_{fid}"] = {
            "type": "categorical",
            "source": "user",
            "vocab_size": 32,
            "padding_idx": 0,
        }
    features[SEQUENCE_FIELD] = {
        "type": "sequence",
        "source": "item",
        "vocab_size": cardinality,
        "padding_idx": 0,
        "max_len": 8,
        "share_embedding": TARGET_FIELD,
    }
    for field in SIDE_SEQUENCE_FIELDS:
        target = field.split("__", 1)[1]
        features[field] = {
            "type": "sequence",
            "source": "item",
            "vocab_size": 32,
            "padding_idx": 0,
            "max_len": 8,
            "share_embedding": target,
        }
    return SimpleNamespace(
        features=features,
        labels=["label"],
        group_id=None,
        dataset_id="synthetic_taac_bridge",
        data_dir="/tmp",
    )


def _quantizer_config(seed, capacity=8):
    return {
        "enabled": True,
        "identity_fields": ["shared_item_id"],
        "codebook_size": int(capacity),
        "codebook_size_by_field": {"shared_item_id": int(capacity)},
        "num_subspaces": 1,
        "num_residual_levels": 1,
        "multi_codebook_mode": "serial_residual",
        "continuous_residual": False,
        "base_embedding_mode": "split_zero_base",
        "private_row_initialization_mode": "train_first_touch_deterministic_code",
        "assignment_stability_mode": "free_nearest",
        "assignment_freeze_fraction": 0.0,
        "assignment_switch_relative_improvement": 0.0,
        "task_codebook_gradient_mode": "legacy_hard_plus_soft",
        "temperature_start": 1.0,
        "temperature_end": 0.3,
        "margin": 0.1,
        "code_init_radius": 0.2,
        "zero_l2_weight": 0.0,
        "commitment_weight": 0.0,
        "codebook_loss_weight": 1.0,
        "regularization_release_fraction": 0.0,
        "regularization_ramp_fraction": 0.0,
        "codebook_release_fraction": 0.0,
        "codebook_ramp_fraction": 0.0,
        "distance_backend": "gemm",
        "distance_row_chunk_size": 64,
        "compact_distance_outputs": True,
        "sparse_batch_diagnostics": True,
        "defer_cumulative_diagnostics": True,
        "audit_enabled": False,
        "code_logit_bias_mode": "none",
        "codebook_transform_mode": "none",
        "initialization_seed": int(seed),
        "isolate_initialization_rng": True,
    }


def _synthetic_model(carrier, seed, coefficient):
    params = {
        "model_id": f"synthetic_{carrier}_{seed}",
        "gpu": -1,
        "learning_rate": 0.001,
        "embedding_dim": 16,
        "d_model": 32,
        "num_layers": 2,
        "ffn_mult": 2,
        "ffn_type": "per_token_swiglu",
        "block_norm": "pre",
        "num_ns_tokens": 11,
        "seq_tokenizer": "recent_k_plus_equal_chunks",
        "num_seq_tokens": 5,
        "recent_k": 2,
        "maxlen": 8,
        "concat_mode": "s_ns",
        "pooling": "ns_tokens_mean",
        "seq_pooling": "attn",
        "sequence_fields": [SEQUENCE_FIELD, *SIDE_SEQUENCE_FIELDS],
        "mlp_dims": [64, 32],
        "dropout": 0.0,
        "embedding_regularizer": 0,
        "net_regularizer": 0,
        "full_table_l2_coefficient": float(coefficient),
        "full_table_l2_target": (
            "prequantization_assignment_table"
            if carrier == "idshare"
            else "identity_output_table"
        ),
        "full_table_l2_first_regularized_row": 4,
        "full_table_l2_application_order": "after_global_clip",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "task": "binary_classification",
        "model_root": "/tmp/idshare-ub-regression",
        "verbose": 0,
        "metrics": ["AUC"],
        "monitor": "AUC",
        "tensorboard": False,
        "idshare_enabled": carrier == "idshare",
        "idshare_quantizer_config": _quantizer_config(seed),
    }
    return src.IDShareUnifiedMixer(_synthetic_feature_map(), **params)


def regression(matrix_path: Path) -> dict:
    matrix = load_json(matrix_path)
    validate_matrix(matrix, matrix_path)
    rows = {}
    for key, cell in matrix["runs"].items():
            carrier = cell["carrier"]
            seed = int(cell["seed"])
            coefficient = float(cell["l2_coefficient"])
            seed_everything(seed=seed)
            model = _synthetic_model(carrier, seed, coefficient)
            batch_size = 4
            batch = {
                "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
                TARGET_FIELD: torch.tensor([4, 5, 1, 6]),
                SEQUENCE_FIELD: torch.tensor(
                    [[0, 0, 4, 5, 6, 7, 8, 9]] * batch_size
                ),
            }
            for name, spec in model.feature_map.features.items():
                if name in batch:
                    continue
                if spec["type"] == "sequence":
                    batch[name] = torch.ones(batch_size, 8, dtype=torch.long)
                else:
                    batch[name] = torch.ones(batch_size, dtype=torch.long)
            loss = model.train_step(batch)
            output = model(batch)
            finite_grads = all(
                bool(torch.isfinite(param.grad).all().item())
                for param in model.parameters()
                if param.grad is not None
            )
            contract = model.carrier_contract()
            l2_report = model.full_table_l2_last_gradient_report
            rows[key] = {
                "forward_shape": list(output["y_pred"].shape),
                "prediction_finite": bool(torch.isfinite(output["y_pred"]).all()),
                "loss_finite": bool(torch.isfinite(loss)),
                "gradients_finite": finite_grads,
                "carrier_contract": contract,
                "l2_coefficient": coefficient,
                "l2_target": cell["l2_target"],
                "all_checks_pass": bool(
                    output["y_pred"].shape == (batch_size, 1)
                    and torch.isfinite(output["y_pred"]).all()
                    and torch.isfinite(loss)
                    and finite_grads
                    and contract["all_checks_pass"]
                    and l2_report is not None
                    and l2_report["regularization_target"]
                    == cell["l2_target"]
                    and l2_report["first_regularized_row"] == 4
                ),
            }
            del model, output, loss
    passed = all(row["all_checks_pass"] for row in rows.values())
    return {
        "status": "passed" if passed else "failed",
        "protocol": PROTOCOL,
        "mode": "no_data_regression",
        **binding_report(),
        "routes_checked": len(rows),
        "cells": rows,
        "datasets_loaded": [],
        "uses_test_dataset": False,
        "uses_test_labels": False,
        "test_or_holdout_read": False,
        "all_execution_checks_pass": passed,
        **verify_source_manifest(),
    }


def _prepare_feature_map(params):
    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    cardinality = int(params["shared_item_table_cardinality"])
    if TARGET_FIELD not in feature_map.features or SEQUENCE_FIELD not in feature_map.features:
        raise RuntimeError("TAAC feature map lacks target/history item fields")
    feature_map.features[TARGET_FIELD]["vocab_size"] = cardinality
    feature_map.features[TARGET_FIELD]["padding_idx"] = 0
    feature_map.features[SEQUENCE_FIELD]["vocab_size"] = cardinality
    feature_map.features[SEQUENCE_FIELD]["padding_idx"] = 0
    feature_map.features[SEQUENCE_FIELD]["share_embedding"] = TARGET_FIELD
    for field in SIDE_SEQUENCE_FIELDS:
        target = field.split("__", 1)[1]
        if field not in feature_map.features or target not in feature_map.features:
            raise RuntimeError(f"TAAC side-sequence field contract drift: {field}")
        feature_map.features[field]["share_embedding"] = target
        feature_map.features[field]["padding_idx"] = 0
    return feature_map


def _best_validation_auc(model):
    values = [row.get("AUC") for row in getattr(model, "_history", [])]
    values = [float(value) for value in values if value is not None]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("finite validation AUC was not recorded")
    return max(values)


def train_one(config_dir: Path, expid: str, max_samples=None, emit_auc=True):
    params = load_config(str(config_dir), expid)
    if params.get("test_data") not in (None, ""):
        raise RuntimeError("test_data must be absent")
    if float(params.get("embedding_regularizer", -1)) != 0.0:
        raise RuntimeError("framework embedding regularizer must remain disabled")
    if float(params.get("net_regularizer", -1)) != 0.0:
        raise RuntimeError("framework network regularizer must remain disabled")
    coefficient = float(params.get("full_table_l2_coefficient", 0.0))
    if coefficient not in DOSES:
        raise RuntimeError("full-table L2 dose is outside the frozen grid")
    if int(params.get("epochs", -1)) != 1:
        raise RuntimeError("bridge is fixed to one epoch")
    if max_samples is not None:
        params["max_samples"] = int(max_samples)
    params["gpu"] = int(os.environ.get("RTP_GPU_INDEX", "0"))
    params["model_root"] = f"/tmp/{PROTOCOL}/{expid}/checkpoints"
    params["tensorboard"] = False
    params["num_workers"] = 0
    set_logger(params)
    seed_everything(seed=int(params["seed"]))
    feature_map = _prepare_feature_map(params)
    model_class = getattr(src, params["model"])
    model = model_class(feature_map, **params)
    model.count_parameters(count_embedding=True, batch_size=1)
    module_name, class_name = params["data_loader_class"].rsplit(".", 1)
    params["data_loader"] = getattr(importlib.import_module(module_name), class_name)
    train_gen, valid_gen = RankDataLoader(
        feature_map, stage="train", **params
    ).make_iterator()
    if train_gen is None or valid_gen is None:
        raise RuntimeError("train/validation loaders are required")
    if getattr(train_gen.dataset, "split", None) != "train":
        raise RuntimeError("training split contract drift")
    if getattr(valid_gen.dataset, "split", None) != "valid":
        raise RuntimeError("validation split contract drift")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.monotonic()
    model.fit(train_gen, validation_data=valid_gen, **params)
    elapsed = time.monotonic() - started
    auc = _best_validation_auc(model)
    contract = model.carrier_contract()
    parameter_count = sum(param.numel() for param in model.parameters())
    trainable_parameter_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(model.device))
        if torch.cuda.is_available()
        else 0
    )
    data_contract = {
        "train": dict(train_gen.dataset.manifest_contract),
        "validation": dict(valid_gen.dataset.manifest_contract),
    }
    result = {
        "experiment_id": expid,
        "carrier": "idshare" if params["idshare_enabled"] else "continuous",
        "seed": int(params["seed"]),
        "l2_coefficient": coefficient,
        "idshare_capacity": 20000 if params["idshare_enabled"] else None,
        "train_rows": int(train_gen.num_samples),
        "validation_rows": int(valid_gen.num_samples),
        "carrier_contract": contract,
        "history_length": len(getattr(model, "_history", [])),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "wall_time_seconds": float(elapsed),
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "data_contract": data_contract,
        "execution_contract": execution_contract(params),
        "best_validation_auc": auc if emit_auc else None,
        "all_checks_pass": bool(
            contract["all_checks_pass"]
            and train_gen.num_samples > 0
            and valid_gen.num_samples > 0
            and math.isfinite(auc)
            and all(
                row.get("raw_data_root") is True
                and row.get("samples_path") is True
                and row.get("test_or_holdout") is False
                for row in data_contract.values()
            )
        ),
    }
    checkpoint_root = Path(params["model_root"])
    del model, train_gen, valid_gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return result


def smoke(matrix_path: Path, config_dir: Path, max_samples: int) -> dict:
    matrix = load_json(matrix_path)
    validate_matrix(matrix, matrix_path)
    smoke_cells = [
        cell for cell in matrix["runs"].values()
        if int(cell["seed"]) == 2021
        and float(cell["l2_coefficient"]) == max(DOSES)
    ]
    if {cell["carrier"] for cell in smoke_cells} != {
        "continuous", "idshare"
    }:
        raise RuntimeError("paired high-dose smoke cells are incomplete")
    expids = [cell["experiment_id"] for cell in smoke_cells]
    rows = {}
    for expid in expids:
        row = train_one(config_dir, expid, max_samples=max_samples, emit_auc=False)
        cell = next(
            item for item in matrix["runs"].values()
            if item["experiment_id"] == expid
        )
        if row["execution_contract"] != cell["execution_contract"]:
            raise RuntimeError("smoke execution contract differs from matrix")
        rows[expid] = row
    passed = all(row["all_checks_pass"] for row in rows.values())
    return {
        "status": "passed" if passed else "failed",
        "protocol": PROTOCOL,
        "mode": "paired_training_smoke",
        **binding_report(),
        "max_samples_per_split": int(max_samples),
        "l2_coefficient": max(DOSES),
        "cells": rows,
        "datasets_loaded": ["train", "validation"],
        "validation_auc_emitted": False,
        "uses_test_dataset": False,
        "uses_test_labels": False,
        "test_or_holdout_read": False,
        "all_execution_checks_pass": passed,
        **verify_source_manifest(),
    }


def formal(matrix_path: Path, config_dir: Path, run_key: str) -> dict:
    matrix = load_json(matrix_path)
    validate_matrix(matrix, matrix_path)
    if run_key not in matrix["runs"]:
        raise ValueError("run key is absent from the frozen matrix")
    cell = matrix["runs"][run_key]
    result = train_one(config_dir, cell["experiment_id"], emit_auc=True)
    checks = {
        "run_key": result["carrier"] == cell["carrier"]
        and result["seed"] == int(cell["seed"]),
        "carrier": result["carrier"] in {"continuous", "idshare"},
        "l2": result["l2_coefficient"]
        == float(cell["l2_coefficient"]),
        "execution": result["all_checks_pass"] is True,
        "configuration": result["execution_contract"]
        == cell["execution_contract"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal execution contract failed: {checks}")
    return {
        "status": "completed",
        "protocol": PROTOCOL,
        "mode": "formal",
        "run_key": run_key,
        **binding_report(),
        **result,
        "datasets_loaded": ["train", "validation"],
        "uses_test_dataset": False,
        "uses_test_labels": False,
        "test_or_holdout_read": False,
        "all_execution_checks_pass": True,
        **verify_source_manifest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("regression", "smoke", "formal"))
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--run-key")
    parser.add_argument("--max-samples", type=int, default=8192)
    args = parser.parse_args()
    if args.mode == "regression":
        report = regression(args.matrix)
    elif args.mode == "smoke":
        if args.config_dir is None:
            parser.error("--config-dir is required for smoke")
        report = smoke(args.matrix, args.config_dir, args.max_samples)
    else:
        if args.config_dir is None or not args.run_key:
            parser.error("--config-dir and --run-key are required for formal")
        report = formal(args.matrix, args.config_dir, args.run_key)
    path = write_report(report)
    print(json.dumps({
        "status": report["status"],
        "mode": report["mode"],
        "report": str(path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
