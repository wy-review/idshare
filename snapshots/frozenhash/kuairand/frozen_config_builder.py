from pathlib import Path
import json
IDENTITY_FIELDS = ["video_id"]

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
