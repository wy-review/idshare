"""
recscale.trainer — 固化训练循环

特性:
- BCEWithLogitsLoss (固定)
- 支持单卡 / DDP 多卡
- 可选 BF16 混合精度
- Checkpoint + Early Stopping
- 结构化 JSON 日志
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from .evaluator import CTREvaluator
from .utils.distributed import is_main_process, gather_predictions
from .utils.reproducibility import make_data_generator
from .utils.zero_anchor_regularization import (
    IDENTITY_OUTPUT_TABLE_TARGET,
    PREQUANTIZATION_ASSIGNMENT_TABLE_TARGET,
    QUANTIZED_OUTPUT_CODEBOOK_TARGET,
    add_identity_full_table_l2_gradients,
    add_zero_anchor_full_table_l2_gradients,
    add_zero_anchor_output_codebook_l2_gradients,
    get_identity_embedding_weights,
    get_zero_anchor_identity_quantizer,
    get_zero_anchor_identity_embedding_weights,
)


class Trainer:
    """
    固化训练器。子类不应覆盖 train() 方法。

    Args:
        model: RecModel 实例
        train_dataset: 训练数据集
        test_dataset: 测试数据集 (可选)
        config: 完整配置 dict
        device: torch.device
        rank: DDP rank (0 for single GPU)
        world_size: DDP world size (1 for single GPU)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        test_dataset,
        config: dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        tc = config["training"]
        self.epochs = tc["epochs"]
        self.log_every = tc.get("log_every", 100)
        self.eval_every = tc.get("eval_every", 0)
        self.use_amp = tc.get("use_amp", False)
        self.grad_clip = tc.get("grad_clip", 1.0)
        self.early_stop_patience = tc.get("early_stop_patience", 0)
        self.early_stop_min_epochs = int(tc.get("early_stop_min_epochs", 1))
        self.early_stop_min_delta = tc.get("early_stop_min_delta", 0.0)
        self.checkpoint_selection_tolerance = float(
            tc.get("checkpoint_selection_tolerance", 0.0)
        )
        self.suppress_effect_metric_logs = bool(
            tc.get("suppress_effect_metric_logs", False)
        )
        self.save_best_checkpoint = bool(tc.get("save_best_checkpoint", True))
        if self.early_stop_min_epochs < 1:
            raise ValueError("training.early_stop_min_epochs must be at least 1")
        if self.early_stop_min_epochs > self.epochs:
            raise ValueError(
                "training.early_stop_min_epochs cannot exceed training.epochs"
            )
        if self.early_stop_patience < 0:
            raise ValueError("training.early_stop_patience must be non-negative")
        if self.early_stop_min_delta < 0.0:
            raise ValueError("training.early_stop_min_delta must be non-negative")
        if self.checkpoint_selection_tolerance < 0.0:
            raise ValueError(
                "training.checkpoint_selection_tolerance must be non-negative"
            )
        if self.early_stop_patience > 0 and self.world_size != 1:
            raise RuntimeError(
                "early stopping is single-process only; world_size must equal 1"
            )
        if self.early_stop_patience > 0 and self.eval_every != 0:
            raise RuntimeError(
                "mid-epoch evaluation must be disabled when early stopping is enabled"
            )

        # Save dir
        self.save_dir = Path(tc.get("save_dir", "./outputs"))
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = None
        self.best_checkpoint_epoch = None
        self.best_checkpoint_step = None
        self.last_metrics = {}
        self.best_metrics = {}
        self.epoch_history = []
        self.epochs_ran = 0
        self.stop_reason = "not_started"
        self.training_step_observer = None

        # DataLoaders
        collate_fn = getattr(train_dataset, "collate_fn", None)
        num_workers = tc.get("num_workers", 4)
        batch_size = tc["batch_size"]
        self.data_seed = int(tc.get("data_seed", config.get("seed", 42)))
        self.data_generator = make_data_generator(self.data_seed)
        self.train_shuffle = bool(tc.get("shuffle", True))

        from torch.utils.data import IterableDataset as _IterableDataset
        train_is_iterable = isinstance(train_dataset, _IterableDataset)

        if train_is_iterable:
            # IterableDataset: 不能 shuffle/sampler，worker 分片由 dataset.__iter__ 自行处理
            train_sampler = None
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
                generator=self.data_generator,
            )
        elif world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=self.train_shuffle,
                seed=self.data_seed,
            )
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
                generator=self.data_generator,
            )
        else:
            train_sampler = None
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=self.train_shuffle,
                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
                generator=self.data_generator,
            )
        self.train_sampler = train_sampler

        temperature_anneal_epochs = tc.get("temperature_anneal_epochs")
        if temperature_anneal_epochs is None:
            self.temperature_anneal_steps = None
        else:
            temperature_anneal_epochs = int(temperature_anneal_epochs)
            if not 1 <= temperature_anneal_epochs <= self.epochs:
                raise ValueError(
                    "training.temperature_anneal_epochs must be in [1, epochs]"
                )
            self.temperature_anneal_steps = (
                temperature_anneal_epochs * len(self.train_loader)
            )

        if test_dataset is not None:
            test_is_iterable = isinstance(test_dataset, _IterableDataset)
            self.test_loader = DataLoader(
                test_dataset, batch_size=batch_size * 2, shuffle=False,
                num_workers=num_workers, collate_fn=collate_fn,
            )
        else:
            self.test_loader = None

        # DDP wrap
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank], find_unused_parameters=False,
            )

        # Loss + Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self._configure_zero_anchor_full_table_l2()
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        if is_main_process():
            print(
                "[Trainer] Reproducible data order: "
                f"data_seed={self.data_seed}, num_workers={num_workers}, "
                f"shuffle={self.train_shuffle}, "
                f"deterministic={bool(tc.get('deterministic', False))}"
            )

        # LR warmup
        self.warmup_steps = int(tc.get("warmup_steps", 0))
        self.base_lr = float(tc.get("lr", 0.001))
        if self.warmup_steps > 0:
            print(f"[Trainer] Warmup: {self.warmup_steps} steps, base_lr={self.base_lr}")

        # Evaluator
        self.evaluator = CTREvaluator()

        # Logging
        self.log_file = None
        if is_main_process():
            self.log_file = open(self.save_dir / "train.log", "w")

    def _build_optimizer(self):
        tc = self.config["training"]
        opt_name = tc.get("optimizer", "adam").lower()
        lr = tc["lr"]
        wd = tc.get("weight_decay", 0.0)
        foreach = tc.get("optimizer_foreach", None)
        embedding_wd = tc.get("embedding_weight_decay")
        embedding_wd_fields = tc.get("embedding_weight_decay_field_indices")
        if embedding_wd is None and embedding_wd_fields is not None:
            raise ValueError(
                "training.embedding_weight_decay_field_indices requires "
                "training.embedding_weight_decay"
            )
        if embedding_wd is None:
            params = self.model.parameters()
        else:
            selected_param_ids = None
            selected_fields = None
            if embedding_wd_fields is not None:
                selected_fields = [int(index) for index in embedding_wd_fields]
                if not selected_fields:
                    raise ValueError(
                        "training.embedding_weight_decay_field_indices must not be empty"
                    )
                if len(set(selected_fields)) != len(selected_fields):
                    raise ValueError(
                        "training.embedding_weight_decay_field_indices must be unique"
                    )
                raw_model = (
                    self.model.module
                    if isinstance(
                        self.model,
                        nn.parallel.DistributedDataParallel,
                    )
                    else self.model
                )
                embeddings = getattr(
                    getattr(getattr(raw_model, "encoder", None), "sparse_arch", None),
                    "embeddings",
                    None,
                )
                if embeddings is None:
                    raise ValueError(
                        "training.embedding_weight_decay_field_indices was set, but "
                        "model.encoder.sparse_arch.embeddings was not found"
                    )
                if min(selected_fields) < 0 or max(selected_fields) >= len(embeddings):
                    raise ValueError(
                        "training.embedding_weight_decay_field_indices must be within "
                        f"[0, {len(embeddings)}), got {selected_fields}"
                    )
                selected_param_ids = {
                    id(embeddings[index].weight) for index in selected_fields
                }

            embedding_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                is_embedding = "sparse_arch.embeddings" in name
                if selected_param_ids is not None:
                    is_embedding = id(param) in selected_param_ids
                if is_embedding:
                    embedding_params.append(param)
                else:
                    other_params.append(param)
            if not embedding_params:
                raise ValueError(
                    "training.embedding_weight_decay was set, but no "
                    "sparse_arch.embeddings parameters were found"
                )
            params = [
                {"params": embedding_params, "weight_decay": float(embedding_wd)},
                {"params": other_params, "weight_decay": float(wd)},
            ]
            if getattr(self, "rank", 0) == 0:
                print(
                    "[Trainer] Weight decay groups: "
                    f"sparse_embeddings={float(embedding_wd):g} "
                    f"fields={selected_fields if selected_fields is not None else 'all'} "
                    f"({len(embedding_params)} tensors), other={float(wd):g} "
                    f"({len(other_params)} tensors)"
                )

        if opt_name == "adam":
            kwargs = {"foreach": foreach} if foreach is not None else {}
            return torch.optim.Adam(params, lr=lr, weight_decay=wd, **kwargs)
        elif opt_name == "adamw":
            kwargs = {"foreach": foreach} if foreach is not None else {}
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, **kwargs)
        elif opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "adagrad":
            return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _configure_zero_anchor_full_table_l2(self) -> None:
        tc = self.config["training"]
        cfg = tc.get("zero_anchor_full_table_l2", {})
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ValueError("training.zero_anchor_full_table_l2 must be a mapping")
        self.zero_anchor_full_table_l2_enabled = bool(
            cfg.get("enabled", False)
        )
        self.zero_anchor_full_table_l2_last_gradient_report = None
        configured_target = cfg.get("target")
        if configured_target is not None and not isinstance(
            configured_target, str
        ):
            raise ValueError("whole-table L2 target must be a string")
        self.zero_anchor_full_table_l2_target_is_explicit = (
            configured_target is not None
        )
        self.zero_anchor_full_table_l2_target = configured_target
        self.zero_anchor_full_table_l2_coefficient = float(
            cfg.get("coefficient", 0.0)
        )
        self.zero_anchor_full_table_l2_application_order = str(
            cfg.get("application_order", "before_global_clip")
        )
        if self.zero_anchor_full_table_l2_application_order not in {
            "before_global_clip",
            "after_global_clip",
        }:
            raise ValueError(
                "whole-table L2 application_order must be "
                "'before_global_clip' or 'after_global_clip'"
            )
        self.zero_anchor_full_table_l2_first_regularized_row = int(
            cfg.get("first_regularized_row", 0)
        )
        if self.zero_anchor_full_table_l2_first_regularized_row < 0:
            raise ValueError(
                "whole-table L2 first_regularized_row must be non-negative"
            )
        release_is_explicit = "release_fraction" in cfg
        ramp_is_explicit = "ramp_fraction" in cfg
        if release_is_explicit != ramp_is_explicit:
            raise ValueError(
                "whole-table L2 release_fraction and ramp_fraction must be "
                "configured together"
            )
        self.zero_anchor_full_table_l2_schedule_is_explicit = bool(
            release_is_explicit
        )
        self.zero_anchor_full_table_l2_release_fraction = float(
            cfg.get("release_fraction", 0.0)
        )
        self.zero_anchor_full_table_l2_ramp_fraction = float(
            cfg.get("ramp_fraction", 0.0)
        )
        if not (
            0.0 <= self.zero_anchor_full_table_l2_release_fraction <= 1.0
            and 0.0 <= self.zero_anchor_full_table_l2_ramp_fraction <= 1.0
            and (
                self.zero_anchor_full_table_l2_release_fraction
                + self.zero_anchor_full_table_l2_ramp_fraction
                <= 1.0
            )
        ):
            raise ValueError(
                "whole-table L2 release/ramp fractions must be in [0,1] "
                "and sum to at most 1"
            )
        self.zero_anchor_full_table_l2_current_multiplier = (
            1.0
            if (
                self.zero_anchor_full_table_l2_release_fraction == 0.0
                and self.zero_anchor_full_table_l2_ramp_fraction == 0.0
            )
            else 0.0
        )
        if self.zero_anchor_full_table_l2_coefficient < 0.0:
            raise ValueError(
                "training.zero_anchor_full_table_l2.coefficient must be non-negative"
            )
        if not self.zero_anchor_full_table_l2_enabled:
            if self.zero_anchor_full_table_l2_coefficient != 0.0:
                raise ValueError(
                    "whole-table L2 coefficient must be zero when disabled"
                )
            if self.zero_anchor_full_table_l2_schedule_is_explicit:
                raise ValueError(
                    "disabled whole-table L2 must not configure a release/ramp "
                    "schedule"
                )
            self.zero_anchor_full_table_l2_fields = []
            return
        if self.zero_anchor_full_table_l2_coefficient <= 0.0:
            raise ValueError(
                "enabled whole-table L2 requires a positive coefficient"
            )
        if tc.get("embedding_weight_decay") is not None:
            raise ValueError(
                "whole-table zero-anchor L2 cannot be combined with "
                "training.embedding_weight_decay"
            )

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        quantizer = get_zero_anchor_identity_quantizer(raw_model)
        if quantizer is None:
            expected_target = IDENTITY_OUTPUT_TABLE_TARGET
            if (
                self.zero_anchor_full_table_l2_target is not None
                and self.zero_anchor_full_table_l2_target != expected_target
            ):
                raise ValueError(
                    "continuous whole-table L2 target must be "
                    f"{expected_target!r}"
                )
            self.zero_anchor_full_table_l2_target = expected_target
            identity_fields = cfg.get("identity_fields")
            if not isinstance(identity_fields, (list, tuple)) or not identity_fields:
                raise ValueError(
                    "continuous whole-table L2 requires non-empty identity_fields"
                )
            field_names = self.config["dataset"].get("sparse_cols", [])
            self.zero_anchor_full_table_l2_fields = [
                field
                for field, _ in get_identity_embedding_weights(
                    raw_model,
                    identity_fields=identity_fields,
                    field_names=field_names,
                )
            ]
            self.zero_anchor_full_table_l2_mode = "continuous_identity"
            if is_main_process():
                print(
                    "[Trainer] Continuous identity whole-table L2: "
                    f"target={self.zero_anchor_full_table_l2_target}, "
                    f"coefficient={self.zero_anchor_full_table_l2_coefficient:g}, "
                    f"fields={self.zero_anchor_full_table_l2_fields}, "
                    "first_regularized_row="
                    f"{self.zero_anchor_full_table_l2_first_regularized_row}, "
                    "release="
                    f"{self.zero_anchor_full_table_l2_release_fraction:g}, "
                    "ramp="
                    f"{self.zero_anchor_full_table_l2_ramp_fraction:g}, "
                    "application_order="
                    f"{self.zero_anchor_full_table_l2_application_order}"
                )
            return
        allowed_targets = {
            PREQUANTIZATION_ASSIGNMENT_TABLE_TARGET,
            QUANTIZED_OUTPUT_CODEBOOK_TARGET,
        }
        if (
            self.zero_anchor_full_table_l2_target is not None
            and self.zero_anchor_full_table_l2_target not in allowed_targets
        ):
            raise ValueError(
                "quantized L2 target must be one of "
                f"{sorted(allowed_targets)}"
            )
        if self.zero_anchor_full_table_l2_target is None:
            self.zero_anchor_full_table_l2_target = (
                PREQUANTIZATION_ASSIGNMENT_TABLE_TARGET
            )
        if not quantizer.regularization_schedule_is_explicit:
            raise ValueError(
                "whole-table zero-anchor L2 requires explicit "
                "regularization_release_fraction and regularization_ramp_fraction"
            )
        if quantizer.zero_l2_weight != 0.0:
            raise ValueError(
                "formal whole-table L2 cannot be mixed with touched-row zero_l2_weight"
            )
        quantizer_release = float(
            quantizer.regularization_release_fraction
        )
        quantizer_ramp = float(quantizer.regularization_ramp_fraction)
        if self.zero_anchor_full_table_l2_schedule_is_explicit:
            if (
                self.zero_anchor_full_table_l2_release_fraction
                != quantizer_release
                or self.zero_anchor_full_table_l2_ramp_fraction
                != quantizer_ramp
            ):
                raise ValueError(
                    "whole-table L2 and quantizer release/ramp schedules must "
                    "match exactly"
                )
        else:
            self.zero_anchor_full_table_l2_release_fraction = (
                quantizer_release
            )
            self.zero_anchor_full_table_l2_ramp_fraction = quantizer_ramp
            self.zero_anchor_full_table_l2_current_multiplier = (
                quantizer.current_regularization_multiplier
            )
        if (
            self.zero_anchor_full_table_l2_target
            == PREQUANTIZATION_ASSIGNMENT_TABLE_TARGET
        ):
            self.zero_anchor_full_table_l2_fields = [
                field
                for field, _ in get_zero_anchor_identity_embedding_weights(
                    raw_model
                )
            ]
            self.zero_anchor_full_table_l2_mode = (
                "zero_anchor_assignment_coordinates"
            )
        else:
            if self.zero_anchor_full_table_l2_first_regularized_row != 1:
                raise ValueError(
                    "output-codebook L2 requires first_regularized_row=1 "
                    "to document exclusion of the fixed zero code"
                )
            if quantizer.num_subspaces != 1 or quantizer.num_residual_levels != 1:
                raise ValueError(
                    "output-codebook L2 currently supports Product-M1 only"
                )
            self.zero_anchor_full_table_l2_fields = list(
                quantizer.identity_fields
            )
            self.zero_anchor_full_table_l2_mode = "zero_anchor_output_codebook"
        if is_main_process():
            print(
                "[Trainer] Zero-anchor whole-table L2: "
                f"target={self.zero_anchor_full_table_l2_target}, "
                f"coefficient={self.zero_anchor_full_table_l2_coefficient:g}, "
                f"fields={self.zero_anchor_full_table_l2_fields}, "
                "first_regularized_row="
                f"{self.zero_anchor_full_table_l2_first_regularized_row}, "
                f"release={self.zero_anchor_full_table_l2_release_fraction:g}, "
                f"ramp={self.zero_anchor_full_table_l2_ramp_fraction:g}, "
                "application_order="
                f"{self.zero_anchor_full_table_l2_application_order}"
            )

    @staticmethod
    def _release_ramp_multiplier(
        current_step: int,
        total_steps: int,
        *,
        release_fraction: float,
        ramp_fraction: float,
    ) -> float:
        denominator = max(int(total_steps) - 1, 1)
        progress = min(
            max(float(current_step) / denominator, 0.0),
            1.0,
        )
        if progress < release_fraction:
            return 0.0
        if ramp_fraction == 0.0:
            return 1.0
        return min(
            max((progress - release_fraction) / ramp_fraction, 0.0),
            1.0,
        )

    def _set_zero_anchor_full_table_l2_progress(
        self,
        current_step: int,
        total_steps: int,
        raw_model: nn.Module,
    ) -> None:
        if not self.zero_anchor_full_table_l2_enabled:
            return
        expected = self._release_ramp_multiplier(
            current_step,
            total_steps,
            release_fraction=(
                self.zero_anchor_full_table_l2_release_fraction
            ),
            ramp_fraction=self.zero_anchor_full_table_l2_ramp_fraction,
        )
        quantizer = get_zero_anchor_identity_quantizer(raw_model)
        if quantizer is not None:
            actual = float(quantizer.current_regularization_multiplier)
            if abs(actual - expected) > 1e-12:
                raise RuntimeError(
                    "quantizer and whole-table L2 schedules diverged: "
                    f"expected={expected}, actual={actual}"
                )
        self.zero_anchor_full_table_l2_current_multiplier = expected

    def _apply_zero_anchor_full_table_l2(self, raw_model: nn.Module) -> None:
        if not self.zero_anchor_full_table_l2_enabled:
            return
        quantizer = get_zero_anchor_identity_quantizer(raw_model)
        if quantizer is None:
            self.zero_anchor_full_table_l2_last_gradient_report = (
                add_identity_full_table_l2_gradients(
                    raw_model,
                    identity_fields=self.zero_anchor_full_table_l2_fields,
                    field_names=self.config["dataset"]["sparse_cols"],
                    coefficient=self.zero_anchor_full_table_l2_coefficient,
                    multiplier=(
                        self.zero_anchor_full_table_l2_current_multiplier
                    ),
                    first_regularized_row=(
                        self.zero_anchor_full_table_l2_first_regularized_row
                    ),
                )
            )
        elif (
            self.zero_anchor_full_table_l2_target
            == PREQUANTIZATION_ASSIGNMENT_TABLE_TARGET
        ):
            self.zero_anchor_full_table_l2_last_gradient_report = (
                add_zero_anchor_full_table_l2_gradients(
                    raw_model,
                    coefficient=self.zero_anchor_full_table_l2_coefficient,
                    multiplier=(
                        self.zero_anchor_full_table_l2_current_multiplier
                    ),
                    first_regularized_row=(
                        self.zero_anchor_full_table_l2_first_regularized_row
                    ),
                )
            )
        else:
            self.zero_anchor_full_table_l2_last_gradient_report = (
                add_zero_anchor_output_codebook_l2_gradients(
                    raw_model,
                    coefficient=self.zero_anchor_full_table_l2_coefficient,
                    multiplier=(
                        self.zero_anchor_full_table_l2_current_multiplier
                    ),
                )
            )
        actual_target = self.zero_anchor_full_table_l2_last_gradient_report[
            "regularization_target"
        ]
        if actual_target != self.zero_anchor_full_table_l2_target:
            raise RuntimeError(
                "whole-table L2 gradient target diverged from configuration: "
                f"configured={self.zero_anchor_full_table_l2_target!r}, "
                f"actual={actual_target!r}"
            )

    def _postprocess_gradients(self, raw_model: nn.Module) -> None:
        """Clip task gradients and apply L2 in the registered order."""
        if (
            self.zero_anchor_full_table_l2_enabled
            and self.zero_anchor_full_table_l2_application_order
            == "before_global_clip"
        ):
            self._apply_zero_anchor_full_table_l2(raw_model)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        )
        if (
            self.zero_anchor_full_table_l2_enabled
            and self.zero_anchor_full_table_l2_application_order
            == "after_global_clip"
        ):
            self._apply_zero_anchor_full_table_l2(raw_model)

    def set_training_step_observer(self, observer) -> None:
        """Attach an optional read-only observer called after optimizer steps."""
        if observer is not None and not callable(observer):
            raise TypeError("training step observer must be callable or None")
        self.training_step_observer = observer

    def train(self):
        """固化训练循环 — 不允许覆盖"""
        checkpoint_best_auc = float("-inf")
        patience_reference_auc = float("-inf")
        no_improve_epochs = 0
        global_step = 0
        t0 = time.time()
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        total_steps = self.epochs * len(self.train_loader)
        temperature_total_steps = (
            self.temperature_anneal_steps or total_steps
        )
        self.stop_reason = "max_epochs"

        if is_main_process():
            params = raw_model.get_num_params()
            print(f"[Trainer] Model: {raw_model.model_name}")
            print(f"[Trainer] Params: {params}")
            print(f"[Trainer] Epochs: {self.epochs}, Batch: {self.config['training']['batch_size']}")
            print(f"[Trainer] Device: {self.device}, World: {self.world_size}, AMP: {self.use_amp}")
            if self.early_stop_patience > 0:
                print(
                    "[Trainer] EarlyStop: "
                    f"min_epochs={self.early_stop_min_epochs}, "
                    f"patience={self.early_stop_patience}, "
                    f"min_delta={self.early_stop_min_delta}"
                )

        for epoch in range(1, self.epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)

                # Temperature annealing for models like UniMixerV2
                if hasattr(raw_model, "set_tau_for_step"):
                    temperature_step = min(
                        global_step,
                        max(temperature_total_steps - 1, 0),
                    )
                    raw_model.set_tau_for_step(
                        temperature_step,
                        temperature_total_steps,
                    )
                self._set_zero_anchor_full_table_l2_progress(
                    global_step,
                    total_steps,
                    raw_model,
                )

                # Forward
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = self.model(batch)
                        if isinstance(out, tuple):
                            logits, aux_loss = out
                        else:
                            logits, aux_loss = out, None
                        loss = self.criterion(logits, batch["label"])
                        if aux_loss is not None:
                            loss = loss + aux_loss
                else:
                    out = self.model(batch)
                    if isinstance(out, tuple):
                        logits, aux_loss = out
                    else:
                        logits, aux_loss = out, None
                    loss = self.criterion(logits, batch["label"])
                    if aux_loss is not None:
                        loss = loss + aux_loss

                # Backward
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                self._postprocess_gradients(raw_model)

                if (
                    hasattr(raw_model, "should_record_discrete_gradient_audit")
                    and raw_model.should_record_discrete_gradient_audit(global_step)
                ):
                    raw_model.record_discrete_gradient_audit()

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.training_step_observer is not None:
                    with torch.no_grad():
                        self.training_step_observer(
                            raw_model,
                            completed_step=global_step + 1,
                            total_steps=total_steps,
                        )

                # LR warmup
                if self.warmup_steps > 0 and global_step < self.warmup_steps:
                    warmup_lr = self.base_lr * (global_step + 1) / self.warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = warmup_lr

                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # Log
                if is_main_process() and global_step % self.log_every == 0:
                    log = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": round(loss.item(), 6),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "time": round(time.time() - t0, 1),
                    }
                    log.update(self._collect_model_diagnostics(raw_model))
                    print(json.dumps(log))
                    if self.log_file:
                        self.log_file.write(json.dumps(log) + "\n")
                        self.log_file.flush()

                # Mid-epoch eval
                if self.eval_every > 0 and global_step % self.eval_every == 0:
                    metrics = self._evaluate()
                    self.last_metrics = metrics
                    if is_main_process() and metrics:
                        if self.suppress_effect_metric_logs:
                            print(
                                f"  [Eval] step={global_step} "
                                "effect metrics withheld until matrix completion"
                            )
                        else:
                            print(
                                f"  [Eval] step={global_step} "
                                f"AUC={metrics['auc']:.4f} "
                                f"LogLoss={metrics['logloss']:.4f}"
                            )
                        if (
                            metrics["auc"]
                            > checkpoint_best_auc
                            + self.checkpoint_selection_tolerance
                        ):
                            checkpoint_best_auc = metrics["auc"]
                            self.best_metrics = metrics
                            self._save_checkpoint(epoch, global_step, metrics)

            # End-of-epoch eval
            avg_loss = epoch_loss / max(epoch_steps, 1)
            metrics = self._evaluate()
            self.last_metrics = metrics

            if is_main_process():
                print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}", end="")
                if metrics:
                    if self.suppress_effect_metric_logs:
                        print(" | effect metrics withheld until matrix completion")
                    else:
                        print(
                            f" | AUC={metrics['auc']:.4f} "
                            f"LogLoss={metrics['logloss']:.4f}"
                        )
                    auc = float(metrics["auc"])
                    if not np.isfinite(auc):
                        raise RuntimeError(
                            f"non-finite validation AUC at epoch {epoch}: {auc}"
                        )
                    if (
                        auc
                        > checkpoint_best_auc
                        + self.checkpoint_selection_tolerance
                    ):
                        checkpoint_best_auc = auc
                        self.best_metrics = metrics
                        self._save_checkpoint(epoch, global_step, metrics)
                    if auc > patience_reference_auc + self.early_stop_min_delta:
                        patience_reference_auc = auc
                        no_improve_epochs = 0
                    elif self.early_stop_patience > 0:
                        no_improve_epochs += 1
                        print(
                            f"  [EarlyStop] no improvement for "
                            f"{no_improve_epochs}/{self.early_stop_patience} epochs"
                        )
                else:
                    print()

            self.epochs_ran = epoch
            self.epoch_history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "average_loss": float(avg_loss),
                    "metrics": dict(metrics),
                    "checkpoint_best_auc": (
                        None
                        if not np.isfinite(checkpoint_best_auc)
                        else float(checkpoint_best_auc)
                    ),
                    "patience_reference_auc": (
                        None
                        if not np.isfinite(patience_reference_auc)
                        else float(patience_reference_auc)
                    ),
                    "no_improve_epochs": int(no_improve_epochs),
                }
            )

            if (
                self.early_stop_patience > 0
                and epoch >= self.early_stop_min_epochs
                and no_improve_epochs >= self.early_stop_patience
            ):
                self.stop_reason = "early_stopping"
                if is_main_process():
                    if self.suppress_effect_metric_logs:
                        print(
                            f"  [EarlyStop] Stop at epoch {epoch}; "
                            "best effect metric withheld"
                        )
                    else:
                        print(
                            f"  [EarlyStop] Stop at epoch {epoch}; "
                            f"best AUC={checkpoint_best_auc:.4f}"
                        )
                break

        elapsed = time.time() - t0
        if is_main_process():
            if self.suppress_effect_metric_logs:
                print(
                    f"\n[Trainer] Done in {elapsed:.1f}s. "
                    "Effect metrics withheld until matrix completion."
                )
            else:
                print(
                    f"\n[Trainer] Done in {elapsed:.1f}s. "
                    f"Best AUC: {checkpoint_best_auc:.4f}"
                )
            if self.log_file:
                self.log_file.close()

        return checkpoint_best_auc

    def evaluate_dataset(self, dataset) -> dict:
        """Evaluate a map-style dataset with the frozen evaluation settings."""
        if dataset is None:
            return {}
        collate_fn = getattr(dataset, "collate_fn", None)
        loader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"] * 2,
            shuffle=False,
            num_workers=self.config["training"].get("num_workers", 4),
            collate_fn=collate_fn,
        )
        return self._evaluate(loader=loader)

    def _evaluate(self, *, loader=None) -> dict:
        """评估循环"""
        evaluation_loader = self.test_loader if loader is None else loader
        if evaluation_loader is None:
            return {}

        self.model.eval()
        score_chunks = []
        label_chunks = []
        k1_band_chunks = []
        k1_side_chunks = []
        has_k1_metadata = None

        with torch.no_grad():
            for batch in evaluation_loader:
                batch = self._to_device(batch)

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = self.model(batch)
                else:
                    out = self.model(batch)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out

                scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                labels = batch["label"].detach().cpu().numpy().reshape(-1)
                score_chunks.append(scores)
                label_chunks.append(labels)
                batch_has_k1 = (
                    "k1_video_train_band" in batch and "k1_side_bits" in batch
                )
                if has_k1_metadata is None:
                    has_k1_metadata = batch_has_k1
                elif has_k1_metadata != batch_has_k1:
                    raise RuntimeError("K1 evaluation metadata is missing from some batches")
                if batch_has_k1:
                    k1_band_chunks.append(
                        batch["k1_video_train_band"]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    k1_side_chunks.append(
                        batch["k1_side_bits"].detach().cpu().numpy().reshape(-1)
                    )

        all_scores = np.concatenate(score_chunks) if score_chunks else np.empty(0)
        all_labels = np.concatenate(label_chunks) if label_chunks else np.empty(0)

        # DDP gather
        if self.world_size > 1:
            all_scores, all_labels = gather_predictions(
                all_scores.tolist(), all_labels.tolist(), self.world_size, self.device
            )
            if has_k1_metadata:
                raise RuntimeError("K1 segmented evaluation currently requires one GPU")

        self.model.train()

        if is_main_process():
            metrics = self.evaluator.compute(all_labels, all_scores)
            if has_k1_metadata:
                from .analysis.kuairand_k1_segmented import (
                    evaluate_kuairand_k1_segments,
                )

                metrics["k1_segments"] = evaluate_kuairand_k1_segments(
                    all_labels,
                    all_scores,
                    np.concatenate(k1_band_chunks),
                    np.concatenate(k1_side_chunks),
                )
            return metrics
        return {}

    def _save_checkpoint(self, epoch: int, step: int, metrics: dict):
        """保存最佳模型"""
        if not self.save_best_checkpoint:
            return
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = self.save_dir / f"best_auc{metrics['auc']:.4f}_step{step}.pt"
        torch.save({
            "model": raw_model.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "config": self.config,
        }, ckpt_path)
        self.best_checkpoint_path = ckpt_path
        self.best_checkpoint_epoch = int(epoch)
        self.best_checkpoint_step = int(step)
        print(f"  [Checkpoint] Saved to {ckpt_path}")

    def restore_best_model(self) -> bool:
        """Restore the best checkpoint before optional post-training diagnostics."""
        if self.best_checkpoint_path is None:
            return False
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = torch.load(
            self.best_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        raw_model.load_state_dict(checkpoint["model"])
        print(f"[Trainer] Restored best checkpoint: {self.best_checkpoint_path}")
        return True

    @staticmethod
    def _collect_model_diagnostics(raw_model: nn.Module) -> dict:
        if not hasattr(raw_model, "get_tokenizer_diagnostics"):
            return {}
        diagnostics = raw_model.get_tokenizer_diagnostics()
        log_values = {}
        for name, value in diagnostics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().item()
            log_values[f"tok_{name}"] = round(float(value), 6)
        return log_values

    def _to_device(self, batch: dict) -> dict:
        """移动 batch 到 device"""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out
