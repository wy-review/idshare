# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
import datetime as _dt
from fuxictr.pytorch.layers import FeatureEmbeddingDict
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import Monitor, not_in_whitelist
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]
        self._history = []  # List[Dict] — per-eval metrics for training_report.json
        self._started_at = None
        self._finished_at = None
        self.use_tensorboard = kwargs.get("tensorboard", True)
        self.writer = None
        self.tb_logdir = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                self.use_tensorboard = False
                logging.warning(
                    "TensorBoard is enabled but 'tensorboard' is not installed. Skip TensorBoard logging. Please install it or set tensorboard: False."
                )
            else:
                self.tb_logdir = os.path.join(self.model_dir, "tb_logs", self.model_id)
                self.writer = SummaryWriter(log_dir=self.tb_logdir)
                logging.info("TensorBoard log dir: {}".format(self.tb_logdir))

    def compile(self, optimizer, loss, lr, optimizer_embedding=None,
                embedding_lr_decay=True):
        """Create optimizer(s) and loss fn.

        Args:
            optimizer: str or dict. Optimizer for non-embedding (dense) params.
                Also used as the fallback single optimizer when
                ``optimizer_embedding`` is None (legacy behavior).
                If the dict's ``type`` is ``muon`` and an extra
                ``muon_fallback`` dict is present, dense params are split by
                ``ndim``: 2D params are trained by Muon, everything else by
                the fallback optimizer (typically AdamW). ``self.optimizer``
                points at the fallback, since fallback carries the "main" lr
                schedule (Muon uses a different lr magnitude).
            loss: str, loss fn name (see get_loss).
            lr: float, default learning rate. Passed to every optimizer as a
                fallback when its dict spec does not specify ``lr``.
            optimizer_embedding: None or dict. When set, enables the dual
                optimizer path: embedding (FeatureEmbeddingDict) params are
                trained by a separate optimizer built from this dict.
            embedding_lr_decay: bool, default True. When dual optimizer is on,
                whether plateau-triggered lr_decay also reduces embedding lr.
                Set to False when embedding_optimizer is self-adaptive (e.g.
                Adagrad) and you prefer to leave its schedule untouched.
        """
        self.loss_fn = get_loss(loss)
        self._embedding_lr_decay = embedding_lr_decay

        if optimizer_embedding is None:
            # Legacy single-optimizer path. Also supports Muon fallback for
            # models where the caller wants Muon on 2D weights but still
            # needs something to handle 1D biases / LayerNorm / embedding.
            net_opts = self._build_net_optimizers(optimizer, list(self.parameters()), lr)
            # self.optimizer points at the "main" optimizer: last one built,
            # which is the fallback when Muon splitting is active, otherwise
            # the sole optimizer.
            self.optimizer = net_opts[-1]
            self._optimizers = net_opts
            if len(net_opts) > 1:
                self._log_muon_setup(net_opts)
            return

        emb_params, net_params = self._split_embedding_params()
        if len(emb_params) == 0:
            raise ValueError(
                "optimizer_embedding is set but no FeatureEmbeddingDict "
                "params were found in the model")
        if len(net_params) == 0:
            raise ValueError(
                "optimizer_embedding is set but no non-embedding (dense) "
                "params were found in the model")

        net_opts = self._build_net_optimizers(optimizer, net_params, lr)
        emb_opt = get_optimizer(optimizer_embedding, emb_params, lr)
        # self.optimizer points at the dense-side "main" (fallback if Muon,
        # else the sole dense optimizer) so tensorboard logging and external
        # callers stay aligned with a traditional-lr quantity.
        self.optimizer = net_opts[-1]
        self.emb_optimizer = emb_opt
        self._optimizers = net_opts + [emb_opt]
        if len(net_opts) > 1:
            self._log_muon_setup(net_opts)
        self._log_optimizer_setup(emb_params, net_params, self.optimizer, emb_opt)

    def _build_net_optimizers(self, spec, params, lr):
        """Build one optimizer for dense params, or two when Muon fallback
        is requested.

        Muon fallback syntax: spec is a dict with ``type: muon`` AND a
        ``muon_fallback`` child dict. In that case:
          - 2D params -> Muon (built from ``spec`` with ``muon_fallback`` stripped)
          - non-2D params -> fallback optimizer (built from ``spec['muon_fallback']``)
        Order of the returned list is [muon, fallback]; callers treat the
        last element as the "main" optimizer for logging / lr schedules.
        """
        use_muon_fallback = (
            isinstance(spec, dict)
            and str(spec.get("type", "")).lower() == "muon"
            and "muon_fallback" in spec
        )
        if not use_muon_fallback:
            return [get_optimizer(spec, params, lr)]

        muon_spec = {k: v for k, v in spec.items() if k != "muon_fallback"}
        fallback_spec = spec["muon_fallback"]
        muon_params = [p for p in params if p.ndim == 2]
        fallback_params = [p for p in params if p.ndim != 2]
        if len(muon_params) == 0:
            raise ValueError(
                "optimizer type=muon with muon_fallback is set, but the "
                "model has no 2D parameters for Muon to train")
        if len(fallback_params) == 0:
            raise ValueError(
                "optimizer type=muon with muon_fallback is set, but every "
                "parameter is 2D; the fallback optimizer has nothing to train. "
                "Drop the muon_fallback key if you truly want Muon-only.")
        muon_opt = get_optimizer(muon_spec, muon_params, lr)
        fallback_opt = get_optimizer(fallback_spec, fallback_params, lr)
        return [muon_opt, fallback_opt]

    def _log_muon_setup(self, net_opts):
        """Log the Muon / fallback param split so users can sanity-check
        which tensors went where."""
        muon_opt, fallback_opt = net_opts
        def _count(groups):
            return sum(p.numel() for g in groups for p in g["params"])
        logging.info(
            "Muon fallback setup:\n"
            "  muon (2D):       {} ({:,} params, lr={})\n"
            "  fallback (rest): {} ({:,} params, lr={})".format(
                type(muon_opt).__name__, _count(muon_opt.param_groups),
                muon_opt.param_groups[0]["lr"],
                type(fallback_opt).__name__, _count(fallback_opt.param_groups),
                fallback_opt.param_groups[0]["lr"],
            )
        )

    def _split_embedding_params(self):
        """Partition parameters into (embedding, dense) lists.

        Uses the same FeatureEmbeddingDict rule that regularization_loss()
        follows, so "embedding" means "parameters owned by a
        FeatureEmbeddingDict module anywhere in the model tree".
        """
        emb_param_ids = set()
        for _, module in self.named_modules():
            if type(module) == FeatureEmbeddingDict:
                for p in module.parameters():
                    emb_param_ids.add(id(p))
        emb_params, net_params = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in emb_param_ids:
                emb_params.append(p)
            else:
                net_params.append(p)
        return emb_params, net_params

    def _log_optimizer_setup(self, emb_params, net_params, net_opt, emb_opt):
        def _count(ps):
            return sum(p.numel() for p in ps)
        logging.info(
            "Dual optimizer setup:\n"
            "  embedding: {} ({} tensors / {:,} params, lr={})\n"
            "  network:   {} ({} tensors / {:,} params, lr={})\n"
            "  embedding_lr_decay={}".format(
                type(emb_opt).__name__, len(emb_params), _count(emb_params),
                emb_opt.param_groups[0]["lr"],
                type(net_opt).__name__, len(net_params), _count(net_params),
                net_opt.param_groups[0]["lr"],
                self._embedding_lr_decay,
            )
        )

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            emb_params = set()
            for m_name, module in self.named_modules():
                if type(module) == FeatureEmbeddingDict:
                    for p_name, param in module.named_parameters():
                        if param.requires_grad:
                            emb_params.add(".".join([m_name, p_name]))
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name not in emb_params:
                        for net_p, net_lambda in net_reg:
                            reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return loss

    def compute_loss(self, return_dict, y_true):
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        return loss

    def reset_parameters(self):
        def default_reset_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def custom_reset_params(m):
            # initialize layers with customized init_weights()
            if hasattr(m, 'init_weights'):
                m.init_weights()
        self.apply(default_reset_params)
        self.apply(custom_reset_params)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        for feature in inputs.keys():
            if feature in self.feature_map.labels:
                continue
            spec = self.feature_map.features[feature]
            if spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(spec["source"], feature_source):
                continue
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)
        # Some optimizers (e.g. Adagrad with initial_accumulator_value > 0)
        # eagerly allocate state tensors in __init__, before the model has
        # been moved to GPU. After self.to(device) the parameters are on GPU
        # but the pre-allocated optimizer state still sits on CPU, which
        # triggers "Expected all tensors to be on the same device" on step().
        # Walk every active optimizer and re-home its state to match its
        # params' device. Adam and most others have no eager state so this
        # is a no-op for them.
        for opt in getattr(self, "_optimizers", [getattr(self, "optimizer", None)]):
            if opt is None:
                continue
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state.get(p)
                    if not state:
                        continue
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.device != p.device:
                            state[k] = v.to(p.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        """Reduce learning rate on plateau.

        Applies the factor to every param_group across all active optimizers.
        When dual optimizer is enabled and embedding_lr_decay=False, the
        embedding optimizer is skipped (useful for self-adaptive optimizers
        like Adagrad). Returns the new lr of the dense optimizer for logging.
        """
        dense_lr = None
        for opt in self._optimizers:
            if opt is getattr(self, "emb_optimizer", None) and not getattr(
                    self, "_embedding_lr_decay", True):
                continue
            for param_group in opt.param_groups:
                new_lr = max(param_group["lr"] * factor, min_lr)
                param_group["lr"] = new_lr
                if opt is self.optimizer and dense_lr is None:
                    dense_lr = new_lr
        return dense_lr
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._params = kwargs  # save for training_report.json
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        self._history = []
        self._started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self._epochs_planned = epochs
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        self._finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        # Reset post-fit eval counter so subsequent evaluate() calls
        # are tracked as valid_metrics (1st) and test_metrics (2nd).
        self._post_fit_eval_count = 0
        self._last_valid_result = {}
        self._last_test_result = {}
        # Record whether validation data was used; when absent, the first
        # post-fit evaluate() should go to test_metrics, not valid_metrics.
        self._had_validation = self.valid_gen is not None
        # Write training_report.json for remote-training-platform compatibility
        try:
            from fuxictr.pytorch.reporting import write_training_report
            report = self._build_report()
            report_path = write_training_report(
                report, output_dir=os.environ.get("JOB_OUTPUT_DIR", os.getcwd()))
            logging.info("Training report written to {}".format(report_path))
        except Exception as e:
            logging.warning("Failed to write training_report.json: {}".format(e))
        # When no validation data, ensure final model is saved as checkpoint
        # (eval_step skips evaluation when valid_gen is None, so no checkpoint
        # may have been saved; without this, load_weights would fail)
        if self.valid_gen is None:
            logging.info("No validation - saving final model as checkpoint.")
            self.save_weights(self.checkpoint)
        if self.writer is not None:
            self.writer.close()
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def _build_report(self):
        """Build training report dict for training_report.json.

        Called at the end of ``fit()``.  Uses accumulated history and
        model state to produce a report compatible with the
        remote-training-platform ``get_job_report`` API.
        """
        from fuxictr.pytorch.reporting import build_training_report
        return build_training_report(
            params=getattr(self, "_params", {}),
            model=self,
            valid_result=getattr(self, "_last_valid_result", {}),
            test_result=getattr(self, "_last_test_result", {}),
            started_at=self._started_at,
            finished_at=self._finished_at,
        )

    def _update_report_with_eval(self, eval_result):
        """Auto-update training_report.json after each evaluate() call.

        Tracks post-fit evaluate() calls and writes an updated report.
        The first evaluate() after fit() is treated as valid_metrics,
        the second as test_metrics.  This makes the report work for ALL
        models without modifying their run_expid.py.
        """
        if not eval_result or not getattr(self, "_started_at", None):
            return  # not in a training session, skip
        # Track which eval call this is (valid vs test)
        if not hasattr(self, "_post_fit_eval_count"):
            self._post_fit_eval_count = 0
        self._post_fit_eval_count += 1
        no_valid = getattr(self, "_had_validation", True) is False
        if self._post_fit_eval_count == 1 and not no_valid:
            self._last_valid_result = eval_result
        else:
            # 2nd eval, or 1st eval when no validation data → test_metrics
            self._last_test_result = eval_result
        # Rewrite the report
        try:
            from fuxictr.pytorch.reporting import write_training_report
            report = self._build_report()
            write_training_report(
                report, output_dir=os.environ.get("JOB_OUTPUT_DIR", os.getcwd()))
        except Exception as e:
            logging.debug("Auto-update training_report.json skipped: {}".format(e))

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        # Record history entry for training_report.json
        self._history.append({
            "epoch": self._epoch_index + 1,
            "step": self._total_steps,
            **logs,
        })
        monitor_value = self._monitor.get_value(logs)
        if self.writer is not None:
            self.writer.add_scalar("val/monitor", monitor_value, self._total_steps)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self._total_steps)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
                if self.writer is not None:
                    self.writer.add_scalar("train/lr", current_lr, self._total_steps)
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self.writer is not None:
                self.writer.add_scalar("val/best_metric", self._best_metric, self._total_steps)
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        if self.valid_gen is None:
            logging.info('No validation data - skipping evaluation, saving checkpoint.')
            self._history.append({
                "epoch": self._epoch_index + 1,
                "step": self._total_steps,
            })
            if not self._save_best_only:
                self.save_weights(self.checkpoint)
            return
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def train_step(self, batch_data):
        for opt in self._optimizers:
            opt.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        for opt in self._optimizers:
            opt.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            loss_value = loss.item()
            train_loss += loss_value
            if self.writer is not None:
                self.writer.add_scalar("train/batch_loss", loss_value, self._total_steps)
            if self._total_steps % self._eval_steps == 0:
                avg_loss = train_loss / self._eval_steps
                logging.info("Train loss: {:.6f}".format(avg_loss))
                if self.writer is not None:
                    self.writer.add_scalar("train/avg_loss", avg_loss, self._total_steps)
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None):
        if data_generator is None:
            logging.info('No validation data - skipping evaluation.')
            return {}
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            if self.writer is not None:
                for k, v in val_logs.items():
                    self.writer.add_scalar(f"val/{k}", v, self._total_steps)
            # Auto-update training_report.json with evaluation results
            self._update_report_with_eval(val_logs)
            return val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True, batch_size=4096):
        sparse_params = 0  # nn.Embedding 参数（categorical/sequence 特征 embedding 表）
        dense_params = 0   # 其他参数（network 层 + numeric Linear(1,d)）
        for _, module in self.named_modules():
            for param in module.parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if isinstance(module, nn.Embedding):
                    sparse_params += param.numel()
                else:
                    dense_params += param.numel()
        total_params = sparse_params + dense_params
        logging.info("Total number of parameters: {:,}.".format(total_params))
        logging.info("  - Sparse (embedding): {:,}".format(sparse_params))
        logging.info("  - Dense  (network):   {:,} ({:.2f}M)".format(dense_params, dense_params / 1e6))
        # Store for training_report.json
        self._sparse_params = sparse_params
        self._dense_params = dense_params
        # GFLOPs/Batch 统计：对齐 MixFormer 论文 (arXiv:2602.14110) Table 1 口径
        #   - #Params(M): dense only（排除 nn.Embedding），论文原文："only counts for the dense parameters"
        #   - GFLOPs: MACs → FLOPs = MACs × 2, G = 1e9 → GFLOPs = MACs × 2 / 1e9
        #   - batch_size: 使用实际训练 batch（MixFormer 论文 batch=1500），非 batch=1
        network_gflops = None
        try:
            from torchinfo import summary
            dummy_inputs = self._make_dummy_inputs(batch_size=batch_size)
            stats = summary(self, input_data=[dummy_inputs], verbose=0)
            network_macs = sum(li.macs for li in stats.summary_list
                               if li.is_leaf_layer and not isinstance(li.module, nn.Embedding))
            # MACs → GFLOPs: 1 MAC = 2 FLOPs, G = 1e9
            network_gflops = round(network_macs * 2 / 1e9, 4)
            logging.info("  - GFLOPs/Batch (batch_size={:,}, dense only): {:.4f}".format(batch_size, network_gflops))
        except Exception as e:
            logging.warning("GFLOPs estimation skipped: {}".format(e))
        self._gflops = network_gflops

    def _make_dummy_inputs(self, batch_size=1):
        """构造 dummy input dict，支持 categorical/numeric/sequence 所有特征类型"""
        dummy = {}
        # label 列（forward 内部会用 get_labels 读取）
        for label in self.feature_map.labels:
            dummy[label] = torch.zeros(batch_size, 1, device=self.device)
        # 特征列
        for fname, fspec in self.feature_map.features.items():
            ftype = fspec.get("type", "categorical")
            if ftype == "numeric":
                dummy[fname] = torch.zeros(batch_size, 1, device=self.device)
            elif ftype in ["categorical", "meta"]:
                dummy[fname] = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            elif ftype == "sequence":
                max_len = fspec.get("max_len", 10)
                dummy[fname] = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        return dummy

