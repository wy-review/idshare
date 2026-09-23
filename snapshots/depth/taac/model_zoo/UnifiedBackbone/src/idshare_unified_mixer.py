"""UnifiedMixer with a matched Continuous/IDShare item carrier."""

from __future__ import annotations

import torch

from recscale.models.zero_anchor_identity_quantizer import (
    ZeroAnchorIdentityQuantizer,
)
from recscale.utils.zero_anchor_regularization import (
    add_identity_full_table_l2_gradients,
    add_zero_anchor_full_table_l2_gradients,
)

from .UnifiedBackbone import _UnifiedBase
from .unified_mixer import UnifiedMixerEncoder


class IDShareUnifiedMixer(_UnifiedBase):
    """Keep the UnifiedMixer backbone fixed and swap only item identity."""

    TARGET_FIELD = "target_item_id"
    SEQUENCE_FIELD = "item_seq"
    IDENTITY_FIELD = "shared_item_id"

    def __init__(
        self,
        feature_map,
        model_id="IDShareUnifiedMixer",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        d_model=64,
        num_layers=2,
        ffn_mult=2,
        ffn_type="per_token_swiglu",
        block_norm="pre",
        num_ns_tokens=11,
        seq_tokenizer="recent_k_plus_equal_chunks",
        num_seq_tokens=5,
        recent_k=2,
        maxlen=100,
        concat_mode="s_ns",
        use_semantic_bias=False,
        pooling="ns_tokens_mean",
        mlp_dims=(128, 64),
        dropout=0.0,
        norm_type="layer_norm",
        embedding_regularizer=None,
        net_regularizer=None,
        seq_pooling="attn",
        use_target_token=False,
        target_token_fields=None,
        idshare_enabled=False,
        idshare_quantizer_config=None,
        idshare_deduplicate_occurrences=True,
        full_table_l2_coefficient=0.0,
        full_table_l2_target=None,
        full_table_l2_first_regularized_row=4,
        full_table_l2_application_order="after_global_clip",
        **kwargs,
    ):
        super().__init__(
            feature_map,
            model_id,
            gpu,
            learning_rate,
            embedding_dim,
            d_model,
            num_ns_tokens,
            seq_tokenizer,
            num_seq_tokens,
            recent_k,
            maxlen,
            concat_mode,
            use_semantic_bias,
            pooling,
            list(mlp_dims),
            dropout,
            embedding_regularizer,
            net_regularizer,
            seq_pooling=seq_pooling,
            use_target_token=use_target_token,
            target_token_fields=target_token_fields,
            **kwargs,
        )
        self.backbone = UnifiedMixerEncoder(
            num_tokens=self.tokenizer.total_tokens,
            d_model=d_model,
            num_layers=num_layers,
            ffn_mult=ffn_mult,
            ffn_type=ffn_type,
            dropout=dropout,
            block_norm=block_norm,
            norm_type=norm_type,
        )

        self.idshare_enabled = bool(idshare_enabled)
        self.idshare_deduplicate_occurrences = bool(
            idshare_deduplicate_occurrences
        )
        self.identity_quantizer = None
        self.full_table_l2_coefficient = float(full_table_l2_coefficient)
        self.full_table_l2_target = full_table_l2_target
        self.full_table_l2_first_regularized_row = int(
            full_table_l2_first_regularized_row
        )
        self.full_table_l2_application_order = str(
            full_table_l2_application_order
        )
        self.full_table_l2_last_gradient_report = None
        expected_target = (
            "prequantization_assignment_table"
            if self.idshare_enabled
            else "identity_output_table"
        )
        if self.full_table_l2_coefficient < 0.0:
            raise ValueError("full-table L2 coefficient must be non-negative")
        if self.full_table_l2_coefficient > 0.0:
            if self.full_table_l2_target != expected_target:
                raise ValueError(
                    "full-table L2 target does not match the identity carrier"
                )
            if self.full_table_l2_application_order != "after_global_clip":
                raise ValueError("full-table L2 must run after global clipping")
            if self.full_table_l2_first_regularized_row != 4:
                raise ValueError("full-table L2 must begin at private row 4")
        self._assert_shared_item_table()
        if self.idshare_enabled:
            if not isinstance(idshare_quantizer_config, dict):
                raise ValueError("IDShare requires idshare_quantizer_config")
            table = self._shared_item_table()
            self.identity_quantizer = ZeroAnchorIdentityQuantizer(
                {"model": {ZeroAnchorIdentityQuantizer.CONFIG_KEY:
                           idshare_quantizer_config}},
                cardinalities=[table.num_embeddings],
                field_names=[self.IDENTITY_FIELD],
                embedding_dim=embedding_dim,
            )
            if self.identity_quantizer.code_logit_bias_enabled:
                raise ValueError("IDShare bridge does not support code-logit bias")

        self._compile_after_backbone()
        self.reset_parameters()
        if self.identity_quantizer is not None:
            self.identity_quantizer.initialize_identity_embeddings(
                [self._shared_item_table()]
            )
        self.model_to_device()

    def _assert_shared_item_table(self):
        layers = self.embedding_layer.embedding_layers
        if self.TARGET_FIELD not in layers or self.SEQUENCE_FIELD not in layers:
            raise ValueError("TAAC target/history identity fields are missing")
        if layers[self.TARGET_FIELD] is not layers[self.SEQUENCE_FIELD]:
            raise ValueError("target_item_id and item_seq must share one table")
        target_spec = self.feature_map.features[self.TARGET_FIELD]
        sequence_spec = self.feature_map.features[self.SEQUENCE_FIELD]
        if target_spec.get("padding_idx") != 0:
            raise ValueError("target item padding_idx must be zero")
        if sequence_spec.get("padding_idx") != 0:
            raise ValueError("history item padding_idx must be zero")
        if sequence_spec.get("share_embedding") != self.TARGET_FIELD:
            raise ValueError("history item share_embedding contract drift")

    def _shared_item_table(self):
        return self.embedding_layer.embedding_layers[self.TARGET_FIELD]

    @property
    def zero_anchor_identity_quantizer(self):
        return self.identity_quantizer

    def resolve_full_table_l2_identity_weights(self, identity_fields):
        if list(identity_fields) != [self.IDENTITY_FIELD]:
            raise ValueError("unexpected full-table L2 identity field")
        return [(self.IDENTITY_FIELD, self._shared_item_table().weight)]

    def _apply_full_table_l2(self):
        if self.full_table_l2_coefficient <= 0.0:
            return
        if self.idshare_enabled:
            report = add_zero_anchor_full_table_l2_gradients(
                self,
                coefficient=self.full_table_l2_coefficient,
                multiplier=1.0,
                first_regularized_row=(
                    self.full_table_l2_first_regularized_row
                ),
            )
        else:
            report = add_identity_full_table_l2_gradients(
                self,
                identity_fields=[self.IDENTITY_FIELD],
                field_names=[self.IDENTITY_FIELD],
                coefficient=self.full_table_l2_coefficient,
                multiplier=1.0,
                first_regularized_row=(
                    self.full_table_l2_first_regularized_row
                ),
            )
        if report["regularization_target"] != self.full_table_l2_target:
            raise RuntimeError("full-table L2 gradient target drift")
        self.full_table_l2_last_gradient_report = report

    def train_step(self, batch_data):
        for optimizer in self._optimizers:
            optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), getattr(self, "_max_gradient_norm", 1.0)
        )
        self._apply_full_table_l2()
        for optimizer in self._optimizers:
            optimizer.step()
        return loss

    def _quantized_item_embeddings(self, item_ids):
        if self.identity_quantizer is None:
            return self._shared_item_table()(item_ids)
        original_shape = item_ids.shape
        flat_ids = item_ids.reshape(-1)
        if self.idshare_deduplicate_occurrences:
            carrier_ids, inverse, counts = torch.unique(
                flat_ids,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
        else:
            carrier_ids = flat_ids
            inverse = None
            counts = None
        sparse_ids = carrier_ids.unsqueeze(1)
        table = self._shared_item_table()
        self.identity_quantizer.initialize_train_first_touch_rows(
            [table], sparse_ids
        )
        fields = table(carrier_ids).unsqueeze(1)
        quantized = self.identity_quantizer(
            fields,
            sparse_ids,
            row_weights=counts,
        )[:, 0, :]
        if inverse is not None:
            quantized = quantized[inverse]
        return quantized.reshape(*original_shape, -1)

    def _forward_tokens(self, inputs):
        X = self.get_inputs(inputs)
        embed_X = {key: value for key, value in X.items()
                   if key in self.feature_map.features}
        if self.identity_quantizer is not None:
            embed_X.pop(self.TARGET_FIELD)
            embed_X.pop(self.SEQUENCE_FIELD)
        feature_emb_dict = self.embedding_layer(embed_X)
        if self.identity_quantizer is not None:
            target = X[self.TARGET_FIELD]
            history = X[self.SEQUENCE_FIELD]
            flat = torch.cat((target.reshape(-1), history.reshape(-1)))
            shared = self._quantized_item_embeddings(flat)
            batch = target.shape[0]
            target_count = target.numel()
            feature_emb_dict[self.TARGET_FIELD] = shared[:target_count].reshape(
                batch, -1
            )
            feature_emb_dict[self.SEQUENCE_FIELD] = shared[target_count:].reshape(
                *history.shape, -1
            )
            if self.training:
                total_steps = max(
                    int(getattr(self, "_steps_per_epoch", 1))
                    * int(getattr(self, "_epochs_planned", 1)),
                    1,
                )
                current_step = max(int(getattr(self, "_total_steps", 1)) - 1, 0)
                self.identity_quantizer.set_progress(current_step, total_steps)
        return self.tokenizer(feature_emb_dict, X)

    def add_loss(self, return_dict, y_true):
        loss = super().add_loss(return_dict, y_true)
        if self.identity_quantizer is not None:
            auxiliary = self.identity_quantizer.get_aux_loss()
            if auxiliary is not None:
                loss = loss + auxiliary
        return loss

    def carrier_contract(self):
        table = self._shared_item_table()
        result = {
            "model": "UnifiedMixer",
            "carrier": "idshare" if self.idshare_enabled else "continuous",
            "target_field": self.TARGET_FIELD,
            "sequence_field": self.SEQUENCE_FIELD,
            "one_shared_table": True,
            "table_cardinality": int(table.num_embeddings),
            "embedding_dim": int(table.embedding_dim),
            "framework_global_l2_enabled": False,
            "full_table_l2": {
                "enabled": self.full_table_l2_coefficient > 0.0,
                "coefficient": self.full_table_l2_coefficient,
                "target": self.full_table_l2_target,
                "first_regularized_row": (
                    self.full_table_l2_first_regularized_row
                ),
                "application_order": self.full_table_l2_application_order,
                "last_gradient_report_present": (
                    self.full_table_l2_last_gradient_report is not None
                ),
                "last_gradient_report": (
                    self.full_table_l2_last_gradient_report
                ),
            },
            "all_checks_pass": True,
        }
        l2_checks = (
            self.full_table_l2_coefficient > 0.0
            and self.full_table_l2_target
            == (
                "prequantization_assignment_table"
                if self.idshare_enabled
                else "identity_output_table"
            )
            and self.full_table_l2_first_regularized_row == 4
            and self.full_table_l2_application_order == "after_global_clip"
            and self.full_table_l2_last_gradient_report is not None
        )
        result["all_checks_pass"] = bool(l2_checks)
        if self.identity_quantizer is not None:
            metadata = self.identity_quantizer.get_metadata()
            result["quantizer_metadata"] = metadata
            result["all_checks_pass"] = bool(
                result["all_checks_pass"]
                and
                metadata.get("num_subspaces") == 1
                and metadata.get("num_residual_levels") == 1
                and metadata.get("continuous_residual") is False
                and metadata.get("base_embedding_mode") == "split_zero_base"
                and metadata.get("oov_uses_quantized_zero_path_by_field")
                == {self.IDENTITY_FIELD: True}
            )
        return result
