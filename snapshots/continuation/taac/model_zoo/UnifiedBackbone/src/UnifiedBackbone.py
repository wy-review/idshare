import torch
from torch import nn

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block

from .tokenizers import UnifiedInputTokenizer, TOKEN_NS
from .unified_former import UnifiedFormerEncoder
from .unified_mixer import UnifiedMixerEncoder


class _UnifiedBase(BaseModel):
    def __init__(self, feature_map, model_id, gpu, learning_rate,
                 embedding_dim, d_model, num_ns_tokens, seq_tokenizer,
                 num_seq_tokens, recent_k, maxlen, concat_mode, use_semantic_bias,
                 pooling, mlp_dims, dropout, embedding_regularizer,
                 net_regularizer, seq_pooling="mean", use_target_token=False,
                 target_token_fields=None, tokenizer_class=None,
                 tokenizer_extra_kwargs=None, **kwargs):
        super().__init__(feature_map,
                         model_id=model_id,
                         gpu=gpu,
                         embedding_regularizer=embedding_regularizer,
                         net_regularizer=net_regularizer,
                         **kwargs)
        sequence_fields = kwargs.get("sequence_fields", kwargs.get("unified_sequence_fields"))
        if sequence_fields is None:
            sequence_fields = kwargs.get("hyformer_sequence_field", "item_seq")
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        tokenizer_class = tokenizer_class or UnifiedInputTokenizer
        tokenizer_extra_kwargs = dict(tokenizer_extra_kwargs or {})
        self.tokenizer = tokenizer_class(
            feature_map=feature_map,
            embedding_dim=embedding_dim,
            d_model=d_model,
            sequence_fields=sequence_fields,
            num_ns_tokens=num_ns_tokens,
            seq_tokenizer=seq_tokenizer,
            num_seq_tokens=num_seq_tokens,
            recent_k=recent_k,
            maxlen=maxlen,
            concat_mode=concat_mode,
            use_semantic_bias=use_semantic_bias,
            seq_pooling=seq_pooling,
            use_target_token=use_target_token,
            target_token_fields=target_token_fields,
            **tokenizer_extra_kwargs,
        )
        self.pooling = pooling
        if pooling not in ("all_tokens_mean", "ns_tokens_mean"):
            raise ValueError("pooling must be all_tokens_mean/ns_tokens_mean")
        self.dnn = MLP_Block(input_dim=d_model,
                             output_dim=1,
                             hidden_units=mlp_dims,
                             hidden_activations="ReLU",
                             output_activation=self.output_activation,
                             dropout_rates=dropout)
        self._compile_args = (kwargs["optimizer"], kwargs["loss"], learning_rate,
                              kwargs.get("optimizer_embedding"),
                              kwargs.get("embedding_lr_decay", True))

    def _compile_after_backbone(self):
        optimizer, loss, learning_rate, optimizer_embedding, embedding_lr_decay = self._compile_args
        self.compile(optimizer, loss, learning_rate,
                     optimizer_embedding=optimizer_embedding,
                     embedding_lr_decay=embedding_lr_decay)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        for feature in inputs.keys():
            if feature in self.feature_map.labels:
                continue
            if feature not in self.feature_map.features:
                if feature == "item_seq__timestamp":
                    X_dict[feature] = inputs[feature].to(self.device)
                continue
            spec = self.feature_map.features[feature]
            if spec["type"] == "meta":
                continue
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict

    def _pool(self, x, token_mask, token_type):
        if self.pooling == "ns_tokens_mean":
            mask = token_mask & (token_type == TOKEN_NS)
        else:
            mask = token_mask
        weights = mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def _forward_tokens(self, inputs):
        X = self.get_inputs(inputs)
        embed_X = {k: v for k, v in X.items() if k in self.feature_map.features}
        feature_emb_dict = self.embedding_layer(embed_X)
        return self.tokenizer(feature_emb_dict, X)

    def forward(self, inputs):
        tokens, token_mask, token_type = self._forward_tokens(inputs)
        x = self.backbone(tokens, token_mask, token_type)
        pooled = self._pool(x, token_mask, token_type)
        y_pred = self.dnn(pooled)
        return {"y_pred": y_pred}


class UnifiedFormer(_UnifiedBase):
    def __init__(self,
                 feature_map,
                 model_id="UnifiedFormer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 d_model=64,
                 num_layers=2,
                 num_heads=4,
                 ffn_mult=4,
                 ffn_type="per_token_swiglu",
                 qkv_type="shared_qkv",
                 attention_mask_type="causal",
                 local_window_size=16,
                 num_ns_tokens=8,
                 seq_tokenizer="equal_chunks",
                 num_seq_tokens=8,
                 recent_k=10,
                 maxlen=50,
                 concat_mode="s_ns",
                 use_semantic_bias=False,
                 pooling="all_tokens_mean",
                 mlp_dims=[128, 64],
                 dropout=0.1,
                 norm_type="layer_norm",
                 use_low_rank_qkv=False,
                 use_basis_hypernet=False,
                 use_score_calibration=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 seq_pooling="mean",
                 use_target_token=False,
                 target_token_fields=None,
                 **kwargs):
        super().__init__(feature_map, model_id, gpu, learning_rate,
                         embedding_dim, d_model, num_ns_tokens, seq_tokenizer,
                         num_seq_tokens, recent_k, maxlen, concat_mode, use_semantic_bias,
                         pooling, mlp_dims, dropout, embedding_regularizer,
                         net_regularizer, seq_pooling=seq_pooling,
                         use_target_token=use_target_token,
                         target_token_fields=target_token_fields, **kwargs)
        self.backbone = UnifiedFormerEncoder(
            num_tokens=self.tokenizer.total_tokens,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_mult=ffn_mult,
            qkv_type=qkv_type,
            ffn_type=ffn_type,
            attention_mask_type=attention_mask_type,
            local_window_size=local_window_size,
            dropout=dropout,
            norm_type=norm_type,
            use_low_rank_qkv=use_low_rank_qkv,
            use_basis_hypernet=use_basis_hypernet,
            use_score_calibration=use_score_calibration,
        )
        self._compile_after_backbone()
        self.reset_parameters()
        self.model_to_device()


class UnifiedMixer(_UnifiedBase):
    def __init__(self,
                 feature_map,
                 model_id="UnifiedMixer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 d_model=64,
                 num_layers=2,
                 ffn_mult=4,
                 ffn_type="per_token_swiglu",
                 block_norm="pre",
                 num_ns_tokens=8,
                 seq_tokenizer="equal_chunks",
                 num_seq_tokens=8,
                 recent_k=10,
                 maxlen=50,
                 concat_mode="s_ns",
                 use_semantic_bias=False,
                 pooling="all_tokens_mean",
                 mlp_dims=[128, 64],
                 dropout=0.1,
                 norm_type="layer_norm",
                 embedding_regularizer=None,
                 net_regularizer=None,
                 seq_pooling="mean",
                 use_target_token=False,
                 target_token_fields=None,
                 **kwargs):
        super().__init__(feature_map, model_id, gpu, learning_rate,
                         embedding_dim, d_model, num_ns_tokens, seq_tokenizer,
                         num_seq_tokens, recent_k, maxlen, concat_mode, use_semantic_bias,
                         pooling, mlp_dims, dropout, embedding_regularizer,
                         net_regularizer, seq_pooling=seq_pooling,
                         use_target_token=use_target_token,
                         target_token_fields=target_token_fields, **kwargs)
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
        self._compile_after_backbone()
        self.reset_parameters()
        self.model_to_device()
