import torch
import torch.nn as nn

from .unified_former import PerTokenSwiGLU, SharedSwiGLU, _make_norm


class TokenMixing(nn.Module):
    def __init__(self, num_tokens, d_model):
        super().__init__()
        if d_model % num_tokens != 0:
            raise ValueError("UnifiedMixer requires d_model divisible by total token count")
        self.num_tokens = num_tokens
        self.num_heads = num_tokens
        self.d_head = d_model // num_tokens
        self.d_model = d_model

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.num_tokens, self.num_heads, self.d_head)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(B, self.num_tokens, self.d_model)


class UnifiedMixerBlock(nn.Module):
    def __init__(self, num_tokens, d_model, ffn_mult=4,
                 ffn_type="per_token_swiglu", dropout=0.0,
                 block_norm="pre", norm_type="layer_norm"):
        super().__init__()
        if block_norm not in ("pre", "post"):
            raise ValueError("block_norm must be pre/post")
        self.block_norm = block_norm
        self.ln1 = _make_norm(norm_type, d_model)
        self.token_mixer = TokenMixing(num_tokens, d_model)
        self.ln2 = _make_norm(norm_type, d_model)
        if ffn_type == "per_token_swiglu":
            self.ffn = PerTokenSwiGLU(num_tokens, d_model, ffn_mult, dropout)
        elif ffn_type == "shared_swiglu":
            self.ffn = SharedSwiGLU(d_model, ffn_mult, dropout)
        else:
            raise ValueError("UnifiedMixer ffn_type supports per_token_swiglu/shared_swiglu")

    def forward(self, x):
        if self.block_norm == "pre":
            x = x + self.token_mixer(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.token_mixer(x))
            x = self.ln2(x + self.ffn(x))
        return x


class UnifiedMixerEncoder(nn.Module):
    def __init__(self, num_tokens, d_model, num_layers=2, ffn_mult=4,
                 ffn_type="per_token_swiglu", dropout=0.0,
                 block_norm="pre", norm_type="layer_norm"):
        super().__init__()
        self.blocks = nn.ModuleList([
            UnifiedMixerBlock(num_tokens, d_model, ffn_mult, ffn_type, dropout,
                              block_norm, norm_type)
            for _ in range(num_layers)
        ])
        self.final_ln = _make_norm(norm_type, d_model)

    def forward(self, tokens, token_mask=None, token_type=None):
        x = tokens
        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x)
            if token_mask is not None:
                x = x * token_mask.unsqueeze(-1).to(x.dtype)
        return self.final_ln(x)
