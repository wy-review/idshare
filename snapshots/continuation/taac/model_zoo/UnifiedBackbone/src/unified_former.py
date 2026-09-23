import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import TOKEN_S


def _make_norm(norm_type, d_model):
    if norm_type == "rms_norm":
        return nn.RMSNorm(d_model)
    return nn.LayerNorm(d_model)


class PerTokenSwiGLU(nn.Module):
    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0):
        super().__init__()
        hidden = d_model * mult
        self.W_gate = nn.Parameter(torch.empty(num_tokens, d_model, hidden))
        self.W_up = nn.Parameter(torch.empty(num_tokens, d_model, hidden))
        self.W_down = nn.Parameter(torch.empty(num_tokens, hidden, d_model))
        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_up)
        nn.init.xavier_uniform_(self.W_down)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.einsum("btd,tdh->bth", x, self.W_gate)
        up = torch.einsum("btd,tdh->bth", x, self.W_up)
        h = gate * F.silu(up)
        h = self.dropout(h)
        y = torch.einsum("bth,thd->btd", h, self.W_down)
        return self.dropout(y)


class SharedSwiGLU(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        hidden = d_model * mult
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.gate(x) * F.silu(self.up(x))
        return self.dropout(self.down(self.dropout(h)))


class SeqSharedNsPerTokenSwiGLU(nn.Module):
    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.seq_ffn = SharedSwiGLU(d_model, mult, dropout)
        self.other_ffn = PerTokenSwiGLU(num_tokens, d_model, mult, dropout)

    def forward(self, x, token_type):
        seq_mask = (token_type == TOKEN_S).unsqueeze(-1)
        return torch.where(seq_mask, self.seq_ffn(x), self.other_ffn(x))


class UnifiedSelfAttention(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads, qkv_type="shared_qkv",
                 dropout=0.0, use_low_rank_qkv=False, use_basis_hypernet=False,
                 use_score_calibration=False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if use_low_rank_qkv:
            raise NotImplementedError("use_low_rank_qkv is a configurable skeleton and defaults to False")
        if use_basis_hypernet:
            raise NotImplementedError("use_basis_hypernet is a configurable skeleton and defaults to False")
        if qkv_type not in ("shared_qkv", "split_qkv", "context_qkv", "rankmixer_split"):
            raise ValueError("qkv_type must be shared_qkv/split_qkv/context_qkv/rankmixer_split")
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_type = qkv_type
        self.use_score_calibration = use_score_calibration
        if use_score_calibration:
            self.score_calibration = nn.Parameter(torch.ones(num_heads, num_tokens, num_tokens))
        if qkv_type == "shared_qkv":
            self.qkv = nn.Linear(d_model, 3 * d_model)
        elif qkv_type == "split_qkv":
            self.W_q = nn.Parameter(torch.empty(num_tokens, d_model, d_model))
            self.W_k = nn.Parameter(torch.empty(num_tokens, d_model, d_model))
            self.W_v = nn.Parameter(torch.empty(num_tokens, d_model, d_model))
            for p in (self.W_q, self.W_k, self.W_v):
                nn.init.xavier_uniform_(p)
        elif qkv_type == "context_qkv":
            self.W_ctx = nn.Parameter(torch.empty(num_tokens, num_tokens * d_model, 3 * d_model))
            self.b_ctx = nn.Parameter(torch.zeros(num_tokens, 3 * d_model))
            nn.init.xavier_uniform_(self.W_ctx)
        else:  # rankmixer_split: split first, then per-head independent projection
            H = num_heads
            d_h = self.head_dim
            self.W_q = nn.Parameter(torch.empty(H, d_h, d_h))
            self.W_k = nn.Parameter(torch.empty(H, d_h, d_h))
            self.W_v = nn.Parameter(torch.empty(H, d_h, d_h))
            for p in (self.W_q, self.W_k, self.W_v):
                nn.init.xavier_uniform_(p)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _qkv(self, x):
        B, T, D = x.shape
        if T != self.num_tokens:
            raise ValueError("Expected {} tokens, got {}".format(self.num_tokens, T))
        if self.qkv_type == "shared_qkv":
            qkv = self.qkv(x)
            return qkv.chunk(3, dim=-1)
        if self.qkv_type == "split_qkv":
            q = torch.einsum("btd,tde->bte", x, self.W_q)
            k = torch.einsum("btd,tde->bte", x, self.W_k)
            v = torch.einsum("btd,tde->bte", x, self.W_v)
            return q, k, v
        if self.qkv_type == "context_qkv":
            flat = x.flatten(start_dim=1)
            qkv = torch.einsum("bc,tco->bto", flat, self.W_ctx) + self.b_ctx
            return qkv.chunk(3, dim=-1)
        # rankmixer_split: split D into H heads, then per-head projection
        H = self.num_heads
        d_h = self.head_dim
        x_heads = x.view(B, T, H, d_h)                # [B, T, H, d_h]
        q = torch.einsum("bthi,hij->bthj", x_heads, self.W_q)  # [B, T, H, d_h]
        k = torch.einsum("bthi,hij->bthj", x_heads, self.W_k)
        v = torch.einsum("bthi,hij->bthj", x_heads, self.W_v)
        return q.reshape(B, T, D), k.reshape(B, T, D), v.reshape(B, T, D)

    def forward(self, x, allowed_mask):
        B, T, D = x.shape
        H = self.num_heads
        q, k, v = self._qkv(x)
        q = q.view(B, T, H, self.head_dim).transpose(1, 2)
        k = k.view(B, T, H, self.head_dim).transpose(1, 2)
        v = v.view(B, T, H, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.use_score_calibration:
            scores = scores * self.score_calibration.unsqueeze(0)
        scores = scores.masked_fill(~allowed_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class UnifiedFormerBlock(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads, ffn_mult=4,
                 qkv_type="shared_qkv", ffn_type="per_token_swiglu",
                 dropout=0.0, norm_type="layer_norm", **attention_kwargs):
        super().__init__()
        self.ln1 = _make_norm(norm_type, d_model)
        self.attn = UnifiedSelfAttention(num_tokens, d_model, num_heads, qkv_type, dropout,
                                         **attention_kwargs)
        self.ln2 = _make_norm(norm_type, d_model)
        self.ffn_type = ffn_type
        if ffn_type == "per_token_swiglu":
            self.ffn = PerTokenSwiGLU(num_tokens, d_model, ffn_mult, dropout)
        elif ffn_type == "shared_swiglu":
            self.ffn = SharedSwiGLU(d_model, ffn_mult, dropout)
        elif ffn_type == "seq_shared_ns_pertoken":
            self.ffn = SeqSharedNsPerTokenSwiGLU(num_tokens, d_model, ffn_mult, dropout)
        else:
            raise ValueError("ffn_type must be per_token_swiglu/shared_swiglu/seq_shared_ns_pertoken")

    def forward(self, x, allowed_mask, token_type):
        x = x + self.attn(self.ln1(x), allowed_mask)
        y = self.ln2(x)
        if self.ffn_type == "seq_shared_ns_pertoken":
            x = x + self.ffn(y, token_type)
        else:
            x = x + self.ffn(y)
        return x


class UnifiedFormerEncoder(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads, num_layers=2,
                 ffn_mult=4, qkv_type="shared_qkv", ffn_type="per_token_swiglu",
                 attention_mask_type="causal", local_window_size=16,
                 dropout=0.0, norm_type="layer_norm",
                 use_low_rank_qkv=False, use_basis_hypernet=False,
                 use_score_calibration=False):
        super().__init__()
        if attention_mask_type not in ("full", "causal", "seq_local"):
            raise ValueError("attention_mask_type must be full/causal/seq_local")
        self.num_tokens = num_tokens
        self.attention_mask_type = attention_mask_type
        self.local_window_size = local_window_size
        self.blocks = nn.ModuleList([
            UnifiedFormerBlock(
                num_tokens, d_model, num_heads, ffn_mult, qkv_type, ffn_type, dropout,
                norm_type=norm_type,
                use_low_rank_qkv=use_low_rank_qkv,
                use_basis_hypernet=use_basis_hypernet,
                use_score_calibration=use_score_calibration,
            ) for _ in range(num_layers)
        ])
        self.final_ln = _make_norm(norm_type, d_model)

    def _pattern_mask(self, token_type):
        device = token_type.device
        T = token_type.numel()
        if self.attention_mask_type == "full":
            return torch.ones(T, T, dtype=torch.bool, device=device)
        if self.attention_mask_type == "causal":
            return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        pos = torch.arange(T, device=device)
        local = (pos[:, None] - pos[None, :]).abs() <= self.local_window_size
        is_seq = token_type == TOKEN_S
        both_seq = is_seq[:, None] & is_seq[None, :]
        return torch.where(both_seq, local, torch.ones_like(local))

    def _allowed_mask(self, token_mask, token_type):
        pattern = self._pattern_mask(token_type[0])
        valid = token_mask[:, :, None] & token_mask[:, None, :]
        return valid & pattern.unsqueeze(0)

    def forward(self, tokens, token_mask, token_type):
        allowed_mask = self._allowed_mask(token_mask, token_type)
        x = tokens
        for block in self.blocks:
            x = block(x, allowed_mask, token_type)
            x = x * token_mask.unsqueeze(-1).to(x.dtype)
        return self.final_ln(x)
