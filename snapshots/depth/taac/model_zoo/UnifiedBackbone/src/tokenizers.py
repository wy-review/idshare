import torch
import torch.nn as nn


TOKEN_S = 0
TOKEN_NS = 1
TOKEN_BOS = 2
TOKEN_SEP = 3


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


class UnifiedInputTokenizer(nn.Module):
    def __init__(self, feature_map, embedding_dim, d_model,
                 sequence_fields, num_ns_tokens=8, seq_tokenizer="equal_chunks",
                 num_seq_tokens=8, recent_k=10, maxlen=50, concat_mode="s_ns",
                 ns_pooling="mean", seq_pooling="mean", use_semantic_bias=False,
                 use_target_token=False, target_token_fields=None):
        super().__init__()
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.sequence_fields = _as_list(sequence_fields)
        self.sequence_timestamp_field = "item_seq__timestamp"
        self.num_ns_tokens = num_ns_tokens
        self.seq_tokenizer = seq_tokenizer
        self.num_seq_tokens = num_seq_tokens
        self.recent_k = recent_k
        self.maxlen = maxlen
        self.concat_mode = concat_mode
        self.ns_pooling = ns_pooling
        self.seq_pooling = seq_pooling
        self.use_semantic_bias = use_semantic_bias
        self.use_target_token = use_target_token

        if num_ns_tokens <= 0:
            raise ValueError("num_ns_tokens must be positive")

        if seq_tokenizer not in ("item", "equal_chunks", "timestamp_diff",
                                 "recent_k_plus_equal_chunks"):
            raise ValueError("seq_tokenizer must be item/equal_chunks/timestamp_diff/recent_k_plus_equal_chunks")
        if seq_tokenizer == "recent_k_plus_equal_chunks" and not (0 < recent_k < num_seq_tokens):
            raise ValueError("recent_k_plus_equal_chunks requires 0 < recent_k < num_seq_tokens")
        if concat_mode not in ("s_ns", "bos_s_ns", "bos_s_sep_ns"):
            raise ValueError("concat_mode must be s_ns/bos_s_ns/bos_s_sep_ns")
        if ns_pooling not in ("mean", "sum") or seq_pooling not in ("mean", "sum", "attn"):
            raise ValueError("ns_pooling must be mean/sum; seq_pooling must be mean/sum/attn")

        sequence_set = set(self.sequence_fields)
        self.non_seq_fields = [f for f, spec in feature_map.features.items()
                               if f not in sequence_set and spec.get("type") != "meta"]
        if len(self.non_seq_fields) == 0:
            raise ValueError("UnifiedInputTokenizer requires at least one non-sequence field")

        # Keep num_ns_tokens as the total non-sequence token budget. When enabled,
        # one slot is reserved for a target token and the remaining fields are
        # grouped into num_ns_tokens - 1 tokens. The disabled path intentionally
        # creates the exact same modules and state_dict as older versions.
        self.target_token_fields = []
        self.target_proj = None
        if use_target_token:
            self.target_token_fields = self._resolve_target_token_fields(target_token_fields)
        target_set = set(self.target_token_fields)
        self.grouped_ns_fields = [f for f in self.non_seq_fields if f not in target_set]
        self.num_grouped_ns_tokens = num_ns_tokens - int(use_target_token)

        if self.grouped_ns_fields and self.num_grouped_ns_tokens == 0:
            raise ValueError(
                "use_target_token=True requires num_ns_tokens >= 2 when other "
                "non-sequence fields are present")
        if self.num_grouped_ns_tokens > len(self.grouped_ns_fields):
            raise ValueError(
                "number of grouped non-sequence tokens={} cannot exceed remaining "
                "non-sequence fields={}".format(
                    self.num_grouped_ns_tokens, len(self.grouped_ns_fields)))

        if use_target_token:
            target_input_dim = len(self.target_token_fields) * embedding_dim
            self.target_proj = nn.Linear(target_input_dim, d_model)

        # Uneven grouping: split grouped_ns_fields into num_grouped_ns_tokens groups,
        # each with potentially different size (same logic as HyFormer's GroupedNSTokenizer).
        N = len(self.grouped_ns_fields)
        if self.num_grouped_ns_tokens:
            self.ns_group_slices = [
                (i * N // self.num_grouped_ns_tokens,
                 (i + 1) * N // self.num_grouped_ns_tokens)
                for i in range(self.num_grouped_ns_tokens)
            ]
        else:
            self.ns_group_slices = []
        group_input_dims = [
            (end - start) * embedding_dim
            for start, end in self.ns_group_slices
        ]
        self.ns_projs = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in group_input_dims
        ])

        seq_in_dim = embedding_dim * len(self.sequence_fields)
        self.seq_in_dim = seq_in_dim

        if seq_pooling == "attn":
            self.seq_attn_query = nn.Parameter(torch.empty(num_seq_tokens, seq_in_dim))
            nn.init.normal_(self.seq_attn_query, std=0.02)
        else:
            self.seq_attn_query = None

        if seq_tokenizer == "item":
            self.seq_proj = nn.Linear(seq_in_dim, d_model)
            self.seq_out_tokens = maxlen
        elif seq_tokenizer == "recent_k_plus_equal_chunks":
            old_tokens = num_seq_tokens - recent_k
            self.old_seq_proj_W = nn.Parameter(torch.empty(old_tokens, seq_in_dim, d_model))
            self.old_seq_proj_b = nn.Parameter(torch.zeros(old_tokens, d_model))
            nn.init.xavier_uniform_(self.old_seq_proj_W)
            self.recent_seq_proj = nn.Linear(seq_in_dim, d_model)
            self.seq_out_tokens = num_seq_tokens
        else:
            self.seq_proj_W = nn.Parameter(torch.empty(num_seq_tokens, seq_in_dim, d_model))
            self.seq_proj_b = nn.Parameter(torch.zeros(num_seq_tokens, d_model))
            nn.init.xavier_uniform_(self.seq_proj_W)
            self.seq_out_tokens = num_seq_tokens

        self.num_special_tokens = 0
        if concat_mode == "bos_s_ns":
            self.num_special_tokens = 1
            self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        elif concat_mode == "bos_s_sep_ns":
            self.num_special_tokens = 2
            self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
            self.sep = nn.Parameter(torch.zeros(1, 1, d_model))
        self.total_tokens = self.seq_out_tokens + num_ns_tokens + self.num_special_tokens

        if use_semantic_bias:
            self.semantic_bias = nn.Parameter(torch.zeros(4, d_model))
        else:
            self.semantic_bias = None

    def _resolve_target_token_fields(self, configured_fields):
        """Resolve target fields in sequence-field order.

        Explicit ``target_token_fields`` takes precedence. Otherwise, use each
        sequence field's ``share_embedding`` target when available, then fall
        back to the repository naming convention:
        ``item_seq -> target_item_id`` and ``item_seq__x -> x``.
        """
        if configured_fields is not None:
            fields = _as_list(configured_fields)
        else:
            fields = []
            primary = self.sequence_fields[0]
            for seq_field in self.sequence_fields:
                spec = self.feature_map.features.get(seq_field, {})
                candidate = spec.get("share_embedding")
                if candidate not in self.non_seq_fields:
                    if seq_field == primary and "target_item_id" in self.non_seq_fields:
                        candidate = "target_item_id"
                    elif seq_field.startswith(primary + "__"):
                        suffix = seq_field[len(primary) + 2:]
                        candidate = suffix if suffix in self.non_seq_fields else None
                    else:
                        candidate = None
                if candidate is not None and candidate not in fields:
                    fields.append(candidate)

        if not fields:
            raise ValueError(
                "use_target_token=True could not infer target fields; set "
                "target_token_fields explicitly")
        duplicates = [f for i, f in enumerate(fields) if f in fields[:i]]
        if duplicates:
            raise ValueError("target_token_fields contains duplicates: {}".format(duplicates))
        missing = [f for f in fields if f not in self.non_seq_fields]
        if missing:
            raise ValueError(
                "target_token_fields must be non-sequence features; missing/invalid: {}".format(
                    missing))
        return fields

    # 这个函数把不同形状的 feature embedding 统一成 (B, embedding_dim)
    def _feature_token(self, field, emb, X):
        if emb.dim() == 2:
            return emb
        if emb.dim() == 3:
            raw = X.get(field)
            if raw is None or raw.dim() != 2:
                mask = emb.new_ones(emb.shape[:-1]).unsqueeze(-1)
            else:
                mask = (raw.long() != 0).unsqueeze(-1).to(emb.dtype)
            pooled = (emb * mask).sum(dim=1)
            if self.ns_pooling == "mean":
                pooled = pooled / mask.sum(dim=1).clamp(min=1)
            return pooled
        raise ValueError("Unsupported embedding rank for field {}: {}".format(field, emb.dim()))

    def _build_ns_tokens(self, feature_emb_dict, X):
        pieces = []
        if self.use_target_token:
            target_input = torch.cat([
                self._feature_token(field, feature_emb_dict[field], X)
                for field in self.target_token_fields
            ], dim=-1)
            pieces.append(self.target_proj(target_input).unsqueeze(1))

        if self.grouped_ns_fields:
            values = []
            for field in self.grouped_ns_fields:
                values.append(self._feature_token(field, feature_emb_dict[field], X))
            flat = torch.cat(values, dim=-1)
            group_inputs = []
            emb = self.embedding_dim
            for start, end in self.ns_group_slices:
                group_inputs.append(flat[:, start * emb : end * emb])
            grouped_tokens = torch.stack(
                [proj(x) for proj, x in zip(self.ns_projs, group_inputs)], dim=1)
            pieces.append(grouped_tokens)

        tokens = torch.cat(pieces, dim=1)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        types = torch.full(tokens.shape[:2], TOKEN_NS, dtype=torch.long, device=tokens.device)
        return tokens, mask, types

    def _seq_input(self, feature_emb_dict, X):
        seq_embs = [feature_emb_dict[f] for f in self.sequence_fields]
        seq_cat = torch.cat(seq_embs, dim=-1)
        primary = self.sequence_fields[0]
        mask = X[primary].long() != 0
        return seq_cat, mask

    def _build_item_seq_tokens(self, seq_cat, seq_mask):
        tokens = self.seq_proj(seq_cat)
        types = torch.full(tokens.shape[:2], TOKEN_S, dtype=torch.long, device=tokens.device)
        return tokens, seq_mask, types

    def _pool_segments(self, seq_cat, seq_mask, segment_ids, num_segments=None):
        B, L, C = seq_cat.shape
        M = num_segments or self.num_seq_tokens
        out = seq_cat.new_zeros(B, M, C)
        mask_out = torch.zeros(B, M, dtype=torch.bool, device=seq_cat.device)

        if self.seq_pooling == "attn":
            queries = self.seq_attn_query[:M]
            for m in range(M):
                mask = (segment_ids == m) & seq_mask
                scores = torch.einsum("blc,c->bl", seq_cat, queries[m])
                scores = scores.masked_fill(~mask, float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                weights = torch.nan_to_num(weights, nan=0.0)
                weights = weights.unsqueeze(-1)
                out[:, m, :] = (seq_cat * weights).sum(dim=1)
                mask_out[:, m] = mask.any(dim=1)
            return out, mask_out

        for m in range(M):
            mask = (segment_ids == m) & seq_mask
            weights = mask.unsqueeze(-1).to(seq_cat.dtype)
            pooled = (seq_cat * weights).sum(dim=1)
            if self.seq_pooling == "mean":
                pooled = pooled / weights.sum(dim=1).clamp(min=1)
            out[:, m, :] = pooled
            mask_out[:, m] = mask.any(dim=1)
        return out, mask_out

    def _equal_segment_ids(self, B, L, device, num_segments=None):
        M = num_segments or self.num_seq_tokens
        pos = torch.arange(L, device=device)
        segment = torch.div(pos * M, L, rounding_mode="floor")
        return segment.clamp(max=M - 1).unsqueeze(0).expand(B, -1)

    def _timestamp_segment_ids(self, X, seq_mask):
        timestamps = X.get(self.sequence_timestamp_field)
        if timestamps is None:
            raise ValueError("timestamp_diff tokenizer requires item_seq__timestamp from DataLoader")
        timestamps = timestamps.long()
        B, L = timestamps.shape
        device = timestamps.device
        segment_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        valid_counts = seq_mask.long().sum(dim=1)
        for b in range(B):
            n = int(valid_counts[b].item())
            if n <= 0:
                continue
            start = L - n
            if n == 1 or self.num_seq_tokens == 1:
                segment_ids[b, start:] = 0
                continue
            ts = timestamps[b, start:]
            diffs = ts[1:] - ts[:-1]
            k = min(self.num_seq_tokens - 1, diffs.numel())
            cut_rel = torch.topk(diffs, k=k).indices + 1
            cuts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                torch.sort(cut_rel).values,
                torch.tensor([n], dtype=torch.long, device=device),
            ])
            for m in range(cuts.numel() - 1):
                segment_ids[b, start + cuts[m]:start + cuts[m + 1]] = min(m, self.num_seq_tokens - 1)
        return segment_ids

    def _build_grouped_seq_tokens(self, seq_cat, seq_mask, X):
        B, L, _ = seq_cat.shape
        if self.seq_tokenizer == "equal_chunks":
            segment_ids = self._equal_segment_ids(B, L, seq_cat.device)
        else:
            segment_ids = self._timestamp_segment_ids(X, seq_mask)
        pooled, mask = self._pool_segments(seq_cat, seq_mask, segment_ids)
        tokens = torch.einsum("bmc,mcd->bmd", pooled, self.seq_proj_W) + self.seq_proj_b
        types = torch.full(tokens.shape[:2], TOKEN_S, dtype=torch.long, device=tokens.device)
        return tokens, mask, types

    def _build_recent_k_seq_tokens(self, seq_cat, seq_mask):
        B, L, _ = seq_cat.shape
        old_tokens = self.num_seq_tokens - self.recent_k
        old_len = max(L - self.recent_k, 0)
        if old_len > 0:
            old_seq = seq_cat[:, :old_len, :]
            old_mask = seq_mask[:, :old_len]
            segment_ids = self._equal_segment_ids(B, old_len, seq_cat.device, num_segments=old_tokens)
            old_pooled, old_mask_out = self._pool_segments(
                old_seq, old_mask, segment_ids, num_segments=old_tokens)
            old_out = torch.einsum("bmc,mcd->bmd", old_pooled, self.old_seq_proj_W) + self.old_seq_proj_b
        else:
            old_out = seq_cat.new_zeros(B, old_tokens, self.d_model)
            old_mask_out = torch.zeros(B, old_tokens, dtype=torch.bool, device=seq_cat.device)

        recent_seq = seq_cat[:, -self.recent_k:, :]
        recent_mask = seq_mask[:, -self.recent_k:]
        recent_out = self.recent_seq_proj(recent_seq)
        tokens = torch.cat([old_out, recent_out], dim=1)
        mask = torch.cat([old_mask_out, recent_mask], dim=1)
        types = torch.full(tokens.shape[:2], TOKEN_S, dtype=torch.long, device=tokens.device)
        return tokens, mask, types

    def _add_semantic_bias(self, tokens, token_type):
        if self.semantic_bias is None:
            return tokens
        return tokens + self.semantic_bias[token_type]

    def forward(self, feature_emb_dict, X):
        seq_cat, seq_mask = self._seq_input(feature_emb_dict, X)
        if self.seq_tokenizer == "item":
            s_tokens, s_mask, s_type = self._build_item_seq_tokens(seq_cat, seq_mask)
        elif self.seq_tokenizer == "recent_k_plus_equal_chunks":
            s_tokens, s_mask, s_type = self._build_recent_k_seq_tokens(seq_cat, seq_mask)
        else:
            s_tokens, s_mask, s_type = self._build_grouped_seq_tokens(seq_cat, seq_mask, X)
        ns_tokens, ns_mask, ns_type = self._build_ns_tokens(feature_emb_dict, X)

        B = s_tokens.size(0)
        pieces = []
        masks = []
        types = []
        if self.concat_mode in ("bos_s_ns", "bos_s_sep_ns"):
            pieces.append(self.bos.expand(B, -1, -1))
            masks.append(torch.ones(B, 1, dtype=torch.bool, device=s_tokens.device))
            types.append(torch.full((B, 1), TOKEN_BOS, dtype=torch.long, device=s_tokens.device))
        pieces.extend([s_tokens])
        masks.extend([s_mask])
        types.extend([s_type])
        if self.concat_mode == "bos_s_sep_ns":
            pieces.append(self.sep.expand(B, -1, -1))
            masks.append(torch.ones(B, 1, dtype=torch.bool, device=s_tokens.device))
            types.append(torch.full((B, 1), TOKEN_SEP, dtype=torch.long, device=s_tokens.device))
        pieces.append(ns_tokens)
        masks.append(ns_mask)
        types.append(ns_type)

        tokens = torch.cat(pieces, dim=1)
        token_mask = torch.cat(masks, dim=1)
        token_type = torch.cat(types, dim=1)
        tokens = self._add_semantic_bias(tokens, token_type)
        return tokens, token_mask, token_type
