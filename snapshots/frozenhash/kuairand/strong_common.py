"""Frozen strong-backbone adapters shared by AdamAR and TailShare."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    tmp.replace(path)


def milestone(name, **data):
    print(json.dumps({'milestone': name, **data}, ensure_ascii=False), flush=True)


def prohibited_path(path, train_only=False):
    p = Path(os.fsdecode(path))
    blocked = {'test', 'holdout'} | ({'val', 'valid', 'validation'} if train_only else set())
    if any(part.lower() in blocked for part in p.parts):
        return True
    if p.suffix.lower() in {'.npy', '.npz', '.parquet', '.pkl', '.csv', '.tsv'}:
        return any(p.name.lower().startswith(k + sep) for k in blocked for sep in ('_', '.'))
    return False


def install_guard(train_only=False):
    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            if prohibited_path(path, train_only) or prohibited_path(os.path.realpath(path), train_only):
                raise RuntimeError('forbidden dataset path')
        elif isinstance(path, (list, tuple)):
            for item in path:
                check(item)

    def audit(event, args):
        if event == 'open' and args:
            check(args[0])

    sys.addaudithook(audit)
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    def wrap(fn):
        def guarded(path, *args, **kwargs):
            check(path)
            return fn(path, *args, **kwargs)
        return guarded
    ds.dataset = wrap(ds.dataset)
    pq.ParquetFile = wrap(pq.ParquetFile)
    pq.read_table = wrap(pq.read_table)
    np.load = wrap(np.load)


class TailSharedEmbedding(nn.Module):
    """Keep head rows, share one distinct trainable vector for seen-tail IDs."""

    def __init__(self, original, mask):
        super().__init__()
        if original.sparse or original.max_norm is not None:
            raise ValueError('only the frozen dense lookup is supported')
        mask = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.bool, device=original.weight.device)
        if mask.shape != (original.num_embeddings,) or bool(mask[:4].any()):
            raise ValueError('tail mask must preserve all four special rows')
        self.weight = original.weight
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.padding_idx = original.padding_idx
        self.register_buffer('tail_mask', mask)
        # No extra RNG consumption: head and backbone initializations are unchanged.
        self.tail = nn.Parameter(torch.zeros(self.embedding_dim, device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, ids):
        private = F.embedding(ids, self.weight, padding_idx=self.padding_idx)
        return torch.where(self.tail_mask[ids].unsqueeze(-1), self.tail, private)


def make_adamar(model, identity, native, alpha):
    from adamar_frozen_compat import install
    install()
    from recscale.utils.adamar import RowwiseAdamAR
    if type(native) is not torch.optim.Adam or len(native.param_groups) != 1 or native.state:
        raise ValueError('adapter requires an unstepped single-group original Adam')
    group = native.param_groups[0]
    if group['weight_decay'] != 0 or group.get('amsgrad') or group.get('maximize'):
        raise ValueError('unexpected original Adam semantics')
    params = list(model.parameters())
    return RowwiseAdamAR([
        {'params': [identity], 'adamar': True, 'first_regularized_row': 4},
        {'params': [p for p in params if p is not identity], 'adamar': False},
    ], lr=group['lr'], alpha=alpha, betas=group['betas'], eps=group['eps'],
       eps_placement='pytorch_outside_sqrt')


def renew_adam(model, native):
    if type(native) is not torch.optim.Adam or len(native.param_groups) != 1 or native.state:
        raise ValueError('unexpected original Adam')
    return torch.optim.Adam(model.parameters(), **native.defaults)


def max_state_error(left, right):
    if list(left.state_dict()) != list(right.state_dict()):
        raise ValueError('state structure differs')
    result = 0.
    for x, y in zip(left.state_dict().values(), right.state_dict().values()):
        a, b = x.detach().reshape(-1), y.detach().reshape(-1)
        if a.shape != b.shape:
            raise ValueError('state shape differs')
        for i in range(0, a.numel(), 1048576):
            delta = (a[i:i+1048576].to(torch.float64) - b[i:i+1048576].to(torch.float64)).abs()
            if not bool(torch.isfinite(delta).all()):
                raise ValueError('non-finite parameter')
            if delta.numel():
                result = max(result, delta.max().item())
    return result


def finite_parameters(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def accumulate_windows(delta, offsets, lengths, positions, cutoffs, maxlen):
    """Difference-array counts equal enumerating every truncated history occurrence."""
    end = np.minimum(np.maximum(cutoffs, 0), lengths[positions])
    start = np.maximum(end - maxlen, 0)
    np.add.at(delta, offsets[positions] + start, 1)
    np.add.at(delta, offsets[positions] + end, -1)


def build_taac_counts(params, destination, protocol_sha):
    from fuxictr_ext.taac2025.taac_dataloader import _load_user_seqs
    import pyarrow.parquet as pq
    destination = Path(destination)
    if destination.exists():
        meta = read(destination.with_suffix('.json'))
        if meta['protocol_sha256'] != protocol_sha or sha(destination) != meta['counts_sha256']:
            raise ValueError('frequency cache binding drift')
        return meta
    root = Path(params['expected_raw_data_root'])
    sequences = _load_user_seqs(root / 'user_seqs')
    users = np.asarray(sorted(sequences), dtype=np.int64)
    lengths = np.fromiter((len(sequences[int(u)]) for u in users), dtype=np.int64, count=len(users))
    offsets = np.r_[0, lengths.cumsum()]
    items = np.concatenate([sequences[int(u)] for u in users])
    del sequences
    delta = np.zeros(len(items) + 1, dtype=np.int64)
    cardinality = params['shared_item_table_cardinality']
    counts = np.zeros(cardinality, dtype=np.int64)
    rows = 0
    samples = params['expected_samples_path_by_split']['train']
    for batch in pq.ParquetFile(samples).iter_batches(batch_size=1048576, columns=['user_id', 'target_item_id', 'seq_cutoff_pos']):
        uid, tid, cutoff = [batch.column(i).to_numpy(zero_copy_only=False).astype(np.int64) for i in range(3)]
        pos = np.searchsorted(users, uid)
        keep = pos < len(users)
        keep[keep] &= users[pos[keep]] == uid[keep]
        pos, tid, cutoff = pos[keep], tid[keep], cutoff[keep]
        real = tid > 0
        if np.any(tid[real] + 3 >= cardinality):
            raise ValueError('target ID exceeds frozen table')
        np.add.at(counts, tid[real] + 3, 1)
        accumulate_windows(delta, offsets, lengths, pos, cutoff, params['maxlen'])
        rows += len(tid)
    if rows != 131222882:
        raise ValueError('TAAC train row count drift')
    carry = 0
    history_occurrences = 0
    for start in range(0, len(items), 1048576):
        end = min(start + 1048576, len(items))
        weights = np.cumsum(delta[start:end]) + carry
        if weights.size:
            carry = int(weights[-1])
        raw = items[start:end]
        valid = (raw > 0) & (weights > 0)
        if np.any(raw[valid] + 3 >= cardinality):
            raise ValueError('history ID exceeds frozen table')
        np.add.at(counts, raw[valid] + 3, weights[valid])
        history_occurrences += int(weights[valid].sum())
    if carry + int(delta[-1]) != 0:
        raise ValueError('unbalanced history intervals')
    mask_path = Path(params['train_input_seen_mask_path'])
    if sha(mask_path) != params['train_input_seen_mask_sha256']:
        raise ValueError('seen mask drift')
    seen = np.load(mask_path, allow_pickle=False)
    if not np.array_equal(counts[4:] > 0, seen[1:]):
        raise ValueError('full occurrence counts disagree with frozen train-input seen mask')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.npy.tmp').open('wb') as f:
        np.save(f, counts, allow_pickle=False)
    destination.with_suffix('.npy.tmp').replace(destination)
    meta = {'protocol_sha256': protocol_sha, 'counts_sha256': sha(destination),
            'rows': rows, 'history_occurrences': history_occurrences,
            'frequency_scope': 'train_target_and_truncated_history_occurrences',
            'seen_mask_verified': True, 'labels_read_for_counts': False,
            'validation_used_for_counts': False, 'tail_id_count': int(((counts > 0) & (counts <= 5)).sum())}
    save(destination.with_suffix('.json'), meta)
    milestone('tail_counts_ready', **meta)
    return meta
