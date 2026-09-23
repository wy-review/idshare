"""Read-only final-assignment support audit; no training and no raw data reads."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from frequency_common import BUCKETS, DTYPE, classify, milestone, read, save, sha

HERE = Path(__file__).resolve().parent
SUPPORT_BOUNDS = [0, 10, 100, 1000]
SUPPORT_NAMES = ['0', '1-10', '11-100', '101-1000', '1001+']


def guard():
    import sys

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        for p in (path, path.resolve()):
            if any(x.lower() in {'test', 'holdout'} for x in p.parts):
                raise RuntimeError('forbidden split')
            if p.suffix.lower() in {'.npy', '.npz', '.parquet', '.pkl', '.csv', '.tsv'}:
                if any(p.name.lower().startswith(x + s) for x in ('test', 'holdout') for s in ('_', '.')):
                    raise RuntimeError('forbidden split file')
            if p.suffix.lower() in {'.parquet', '.csv', '.tsv'}:
                raise RuntimeError('support audit does not need raw datasets')
    sys.addaudithook(audit)


def verify(expected):
    assert sha(HERE / 'SUPPORT_MANIFEST.json') == expected
    m = read(HERE / 'SUPPORT_MANIFEST.json')
    for name, digest in m['files'].items():
        assert sha(HERE / name) == digest, name
    return read(HERE / 'SUPPORT_PROTOCOL.json')


def quantizer(p, seed, state=None, device='cuda'):
    from recscale.models.zero_anchor_identity_quantizer import ZeroAnchorIdentityQuantizer
    cfg = dict(p['quantizer_configs'][str(seed)])
    q = ZeroAnchorIdentityQuantizer(
        {'model': {ZeroAnchorIdentityQuantizer.CONFIG_KEY: cfg}},
        cardinalities=[p['identity_rows']], field_names=['shared_item_id'], embedding_dim=16)
    if state is not None:
        prefix = 'identity_quantizer.'
        qs = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        q.load_state_dict(qs, strict=True)
    q.eval().to(device)
    return q


def export_routes(q, table, counts, destination, chunk=4096):
    import torch
    routes = np.lib.format.open_memmap(destination / 'id_to_code.npy', mode='w+',
                                     dtype=np.int32, shape=(len(counts),))
    routes[:] = -1
    seen = np.flatnonzero(counts > 0)
    assert (seen >= 4).all()
    code = q.codebook(0).detach()
    output = code + q.base_embeddings[0].detach()
    output[0] = q.zero_state_base_embeddings[0].detach()
    np.save(destination / 'shared_outputs.npy', output.cpu().numpy(), allow_pickle=False)
    max_error = 0.
    with torch.no_grad():
        for start in range(0, len(seen), chunk):
            ids_np = seen[start:start + chunk]
            ids = torch.from_numpy(ids_np.copy())
            rows = table.index_select(0, ids).to(code.device)
            # Use the frozen quantizer with the source configuration's GEMM chunks.
            result = q.quantize_residuals(rows, 0)
            route = result['indices']
            routes[ids_np] = route.cpu().numpy()
            if start == 0 or start + chunk >= len(seen):
                native = q(rows[:, None, :], ids.to(code.device)[:, None])[:, 0, :]
                err = float((native - output[route]).abs().max())
                max_error = max(max_error, err)
                assert err <= 1e-6
            if start % (chunk * 256) == 0:
                milestone('route_export', done=start, total=len(seen))
    routes.flush()
    return routes, {'seen_ids': len(seen), 'unseen_exported': False,
                    'native_embedding_max_abs_error': max_error,
                    'route_sha256': sha(destination / 'id_to_code.npy'),
                    'shared_outputs_sha256': sha(destination / 'shared_outputs.npy')}


def distributions(values, weights):
    if not len(values):
        return None
    return {'mean_id_equal': float(np.mean(values)),
            'median': float(np.median(values)), 'p10': float(np.quantile(values, .1)),
            'p90': float(np.quantile(values, .9)), 'max': int(np.max(values)),
            'mean_lookup_weighted': float(np.average(values, weights=weights))}


def support_stats(counts, routes, k):
    seen = np.flatnonzero(counts > 0)
    codes, own = routes[seen], counts[seen]
    assert ((codes >= 0) & (codes < k)).all()
    number = np.bincount(codes, minlength=k)
    total = np.bincount(codes, weights=own, minlength=k).astype(np.int64)
    tail = np.bincount(codes[own <= 10], weights=own[own <= 10], minlength=k).astype(np.int64)
    freq_bucket = np.searchsorted([0, 1, 5, 10, 50, 100, 500], own, side='left')
    result = {}
    for idx, name in enumerate(BUCKETS[1:8], 1):
        selected = freq_bucket == idx
        c, n = codes[selected], own[selected]
        other = total[c] - n
        other_tail = tail[c] - np.where(n <= 10, n, 0)
        other_high = other - other_tail
        assert (other >= 0).all() and (other_tail >= 0).all() and (other_high >= 0).all()
        result[name] = {'ids': int(selected.sum()), 'own_lookup_sum': int(n.sum()),
            'other_ids': distributions(number[c] - 1, n),
            'other_lookups': distributions(other, n),
            'fraction_with_other_ids': float(np.mean(number[c] > 1)) if len(c) else None,
            'fraction_with_1_to_10_lookup_partners': float(np.mean(other_tail > 0)) if len(c) else None,
            'fraction_with_above_10_lookup_partners': float(np.mean(other_high > 0)) if len(c) else None,
            'partner_lookup_fraction_from_1_to_10': float(other_tail.sum() / other.sum()) if other.sum() else None,
            'zero_code_fraction': float(np.mean(c == 0)) if len(c) else None}
    assert sum(x['ids'] for x in result.values()) == len(seen)
    assert int(total.sum()) == int(own.sum())
    return result, total


def prediction_support(p, seed, counts, routes, total):
    files = p['sources'][str(seed)]
    values = {}
    for arm in ('continuous', 'idshare'):
        f = Path(files[arm]['directory']) / 'validation_predictions.npy'
        assert sha(f) == files[arm]['prediction_sha256']
        values[arm] = np.load(f, mmap_mode='r', allow_pickle=False)
        assert values[arm].dtype == DTYPE and len(values[arm]) == p['validation_rows']
    c, i = values['continuous'], values['idshare']
    # Only observed long-tail targets enter the support/effect association.
    use = (i['lookup_count'] >= 1) & (i['lookup_count'] <= 10)
    for name in ('target_id', 'label', 'lookup_count', 'bucket'):
        assert np.array_equal(c[name], i[name])
    raw, labels = i['target_id'][use], i['label'][use]
    own = i['lookup_count'][use]
    freq, buckets = classify(raw, counts)
    assert np.array_equal(freq, own) and np.array_equal(buckets, i['bucket'][use])
    codes = routes[raw + 3]
    assert (codes >= 0).all()
    other = total[codes] - own
    group = np.searchsorted(SUPPORT_BOUNDS, other, side='left')
    loss = {}
    for arm, data in values.items():
        pred = np.clip(data['prediction'][use].astype(np.float64), 1e-15, 1 - 1e-15)
        loss[arm] = -(labels * np.log(pred) + (1 - labels) * np.log1p(-pred))
    difference = loss['idshare'] - loss['continuous']
    output = {}
    for b in (1, 2, 3):
        for g, support_name in enumerate(SUPPORT_NAMES):
            sel = (buckets == b) & (group == g)
            key = BUCKETS[b] + '/' + support_name
            row = {'n': int(sel.sum()), 'positive': int(labels[sel].sum())}
            if row['n']:
                ids, inv = np.unique(raw[sel], return_inverse=True)
                per_id = np.bincount(inv, weights=difference[sel]) / np.bincount(inv)
                row.update(ids=len(ids), logloss_idshare=float(loss['idshare'][sel].mean()),
                           logloss_continuous=float(loss['continuous'][sel].mean()),
                           paired_logloss=float(difference[sel].mean()),
                           paired_logloss_id_equal=float(per_id.mean()))
            output[key] = row
    assert sum(x['n'] for x in output.values()) == int(use.sum())
    return output


def run(p, seed, manifest_sha, output):
    import torch
    guard()
    counts_path = Path(p['counts_path'])
    assert sha(counts_path) == p['counts_sha256']
    counts = np.load(counts_path, mmap_mode='r', allow_pickle=False)
    assert counts.dtype == np.int64 and counts.shape == (p['identity_rows'],)
    assert not counts[:4].any() and (counts >= 0).all()
    record = p['sources'][str(seed)]['idshare']
    path = Path(record['directory']) / 'selected_model.pt'
    assert sha(path) == record['checkpoint_sha256']
    destination = Path(p['retention_root']) / f's{seed}'
    destination.mkdir(parents=True, exist_ok=False)
    state = torch.load(path, map_location='cpu', weights_only=True)
    table = state['embedding_layer.embedding_layers.target_item_id.weight']
    assert table.shape == (p['identity_rows'], 16)
    assert torch.equal(table, state['embedding_layer.embedding_layers.item_seq.weight'])
    q = quantizer(p, seed, state)
    torch.backends.cuda.matmul.allow_tf32 = False
    milestone('artifacts_verified', seed=seed, training=False)
    routes, export = export_routes(q, table, counts, destination)
    coverage, total = support_stats(counts, routes, p['capacity'])
    effects = prediction_support(p, seed, counts, routes, total)
    report = {'status': 'completed', 'mode': 'support', 'seed': seed,
              'source_manifest_sha256': manifest_sha,
              'protocol_sha256': sha(HERE / 'SUPPORT_PROTOCOL.json'),
              'structural_checks': {'sources_verified': True, 'mapping_verified': True,
                                    'pair_order_identical': True, 'counts_partition_complete': True},
              'datasets_loaded': [], 'artifacts_read': ['train_counts', 'checkpoints', 'validation_predictions'],
              'uses_test_dataset': False, 'uses_test_labels': False, 'training_steps': 0,
              'paper_modified': False, 'source_checkpoint_sha256': record['checkpoint_sha256'],
              'export': export, 'output_directory': str(destination), 'coverage': coverage,
              'support_logloss': effects, 'historical_update_count_claimed': False,
              'causal_effect_claimed': False}
    save(destination / 'support_report.json', report)
    save(output / 'training_report.json', report)
    milestone('support_complete', seed=seed, values_blinded=True)


def check(p, manifest_sha, output, device):
    import torch
    import tempfile
    small = dict(p, identity_rows=64)
    small['quantizer_configs'] = {key: dict(value, codebook_size=8,
                codebook_size_by_field={'shared_item_id': 8}) for key, value in p['quantizer_configs'].items()}
    results = []
    for seed in p['seeds']:
        torch.manual_seed(seed)
        q = quantizer(small, seed, device=device)
        table = torch.randn(64, 16) * .1
        table[:4] = 0
        counts = np.zeros(64, dtype=np.int64)
        counts[4:] = np.arange(1, 61)
        with tempfile.TemporaryDirectory() as tmp:
            routes, info = export_routes(q, table, counts, Path(tmp), chunk=23)
            stats, total = support_stats(counts, routes, 8)
            assert sum(v['ids'] for v in stats.values()) == 60
            assert (routes[:4] == -1).all() and total.sum() == counts.sum()
            results.append({'seed': seed, **info})
    save(output / 'training_report.json', {'status': 'passed', 'mode': 'check', 'device': device,
         'checks': results, 'source_manifest_sha256': manifest_sha,
         'protocol_sha256': sha(HERE / 'SUPPORT_PROTOCOL.json'),
         'datasets_loaded': [], 'uses_test_dataset': False, 'uses_test_labels': False,
         'real_effects_computed': False, 'training_steps': 0})
    milestone('regression_passed', device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'support'])
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = verify(args.manifest_sha)
    if args.mode == 'check':
        check(protocol, args.manifest_sha, args.output, args.device)
    else:
        assert args.seed in protocol['seeds']
        run(protocol, args.seed, args.manifest_sha, args.output)
