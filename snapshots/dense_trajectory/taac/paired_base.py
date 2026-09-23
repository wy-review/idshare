"""Matched fixed-route sharing vs untying from the same retained checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from frequency_common import (BUCKETS, DTYPE, classify, install_guard, metrics,
                              milestone, read, save, sha, summarize_predictions)

HERE = Path(__file__).resolve().parent


class FixedOutputCarrier(nn.Module):
    def __init__(self, route, outputs, special, arm):
        super().__init__()
        assert arm in ('shared', 'untied')
        self.arm = arm
        self.register_buffer('route', route.clone())
        self.register_buffer('special', special.clone())
        if arm == 'shared':
            initial = outputs.clone()
        else:
            initial = torch.empty(len(route), outputs.shape[1], device=outputs.device)
            for start in range(0, len(route), 65536):
                end = min(start + 65536, len(route))
                initial[start:end] = outputs[route[start:end].clamp_min(0)]
        self.weight = nn.Parameter(initial)

    def forward(self, ids):
        real = ids >= 4
        chosen = self.route[ids]
        assert bool(((chosen >= 0) | ~real).all())
        index = chosen.clamp_min(0) if self.arm == 'shared' else ids
        emitted = torch.nn.functional.embedding(index, self.weight)
        return torch.where(real[..., None], emitted, self.special[ids.clamp(max=3)])


def parameter_digest(model):
    h = hashlib.sha256()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if name.startswith('probe_carrier.'):
            continue
        assert not tensor.requires_grad
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def install_carrier(model, carrier):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model.probe_carrier = carrier

    def emit(self, ids):
        return self.probe_carrier(ids)
    model._quantized_item_embeddings = types.MethodType(emit, model)
    allowed = [n for n, p in model.named_parameters() if p.requires_grad]
    assert allowed == ['probe_carrier.weight']
    return model


def verify(expected):
    assert sha(HERE / 'CONTINUATION_MANIFEST.json') == expected
    manifest = read(HERE / 'CONTINUATION_MANIFEST.json')
    for name, digest in manifest['files'].items():
        assert sha(HERE / name) == digest, name
    return read(HERE / 'CONTINUATION_PROTOCOL.json')


def source_material(p, seed):
    record = p['sources'][str(seed)]['idshare']
    checkpoint = Path(record['directory']) / 'selected_model.pt'
    assert sha(checkpoint) == record['checkpoint_sha256']
    support = p['support_exports'][str(seed)]
    root = Path(support['output_directory'])
    route_path, outputs_path = root / 'id_to_code.npy', root / 'shared_outputs.npy'
    assert sha(route_path) == support['export']['route_sha256']
    assert sha(outputs_path) == support['export']['shared_outputs_sha256']
    assert sha(p['counts_path']) == p['counts_sha256']
    counts = np.load(p['counts_path'], mmap_mode='r', allow_pickle=False)
    route = np.load(route_path, allow_pickle=False)
    outputs = np.load(outputs_path, allow_pickle=False)
    assert counts.shape == route.shape == (p['identity_rows'],)
    assert outputs.shape == (p['capacity'], 16)
    assert np.array_equal(route >= 0, counts > 0)
    return torch.load(checkpoint, map_location='cpu', weights_only=True), counts, route, outputs


def construct(p, seed):
    sys.path.insert(0, str(HERE / 'model_zoo/UnifiedBackbone'))
    import run_idshare_bridge as bridge
    state, counts, routes, outputs = source_material(p, seed)
    cell = read(HERE / 'PARENT_MATRIX.json')['runs'][f'idshare_s{seed}']
    params = bridge.load_config(str(HERE / 'model_zoo/UnifiedBackbone/config/idshare_bridge_r01'), cell['experiment_id'])
    assert bridge.execution_contract(params) == cell['execution_contract']
    assert params.get('test_data') is None and params['learning_rate'] == p['learning_rate']
    assert params['max_gradient_norm'] == p['gradient_clip']
    params.update(gpu=0, model_root='/tmp/idshare-sharing-continuation', tensorboard=False, verbose=0)
    feature_map = bridge._prepare_feature_map(params)
    models = {}
    native = None
    route = torch.from_numpy(routes.astype(np.int64)).cuda()
    output = torch.from_numpy(outputs).cuda()
    for arm in ('shared', 'untied'):
        model = bridge.src.IDShareUnifiedMixer(feature_map, **params)
        model.load_state_dict(state, strict=True)
        model.eval()
        if arm == 'shared':
            native = model._quantized_item_embeddings
        with torch.no_grad():
            special = model._quantized_item_embeddings(torch.arange(4, device=model.device))
        install_carrier(model, FixedOutputCarrier(route, output, special, arm))
        models[arm] = model
    milestone('source_and_export_verified', seed=seed)
    return models, native, counts, feature_map, params


def load_dataset(feature_map, params, split, preloaded=None, max_samples=None):
    from fuxictr_ext.taac2025.idshare_taac_dataloader import TaacSharedIdentityDataset
    fields = ['train_input_seen_mask_path', 'train_input_seen_mask_sha256', 'shared_item_table_cardinality',
              'expected_raw_data_root', 'expected_samples_path_by_split', 'maxlen', 'user_array_maxlen', 'sequence_side_fields']
    kwargs = {key: params[key] for key in fields}
    path = params['train_data' if split == 'train' else 'valid_data']
    result = TaacSharedIdentityDataset(feature_map, path, split=split,
                                     preloaded=preloaded, max_samples=max_samples, **kwargs)
    assert all(result.manifest_contract[k] for k in ('raw_data_root', 'samples_path', 'split', 'direct_split'))
    return result


def optimizers(models, p):
    return {arm: torch.optim.Adam([m.probe_carrier.weight], lr=p['learning_rate'],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
            for arm, m in models.items()}


def train_pair(models, opts, batch, p):
    losses = {}
    for arm, model in models.items():
        assert not model.training
        opts[arm].zero_grad(set_to_none=True)
        result = model(batch)['y_pred']
        loss = torch.nn.functional.binary_cross_entropy(result, model.get_labels(batch))
        assert torch.isfinite(loss)
        loss.backward()
        gradient = model.probe_carrier.weight.grad
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        torch.nn.utils.clip_grad_norm_([model.probe_carrier.weight], p['gradient_clip'])
        opts[arm].step()
        losses[arm] = float(loss.detach())
    return losses


def initial_checks(models, native, batch):
    model = models['shared']
    with torch.no_grad():
        ids = torch.cat([batch['target_item_id'].flatten(), batch['item_seq'].flatten()]).to(model.device)
        old = native(ids)
        shared, untied = [m.probe_carrier(ids) for m in models.values()]
        err = float((old - shared).abs().max())
        assert err <= 1e-6 and torch.equal(shared, untied)
        a, b = [m(batch)['y_pred'] for m in models.values()]
        pred_error = float((a - b).abs().max())
        assert pred_error <= 1e-7
    return {'native_embedding_max_error': err, 'pair_prediction_max_error': pred_error}


def evaluate_pair(models, loader, counts, destination, label, reference=None):
    n = len(loader.dataset)
    arrays = {arm: np.lib.format.open_memmap(destination / f'{label}_{arm}.npy', mode='w+', dtype=DTYPE, shape=(n,))
              for arm in models}
    cursor, maximum = 0, 0.
    is_subset = isinstance(loader.dataset, Subset)
    dataset = loader.dataset.dataset if is_subset else loader.dataset
    subset_indices = loader.dataset.indices if is_subset else None
    with torch.no_grad():
        for batch in loader:
            size = len(batch['label'])
            end = cursor + size
            idx = subset_indices[cursor:end] if is_subset else slice(cursor, end)
            raw = np.asarray(dataset.tid[idx], dtype=np.int64)
            labels = batch['label'].numpy().reshape(-1)
            assert np.array_equal(labels, dataset.label[idx])
            assert np.array_equal(batch['target_item_id'].numpy().reshape(-1), dataset._remap(raw))
            freq, bucket = classify(raw, counts)
            preds = {}
            for arm, model in models.items():
                pred = model(batch)['y_pred'].cpu().numpy().reshape(-1)
                assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
                preds[arm] = pred
                data = arrays[arm]
                data['target_id'][cursor:end], data['label'][cursor:end] = raw, labels
                data['lookup_count'][cursor:end], data['bucket'][cursor:end] = freq, bucket
                data['prediction'][cursor:end] = pred
            if label.endswith('step0'):
                assert np.max(np.abs(preds['shared'] - preds['untied'])) <= 1e-7
            if reference is not None:
                ref = reference[cursor:end]
                assert np.array_equal(ref['target_id'], raw) and np.array_equal(ref['label'], labels)
                maximum = max(maximum, float(np.max(np.abs(preds['shared'] - ref['prediction']))))
                assert maximum <= 1e-5
            cursor = end
            if cursor % (2048 * 1000) == 0:
                milestone('evaluation_progress', stage=label, rows=cursor, total=n)
    assert cursor == n
    result = {}
    for arm, a in arrays.items():
        a.flush()
        path = destination / f'{label}_{arm}.npy'
        value = summarize_predictions(path)
        long_tail = (a['lookup_count'] >= 1) & (a['lookup_count'] <= 10)
        value['buckets']['1-10'] = metrics(a['label'][long_tail], a['prediction'][long_tail])
        result[arm] = {'metrics': value, 'prediction_sha256': sha(path)}
    return {'arms': result, 'rows': n, 'reference_prediction_max_error': maximum}


def synthetic_check(device='cuda'):
    route = torch.tensor([-1, -1, -1, -1, 0, 1, 1, 2], device=device)
    outputs = torch.tensor([[.1, .2], [.3, .4], [.5, .6]], device=device)
    special = torch.tensor([[0., 0.], [.1, .2], [.7, .8], [.9, 1.]], device=device)
    a, b = [FixedOutputCarrier(route, outputs, special, arm) for arm in ('shared', 'untied')]
    ids = torch.arange(8, device=device)
    assert torch.equal(a(ids), b(ids))
    start = a(ids).detach().clone()
    for carrier in (a, b):
        opt = torch.optim.Adam([carrier.weight], lr=.001, foreach=False)
        loss = carrier(torch.tensor([4, 5], device=device)).square().sum()
        loss.backward()
        opt.step()
    assert torch.equal(a(ids)[:4], start[:4]) and torch.equal(b(ids)[:4], start[:4])
    assert not torch.equal(a(ids)[6], start[6]) and torch.equal(b(ids)[6], start[6])
    assert torch.equal(a(ids)[7], start[7]) and torch.equal(b(ids)[7], start[7])
    return {'initial_equal': True, 'other_id_updates_shared_partner': True,
            'private_partner_unchanged': True, 'special_outputs_fixed': True}


def run(p, seed, mode, manifest_sha, output):
    install_guard()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    regression = synthetic_check()
    models, native, counts, feature_map, params = construct(p, seed)
    smoke = mode == 'check'
    milestone('loading_train', seed=seed, smoke=smoke)
    train = load_dataset(feature_map, params, 'train', max_samples=8192 if smoke else None)
    assert len(train) == (8192 if smoke else p['train_rows'])
    if smoke:
        loader = DataLoader(train, batch_size=2048, shuffle=False, num_workers=0)
        milestone('train_ready', rows=len(train))
        first = next(iter(loader))
        initial = initial_checks(models, native, first)
        before = {a: parameter_digest(m) for a, m in models.items()}
        weights = {a: hashlib.sha256(m.probe_carrier.weight.detach().cpu().numpy().tobytes()).hexdigest() for a, m in models.items()}
        special = {a: m.probe_carrier.special.clone() for a, m in models.items()}
        opts = optimizers(models, p)
        for batch in loader:
            train_pair(models, opts, batch, p)
        after = {a: parameter_digest(m) for a, m in models.items()}
        assert before == after
        assert all(torch.equal(special[a], m.probe_carrier.special) for a, m in models.items())
        assert all(weights[a] != hashlib.sha256(m.probe_carrier.weight.detach().cpu().numpy().tobytes()).hexdigest() for a, m in models.items())
        report = {'status': 'passed', 'mode': mode, 'seed': seed, 'steps_per_arm': 4,
                  'datasets_loaded': ['train'], 'real_effects_computed': False,
                  'initial': initial, 'synthetic': regression, 'frozen_parameters_unchanged': True,
                  'only_output_parameters_updated': True, 'special_outputs_fixed': True}
    else:
        milestone('loading_validation', seed=seed)
        preloaded = {k: getattr(train, k) for k in ('item_feat', 'user_feat', 'user_seqs')}
        valid = load_dataset(feature_map, params, 'valid', preloaded=preloaded)
        assert len(valid) == p['validation_rows']
        valid_loader = DataLoader(valid, batch_size=2048, shuffle=False, num_workers=0)
        generator = torch.Generator().manual_seed(100000 + seed)
        order = torch.randperm(len(train), generator=generator)[:p['steps'] * 2048].numpy().copy()
        subset = Subset(train, order)
        loader = DataLoader(subset, batch_size=2048, shuffle=False, num_workers=0)
        probe = DataLoader(Subset(train, order[:16384]), batch_size=2048, shuffle=False, num_workers=0)
        destination = Path(p['retention_root']) / f's{seed}'
        destination.mkdir(parents=True, exist_ok=False)
        np.save(destination / 'training_row_order.npy', order, allow_pickle=False)
        initial = initial_checks(models, native, next(iter(loader)))
        before = {a: parameter_digest(m) for a, m in models.items()}
        reference_path = Path(p['sources'][str(seed)]['idshare']['directory']) / 'validation_predictions.npy'
        assert sha(reference_path) == p['sources'][str(seed)]['idshare']['prediction_sha256']
        reference = np.load(reference_path, mmap_mode='r', allow_pickle=False)
        milestone('train_validation_ready', seed=seed, train_rows=len(train), validation_rows=len(valid))
        stages = {'0': {'validation': evaluate_pair(models, valid_loader, counts, destination, 'validation_step0', reference),
                        'train_probe': evaluate_pair(models, probe, counts, destination, 'train_probe_step0')}}
        opts = optimizers(models, p)
        for step, batch in enumerate(loader, 1):
            losses = train_pair(models, opts, batch, p)
            if step == 1 or step % 128 == 0:
                milestone('training', seed=seed, step=step, finite_loss=losses)
            if step in p['readout_steps'][1:]:
                assert before == {a: parameter_digest(m) for a, m in models.items()}
                stages[str(step)] = {
                    'validation': evaluate_pair(models, valid_loader, counts, destination, f'validation_step{step}'),
                    'train_probe': evaluate_pair(models, probe, counts, destination, f'train_probe_step{step}')}
        assert step == p['steps']
        paths = {}
        for arm, model in models.items():
            f = destination / (arm + '_final.pt')
            torch.save({'carrier': model.probe_carrier.state_dict(), 'optimizer': opts[arm].state_dict(), 'step': step}, f)
            paths[arm] = sha(f)
        report = {'status': 'completed', 'mode': mode, 'seed': seed, 'steps_per_arm': step,
                  'datasets_loaded': ['train', 'validation'], 'initial': initial, 'synthetic': regression,
                  'frozen_parameters_unchanged': True, 'frozen_parameter_digests': before,
                  'same_batches_for_both_arms': True, 'sample_order_sha256': sha(destination / 'training_row_order.npy'),
                  'output_directory': str(destination), 'final_checkpoint_sha256': paths,
                  'stages': stages, 'best_step_selected': False,
                  'source_checkpoint_sha256': p['sources'][str(seed)]['idshare']['checkpoint_sha256']}
    report.update(source_manifest_sha256=manifest_sha, protocol_sha256=sha(HERE / 'CONTINUATION_PROTOCOL.json'),
                  uses_test_dataset=False, uses_test_labels=False, paper_modified=False)
    if not smoke:
        save(destination / 'continuation_report.json', report)
    save(output / 'training_report.json', report)
    milestone('continuation_complete', seed=seed, mode=mode, effects_blinded=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'paired'])
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    p = verify(args.manifest_sha)
    assert args.seed in p['seeds']
    run(p, args.seed, args.mode, args.manifest_sha, args.output)
