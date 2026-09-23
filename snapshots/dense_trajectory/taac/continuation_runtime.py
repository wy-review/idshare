"""Capture full training and paired continuation without interleaved evaluation."""

import argparse
import copy
from contextlib import contextmanager
import gc
import logging
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import paired_base as paired
import support_runtime as support
import frequency_runtime as frequency
from frequency_common import install_guard, milestone, read, save, sha

HERE = Path(__file__).resolve().parent


def snapshot_steps(total):
    return sorted({0, *[(total * i + 9) // 10 for i in range(1, 11)]})


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2]) and len(a[3]) == len(b[3])
    assert all(torch.equal(x, y) for x, y in zip(a[3], b[3]))


def tensor_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            tensor_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            tensor_tree_equal(x, y)
    else:
        assert a == b


def persist_state(path, state):
    before = rng_state()
    torch.save(state, path)
    assert_rng_equal(before, rng_state())
    milestone('snapshot_saved', file=path.name)
    return sha(path)


@contextmanager
def capture_training(bridge, counts, destination, metadata, retained):
    # The existing fit/capture still owns the single original validation pass.
    cls = bridge.src.IDShareUnifiedMixer
    with frequency.instrument(bridge, counts, destination, metadata):
        original_step, original_fit = cls.train_step, cls.fit

        def step(model, batch):
            total = model._steps_per_epoch
            current = model._total_steps
            folder = destination / 'pre'
            folder.mkdir(exist_ok=True)
            if current == 1:
                persist_state(folder / 'step0.pt', model.state_dict())
            loss = original_step(model, batch)
            assert bool(torch.isfinite(loss))
            if current in snapshot_steps(total):
                persist_state(folder / f'step{current}.pt', model.state_dict())
            return loss

        def fit(model, loader, *args, **kwargs):
            result = original_fit(model, loader, *args, **kwargs)
            persist_state(destination / 'pre/final_optimizers.pt', [o.state_dict() for o in model._optimizers])
            retained.update(model=model, train=loader.dataset, validation=kwargs['validation_data'].dataset,
                            total_steps=model._total_steps)
            return result

        cls.train_step, cls.fit = step, fit
        try:
            yield
        finally:
            cls.train_step, cls.fit = original_step, original_fit


def discard_optimizer(model):
    model._optimizers = []
    for attr in ('optimizer', 'emb_optimizer'):
        if hasattr(model, attr):
            delattr(model, attr)


def new_model(bridge, params, feature_map, state):
    model = bridge.src.IDShareUnifiedMixer(feature_map, **params)
    model.load_state_dict(state, strict=True)
    discard_optimizer(model)
    model.eval()
    return model


def make_pair(bridge, params, feature_map, state, counts, destination):
    native_model = new_model(bridge, params, feature_map, state)
    q = native_model.identity_quantizer
    table = state['embedding_layer.embedding_layers.target_item_id.weight'].cpu()
    routes, export = support.export_routes(q, table, counts, destination)
    route = torch.from_numpy(np.array(routes, dtype=np.int64)).to(native_model.device)
    outputs = torch.from_numpy(np.load(destination / 'shared_outputs.npy', allow_pickle=False)).to(native_model.device)
    with torch.no_grad():
        special = native_model._quantized_item_embeddings(torch.arange(4, device=native_model.device))
    models = {}
    for arm in ('shared', 'untied'):
        model = new_model(bridge, params, feature_map, state)
        paired.install_carrier(model, paired.FixedOutputCarrier(route, outputs, special, arm))
        models[arm] = model
    return models, native_model, export


def loader(dataset, indices=None):
    return DataLoader(dataset if indices is None else Subset(dataset, indices),
                      batch_size=2048, shuffle=False, num_workers=0)


def evaluate_stages(models, validation, probe, counts, destination, label, reference=None):
    single = set(models) == {'idshare'}
    evaluated = {'shared': models['idshare']} if single else models
    ref = np.load(reference, mmap_mode='r', allow_pickle=False) if reference is not None else None
    result = {split: paired.evaluate_pair(evaluated, data, counts, destination, f'{split}_{label}',
                                        reference=ref if split == 'validation' else None)
              for split, data in (('train_probe', probe), ('validation', validation))}
    if single:
        for value in result.values():
            value['arms']['idshare'] = value['arms'].pop('shared')
    return result


def formal(p, seed, output, manifest_sha):
    bridge = frequency.bridge_module()
    counts = frequency.load_counts(p)
    cell = p['runs'][f'idshare_s{seed}']
    destination = Path(p['retention_root']) / f's{seed}'
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'post').mkdir()
    config_dir = HERE / 'model_zoo/UnifiedBackbone/config/idshare_bridge_r01'
    original_config, original_logger = bridge.load_config, bridge.set_logger

    def config(*args, **kwargs):
        c = original_config(*args, **kwargs)
        assert bridge.execution_contract(c) == cell['execution_contract']
        c['verbose'] = 0
        return c

    def logger(c):
        original_logger(c)
        for handler in logging.getLogger().handlers:
            handler.addFilter(frequency.QuietEffects())

    bridge.load_config, bridge.set_logger, bridge.PROTOCOL = config, logger, p['protocol']
    retained, metadata = {}, {}
    milestone('package_verified', seed=seed, phase='original_training')
    with capture_training(bridge, counts, destination, metadata, retained):
        training = bridge.train_one(config_dir, cell['experiment_id'])
    assert training['all_checks_pass'] and training['execution_contract'] == cell['execution_contract']
    assert training['parameter_count'] == cell['parameter_count']
    assert retained['total_steps'] == p['original_steps']
    assert len(retained['train']) == p['train_rows'] and len(retained['validation']) == p['validation_rows']
    original = retained.pop('model')
    discard_optimizer(original)
    state = torch.load(destination / 'selected_model.pt', map_location='cpu', weights_only=True)
    tensor_tree_equal(state, torch.load(destination / 'pre' / f"step{p['original_steps']}.pt", map_location='cpu', weights_only=True))
    params = config(str(config_dir), cell['experiment_id'])
    params.update(gpu=0, model_root=f"/tmp/{p['protocol']}/offline_s{seed}", tensorboard=False, verbose=0)
    feature_map = original.feature_map
    del original
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.allow_tf32 = False
    order = torch.randperm(p['train_rows'], generator=torch.Generator().manual_seed(100000 + seed))[:p['steps'] * p['batch_size']].numpy()
    np.save(destination / 'training_row_order.npy', order, allow_pickle=False)
    probe_indices = order[:p['train_probe_rows']].copy()
    np.save(destination / 'probe_indices.npy', probe_indices, allow_pickle=False)
    train = loader(retained['train'], order)
    probe = loader(retained['train'], probe_indices)
    validation = loader(retained['validation'])
    models, native, export = make_pair(bridge, params, feature_map, state, counts, destination)
    initial = paired.initial_checks(models, native._quantized_item_embeddings, next(iter(probe)))
    with torch.no_grad():
        batch = next(iter(probe))
        original_prediction = native(batch)['y_pred']
        assert float((original_prediction - models['shared'](batch)['y_pred']).abs().max()) <= 1e-6
    del native, state
    fixed = {a: paired.parameter_digest(m) for a, m in models.items()}
    opts = paired.optimizers(models, p)
    for arm, model in models.items():
        persist_state(destination / 'post' / f'{arm}_step0.pt', model.probe_carrier.state_dict())
    milestone('train_validation_ready', phase='paired', train_rows=p['train_rows'], validation_rows=p['validation_rows'])
    for step, batch in enumerate(train, 1):
        losses = paired.train_pair(models, opts, batch, p)
        if step % 256 == 0:
            milestone('training', phase='paired', step=step, finite_loss=True)
        if step in p['readout_steps']:
            for arm, model in models.items():
                persist_state(destination / 'post' / f'{arm}_step{step}.pt', model.probe_carrier.state_dict())
    assert step == p['steps']
    assert all(paired.parameter_digest(models[a]) == fixed[a] for a in models)
    persist_state(destination / 'post/final_optimizers.pt', {a: o.state_dict() for a, o in opts.items()})
    del opts
    gc.collect()
    torch.cuda.empty_cache()
    stages = {}
    for step in p['readout_steps']:
        for arm, model in models.items():
            model.probe_carrier.load_state_dict(torch.load(destination / 'post' / f'{arm}_step{step}.pt', map_location=model.device, weights_only=True))
        stages[str(step)] = evaluate_stages(models, validation, probe, counts, destination, f'step{step}',
                  destination / 'validation_predictions.npy' if step == 0 else None)
        save(destination / 'post_metrics.json', stages)
    assert all(paired.parameter_digest(models[a]) == fixed[a] for a in models)
    del models
    gc.collect()
    torch.cuda.empty_cache()
    pre = {}
    model = new_model(bridge, params, feature_map, torch.load(destination / 'selected_model.pt', map_location='cpu', weights_only=True))
    for step in p['pre_readout_steps']:
        state = torch.load(destination / 'pre' / f'step{step}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        pre[str(step)] = evaluate_stages({'idshare': model}, validation, probe, counts, destination, f'pre{step}',
                    destination / 'validation_predictions.npy' if step == p['original_steps'] else None)
        tensor_tree_equal(state, {n: v.cpu() for n, v in model.state_dict().items()})
        save(destination / 'pre_metrics.json', pre)
    report = {'status': 'completed', 'mode': 'dense_trajectory', 'seed': seed,
        'source_manifest_sha256': manifest_sha, 'protocol_sha256': sha(HERE / 'CONTINUATION_PROTOCOL.json'),
        'datasets_loaded': ['train', 'validation'], 'uses_test_dataset': False, 'uses_test_labels': False,
        'paper_modified': False, 'old_results_replaced': False, 'best_step_selected': False,
        'training': training, 'initial': initial, 'export': export, 'source_checkpoint_sha256': metadata['checkpoint_sha256'],
        'original_steps': p['original_steps'], 'steps_per_arm': p['steps'], 'same_batches_for_both_arms': True,
        'frozen_parameters_unchanged': True, 'offline_evaluation': True,
        'same_train_probe_across_phases': True, 'probe_indices_sha256': sha(destination / 'probe_indices.npy'),
        'training_row_order_sha256': sha(destination / 'training_row_order.npy'),
        'output_directory': str(destination), 'pre': pre, 'stages': stages,
        'snapshot_sha256': {str(f.relative_to(destination)): sha(f) for f in sorted(destination.rglob('*.pt'))}}
    save(destination / 'dense_report.json', report)
    save(output / 'training_report.json', report)
    milestone('dense_complete', seed=seed, effect_values_blinded=True)


def engineering(p, output, manifest_sha):
    bridge = frequency.bridge_module()
    seed = 2021
    cell = p['runs'][f'idshare_s{seed}']
    params = bridge.load_config(str(HERE / 'model_zoo/UnifiedBackbone/config/idshare_bridge_r01'), cell['experiment_id'])
    assert bridge.execution_contract(params) == cell['execution_contract']
    params.update(gpu=0, model_root=f"/tmp/{p['protocol']}/engineering", tensorboard=False, verbose=0)
    synthetic_evaluation(bridge)
    feature_map = bridge._prepare_feature_map(params)
    train = paired.load_dataset(feature_map, params, 'train', max_samples=8192)
    batches = loader(train)
    synthetic = paired.synthetic_check()
    start, states, rngs, opts = time.monotonic(), [], [], []
    with tempfile.TemporaryDirectory(prefix='dense-check-') as tmp:
        tmp = Path(tmp)
        for capture in (False, True):
            bridge.seed_everything(seed)
            model = bridge.src.IDShareUnifiedMixer(feature_map, **params)
            model.train()
            model._max_gradient_norm = params['max_gradient_norm']
            model._steps_per_epoch, model._epochs_planned = p['original_steps'], 1
            for step, batch in enumerate(batches, 1):
                model._total_steps = step
                if capture and step == 1:
                    persist_state(tmp / 'initial.pt', model.state_dict())
                loss = model.train_step(batch)
                assert bool(torch.isfinite(loss))
                if capture:
                    persist_state(tmp / f'step{step}.pt', model.state_dict())
            assert step == 4
            rngs.append(rng_state())
            states.append({n: x.detach().cpu().clone() for n, x in model.state_dict().items()})
            opts.append(copy.deepcopy([{k: v for k, v in o.state_dict().items()} for o in model._optimizers]))
            # Move optimizer tensors off GPU before the second model is constructed.
            def cpu_tree(v):
                if isinstance(v, torch.Tensor): return v.cpu()
                if isinstance(v, dict): return {k: cpu_tree(x) for k, x in v.items()}
                if isinstance(v, list): return [cpu_tree(x) for x in v]
                return v
            opts[-1] = cpu_tree(opts[-1])
            del model
            gc.collect()
            torch.cuda.empty_cache()
        tensor_tree_equal(states[0], states[1])
        tensor_tree_equal(opts[0], opts[1])
        assert_rng_equal(rngs[0], rngs[1])
        tensor_tree_equal(states[1], torch.load(tmp / 'step4.pt', map_location='cpu', weights_only=True))
        counts = frequency.load_counts(p)
        models, native, export = make_pair(bridge, params, feature_map, states[1], counts, tmp)
        initial = paired.initial_checks(models, native._quantized_item_embeddings, next(iter(batches)))
        fixed = {a: paired.parameter_digest(m) for a, m in models.items()}
        optimizers = paired.optimizers(models, p)
        for step, batch in enumerate(batches, 1):
            paired.train_pair(models, optimizers, batch, p)
            for arm, model in models.items():
                path = tmp / f'{arm}{step}.pt'
                persist_state(path, model.probe_carrier.state_dict())
                tensor_tree_equal(model.probe_carrier.state_dict(), torch.load(path, map_location=model.device, weights_only=True))
        assert all(paired.parameter_digest(models[a]) == fixed[a] for a in models)
    save(output / 'training_report.json', {'status': 'passed', 'mode': 'dense_check',
        'source_manifest_sha256': manifest_sha, 'protocol_sha256': sha(HERE / 'CONTINUATION_PROTOCOL.json'),
        'datasets_loaded': ['train'], 'uses_test_dataset': False, 'uses_test_labels': False,
        'paper_modified': False, 'real_effects_computed': False, 'synthetic': synthetic,
        'initial': initial, 'export': export, 'steps_per_arm': 4,
        'checks': dict(snapshot_weights_exact=True, snapshot_optimizer_exact=True,
                       snapshot_rng_exact=True, snapshot_reload_exact=True,
                       frozen_parameters_unchanged=True, train_only=True),
        'wall_seconds_after_data_load': time.monotonic() - start})
    milestone('engineering_passed', real_effects_computed=False)


def synthetic_evaluation(bridge):
    """Exercise the actual prediction capture and both offline reader routes."""
    from torch.utils.data import Dataset

    class DatasetStub(Dataset):
        split = 'valid'
        tid = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int64)
        label = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float32)

        @staticmethod
        def _remap(ids):
            return ids + 3

        def __len__(self):
            return 8

        def __getitem__(self, i):
            row = {'label': self.label[i], 'target_item_id': self._remap(self.tid[i]),
                   'item_seq': np.array([0, 0, 4, 5, 6, 7, 8, 9], dtype=np.int64)}
            for name, spec in bridge._synthetic_feature_map().features.items():
                if name not in row:
                    row[name] = np.ones(8, dtype=np.int64) if spec['type'] == 'sequence' else np.int64(1)
            return row

    data = DatasetStub()
    batches = DataLoader(data, batch_size=4, shuffle=False, num_workers=0)
    batches.num_samples = len(data)
    counts = np.zeros(64, dtype=np.int64)
    counts[4:11] = [1, 2, 5, 10, 50, 100, 501]
    with tempfile.TemporaryDirectory(prefix='dense-synthetic-') as tmp:
        root = Path(tmp)
        models = []
        for instrumented in (False, True):
            bridge.seed_everything(2021)
            model = bridge._synthetic_model('idshare', 2021)
            model.device = torch.device('cuda')
            model.to('cuda')
            model.checkpoint = str(root / f'native_{instrumented}.pt')
            if instrumented:
                retained, metadata = {}, {}
                with capture_training(bridge, counts, root, metadata, retained):
                    model.fit(batches, epochs=1, validation_data=batches, max_gradient_norm=1.)
                tensor_tree_equal(torch.load(root / 'selected_model.pt', map_location='cpu', weights_only=True),
                                  torch.load(root / 'pre/step2.pt', map_location='cpu', weights_only=True))
            else:
                model.fit(batches, epochs=1, validation_data=batches, max_gradient_norm=1.)
                before = rng_state()
            models.append(model)
        assert_rng_equal(before, rng_state())
        tensor_tree_equal(models[0].state_dict(), models[1].state_dict())
        tensor_tree_equal(models[0].optimizer.state_dict(), models[1].optimizer.state_dict())
        result = evaluate_stages({'idshare': models[1]}, batches, batches, counts, root, 'pre2', root / 'validation_predictions.npy')
        assert result['validation']['reference_prediction_max_error'] <= 1e-5
        result = evaluate_stages({'shared': models[0], 'untied': models[1]}, batches, batches, counts, root, 'step0', root / 'validation_predictions.npy')
        assert result['validation']['reference_prediction_max_error'] <= 1e-5
    milestone('synthetic_full_capture_passed', datasets_loaded=[])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'paired'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install_guard()
    assert sha(HERE / 'CONTINUATION_MANIFEST.json') == args.manifest_sha
    for name, digest in read(HERE / 'CONTINUATION_MANIFEST.json')['files'].items():
        assert sha(HERE / name) == digest, name
    p = read(HERE / 'CONTINUATION_PROTOCOL.json')
    assert args.seed in p['seeds'] and torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    retention = Path(p['retention_root'])
    retention.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(retention).free / 1024 ** 3
    assert free_gb >= 256, 'insufficient artifact storage for dense snapshots/predictions'
    milestone('artifact_storage_ready', available_gb=round(free_gb, 1))
    args.output.mkdir(parents=True, exist_ok=True)
    (engineering(p, args.output, args.manifest_sha) if args.mode == 'check' else
     formal(p, args.seed, args.output, args.manifest_sha))


if __name__ == '__main__':
    main()
