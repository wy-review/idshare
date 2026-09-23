#!/usr/bin/env python3
"""Capture the original validation pass; leave training computations unchanged."""

import argparse
from contextlib import contextmanager
import copy
import hashlib
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

from frequency_common import (BUCKETS, DTYPE, classify, install_guard, metrics,
                              milestone, read, save, sha, summarize_predictions)

ROOT = Path(__file__).resolve().parent


def bridge_module():
    sys.path.insert(0, str(ROOT / 'model_zoo/UnifiedBackbone'))
    import run_idshare_bridge
    return run_idshare_bridge


def verify(expected):
    assert sha(ROOT / 'FREQUENCY_SOURCE_MANIFEST.json') == expected
    m = read(ROOT / 'FREQUENCY_SOURCE_MANIFEST.json')
    for name, digest in m['files'].items():
        assert sha(ROOT / name) == digest, f'source drift: {name}'
    assert all(m['files'][name] == value for name, value in m['unchanged_parent_files'].items())
    p = read(ROOT / 'FREQUENCY_PROTOCOL.json')
    assert p['parent_package_sha256'] == m['parent_package_sha256']
    assert sha(ROOT / 'PARENT_MATRIX.json') == p['parent_matrix_sha256']
    return p


class QuietEffects(logging.Filter):
    def filter(self, record):
        line = record.getMessage().lower()
        return not any(x in line for x in ('auc', 'logloss', 'monitor(', '[metrics]'))


def load_counts(p):
    path = Path(p['counts_path'])
    assert sha(path) == p['counts_metadata']['counts_sha256']
    assert read(path.with_suffix('.json')) == p['counts_metadata']
    counts = np.load(path, mmap_mode='r', allow_pickle=False)
    assert counts.dtype == np.int64 and counts.shape == (p['identity_rows'],)
    assert not counts[:4].any() and (counts >= 0).all()
    c = next(iter(read(ROOT / 'PARENT_MATRIX.json')['runs'].values()))['execution_contract']
    mask_path = Path(c['train_input_seen_mask_path'])
    assert sha(mask_path) == c['train_input_seen_mask_sha256']
    seen = np.load(mask_path, mmap_mode='r', allow_pickle=False)
    assert np.array_equal(counts[4:] > 0, seen[1:])
    return counts


class PredictionCapture:
    def __init__(self, destination, counts):
        self.destination = Path(destination)
        self.counts = counts
        self.calls = 0
        self.result = None

    def evaluate(self, native_evaluate, model, loader, *args, **kwargs):
        from torch.utils.data import SequentialSampler
        assert isinstance(loader.sampler, SequentialSampler), 'validation order must be sequential'
        assert loader.dataset.split == 'valid'
        assert self.calls == 0, 'one-epoch protocol expects one validation pass'
        self.calls += 1
        n = loader.num_samples
        data = np.lib.format.open_memmap(self.destination, mode='w+', dtype=DTYPE, shape=(n,))
        cursor = 0
        order = hashlib.sha256()
        native_forward = model.forward

        def forward(batch, *a, **kw):
            nonlocal cursor
            result = native_forward(batch, *a, **kw)
            assert not model.training
            pred = result['y_pred'].detach().cpu().numpy().reshape(-1)
            y = model.get_labels(batch).detach().cpu().numpy().reshape(-1)
            end = cursor + len(pred)
            assert end <= n
            raw = np.asarray(loader.dataset.tid[cursor:end], dtype=np.int64)
            assert np.array_equal(y, loader.dataset.label[cursor:end])
            assert np.array_equal(batch['target_item_id'].cpu().numpy().reshape(-1), loader.dataset._remap(raw))
            assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
            assert np.isin(y, [0, 1]).all()
            frequency, buckets = classify(raw, self.counts)
            data['target_id'][cursor:end] = raw
            data['lookup_count'][cursor:end] = frequency
            data['bucket'][cursor:end] = buckets
            data['label'][cursor:end] = y
            data['prediction'][cursor:end] = pred
            # Interleaving makes this fingerprint independent of batch boundaries.
            pairs = np.empty((len(raw), 2), dtype='<i8')
            pairs[:, 0], pairs[:, 1] = raw, y
            order.update(pairs.tobytes())
            cursor = end
            return result

        existed, original = 'forward' in model.__dict__, model.__dict__.get('forward')
        model.forward = forward
        try:
            result = native_evaluate(loader, *args, **kwargs)
        finally:
            if existed:
                model.forward = original
            else:
                del model.forward
            data.flush()
        assert cursor == n
        self.result = {'rows': n, 'validation_order_sha256': order.hexdigest(),
                       'prediction_sha256': sha(self.destination), 'captures': self.calls}
        return result


@contextmanager
def instrument(bridge, counts, destination, metadata):
    cls = bridge.src.IDShareUnifiedMixer
    native_fit, native_step = cls.fit, cls.train_step
    fit_owned, step_owned = 'fit' in cls.__dict__, 'train_step' in cls.__dict__
    capture = PredictionCapture(destination / 'validation_predictions.npy', counts)

    def train_step(model, batch):
        loss = native_step(model, batch)
        if model._total_steps == 1 or model._total_steps % 4000 == 0:
            value = float(loss.detach().item())
            assert np.isfinite(value)
            milestone('training', step=model._total_steps, loss=value)
        return loss

    def fit(self, loader, *args, **kwargs):
        valid = kwargs['validation_data']
        milestone('train_validation_ready', train_rows=loader.num_samples, validation_rows=valid.num_samples)
        native_evaluate = self.evaluate
        eval_owned, old_eval = 'evaluate' in self.__dict__, self.__dict__.get('evaluate')

        def evaluate(this, gen, *a, **kw):
            milestone('validation_capture_started')
            return capture.evaluate(native_evaluate, this, gen, *a, **kw)

        self.evaluate = types.MethodType(evaluate, self)
        try:
            result = native_fit(self, loader, *args, **kwargs)
        finally:
            if eval_owned:
                self.evaluate = old_eval
            else:
                del self.evaluate
        assert len(self._history) == capture.calls == 1
        assert self._history[0]['epoch'] == 1
        checkpoint = destination / 'selected_model.pt'
        shutil.copy2(self.checkpoint, checkpoint)
        assert sha(checkpoint) == sha(self.checkpoint)
        metadata.update(capture.result, checkpoint_sha256=sha(checkpoint),
                        total_steps=self._total_steps, selected_epoch=1,
                        history_auc=bridge._best_validation_auc(self),
                        finite_parameters=all(bool(torch.isfinite(x).all()) for x in self.parameters()))
        milestone('artifacts_retained', rows=capture.result['rows'])
        return result

    cls.fit, cls.train_step = fit, train_step
    try:
        yield capture
    finally:
        if fit_owned:
            cls.fit = native_fit
        else:
            del cls.fit
        if step_owned:
            cls.train_step = native_step
        else:
            del cls.train_step


def formal(p, key, output, manifest_sha):
    bridge = bridge_module()
    cell = p['runs'][key]
    counts = load_counts(p)
    destination = Path(p['retention_root']) / key / 'fix1'
    destination.mkdir(parents=True, exist_ok=False)
    native_config, native_logger = bridge.load_config, bridge.set_logger

    def config(*args, **kwargs):
        c = native_config(*args, **kwargs)
        assert bridge.execution_contract(c) == cell['execution_contract']
        c['verbose'] = 0
        return c

    def logger(c):
        native_logger(c)
        for handler in logging.getLogger().handlers:
            handler.addFilter(QuietEffects())

    bridge.load_config, bridge.set_logger = config, logger
    bridge.PROTOCOL = p['protocol']
    metadata = {}
    milestone('package_verified', run_key=key, original_training_unchanged=True)
    with instrument(bridge, counts, destination, metadata):
        row = bridge.train_one(ROOT / 'model_zoo/UnifiedBackbone/config/idshare_bridge_r01', cell['experiment_id'])
    assert row['execution_contract'] == cell['execution_contract']
    assert row['all_checks_pass'] and row['parameter_count'] == cell['parameter_count']
    assert row['train_rows'] == p['train_rows'] and row['validation_rows'] == p['validation_rows']
    assert metadata['total_steps'] == (p['train_rows'] + 2047) // 2048
    analysis = summarize_predictions(destination / 'validation_predictions.npy')
    assert abs(analysis['overall']['auc'] - row['best_validation_auc']) <= 1e-12
    assert abs(metadata.pop('history_auc') - row['best_validation_auc']) <= 1e-12
    payload = {'status': 'completed', 'mode': 'frequency_replay', 'protocol': p['protocol'],
               'run_key': key, 'seed': cell['seed'], 'carrier': cell['carrier'],
               'source_manifest_sha256': manifest_sha, 'protocol_sha256': sha(ROOT / 'FREQUENCY_PROTOCOL.json'),
               'parent_package_sha256': p['parent_package_sha256'], 'parent_job_id': cell['parent_job_id'],
               'counts_sha256': p['counts_metadata']['counts_sha256'], 'bucket_names': BUCKETS,
               'datasets_loaded': ['train', 'validation'], 'uses_test_dataset': False, 'uses_test_labels': False,
               'training_code_unchanged': True, 'old_results_replaced': False, 'paper_modified': False,
               'captured_original_validation_pass': True, 'overall_matches_original_validation': True,
               'artifacts': {'directory': str(destination), **metadata},
               'training': row, 'frequency': analysis}
    save(destination / 'frequency_report.json', payload)
    save(output / 'training_report.json', payload)
    milestone('diagnostic_complete', rows=p['validation_rows'], effect_values_blinded=True)


def regression(p, output, manifest_sha, device):
    bridge = bridge_module()
    from torch.utils.data import DataLoader, Dataset

    class SyntheticDataset(Dataset):
        split = 'valid'
        tid = np.array([1, 2, 0, 3, 4, 5, 6, 7], dtype=np.int64)
        label = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)

        @staticmethod
        def _remap(ids):
            return np.where(ids > 0, ids + 3, 0)

        def __len__(self):
            return len(self.tid)

        def __getitem__(self, i):
            row = {'label': self.label[i], 'target_item_id': self._remap(self.tid[i]),
                   'item_seq': np.array([0, 0, 4, 5, 6, 7, 8, 9], dtype=np.int64)}
            for name, spec in bridge._synthetic_feature_map().features.items():
                if name not in row:
                    row[name] = np.ones(8, dtype=np.int64) if spec['type'] == 'sequence' else np.int64(1)
            return row

    counts = np.zeros(64, dtype=np.int64)
    counts[4:11] = [1, 2, 5, 10, 50, 100, 501]
    rows = []
    dataset = SyntheticDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    loader.num_samples = len(dataset)
    for key, cell in p['runs'].items():
        c = bridge.load_config(str(ROOT / 'model_zoo/UnifiedBackbone/config/idshare_bridge_r01'), cell['experiment_id'])
        assert bridge.execution_contract(c) == cell['execution_contract']
        bridge.seed_everything(cell['seed'])
        model = bridge._synthetic_model(cell['carrier'], cell['seed'])
        model.device = torch.device(device)
        model.to(device)
        with tempfile.TemporaryDirectory(prefix='frequency-regression-') as tmp:
            tmp = Path(tmp)
            model.checkpoint = str(tmp / 'native.pt')
            wrapped = copy.deepcopy(model)
            wrapped.checkpoint = str(tmp / 'wrapped.pt')
            metadata = {}
            bridge.seed_everything(cell['seed'])
            model.fit(loader, validation_data=loader, **c)
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
            bridge.seed_everything(cell['seed'])
            with instrument(bridge, counts, tmp, metadata):
                wrapped.fit(loader, validation_data=loader, **c)
            assert torch.equal(rng, torch.get_rng_state()), 'CPU RNG drift'
            if cuda_rng is not None:
                assert torch.equal(cuda_rng, torch.cuda.get_rng_state()), 'CUDA RNG drift'
            assert model.state_dict().keys() == wrapped.state_dict().keys()
            assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), wrapped.state_dict().values()))
            result = summarize_predictions(tmp / 'validation_predictions.npy')
            assert result['overall']['auc'] == bridge._best_validation_auc(model)
            assert metadata['captures'] == 1 and metadata['rows'] == len(dataset)
            rows.append({'key': key, 'parameter_max_abs_error': 0., 'rng_unchanged': True,
                         'original_validation_capture_verified': True, 'full_config_fit_verified': True})
    entry_checks = entry_regression(p, device, dataset)
    save(output / 'training_report.json', {'status': 'passed', 'mode': 'no_data_regression',
         'source_manifest_sha256': manifest_sha, 'protocol_sha256': sha(ROOT / 'FREQUENCY_PROTOCOL.json'),
         'checks': rows, 'routes_checked': 6, 'entry_checks': entry_checks,
         'full_entry_checked': True, 'device': device, 'datasets_loaded': [],
         'uses_test_dataset': False, 'uses_test_labels': False, 'real_auc_computed': False})
    milestone('regression_passed', routes=6, datasets_loaded=[], original_training_unchanged=True)


def entry_regression(p, device, dataset):
    from unittest.mock import patch
    from torch.utils.data import DataLoader

    bridge = bridge_module()
    cls = bridge.src.IDShareUnifiedMixer
    config_dir = ROOT / 'model_zoo/UnifiedBackbone/config/idshare_bridge_r01'
    counts = np.zeros(64, dtype=np.int64)
    counts[4:11] = [1, 2, 5, 10, 50, 100, 501]
    rows = []
    for key, cell in p['runs'].items():
        config = bridge.load_config(str(config_dir), cell['experiment_id'])
        assert config['model'] == cls.__name__
        assert bridge.execution_contract(config) == cell['execution_contract']
        config['verbose'] = 0
        train, valid = copy.deepcopy(dataset), copy.deepcopy(dataset)
        train.split, valid.split = 'train', 'valid'
        for data in (train, valid):
            data.manifest_contract = dict(raw_data_root=True, samples_path=True,
                                          split=True, direct_split=True, test_or_holdout=False)
        loaders = [DataLoader(data, batch_size=4, shuffle=False, num_workers=0)
                   for data in (train, valid)]
        for loader in loaders:
            loader.num_samples = len(dataset)
        with tempfile.TemporaryDirectory(prefix='frequency-fix1-entry-') as tmp:
            tmp = Path(tmp)
            models, states = [], []
            metadata = {}

            def constructor(feature_map, **kwargs):
                assert kwargs['model'] == cls.__name__
                assert bridge.execution_contract(kwargs) == cell['execution_contract']
                with patch.object(bridge.src, 'IDShareUnifiedMixer', cls):
                    model = bridge._synthetic_model(cell['carrier'], cell['seed'])
                model.device = torch.device(device)
                model.to(device)
                model.checkpoint = str(tmp / f'entry_{len(models)}.pt')
                model.count_parameters = lambda **kw: None
                models.append(model)
                return model

            def execute():
                with patch.object(bridge, 'load_config', side_effect=lambda *a, **kw: copy.deepcopy(config)), \
                     patch.object(bridge, 'set_logger'), \
                     patch.object(bridge, '_prepare_feature_map', return_value=bridge._synthetic_feature_map()), \
                     patch.object(bridge, 'RankDataLoader', return_value=types.SimpleNamespace(make_iterator=lambda: loaders)), \
                     patch.object(bridge, 'PROTOCOL', tmp.name), \
                     patch.object(bridge.src, 'IDShareUnifiedMixer', side_effect=constructor):
                    row = bridge.train_one(config_dir, cell['experiment_id'])
                assert row['all_checks_pass'] and row['execution_contract'] == cell['execution_contract']
                states.append((torch.get_rng_state().clone(),
                               torch.cuda.get_rng_state().clone() if device == 'cuda' else None))
                return row

            native = execute()
            with instrument(bridge, counts, tmp, metadata):
                captured = execute()
            assert torch.equal(states[0][0], states[1][0])
            if device == 'cuda':
                assert torch.equal(states[0][1], states[1][1])
            assert all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values()))
            assert native['best_validation_auc'] == captured['best_validation_auc']
            assert metadata['captures'] == 1 and metadata['rows'] == len(dataset)
            assert summarize_predictions(tmp / 'validation_predictions.npy')['overall']['auc'] == native['best_validation_auc']
            rows.append({'key': key, 'train_one_full_config_verified': True,
                         'parameter_max_abs_error': 0., 'rng_unchanged': True,
                         'original_validation_capture_verified': True})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['formal', 'check'])
    parser.add_argument('--run-key')
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    install_guard()
    p = verify(args.manifest_sha)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'check':
        regression(p, args.output, args.manifest_sha, args.device)
    else:
        assert args.device == 'cuda' and torch.cuda.is_available()
        formal(p, args.run_key, args.output, args.manifest_sha)


if __name__ == '__main__':
    main()
