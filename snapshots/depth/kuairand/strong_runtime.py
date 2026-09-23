#!/usr/bin/env python3
"""Adapters around the unchanged strong-backbone training implementations."""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from strong_common import (TailSharedEmbedding, accumulate_windows, build_taac_counts,
                           finite_parameters, install_guard, make_adamar,
                           max_state_error, milestone, read, renew_adam, save, sha)

ROOT = Path(__file__).resolve().parent


class BlindedStdout:
    """Suppress effect-bearing checkpoint names from the unchanged trainer."""

    def __init__(self, stream):
        self.stream, self.pending = stream, ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if 'auc' not in line.lower() and 'logloss' not in line.lower():
                self.stream.write(line + '\n')
        return len(text)

    def flush(self):
        self.stream.flush()


def verify(expected):
    assert sha(ROOT / 'SOURCE_MANIFEST.json') == expected, 'manifest drift'
    manifest = read(ROOT / 'SOURCE_MANIFEST.json')
    for name, digest in manifest['files'].items():
        assert sha(ROOT / name) == digest, f'source drift: {name}'
    assert all(manifest['files'][k] == v for k, v in manifest['unchanged_original_files'].items())
    p = read(ROOT / 'PROTOCOL.json')
    assert p['parent_package_sha256'] == manifest['parent_package_sha256']
    assert sha(ROOT / 'PARENT_MATRIX.json') == p['parent_matrix_sha256']
    return p


def seed(setting, value):
    if setting == 'taac':
        from fuxictr.pytorch.torch_utils import seed_everything
        seed_everything(seed=value)
    else:
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def config(p, run, output, local=False, train_only=False):
    setting = p['setting']
    parent = read(ROOT / 'PARENT_MATRIX.json')
    if setting == 'taac':
        sys.path.insert(0, str(ROOT / 'model_zoo/UnifiedBackbone'))
        from fuxictr.utils import load_config
        cell = next(c for c in parent['runs'].values()
                    if c['carrier'] == 'continuous' and c['seed'] == run['seed'])
        c = load_config(str(ROOT / 'model_zoo/UnifiedBackbone/config/idshare_l2_response_r02'), cell['experiment_id'])
        c.update(full_table_l2_coefficient=0., idshare_enabled=False,
                 gpu=-1 if local else 0, model_root=str(output / 'checkpoints'),
                 model_id=run['run_key'], tensorboard=False, num_workers=0, verbose=0)
        for key in ['train_data', 'valid_data']:
            c[key] = str(ROOT / c[key])
        assert c.get('test_data') in (None, '')
        assert c['embedding_regularizer'] == c['net_regularizer'] == 0
        assert c['epochs'] == 1 and c['batch_size'] == 2048 and c['maxlen'] == 100
        assert c['d_model'] == 64 and c['num_layers'] == 2 and c['num_seq_tokens'] == 5
        if local:
            c['shared_item_table_cardinality'] = 64
        return c
    from frozen_config_builder import build_core_config
    arm = copy.deepcopy(next(a for a in parent['arms'].values() if a['family'] == 'continuous'))
    arm.update(full_table_l2_coefficient=0., zero_anchor=False)
    path = Path(parent['processed_manifest'])
    if local:
        # Synthetic schema only; no dataset file is opened by this local regression.
        fields = ['video_id'] + [f'f{i}' for i in range(36)]
        c = {'seed': run['seed'], 'dataset': {'sparse_cols': fields, 'cardinalities': [64]*37, 'dense_cols': []},
             'model': {'name': 's2drec', 'embedding_dim': 16, 'embedding_init': 'uniform',
                       'tokenizer_type': 'per_field_proj', 'per_field_proj_mode': 'split',
                       'num_tokens': 37, 'd_model': 74, 'num_mixer_layers': 2, 'ffn_dim': 256,
                       'dropout': 0., 'head_hidden_units': [512, 256], 'head_dropout': 0.,
                       'tokenizer_seed': 2021, 'sparse_embedding_zero_init_rows_by_field': {'video_id': [1]}},
             'training': {'epochs': 10, 'batch_size': 4096, 'lr': .002, 'optimizer': 'adam',
                          'optimizer_foreach': False, 'weight_decay': 0., 'grad_clip': 1.,
                          'num_workers': 0, 'shuffle': False, 'data_seed': 20260724,
                          'early_stop_min_epochs': 4, 'early_stop_patience': 3,
                          'early_stop_min_delta': .0001, 'checkpoint_selection_tolerance': 1e-12}}
    else:
        assert sha(path) == parent['expected_processed_manifest_sha256'], 'processed manifest drift'
        c = build_core_config(path, run=run, arm=arm)
        if train_only:
            c['dataset']['verify_processed_sha256_splits'] = ['train']
    c['model'].update(backbone_type='rankmixer_v2', inter_layer_residual=False, carry_path_ablation=False)
    c['training'].update(save_dir=str(output / 'checkpoints'), suppress_effect_metric_logs=True)
    assert c['training']['epochs'] == 10 and c['training']['early_stop_min_epochs'] == 4
    assert c['training']['early_stop_patience'] == 3 and c['training']['optimizer_foreach'] is False
    return c


def kuai_counts(dataset, p):
    destination = Path(p['cache_path'])
    binding = sha(ROOT / 'PROTOCOL.json')
    if destination.exists():
        meta = read(destination.with_suffix('.json'))
        assert meta['protocol_sha256'] == binding and sha(destination) == meta['counts_sha256']
        return meta
    column = dataset.field_names.index('video_id') - dataset.user_field_count
    assert 0 <= column < dataset.video_field_count and dataset.split == 'train'
    counts = np.zeros(p['expected_identity_rows'], dtype=np.int64)
    rows = 0
    for index in range(len(dataset.shards)):
        shard = dataset._open_shard(index)
        for start in range(0, len(shard), 1048576):
            vids = shard['video_index'][start:start+1048576]
            ids = np.asarray(dataset.video_sparse[vids, column], dtype=np.int64)
            assert (ids >= 0).all() and (ids < len(counts)).all()
            np.add.at(counts, ids, 1)
            rows += len(ids)
    assert rows == p['expected_train_rows']
    counts[:4] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.npy.tmp').open('wb') as handle:
        np.save(handle, counts, allow_pickle=False)
    destination.with_suffix('.npy.tmp').replace(destination)
    meta = {'protocol_sha256': binding, 'counts_sha256': sha(destination), 'rows': rows,
            'frequency_scope': 'train_video_id_occurrences', 'labels_read_for_counts': False,
            'validation_used_for_counts': False, 'tail_id_count': int(((counts > 0) & (counts <= 5)).sum())}
    save(destination.with_suffix('.json'), meta)
    milestone('tail_counts_ready', **meta)
    return meta


def tail_mask(p, local):
    if local:
        mask = np.zeros(64, dtype=bool)
        mask[4:8] = True
        return mask
    path = Path(p['cache_path'])
    meta = read(path.with_suffix('.json'))
    assert meta['protocol_sha256'] == sha(ROOT / 'PROTOCOL.json') and sha(path) == meta['counts_sha256']
    counts = np.load(path, mmap_mode='r', allow_pickle=False)
    assert counts.shape == (p['expected_identity_rows'],)
    mask = (counts > 0) & (counts <= p['tailshare']['tau'])
    mask[:4] = False
    assert mask.sum() == meta['tail_id_count'] and mask.any()
    return mask


def identity(model, c, setting):
    if setting == 'taac':
        table = model.embedding_layer.embedding_layers['target_item_id']
        assert table is model.embedding_layer.embedding_layers['item_seq']
        return table
    index = c['dataset']['sparse_cols'].index('video_id')
    return model.encoder.sparse_arch.embeddings[index]


def adapt(model, c, p, method, native, alpha=None, local=False):
    table = identity(model, c, p['setting'])
    expected = 64 if local else p['expected_identity_rows']
    assert list(table.weight.shape) == [expected, 16]
    if method == 'adamar':
        optimizer = make_adamar(model, table.weight, native, alpha)
    elif method == 'tailshare':
        shared = TailSharedEmbedding(table, tail_mask(p, local))
        if p['setting'] == 'taac':
            model.embedding_layer.embedding_layers['target_item_id'] = shared
            model.embedding_layer.embedding_layers['item_seq'] = shared
        else:
            index = c['dataset']['sparse_cols'].index('video_id')
            model.encoder.sparse_arch.embeddings[index] = shared
        optimizer = renew_adam(model, native)
    else:
        assert method == 'native'
        optimizer = native
    assert identity(model, c, p['setting']).weight is table.weight
    return optimizer


def make_model(c, p, method, alpha, output, local=False, train=None, validation=None, fmap=None):
    seed(p['setting'], c.get('seed', 42))
    if p['setting'] == 'taac':
        import src
        from run_idshare_bridge import _synthetic_feature_map, _prepare_feature_map
        fmap = fmap or (_synthetic_feature_map() if local else _prepare_feature_map(c))
        model = src.IDShareUnifiedMixer(fmap, **c)
        # Match the original pre-loader dummy forward, before changing the identity interface.
        if not local:
            model.count_parameters(count_embedding=True, batch_size=1)
        opt = adapt(model, c, p, method, model.optimizer, alpha, local)
        model.optimizer, model._optimizers = opt, [opt]
        model._max_gradient_norm = c.get('max_gradient_norm', 10.)
        owner = model
    else:
        from recscale.models.s2drec import S2DRecModel
        from recscale.trainer import Trainer
        model = S2DRecModel(c).to('cpu' if local else 'cuda')
        native = torch.optim.Adam(model.parameters(), lr=c['training']['lr'], foreach=False)
        opt = adapt(model, c, p, method, native, alpha, local)
        # Trainer owns the unchanged batching, early stop, checkpoint selection and clipping.
        owner = Trainer(model, train, validation, c, torch.device('cpu' if local else 'cuda'))
        owner.optimizer = opt
    n = sum(x.numel() for x in model.parameters())
    if not local:
        assert n == p['expected_base_parameters'] + (16 if method == 'tailshare' else 0), f'parameter count drift: {n}'
    return model, owner


def step(model, owner, batch, setting):
    model.train()
    if setting == 'taac':
        loss = model.train_step(batch)
    else:
        batch = owner._to_device(batch)
        out = model(batch)
        logits, aux = out if isinstance(out, tuple) else (out, None)
        loss = owner.criterion(logits, batch['label'])
        if aux is not None:
            loss = loss + aux
        owner.optimizer.zero_grad()
        loss.backward()
        owner._postprocess_gradients(model)
        owner.optimizer.step()
    value = float(loss.detach())
    assert math.isfinite(value), 'non-finite training loss'
    return value


def unit_checks():
    original = torch.nn.Embedding(16, 3, padding_idx=0)
    mask = np.zeros(16, dtype=bool)
    mask[4:6] = True
    shared = TailSharedEmbedding(original, mask)
    ids = torch.tensor([0, 1, 4, 5, 4, 8])
    out = shared(ids)
    assert torch.equal(out[[0, 1, 5]], original(ids[[0, 1, 5]]))
    assert torch.equal(out[2], out[3]) and torch.equal(out[2], out[4])
    out.sum().backward()
    assert torch.equal(shared.tail.grad, torch.full((3,), 3.))
    assert not bool(shared.weight.grad[4:6].any()) and not bool(shared.weight.grad[0].any())
    assert torch.equal(shared.weight.grad[8], torch.ones(3))
    lengths = np.array([3, 5]); offsets = np.array([0, 3, 8]); delta = np.zeros(9, dtype=np.int64)
    accumulate_windows(delta, offsets, lengths, np.array([0, 0, 1, 1]), np.array([2, 9, 0, 4]), 2)
    assert np.array_equal(np.cumsum(delta)[:-1], [1, 2, 1, 0, 0, 1, 1, 0])
    return {'tail_forward_shared': True, 'tail_gradient_pooled': True,
            'head_and_special_rows_preserved': True, 'history_count_intervals_verified': True}


class SyntheticKuai(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'sparse': torch.full((37,), 4 + index % 8, dtype=torch.long),
                'label': torch.tensor(float(index % 2))}


def checks(p, output, local):
    report = {'unit_checks': unit_checks(), 'datasets_loaded': [] if local else ['train'],
              'auc_computed': False, 'validation_files_read': False, 'local_check': local}
    run = {'seed': 42, 'run_key': 'engineering_check'}
    c = config(p, run, output, local=local, train_only=True)
    setting = p['setting']
    fmap = None
    if setting == 'taac':
        from run_idshare_bridge import _prepare_feature_map, _synthetic_feature_map
        fmap = _synthetic_feature_map() if local else _prepare_feature_map(c)
        if not local:
            report['frequency_cache'] = build_taac_counts(c, p['cache_path'], sha(ROOT / 'PROTOCOL.json'))
            from fuxictr_ext.taac2025.idshare_taac_dataloader import TaacSharedIdentityDataLoader
            limited = dict(c, max_samples=c['batch_size'] * p['check_steps_per_arm'])
            train = TaacSharedIdentityDataLoader(fmap, data_path=c['train_data'], split='train', **limited)
        else:
            batch = {'label': torch.tensor([0., 1., 0., 1.])}
            for name, spec in fmap.features.items():
                batch[name] = (torch.tensor([[0, 4, 5, 6, 7, 8, 9, 10]]*4)
                               if spec['type'] == 'sequence' else torch.tensor([4, 5, 8, 9]))
            train = [batch] * 4
    else:
        if local:
            train = SyntheticKuai()
        else:
            from recscale.datasets.kuairand27k_k1 import KuaiRand27KK1MMapDataset
            train = KuaiRand27KK1MMapDataset(c, split='train')
            assert len(train) == p['expected_train_rows']
            report['frequency_cache'] = kuai_counts(train, p)
            report['train_sha_verification'] = train.sha256_verification
    milestone('train_ready', setting=setting, local=local)
    native, native_owner = make_model(c, p, 'native', None, output, local, train, fmap=fmap)
    paired, paired_owner = make_model(c, p, 'adamar', 0., output, local, train, fmap=fmap)
    assert max_state_error(native, paired) == 0.
    steps = 4 if local else p['check_steps_per_arm']
    iterator = iter(train if setting == 'taac' else native_owner.train_loader)
    batches = []
    for _ in range(steps):
        try:
            batches.append(next(iterator))
        except StopIteration:
            assert local
            batches.append(batches[0])
    del iterator
    losses, differences, durations = [], [], []
    for batch in batches:
        if not local:
            torch.cuda.synchronize()
        start = time.monotonic()
        x = step(native, native_owner, batch, setting)
        if not local:
            torch.cuda.synchronize()
        durations.append(time.monotonic()-start)
        y = step(paired, paired_owner, batch, setting)
        losses.append(x); differences.append(abs(x-y))
    error = max_state_error(native, paired)
    assert error <= p['alpha_zero_atol'] and max(differences) <= p['alpha_zero_atol'], (error, max(differences))
    report['alpha_zero_comparison'] = {'passed': True, 'steps': steps, 'parameter_max_abs_error': error,
        'loss_max_abs_error': max(differences), 'loss_first': losses[0], 'loss_last': losses[-1],
        'seconds_per_step': float(np.median(durations[1:]))}
    report['base_parameter_count'] = sum(x.numel() for x in native.parameters())
    for owner in (native_owner, paired_owner):
        if getattr(owner, 'log_file', None):
            owner.log_file.close()
    del native, native_owner, paired, paired_owner, owner
    gc.collect()
    if not local:
        torch.cuda.empty_cache()
    for method in ['adamar', 'tailshare']:
        model, owner = make_model(c, p, method, p['check_alpha'], output, local, train, fmap=fmap)
        losses, durations = [], []
        for batch in batches:
            if not local:
                torch.cuda.synchronize()
            start = time.monotonic()
            losses.append(step(model, owner, batch, setting))
            if not local:
                torch.cuda.synchronize()
            durations.append(time.monotonic()-start)
        assert finite_parameters(model)
        report[method] = {'steps': steps, 'finite': True, 'loss_first': losses[0], 'loss_last': losses[-1],
                          'parameter_count': sum(x.numel() for x in model.parameters()),
                          'seconds_per_step': float(np.median(durations[1:]))}
        if method == 'adamar':
            clock = owner.optimizer.state[identity(model, c, setting).weight]['last_valid_step']
            assert clock.dtype == torch.int64 and int(clock.max()) <= steps
            report[method].update(alpha=p['check_alpha'], clock_dtype=str(clock.dtype), last_valid_step_max=int(clock.max()))
        if getattr(owner, 'log_file', None):
            owner.log_file.close()
        del model, owner
        gc.collect()
        if not local:
            torch.cuda.empty_cache()
    return {**report, 'status': 'passed', 'production_param_count_enforced': not local,
            'identity_shape': [64 if local else p['expected_identity_rows'], 16],
            'cuda_verified': not local and torch.cuda.is_available(), 'identical_batches_all_arms': True}


def formal(p, run, output, selection_sha):
    c = config(p, run, output)
    setting = p['setting']
    seed(setting, run['seed'])
    save(output / 'resolved_config.json', c)
    if setting == 'taac':
        model, owner = make_model(c, p, run['method'], run.get('alpha'), output)
        from fuxictr.pytorch.dataloaders import RankDataLoader
        from fuxictr_ext.taac2025.idshare_taac_dataloader import TaacSharedIdentityDataLoader
        params = dict(c, data_loader=TaacSharedIdentityDataLoader)
        train, validation = RankDataLoader(model.feature_map, stage='train', **params).make_iterator()
        assert train.num_samples == p['expected_train_rows'] and validation.num_samples == p['expected_validation_rows']
        milestone('train_validation_ready', train_rows=train.num_samples, validation_rows=validation.num_samples)
        original_step = model.train_step
        counter = [0]
        def observed_step(batch):
            loss = original_step(batch)
            counter[0] += 1
            if counter[0] == 1 or counter[0] % 4000 == 0:
                value = float(loss.detach())
                assert math.isfinite(value), 'non-finite training loss'
                milestone('training', step=counter[0], loss=value)
            return loss
        model.train_step = observed_step
        model.fit(train, validation_data=validation, **c)
        from run_idshare_bridge import _best_validation_auc
        auc = _best_validation_auc(model)
        history = model._history
        epochs, best_epoch, total_steps = 1, 1, model._total_steps
        data_checks = {'train': train.dataset.manifest_contract, 'validation': validation.dataset.manifest_contract}
    else:
        from recscale.datasets.kuairand27k_k1 import KuaiRand27KK1MMapDataset
        train = KuaiRand27KK1MMapDataset(c, split='train')
        vc = copy.deepcopy(c); vc['dataset']['verify_processed_sha256'] = False
        validation = KuaiRand27KK1MMapDataset(vc, split='val')
        assert len(train) == p['expected_train_rows'] and len(validation) == p['expected_validation_rows']
        milestone('train_validation_ready', train_rows=len(train), validation_rows=len(validation))
        model, owner = make_model(c, p, run['method'], run.get('alpha'), output, train=train, validation=validation)
        owner.train()
        assert owner.best_checkpoint_path is not None
        checkpoint = torch.load(owner.best_checkpoint_path, map_location=owner.device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        metrics = owner.evaluate_dataset(validation)
        auc = float(metrics['auc'])
        assert abs(auc - owner.best_metrics['auc']) <= 1e-10
        history = owner.epoch_history
        epochs, best_epoch = owner.epochs_ran, owner.best_checkpoint_epoch
        total_steps = history[-1]['global_step']
        data_checks = {'train_sha_verification': train.sha256_verification,
                       'processed_manifest_sha256': sha(c['dataset']['processed_manifest'])}
    assert math.isfinite(auc) and finite_parameters(model)
    cache = read(Path(p['cache_path']).with_suffix('.json')) if run['method'] == 'tailshare' else None
    return {'status': 'completed', 'run_key': run['run_key'], 'method': run['method'], 'seed': run['seed'],
            'alpha': run.get('alpha'), 'tau': run.get('tau'), 'selection_sha256': selection_sha,
            'datasets_loaded': ['train', 'validation'], 'train_rows': p['expected_train_rows'],
            'validation_rows': p['expected_validation_rows'], 'global_l2': 0., 'weight_decay': 0.,
            'parameter_count': sum(x.numel() for x in model.parameters()), 'production_param_count_enforced': True,
            'identity_shape': list(identity(model, c, setting).weight.shape), 'data_checks': data_checks,
            'frequency_cache': cache, 'epochs_ran': epochs, 'selected_epoch': best_epoch, 'total_steps': total_steps,
            'best_validation_auc': auc, 'history': history, 'finite_parameters': True,
            'resolved_config_sha256': sha(output / 'resolved_config.json'), 'resolved_config': c}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['local-check', 'check', 'formal'])
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-key')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--selection-sha')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    p = verify(args.manifest_sha)
    local = args.mode == 'local-check'
    assert local or torch.cuda.is_available(), 'GPU required'
    if args.mode == 'formal':
        sys.stdout = BlindedStdout(sys.stdout)
    from adamar_frozen_compat import install
    install()
    install_guard(train_only=args.mode != 'formal')
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(2)
    binding = {'protocol': p['protocol'], 'setting': p['setting'], 'mode': args.mode,
               'source_manifest_sha256': args.manifest_sha, 'protocol_sha256': sha(ROOT / 'PROTOCOL.json'),
               'parent_package_sha256': p['parent_package_sha256'],
               'uses_test_dataset': False, 'uses_test_labels': False}
    milestone('package_verified', **binding)
    started = time.monotonic()
    try:
        if args.mode != 'formal':
            result = checks(p, args.output, local)
        else:
            assert args.run_key in p['runs']
            run = copy.deepcopy(p['runs'][args.run_key])
            if run['stage'] == 'confirmation':
                assert args.selection_sha and len(args.selection_sha) == 64
                assert args.alpha in [10.**x for x in p['exponents']]
                run['alpha'] = args.alpha
            else:
                assert args.alpha is None and args.selection_sha is None
            result = formal(p, run, args.output, args.selection_sha)
        save(args.output / 'training_report.json', {**binding, **result, 'wall_seconds': time.monotonic()-started})
        milestone('finished', setting=p['setting'], mode=args.mode, status=result['status'])
    except Exception as error:
        save(args.output / 'training_report.json', {**binding, 'status': 'failed',
             'error_type': type(error).__name__, 'error': str(error), 'wall_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
