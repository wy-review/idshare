#!/usr/bin/env python3
"""Depth overlay: retain the original model, loader, optimizer and selection."""

import argparse
import copy
import gc
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import strong_runtime as base
from strong_common import finite_parameters, install_guard, milestone, read, save, sha

ROOT = Path(__file__).resolve().parent
ORIGINAL_CONFIG = base.config


def config(p, run, output, local=False, train_only=False):
    c = ORIGINAL_CONFIG(p, run, output, local=local, train_only=train_only)
    if p['setting'] == 'taac':
        c['idshare_enabled'] = run['method'] == 'idshare'
        c['num_layers'] = run['depth']
        c['full_table_l2_target'] = ('prequantization_assignment_table' if c['idshare_enabled'] else 'identity_output_table')
    else:
        if run['method'] == 'idshare':
            from frozen_config_builder import build_core_config
            parent = read(ROOT / 'PARENT_MATRIX.json')
            arm = copy.deepcopy(next(v for v in parent['arms'].values() if v['family'] == 'product_m1'))
            arm['full_table_l2_coefficient'] = 0.
            if local:
                # Synthetic IDs use the frozen quantizer's production K and defaults.
                q = {'enabled': True, 'identity_fields': ['video_id'], 'codebook_size': p['codebook_size'],
                     'num_subspaces': 1, 'num_residual_levels': 1, 'continuous_residual': False,
                     'base_embedding_mode': 'split_zero_base',
                     'private_row_initialization_mode': 'train_first_touch_deterministic_code',
                     'assignment_stability_mode': 'free_nearest',
                     'task_codebook_gradient_mode': 'legacy_hard_plus_soft',
                     'temperature_start': 1., 'temperature_end': .3, 'margin': .1, 'code_init_radius': .2,
                     'zero_l2_weight': 0., 'commitment_weight': 0., 'codebook_loss_weight': 1.,
                     'distance_backend': 'gemm', 'distance_row_chunk_size': 1024,
                     'compact_distance_outputs': True, 'sparse_batch_diagnostics': True,
                     'defer_cumulative_diagnostics': True, 'initialization_seed': run['seed'],
                     'isolate_initialization_rng': True}
                c['model']['zero_anchor_identity_quantization'] = q
            else:
                c = build_core_config(Path(parent['processed_manifest']), run=run, arm=arm)
                c['model'].update(backbone_type='rankmixer_v2', inter_layer_residual=False, carry_path_ablation=False)
                c['training'].update(save_dir=str(output / 'checkpoints'), suppress_effect_metric_logs=True)
                if train_only:
                    c['dataset']['verify_processed_sha256_splits'] = ['train']
        c['model']['num_mixer_layers'] = run['depth']
    assert run['method'] in ['continuous', 'idshare'] and run['depth'] in p['depths'] and run['seed'] == 42
    return c


def blocks(model, setting):
    return model.backbone.blocks if setting == 'taac' else model.blocks


def model_info(model, c, p, local):
    setting = p['setting']
    depth = c['num_layers'] if setting == 'taac' else c['model']['num_mixer_layers']
    bs = blocks(model, setting)
    assert len(bs) == depth
    assert all(sum(x.numel() for x in b.parameters()) == p['parameters_per_block'] for b in bs)
    q = model.identity_quantizer if setting == 'taac' else model.encoder.zero_anchor_identity_quantizer
    method = 'idshare' if q is not None else 'continuous'
    count = sum(x.numel() for x in model.parameters())
    if not local:
        assert count == p['depth2_parameter_counts'][method] + (depth-2)*p['parameters_per_block'], count
        assert list(base.identity(model, c, setting).weight.shape) == [p['expected_identity_rows'], 16]
    return {'depth': depth, 'method': method, 'parameter_count': count,
            'mixing_block_parameters': depth * p['parameters_per_block'],
            'production_param_count_enforced': not local}


def make_model(c, p, method, alpha, output, local=False, train=None, validation=None, fmap=None):
    setting = p['setting']
    base.seed(setting, c['seed'])
    if setting == 'taac':
        import src
        from run_idshare_bridge import _prepare_feature_map, _synthetic_feature_map
        fmap = fmap or (_synthetic_feature_map() if local else _prepare_feature_map(c))
        model = src.IDShareUnifiedMixer(fmap, **c)
        if not local:
            # Preserve the frozen TAAC pre-loader dummy-forward RNG consumption.
            model.count_parameters(count_embedding=True, batch_size=1)
        Path(model.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model._max_gradient_norm = c.get('max_gradient_norm', 1.)
        owner = model
    else:
        from recscale.models.s2drec import S2DRecModel
        from recscale.trainer import Trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = S2DRecModel(c).to(device)
        owner = Trainer(model, train, validation, c, device)
    info = model_info(model, c, p, local)
    assert info['method'] == method
    milestone('model_checked', **info)
    return model, owner


def checks(p, output, run_key=None, gpu=False):
    routes = ([p['runs'][run_key]] if run_key else
              [dict(run_key=f'd{d}_{m}_s42', depth=d, method=m, seed=42) for d in p['depths'] for m in ['continuous', 'idshare']])
    results, local_counts = [], {}
    for run in routes:
        target = output / run['run_key']
        target.mkdir(parents=True, exist_ok=True)
        c = config(p, run, target, local=True, train_only=True)
        anchor_config = config(p, dict(run, depth=2), target, local=True, train_only=True)
        normalized = copy.deepcopy(c)
        if p['setting'] == 'taac':
            normalized['num_layers'] = 2
            q = c['idshare_quantizer_config'] if run['method'] == 'idshare' else None
        else:
            normalized['model']['num_mixer_layers'] = 2
            q = c['model'].get('zero_anchor_identity_quantization')
        assert normalized == anchor_config, 'non-depth configuration drift'
        if run['method'] == 'idshare':
            assert q['codebook_size'] == p['codebook_size']
            assert q['private_row_initialization_mode'] == 'train_first_touch_deterministic_code'
            assert q['task_codebook_gradient_mode'] == 'legacy_hard_plus_soft'
            assert q['codebook_loss_weight'] == 1. and q['commitment_weight'] == 0.
        if gpu and p['setting'] == 'taac':
            c['gpu'] = 0
        data = base.SyntheticKuai() if p['setting'] == 'kuairand' else None
        model, owner = make_model(c, p, run['method'], None, target, local=True, train=data)
        if p['setting'] == 'taac':
            batch = {'label': torch.tensor([0., 1., 0., 1.])}
            for name, spec in model.feature_map.features.items():
                batch[name] = (torch.tensor([[0, 4, 5, 6, 7, 8, 9, 10]]*4)
                               if spec['type'] == 'sequence' else torch.tensor([4, 5, 8, 9]))
        else:
            batch = next(iter(owner.train_loader))
        fired = [0] * run['depth']
        handles = []
        for i, block in enumerate(blocks(model, p['setting'])):
            def seen(module, args, result, i=i):
                fired[i] += 1
            handles.append(block.register_forward_hook(seen))
        losses = [base.step(model, owner, batch, p['setting']) for _ in range(p['startup_synthetic_steps'])]
        for handle in handles:
            handle.remove()
        assert all(n == p['startup_synthetic_steps'] for n in fired)
        assert all(any(x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.abs().sum() > 0)
                       for x in block.parameters()) for block in blocks(model, p['setting']))
        if run['method'] == 'idshare':
            quantizer = model.identity_quantizer if p['setting'] == 'taac' else model.encoder.zero_anchor_identity_quantizer
            assert any(x.grad is not None and bool(x.grad.abs().sum() > 0) for x in quantizer.parameters())
            routing_grad = base.identity(model, c, p['setting']).weight.grad
            assert routing_grad is not None and bool(torch.isfinite(routing_grad).all()) and bool(routing_grad.abs().sum() > 0)
        assert finite_parameters(model) and all(math.isfinite(x) for x in losses)
        info = model_info(model, c, p, True)
        local_counts[(run['depth'], run['method'])] = info['parameter_count']
        results.append(dict(run_key=run['run_key'], **info, loss_first=losses[0], loss_last=losses[-1],
                            block_forward_counts=fired, finite_parameters=True, all_blocks_receive_gradients=True,
                            learned_routing_and_codebook_gradients=run['method'] == 'idshare'))
        if getattr(owner, 'log_file', None):
            owner.log_file.close()
        del model, owner, data, batch, block, handles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not run_key:
        for method in ['continuous', 'idshare']:
            for depth in p['new_depths']:
                assert local_counts[(depth, method)] - local_counts[(2, method)] == (depth-2)*p['parameters_per_block']
    return {'status': 'passed', 'datasets_loaded': [], 'auc_computed': False,
            'routes_checked': len(results), 'routes': results, 'cuda_verified': gpu,
            'production_param_count_enforced': False, 'steps_per_route': p['startup_synthetic_steps']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['local-check', 'startup-gpu', 'formal'])
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-key')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    p = base.verify(args.manifest_sha)
    assert args.mode == 'local-check' or torch.cuda.is_available(), 'GPU required'
    from adamar_frozen_compat import install
    install()
    install_guard(train_only=args.mode != 'formal')
    torch.set_num_threads(2)
    logging.disable(logging.CRITICAL)
    sys.stdout = base.BlindedStdout(sys.stdout)
    binding = {'protocol': p['protocol'], 'setting': p['setting'], 'mode': args.mode,
               'source_manifest_sha256': args.manifest_sha, 'protocol_sha256': sha(ROOT / 'PROTOCOL.json'),
               'parent_package_sha256': p['parent_package_sha256'], 'uses_test_dataset': False, 'uses_test_labels': False}
    milestone('package_verified', **binding)
    started = time.monotonic()
    report_name = 'training_report.json' if args.mode == 'formal' else 'startup_check.json'
    try:
        if args.mode != 'formal':
            result = checks(p, output, args.run_key, args.mode == 'startup-gpu')
        else:
            assert args.run_key in p['runs']
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'startup-gpu',
                            '--manifest-sha', args.manifest_sha, '--run-key', args.run_key,
                            '--output', str(output / 'startup')], check=True)
            startup = read(output / 'startup/startup_check.json')
            assert startup['status'] == 'passed' and startup['cuda_verified'] and startup['datasets_loaded'] == []
            base.config, base.make_model = config, make_model
            torch.cuda.reset_peak_memory_stats()
            formal_started = time.monotonic()
            result = base.formal(p, p['runs'][args.run_key], output, None)
            result.update(depth=p['runs'][args.run_key]['depth'],
                          mixing_block_parameters=p['runs'][args.run_key]['depth']*p['parameters_per_block'],
                          formal_wall_seconds=time.monotonic()-formal_started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                          startup_check=startup, startup_check_sha256=sha(output / 'startup/startup_check.json'))
        save(output / report_name, {**binding, **result, 'wall_seconds': time.monotonic()-started})
        milestone('finished', setting=p['setting'], mode=args.mode, status=result['status'])
    except Exception as error:
        save(output / report_name, {**binding, 'status': 'failed', 'error_type': type(error).__name__,
                                   'error': str(error), 'wall_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
