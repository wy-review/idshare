#!/usr/bin/env python3
"""K-only configuration overlay on the immutable depth training runtime."""

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
from unittest.mock import patch

import torch
import depth_runtime as depth
import strong_runtime as base
from strong_common import finite_parameters, install_guard, milestone, read, save, sha

ROOT = Path(__file__).resolve().parent


def quantizer_config(c, setting):
    return c['idshare_quantizer_config'] if setting == 'taac' else c['model']['zero_anchor_identity_quantization']


def expected_quantizer(p, run):
    q = copy.deepcopy(p['anchor_quantizers'][str(run['seed'])])
    q['codebook_size'] = run['k']
    if 'codebook_size_by_field' in q:
        q['codebook_size_by_field'] = {f: run['k'] for f in q['identity_fields']}
    return q


def config(p, run, output, local=False, train_only=False):
    assert run['depth'] == 2 and run['method'] == 'idshare'
    assert run['seed'] in p['seeds'] and run['k'] in p['ks']
    c = depth.config(p, run, output, local=local, train_only=train_only)
    q = quantizer_config(c, p['setting'])
    # Formal configuration must match the accepted same-seed anchor before changing K.
    if not local or p['setting'] == 'taac':
        assert q == p['anchor_quantizers'][str(run['seed'])], 'anchor quantizer drift'
    new_q = expected_quantizer(p, run)
    if p['setting'] == 'taac':
        c['idshare_quantizer_config'] = new_q
    else:
        c['model']['zero_anchor_identity_quantization'] = new_q
    return c


def model_info(model, c, p, local):
    bs = depth.blocks(model, p['setting'])
    assert len(bs) == 2
    assert all(sum(x.numel() for x in b.parameters()) == p['parameters_per_block'] for b in bs)
    quantizer = model.identity_quantizer if p['setting'] == 'taac' else model.encoder.zero_anchor_identity_quantizer
    k = quantizer_config(c, p['setting'])['codebook_size']
    assert quantizer.codebook_size == k and len(quantizer.identity_fields) == 1
    if p['setting'] == 'taac':
        assert tuple(quantizer.codebook_sizes_by_field) == (k,)
    count = sum(x.numel() for x in model.parameters())
    if not local:
        assert count == p['depth2_parameter_counts']['idshare'] + 18*(k-p['anchor_k'])
        assert list(base.identity(model, c, p['setting']).weight.shape) == [p['expected_identity_rows'], 16]
    return dict(depth=2, method='idshare', k=k, parameter_count=count,
                mixing_block_parameters=2*p['parameters_per_block'], production_param_count_enforced=not local)


def make_model(*args, **kwargs):
    depth.model_info = model_info
    return depth.make_model(*args, **kwargs)


def formal_config_check(p, run, target):
    if p['setting'] == 'taac':
        old = depth.config(p, run, target, local=False)
        c = config(p, run, target, local=False)
    else:
        import frozen_config_builder as builder
        parent = read(ROOT/'PARENT_MATRIX.json')
        fake = dict(field_names=['video_id'] + [f'f{i}' for i in range(36)], cardinalities=[64]*37)
        actual_read = Path.read_text
        def guarded(path, *args, **kwargs):
            if str(path) == parent['processed_manifest']:
                return json.dumps(fake)
            assert path.resolve().is_relative_to(ROOT), 'outside package access forbidden'
            return actual_read(path, *args, **kwargs)
        # Exercise the actual formal builder, replacing only manifest metadata.
        with patch.object(Path, 'read_text', guarded), patch.object(base, 'sha', lambda path: parent['expected_processed_manifest_sha256']):
            old = depth.config(p, run, target, local=False)
            c = config(p, run, target, local=False)
    normalized = copy.deepcopy(c)
    if p['setting'] == 'taac':
        normalized['idshare_quantizer_config'] = quantizer_config(old, 'taac')
    else:
        normalized['model']['zero_anchor_identity_quantization'] = quantizer_config(old, 'kuairand')
    assert normalized == old, 'non-K formal configuration drift'
    assert quantizer_config(old, p['setting']) == p['anchor_quantizers'][str(run['seed'])]
    assert quantizer_config(c, p['setting']) == expected_quantizer(p, run)
    assert c['seed'] == run['seed']
    return True


def checks(p, output, run_key=None, gpu=False):
    runs = ([p['runs'][run_key]] if run_key else
            [dict(run_key=f'k{k}_idshare_s{s}', method='idshare', depth=2, seed=s, k=k)
             for s in p['seeds'] for k in p['ks']])
    rows, counts = [], {}
    for run in runs:
        target = output/run['run_key']; target.mkdir(parents=True, exist_ok=True)
        formal_config_check(p, run, target)
        c = config(p, run, target, local=True, train_only=True)
        assert quantizer_config(c, p['setting']) == expected_quantizer(p, run)
        if gpu and p['setting'] == 'taac':
            c['gpu'] = 0
        data = base.SyntheticKuai() if p['setting'] == 'kuairand' else None
        model, owner = make_model(c, p, 'idshare', None, target, local=True, train=data)
        if p['setting'] == 'taac':
            batch = {'label': torch.tensor([0., 1., 0., 1.])}
            for name, spec in model.feature_map.features.items():
                batch[name] = (torch.tensor([[0, 4, 5, 6, 7, 8, 9, 10]]*4)
                               if spec['type'] == 'sequence' else torch.tensor([4, 5, 8, 9]))
        else:
            batch = next(iter(owner.train_loader))
        fired = [0, 0]
        handles = []
        for i, block in enumerate(depth.blocks(model, p['setting'])):
            def seen(module, args, result, i=i):
                fired[i] += 1
            handles.append(block.register_forward_hook(seen))
        losses = [base.step(model, owner, batch, p['setting']) for _ in range(p['startup_synthetic_steps'])]
        for handle in handles:
            handle.remove()
        assert fired == [p['startup_synthetic_steps']]*2
        assert all(any(x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.abs().sum() > 0)
                       for x in block.parameters()) for block in depth.blocks(model, p['setting']))
        q = model.identity_quantizer if p['setting'] == 'taac' else model.encoder.zero_anchor_identity_quantizer
        assert any(x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.abs().sum() > 0) for x in q.parameters())
        grad = base.identity(model, c, p['setting']).weight.grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.abs().sum() > 0)
        assert finite_parameters(model) and all(math.isfinite(x) for x in losses)
        info = model_info(model, c, p, True)
        counts[(run['seed'], run['k'])] = info['parameter_count']
        rows.append(dict(run_key=run['run_key'], seed=run['seed'], **info,
            formal_config_checked=True, quantizer_config=quantizer_config(c, p['setting']),
            block_forward_counts=fired, all_blocks_receive_gradients=True,
            learned_routing_and_codebook_gradients=True, finite_parameters=True,
            loss_first=losses[0], loss_last=losses[-1]))
        if getattr(owner, 'log_file', None):
            owner.log_file.close()
        del model, owner, q, grad, batch, data, block, handles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not run_key:
        for s in p['seeds']:
            for k in p['ks']:
                assert counts[(s, k)]-counts[(s, p['anchor_k'])] == 18*(k-p['anchor_k'])
    return dict(status='passed', datasets_loaded=[], auc_computed=False, real_manifest_read=False,
        formal_config_checked=True, routes_checked=len(rows), routes=rows, cuda_verified=gpu,
        steps_per_route=p['startup_synthetic_steps'])


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
    assert args.mode == 'local-check' or torch.cuda.is_available()
    from adamar_frozen_compat import install
    install()
    install_guard(train_only=args.mode != 'formal')
    torch.set_num_threads(2)
    logging.disable(logging.CRITICAL)
    sys.stdout = base.BlindedStdout(sys.stdout)
    binding = dict(protocol=p['protocol'], setting=p['setting'], mode=args.mode,
        source_manifest_sha256=args.manifest_sha, protocol_sha256=sha(ROOT/'PROTOCOL.json'),
        parent_package_sha256=p['parent_package_sha256'], uses_test_dataset=False, uses_test_labels=False)
    milestone('package_verified', **binding)
    started = time.monotonic()
    name = 'training_report.json' if args.mode == 'formal' else 'startup_check.json'
    try:
        if args.mode != 'formal':
            result = checks(p, output, args.run_key, args.mode == 'startup-gpu')
        else:
            run = p['runs'][args.run_key]
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'startup-gpu',
                '--manifest-sha', args.manifest_sha, '--run-key', args.run_key,
                '--output', str(output/'startup')], check=True)
            startup = read(output/'startup/startup_check.json')
            assert startup['status'] == 'passed' and startup['cuda_verified'] and startup['datasets_loaded'] == []
            base.config, base.make_model = config, make_model
            torch.cuda.reset_peak_memory_stats()
            formal_started = time.monotonic()
            result = base.formal(p, run, output, None)
            result.update(depth=2, k=run['k'], mixing_block_parameters=2*p['parameters_per_block'],
                formal_wall_seconds=time.monotonic()-formal_started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                startup_check=startup, startup_check_sha256=sha(output/'startup/startup_check.json'))
        save(output/name, {**binding, **result, 'wall_seconds': time.monotonic()-started})
        milestone('finished', setting=p['setting'], mode=args.mode, status=result['status'])
    except Exception as error:
        save(output/name, {**binding, 'status': 'failed', 'error_type': type(error).__name__,
                          'error': str(error), 'wall_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
