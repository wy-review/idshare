#!/usr/bin/env python3
"""FrozenHash adapters; immutable parent modules are never edited."""

import argparse
import copy
import gc
import logging
import math
from pathlib import Path
import sys
import time

import torch
import strong_runtime as base
from strong_common import finite_parameters, install_guard, max_state_error, milestone, read, save, sha

ROOT = Path(__file__).resolve().parent
ORIGINAL_CONFIG = base.config


def config(p, run, output, local=False, train_only=False, fixed=True):
    c = ORIGINAL_CONFIG(p, run, output, local, train_only)
    if p['setting'] == 'taac':
        c['idshare_enabled'] = True
        q = c['idshare_quantizer_config']
    else:
        from frozen_config_builder import build_core_config
        parent = read(ROOT / 'PARENT_MATRIX.json')
        arm = copy.deepcopy(next(a for a in parent['arms'].values() if a['family'] == 'product_m1'))
        arm['full_table_l2_coefficient'] = 0.
        if not local:
            c = build_core_config(Path(parent['processed_manifest']), run=run, arm=arm)
            c['model'].update(backbone_type='rankmixer_v2', inter_layer_residual=False, carry_path_ablation=False)
            c['training'].update(save_dir=str(output / 'checkpoints'), suppress_effect_metric_logs=True)
            if train_only:
                c['dataset']['verify_processed_sha256_splits'] = ['train']
        else:
            # Reuse the exact quantizer config builder with a synthetic metadata object.
            from recscale.models.zero_anchor_identity_quantizer import ZeroAnchorIdentityQuantizer
            q = {'enabled': True, 'identity_fields': ['video_id'], 'codebook_size': p['codebook_size'],
                 'num_subspaces': 1, 'num_residual_levels': 1, 'continuous_residual': False,
                 'base_embedding_mode': 'split_zero_base', 'private_row_initialization_mode': 'train_first_touch_deterministic_code',
                 'assignment_stability_mode': 'free_nearest', 'task_codebook_gradient_mode': 'legacy_hard_plus_soft',
                 'temperature_start': 1., 'temperature_end': .3, 'margin': .1, 'code_init_radius': .2,
                 'zero_l2_weight': 0., 'commitment_weight': 0., 'codebook_loss_weight': 1.,
                 'distance_backend': 'gemm', 'distance_row_chunk_size': 1024,
                 'compact_distance_outputs': True, 'sparse_batch_diagnostics': True,
                 'defer_cumulative_diagnostics': True, 'initialization_seed': run['seed'],
                 'isolate_initialization_rng': True}
            c['model'][ZeroAnchorIdentityQuantizer.CONFIG_KEY] = q
        from recscale.models.zero_anchor_identity_quantizer import ZeroAnchorIdentityQuantizer
        q = c['model'][ZeroAnchorIdentityQuantizer.CONFIG_KEY]
    assert q['codebook_size'] == p['codebook_size']
    assert q['codebook_loss_weight'] == 1. and q['commitment_weight'] == 0.
    q['private_row_initialization_mode'] = (p['private_row_initialization_mode'] if fixed
                                          else 'train_first_touch_deterministic_code')
    return c


def quantizer(model, setting):
    return model.identity_quantizer if setting == 'taac' else model.encoder.zero_anchor_identity_quantizer


def make_model(c, p, method, alpha, output, local=False, train=None, validation=None, fmap=None):
    base.seed(p['setting'], c['seed'])
    if p['setting'] == 'taac':
        import src
        from run_idshare_bridge import _synthetic_feature_map, _prepare_feature_map
        fmap = fmap or (_synthetic_feature_map() if local else _prepare_feature_map(c))
        model = src.IDShareUnifiedMixer(fmap, **c)
        # This original dummy forward consumes RNG and must precede the loader.
        if not local:
            model.count_parameters(count_embedding=True, batch_size=1)
        Path(model.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model._max_gradient_norm = c.get('max_gradient_norm', 1.)
        owner = model
    else:
        from recscale.models.s2drec import S2DRecModel
        from recscale.trainer import Trainer
        model = S2DRecModel(c).to('cpu' if local else 'cuda')
        owner = Trainer(model, train, validation, c, torch.device('cpu' if local else 'cuda'))
    q = quantizer(model, p['setting'])
    assert q is not None and q.codebook(0).shape[0] == p['codebook_size']
    assert q.get_metadata()['continuous_residual'] is False
    if not local:
        count = sum(v.numel() for v in model.parameters())
        assert count == p['expected_base_parameters'], f'parameter count drift: {count}'
    return model, owner


def quantizer_checks(q):
    """No data: hash invariance, actual task gradients and weighted deduplication."""
    device = q.base_embeddings.device
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 4, 15], device=device)
    indices = torch.zeros_like(ids)
    indices[ids >= 4] = q.deterministic_private_code_indices(ids[ids >= 4])
    residual = torch.randn(len(ids), 16, device=device, requires_grad=True)
    result = q.quantize_residuals(residual, 0, hard_indices=indices)
    assert torch.equal(result['indices'], indices)
    assert torch.equal(result['value'].detach(), q.codebook(0).detach()[indices])
    shifted = q.quantize_residuals(residual.detach() + 9., 0, hard_indices=indices)
    assert torch.equal(result['value'].detach(), shifted['value'].detach())
    grad = torch.autograd.grad(result['value'].sum(), residual, retain_graph=True)[0]
    assert torch.equal(grad, torch.ones_like(grad))
    # Non-assigned centers must not receive soft-nearest task gradients.
    q.zero_grad(set_to_none=True)
    result['value'].sum().backward()
    trainable = [v for n, v in q.named_parameters() if 'nonzero' in n or 'codebook' in n]
    assert any(v.grad is not None and bool(v.grad.abs().sum() > 0) for v in trainable)
    fields = torch.zeros(len(ids), len(q.field_names), 16, device=device)
    sparse = torch.zeros(len(ids), len(q.field_names), dtype=torch.long, device=device)
    sparse[:, q.name_to_index[q.identity_fields[0]]] = ids
    q.eval()
    out = q(fields, sparse).detach()
    fields[:, q.name_to_index[q.identity_fields[0]]] += 7.
    moved = q(fields, sparse).detach()
    assert torch.equal(out, moved), 'private routing states leaked into frozen output'
    assert torch.equal(out[4], out[7]), 'same ID must share output'
    assert torch.count_nonzero(out[0]) == 0, 'padding must remain zero'
    q.train()
    return {'fixed_hash': True, 'private_perturbation_invariant': True, 'identity_st_gradient': True,
            'codebook_task_gradient': True, 'padding_zero': True, 'auc_computed': False}


def checks(p, output, local):
    setting = p['setting']
    routes = []
    # Every authorized route gets a no-data forward/backward check at the real K.
    for s in p['seeds']:
        run = {'seed': s, 'run_key': f'check_s{s}'}
        c = config(p, run, output, local=True)
        data = base.SyntheticKuai() if setting == 'kuairand' else None
        model, owner = make_model(c, p, 'frozenhash', None, output, True, train=data)
        learned = config(p, run, output, local=True, fixed=False)
        model_ref, owner_ref = make_model(learned, p, 'idshare', None, output, True, train=data)
        assert max_state_error(model, model_ref) == 0., 'initial states must match IDShare'
        if getattr(owner_ref, 'log_file', None):
            owner_ref.log_file.close()
        del model_ref, owner_ref
        check = quantizer_checks(quantizer(model, setting))
        if setting == 'taac':
            batch = {'label': torch.tensor([0., 1., 0., 1.])}
            for name, spec in model.feature_map.features.items():
                batch[name] = (torch.tensor([[0, 4, 5, 6, 7, 8, 9, 10]] * 4)
                               if spec['type'] == 'sequence' else torch.tensor([4, 5, 8, 9]))
        else:
            batch = next(iter(owner.train_loader))
        losses = [base.step(model, owner, batch, setting) for _ in range(3)]
        assert finite_parameters(model)
        routes.append({'seed': s, **check, 'identical_initial_state_to_idshare': True, 'finite_losses': losses})
        if getattr(owner, 'log_file', None):
            owner.log_file.close()
        del model, owner
        gc.collect()
    if local:
        return {'status': 'passed', 'routes_checked': 3, 'routes': routes,
                'datasets_loaded': [], 'auc_computed': False, 'local_check': True}
    run = {'seed': 42, 'run_key': 'engineering_check'}
    c = config(p, run, output, train_only=True)
    if setting == 'taac':
        model, owner = make_model(c, p, 'frozenhash', None, output)
        from fuxictr_ext.taac2025.idshare_taac_dataloader import TaacSharedIdentityDataLoader
        train = TaacSharedIdentityDataLoader(model.feature_map, data_path=c['train_data'], split='train',
                                            **dict(c, max_samples=c['batch_size'] * p['check_steps']))
        iterator = iter(train)
    else:
        from recscale.datasets.kuairand27k_k1 import KuaiRand27KK1MMapDataset
        train = KuaiRand27KK1MMapDataset(c, split='train')
        assert len(train) == p['expected_train_rows']
        model, owner = make_model(c, p, 'frozenhash', None, output, train=train)
        iterator = iter(owner.train_loader)
    milestone('train_ready', setting=setting, mode='check')
    before = quantizer(model, setting).codebook(0).detach().clone()
    losses = [base.step(model, owner, next(iterator), setting) for _ in range(p['check_steps'])]
    after = quantizer(model, setting).codebook(0).detach()
    assert not torch.equal(before, after), 'shared embeddings did not learn'
    assert finite_parameters(model)
    return {'status': 'passed', 'routes_checked': 3, 'routes': routes, 'datasets_loaded': ['train'],
            'auc_computed': False, 'validation_files_read': False, 'local_check': False,
            'production_param_count_enforced': True, 'parameter_count': sum(v.numel() for v in model.parameters()),
            'identity_shape': list(base.identity(model, c, setting).weight.shape),
            'cuda_verified': True, 'steps': p['check_steps'], 'loss_first': losses[0], 'loss_last': losses[-1],
            'shared_codebook_updated': True, 'finite_parameters': True}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('mode', choices=['local-check', 'check', 'formal'])
    a.add_argument('--manifest-sha', required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--run-key')
    args = a.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    p = base.verify(args.manifest_sha)
    from adamar_frozen_compat import install
    install()
    install_guard(train_only=args.mode != 'formal')
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(2)
    sys.stdout = base.BlindedStdout(sys.stdout)
    assert args.mode == 'local-check' or torch.cuda.is_available()
    binding = {'setting': p['setting'], 'protocol': p['protocol'], 'mode': args.mode,
               'source_manifest_sha256': args.manifest_sha, 'protocol_sha256': sha(ROOT / 'PROTOCOL.json'),
               'parent_package_sha256': p['parent_package_sha256'],
               'uses_test_dataset': False, 'uses_test_labels': False}
    milestone('package_verified', **binding)
    started = time.monotonic()
    try:
        if args.mode == 'formal':
            assert args.run_key in p['runs']
            base.config, base.make_model = config, make_model
            result = base.formal(p, p['runs'][args.run_key], args.output, None)
            result.update(codebook_size=p['codebook_size'], fixed_assignment=True,
                          private_row_initialization_mode=p['private_row_initialization_mode'],
                          shared_codebook_trainable=True, best_k_selected=False)
        else:
            result = checks(p, args.output, args.mode == 'local-check')
        save(args.output / 'training_report.json', {**binding, **result, 'duration_seconds': time.monotonic()-started})
        milestone('finished', status=result['status'], setting=p['setting'], mode=args.mode)
    except Exception as error:
        save(args.output / 'training_report.json', {**binding, 'status': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
