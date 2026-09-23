#!/usr/bin/env python3
"""Portable, one-configuration entry point using archived scientific modules.

No scheduler, remote APIs, automatic sweep, or dataset download is involved.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def backend(row):
    setting = row['setting']
    if row['family'] == 'additional_backbone': return ROOT / 'snapshots/legacy' / setting
    if row['method'] == 'frozenhash': return ROOT / 'snapshots/frozenhash' / setting
    if row['family'] == 'capacity': return ROOT / 'snapshots/capacity' / setting
    if row['family'] == 'depth': return ROOT / 'snapshots/depth' / setting
    if row['method'] == 'adamar': return ROOT / 'snapshots/adamar_extension' / setting
    return ROOT / 'snapshots/baselines' / setting


def taac_config(row, source, paths, output, synthetic):
    import yaml
    if row['backbone'] == 'din':
        candidates = list(source.glob('**/configs/resolved*/*.yaml'))
        for file in candidates:
            c = yaml.safe_load(file.read_text())
            q = c['model'].get('zero_anchor_identity_quantization', {})
            dose = c['training'].get('zero_anchor_full_table_l2', {}).get('coefficient', 0.)
            if c['seed'] == row['seed'] and bool(q.get('enabled')) == (row['method'] == 'idshare') and float(dose) == row['global_l2']:
                if synthetic:
                    c['dataset'].update(num_items=63, sparse_cols=['f1','f2'], cardinalities=[64,64],
                                        dense_cols=[], shared_item_identity_remap_enabled=True)
                    c['training']['num_workers'] = 0
                else:
                    c['dataset']['path'] = paths['root']
                    c['dataset']['train_input_seen_mask_path'] = paths['seen_mask']
                c['training']['save_dir'] = str(output / 'checkpoints')
                return c
        raise ValueError('No archived DIN configuration matches this cell')
    cfgroot = source / 'model_zoo/UnifiedBackbone/config/idshare_l2_response_r02'
    configs = yaml.safe_load((cfgroot / 'model_config.yaml').read_text())
    choice = next(v for k, v in configs.items() if k != 'Base' and v.get('seed') == row['seed'] and not v.get('idshare_enabled'))
    c = copy.deepcopy(configs['Base'])
    c.update(copy.deepcopy(choice))
    ds = yaml.safe_load((cfgroot / 'dataset_config.yaml').read_text())[c['dataset_id']]
    c.update(ds)
    c.update(model_id=row['id'], model_root=str(output / 'checkpoints'), gpu=-1 if synthetic else 0,
             full_table_l2_coefficient=row['global_l2'], idshare_enabled=row['method'] in ['idshare', 'frozenhash'],
             full_table_l2_target='prequantization_assignment_table' if row['method'] in ['idshare', 'frozenhash'] else 'identity_output_table',
             num_layers=row['depth'], tensorboard=False)
    if synthetic:
        c['num_workers'] = 0
    q = c['idshare_quantizer_config']
    protocol = read(source / 'PROTOCOL.json')
    if row['family'] == 'capacity' and 'anchor_quantizers' in protocol:
        q = copy.deepcopy(protocol['anchor_quantizers'][str(row['seed'])])
        c['idshare_quantizer_config'] = q
    q['codebook_size'] = row['k']
    q['codebook_size_by_field'] = {'shared_item_id': row['k']}
    if row['method'] == 'frozenhash':
        q['private_row_initialization_mode'] = protocol['private_row_initialization_mode']
    if synthetic:
        c['shared_item_table_cardinality'] = 64
    else:
        data_root = str(Path(paths['root']).resolve())
        c['data_root'] = str(Path(paths['feature_map_root']).resolve())
        c['train_input_seen_mask_path'] = str(Path(paths['seen_mask']).resolve())
        # Preserve the published mask checksum; a different mask is not the paper data.
        if sha(c['train_input_seen_mask_path']) != c['train_input_seen_mask_sha256']:
            raise ValueError('Training-seen identity mask differs from the frozen data contract')
        c['expected_raw_data_root'] = data_root
        c['expected_samples_path_by_split'] = {s: str(Path(data_root) / d / 'samples.parquet')
                                               for s, d in [('train', 'train'), ('validation', 'val')]}
        for key, split in [('train_data', 'train'), ('valid_data', 'validation')]:
            p = output / f'{split}_manifest.json'
            p.write_text(json.dumps(dict(raw_data_root=data_root,
                       samples_path=c['expected_samples_path_by_split'][split], split=split, index_array=None)))
            c[key] = str(p)
        c['test_data'] = None
    return c


def kuai_config(row, source, paths, output, synthetic):
    builder_path = source / 'frozen_config_builder.py'
    if not builder_path.exists():
        # Extracted verbatim from the corresponding frozen legacy training module.
        builder_path = ROOT / 'configs/build_legacy_kuairand_config.py'
    spec = importlib.util.spec_from_file_location('config_builder', builder_path)
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    parent_path = source / 'PARENT_MATRIX.json'
    if not parent_path.exists(): parent_path = ROOT / 'configs/legacy_kuairand_matrix.json'
    parent = read(parent_path)
    arm = copy.deepcopy(next(v for v in parent['arms'].values()
                             if v['family'] == ('product_m1' if row['method'] in ['idshare', 'frozenhash'] else 'continuous')))
    arm['full_table_l2_coefficient'] = row['global_l2']
    manifest = Path(paths['processed_manifest']) if not synthetic else output / 'synthetic_manifest.json'
    if synthetic:
        fields = ['video_id'] + [f'f{i}' for i in range(36)]
        manifest.write_text(json.dumps(dict(field_names=fields, cardinalities=[64]*37)))
    c = builder.build_core_config(manifest, run=dict(seed=row['seed'], run_key=row['id']), arm=arm)
    c['training'].update(save_dir=str(output / 'checkpoints'), num_workers=0 if synthetic else c['training']['num_workers'],
                         suppress_effect_metric_logs=False)
    if row['backbone'] == 'rankmixer':
        c['model'].update(backbone_type='rankmixer_v2', inter_layer_residual=False, carry_path_ablation=False)
    c['model']['num_mixer_layers'] = row['depth']
    q = c['model'].get('zero_anchor_identity_quantization')
    if q:
        q['codebook_size'] = row['k']
        if 'codebook_size_by_field' in q: q['codebook_size_by_field'] = {'video_id': row['k']}
        if row['method'] == 'frozenhash':
            q['private_row_initialization_mode'] = read(source / 'PROTOCOL.json')['private_row_initialization_mode']
    return c


def train(row, source, paths, output, synthetic=False):
    import numpy as np
    import torch
    if synthetic:
        torch.set_num_threads(2)
    helpers = ROOT / 'snapshots/baselines' / row['setting']
    sys.path.insert(0, str(helpers))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'model_zoo/UnifiedBackbone'))
    from strong_common import TailSharedEmbedding, make_adamar, renew_adam, install_guard
    from strong_runtime import seed, step, SyntheticKuai
    install_guard()
    c = (taac_config if row['setting'] == 'taac' else kuai_config)(row, source, paths, output, synthetic)
    (output / 'resolved_config.json').write_text(json.dumps(c, indent=2))
    seed(row['setting'], row['seed'])
    fuxi = row['setting'] == 'taac' and row['backbone'] != 'din'
    if fuxi:
        from run_idshare_bridge import _prepare_feature_map, _synthetic_feature_map, _best_validation_auc
        from src import IDShareUnifiedMixer
        fmap = _synthetic_feature_map() if synthetic else _prepare_feature_map(c)
        model = IDShareUnifiedMixer(fmap, **c)
        if not synthetic: model.count_parameters(count_embedding=True, batch_size=1)
        Path(model.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model._max_gradient_norm = c['max_gradient_norm']
        table = model.embedding_layer.embedding_layers['target_item_id']
        owner = model
    else:
        from recscale.trainer import Trainer
        if row['setting'] == 'kuairand':
            from recscale.models.s2drec import S2DRecModel
            from recscale.datasets.kuairand27k_k1 import KuaiRand27KK1MMapDataset
            train_data = SyntheticKuai() if synthetic else KuaiRand27KK1MMapDataset(c, split='train')
            vc = copy.deepcopy(c); vc['dataset']['verify_processed_sha256'] = False
            val_data = None if synthetic else KuaiRand27KK1MMapDataset(vc, split='val')
            if not synthetic:
                contract = read(helpers / 'PROTOCOL.json')
                identity_rows = c['dataset']['cardinalities'][c['dataset']['sparse_cols'].index('video_id')]
                if (len(train_data), len(val_data), identity_rows) != (
                        contract['expected_train_rows'], contract['expected_validation_rows'],
                        contract['expected_identity_rows']):
                    raise ValueError('KuaiRand split or identity vocabulary differs from paper')
            model = S2DRecModel(c)
        else:
            from recscale.models.din import DINModel
            if synthetic:
                train_data = [dict(sparse=torch.tensor([4,5]), seq=torch.tensor([0,4,5,6]),
                                   target=torch.tensor(4+i), label=torch.tensor(float(i % 2))) for i in range(8)]
                val_data = None
            else:
                from recscale.datasets_timesplit_train_val_test.taac_temporal import TAAC2025TimeTemporalDataset
                train_data = TAAC2025TimeTemporalDataset(c, split='train')
                val_data = TAAC2025TimeTemporalDataset(c, split='val')
            model = DINModel(c)
        device = torch.device('cpu' if synthetic else 'cuda')
        model.to(device)
        owner = Trainer(model, train_data, val_data, c, device)
        table = model.encoder.sparse_arch.embeddings[c['dataset']['sparse_cols'].index('video_id')] if row['setting'] == 'kuairand' else None
    native = owner.optimizer
    if row['method'] == 'adamar':
        owner.optimizer = make_adamar(model, table.weight, native, row['alpha'])
    if row['method'] == 'tailshare':
        if synthetic:
            mask = np.zeros(64, dtype=bool); mask[4:8] = True
        else:
            counts_path = Path(paths['lookup_counts'])
            meta = read(counts_path.with_suffix('.json'))
            if meta['protocol_sha256'] != sha(helpers / 'PROTOCOL.json') or meta['counts_sha256'] != sha(counts_path):
                raise ValueError('Train-only lookup-count provenance mismatch; run preprocessing/count_lookups.py')
            if meta.get('validation_used_for_counts') is not False or meta.get('labels_read_for_counts') is not False:
                raise ValueError('Lookup counts must be training-only and label-free')
            counts = np.load(paths['lookup_counts'], mmap_mode='r', allow_pickle=False)
            if counts.shape != (table.num_embeddings,): raise ValueError('Counts/embedding cardinality mismatch')
            mask = (counts > 0) & (counts <= row['tau']); mask[:4] = False
        shared = TailSharedEmbedding(table, mask)
        if fuxi:
            model.embedding_layer.embedding_layers['target_item_id'] = shared
            model.embedding_layer.embedding_layers['item_seq'] = shared
        else:
            model.encoder.sparse_arch.embeddings[c['dataset']['sparse_cols'].index('video_id')] = shared
        owner.optimizer = renew_adam(model, native)
    if fuxi: model._optimizers = [owner.optimizer]
    if synthetic:
        if fuxi:
            batch = {'label': torch.tensor([0., 1., 0., 1.])}
            for name, field in model.feature_map.features.items():
                batch[name] = torch.tensor([[0, 4, 5, 6, 7, 8, 9, 10]]*4) if field['type'] == 'sequence' else torch.tensor([4,5,8,9])
        else: batch = next(iter(owner.train_loader))
        losses = [step(model, owner, batch, 'taac' if fuxi else 'kuairand') for _ in range(2)]
        if not all(np.isfinite(losses)): raise RuntimeError('Non-finite synthetic loss')
        return dict(mode='synthetic', datasets_loaded=[], losses=losses, parameters=sum(p.numel() for p in model.parameters()))
    if fuxi:
        from fuxictr.pytorch.dataloaders import RankDataLoader
        from fuxictr_ext.taac2025.idshare_taac_dataloader import TaacSharedIdentityDataLoader
        tg, vg = RankDataLoader(model.feature_map, stage='train', **dict(c, data_loader=TaacSharedIdentityDataLoader)).make_iterator()
        if (tg.num_samples, vg.num_samples) != (131222882, 21951746): raise ValueError('TAAC split sizes differ from paper')
        model.fit(tg, validation_data=vg, **c)
        auc = _best_validation_auc(model)
        history = model._history
    else:
        owner.train(); auc = owner.best_metrics['auc']; history = owner.epoch_history
    return dict(mode='formal', datasets_loaded=['train','validation'], best_validation_auc=float(auc), history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--experiment')
    parser.add_argument('--paths', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    catalog = read(ROOT / 'configs/experiments.json')
    if args.list:
        for row in catalog: print(row['id'])
        return
    matches = [r for r in catalog if r['id'] == args.experiment]
    if len(matches) != 1: parser.error('Choose exactly one --experiment from --list')
    if not args.output: parser.error('--output is required')
    if not args.synthetic and not args.paths: parser.error('--paths is required for real data')
    row = matches[0]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {} if args.synthetic else read(args.paths)[row['setting']]
    result = train(row, backend(row), paths, output, args.synthetic)
    (output / 'result.json').write_text(json.dumps(dict(experiment=row, **result), indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
