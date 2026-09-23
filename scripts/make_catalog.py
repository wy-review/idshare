#!/usr/bin/env python3
"""Generate explicit measured configurations, without launching any jobs."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
L2 = [0., 1e-11, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8]
SEEDS = {'taac': [2021, 42, 2024], 'kuairand': [42, 2024, 933888]}


def build_catalog():
    rows = []
    def add(setting, method, seed, *, family='main', backbone='rankmixer',
            coefficient=0., k=None, depth=2, alpha=None, stage=None):
        k = k or (20000 if setting == 'taac' else 40000)
        name = f'{family}_{setting}_{backbone}_{method}_s{seed}'
        if family == 'l2' or family == 'additional_backbone':
            name += f'_l2_{coefficient:g}'
        if family == 'capacity': name += f'_k{k}'
        if family == 'depth': name += f'_d{depth}'
        if family == 'adamar_development': name += f'_alpha_{alpha:g}'
        rows.append(dict(id=name, setting=setting, method=method, seed=seed,
                         family=family, backbone=backbone, global_l2=coefficient,
                         k=k, depth=depth, alpha=alpha, tau=5 if method == 'tailshare' else None,
                         stage=stage, best_k_selected=False, best_depth_selected=False))
    for setting, seeds in SEEDS.items():
        for seed in seeds:
            for method in ['continuous', 'idshare', 'adamar', 'tailshare', 'frozenhash']:
                add(setting, method, seed, alpha=(1e-6 if setting == 'taac' else .1) if method == 'adamar' else None)
            for method in ['continuous', 'idshare']:
                for dose in L2: add(setting, method, seed, family='l2', coefficient=dose)
                for depth in [2, 4, 8]: add(setting, method, seed, family='depth', depth=depth)
            for k in [10000, 20000, 40000, 80000]: add(setting, 'idshare', seed, family='capacity', k=k)
        for alpha in [10**-6.5, 10**-4.5, 10**-2.5, .1] + ([1e-8, 1e-6] if setting == 'taac' else [10**-.5, 10**-.05]):
            add(setting, 'adamar', 42, family='adamar_development', alpha=alpha)
        for seed in seeds:
            for method in ['continuous', 'idshare']:
                for dose in L2:
                    add(setting, method, seed, family='additional_backbone',
                        backbone='din' if setting == 'taac' else 'tokenmixer_large', coefficient=dose)
    # Reuse is explicit: the catalog is a set of analysis views, not a job queue.
    seen = {}
    for row in rows:
        identity = tuple(row[k] for k in ['setting', 'method', 'seed', 'backbone', 'global_l2', 'k', 'depth', 'alpha'])
        row['reuse'] = seen.get(identity)
        seen.setdefault(identity, row['id'])
    return rows


if __name__ == '__main__':
    rows = build_catalog()
    destination = ROOT / 'configs/experiments.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2) + '\n')
    print(f'{len(rows)} analysis cells; {sum(r["reuse"] is None for r in rows)} distinct configurations')
