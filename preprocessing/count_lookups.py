#!/usr/bin/env python3
"""Build train-only lookup counts using the frozen baseline counting routines."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run import backend, kuai_config, read, sha, taac_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=['taac', 'kuairand'], required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New metadata work directory')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = read(args.paths)[args.setting]
    row = next(r for r in read(ROOT / 'configs/experiments.json')
               if r['setting'] == args.setting and r['method'] == 'continuous'
               and r['family'] == 'main' and r['seed'] == 42)
    source = backend(row)
    sys.path.insert(0, str(source))
    from strong_common import install_guard, build_taac_counts
    install_guard()
    p = read(source / 'PROTOCOL.json')
    p['cache_path'] = str(Path(paths['lookup_counts']).resolve())
    if args.setting == 'taac':
        c = taac_config(row, source, paths, output, False)
        result = build_taac_counts(c, p['cache_path'], sha(source / 'PROTOCOL.json'))
    else:
        from strong_runtime import kuai_counts
        from recscale.datasets.kuairand27k_k1 import KuaiRand27KK1MMapDataset
        c = kuai_config(row, source, paths, output, False)
        c['dataset']['verify_processed_sha256_splits'] = ['train']
        dataset = KuaiRand27KK1MMapDataset(c, split='train')
        result = kuai_counts(dataset, p)
    (output / 'counts_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
