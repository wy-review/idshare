#!/usr/bin/env python3
"""Check provenance, source syntax, and accidental private artifact inclusion."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
RAW_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.pkl', '.parquet', '.csv', '.log', '.pyc'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    records = json.loads((ROOT / 'provenance/SOURCES.json').read_text())
    problems = []
    for row in records:
        path = ROOT / row['path']
        if not path.is_file() or sha(path) != row['public_sha256']:
            problems.append((row['path'], 'source hash mismatch'))
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and '.git' not in p.relative_to(ROOT).parts)
    for path in files:
        name = str(path.relative_to(ROOT))
        if path.is_symlink() or path.suffix in RAW_SUFFIXES or path.name.startswith('.env'):
            problems.append((name, 'private/generated artifact or symlink'))
            continue
        data = path.read_text()
        # Print only the file name/reason, never the matched credential.
        if re.search(r'rtp_[A-Za-z0-9_-]{24,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY', data):
            problems.append((name, 'credential-like material'))
        if re.search(r'/Users/[A-Za-z0-9_.-]+|/share_[0-9]{5,}', data) or ('/' + 'apdcephfs') in data:
            problems.append((name, 'private machine path'))
        if path.suffix == '.py':
            ast.parse(data, filename=name)
    if problems:
        print(json.dumps({'status': 'failed', 'problems': problems}, indent=2))
        raise SystemExit(1)
    return files, dict(status='passed', traced_sources=len(records), files=len(files),
                      bytes=sum(p.stat().st_size for p in files), raw_artifacts_included=False,
                      full_data_retraining_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    files, result = audit()
    if args.write_manifest:
        lines = [f'{sha(p)}  {p.relative_to(ROOT)}' for p in files if p.name != 'SHA256SUMS']
        (ROOT / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))
