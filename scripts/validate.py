#!/usr/bin/env python3
"""Run the public data-free release checks and write a portable test summary."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory outside this release')
    parser.add_argument('--report', type=Path, help='Optional portable summary JSON, without local log paths')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Validation logs must be outside the distributable directory')
    output.mkdir(parents=True, exist_ok=False)
    checks = [('source_audit', ['scripts/audit_release.py']),
              ('unit_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v']),
              ('smoke', ['scripts/check_smoke.py', '--output', str(output / 'smoke')]),
              ('reference', ['analysis/reproduce.py', '--output', str(output / 'reference'), '--figures'])]
    for kind in ['frequency', 'support', 'continuation', 'dense_trajectory']:
        checks.append((kind, ['scripts/diagnostic.py', '--kind', kind, '--synthetic',
                              '--output', str(output / kind)]))
    results = []
    for name, command in checks:
        process = subprocess.run([sys.executable, '-B', *command], cwd=ROOT,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                 capture_output=True, text=True)
        (output / (name + '.log')).write_text(process.stdout + process.stderr)
        status = 'passed' if process.returncode == 0 else 'failed'
        synthetic_report = output / name / 'synthetic_check.json'
        if status == 'passed' and synthetic_report.exists():
            if json.loads(synthetic_report.read_text()).get('status') == 'skipped':
                status = 'skipped'
        results.append(dict(check=name, status=status, returncode=process.returncode))
        print(name, status, flush=True)
    versions = {k: importlib.metadata.version(k) for k in
                ['torch','numpy','pandas','pyarrow','polars','scipy','scikit-learn','matplotlib','PyYAML','h5py']}
    state = 'failed' if any(x['status'] == 'failed' for x in results) else (
        'passed_with_declared_skips' if any(x['status'] == 'skipped' for x in results) else 'passed')
    summary = dict(status=state,
                   environment=dict(python=platform.python_version(), system=platform.system(), packages=versions),
                   checks=results, datasets_loaded=[], full_data_retraining_verified=False,
                   full_preprocessing_verified=False, rtp_submissions=0, upload_performed=False)
    smoke = output / 'smoke/summary.json'
    if smoke.exists():
        rows = json.loads(smoke.read_text())
        summary['smoke_cases'] = rows
        summary['smoke_passed'] = sum(x['passed'] for x in rows)
    for target in [output / 'validation.json', *([args.report] if args.report else [])]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + '\n')
    if summary['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
