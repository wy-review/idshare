#!/usr/bin/env python3
"""Run isolated CPU smoke checks, with no real data and no remote services."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cells=[f'main_{d}_rankmixer_{m}_s42' for d in ['taac','kuairand']
           for m in ['continuous','idshare','adamar','tailshare','frozenhash']]
    cells += ['depth_taac_rankmixer_idshare_s42_d8','capacity_taac_rankmixer_idshare_s42_k80000',
              'additional_backbone_taac_din_continuous_s42_l2_0',
              'additional_backbone_taac_din_idshare_s42_l2_0',
              'additional_backbone_kuairand_tokenmixer_large_idshare_s42_l2_0',
              'l2_taac_rankmixer_idshare_s42_l2_1e-09',
              'l2_kuairand_rankmixer_continuous_s42_l2_1e-09']
    results=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for cell in cells:
        target=a.output.resolve()/cell
        process=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run.py'),'--experiment',cell,
                                '--synthetic','--output',str(target)],cwd=ROOT,env=env,capture_output=True,text=True)
        (a.output/(cell+'.log')).write_text(process.stdout+process.stderr)
        results.append(dict(experiment=cell,passed=process.returncode==0))
        print(cell, 'PASS' if process.returncode==0 else 'FAIL',flush=True)
        if process.returncode: print(process.stderr[-1800:],flush=True)
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))
    if not all(r['passed'] for r in results): raise SystemExit(1)


if __name__=='__main__': main()
