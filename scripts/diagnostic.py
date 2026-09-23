#!/usr/bin/env python3
"""Run one diagnostic from an explicitly bound, portable protocol.

Source snapshots stay read-only. Relocated files and new provenance are written
under --output/runtime. Original remote attestation is not asserted for new runs.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
PROTOCOLS={'frequency':'FREQUENCY_PROTOCOL.json','support':'SUPPORT_PROTOCOL.json',
           'continuation':'CONTINUATION_PROTOCOL.json','dense_trajectory':'CONTINUATION_PROTOCOL.json'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=list(PROTOCOLS),required=True)
    parser.add_argument('--protocol',type=Path,help='Locally rebound protocol; required except for --synthetic')
    parser.add_argument('--bindings',type=Path,help='JSON mapping of exact archived path prefixes to local paths')
    parser.add_argument('--seed',type=int,choices=[2021,42,2024],default=42)
    parser.add_argument('--carrier',choices=['continuous','idshare'],default='idshare')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args()
    if not args.synthetic and not args.protocol: parser.error('--protocol is required for real artifacts')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'snapshots'/args.kind/'taac'
    runtime=output/'runtime'
    shutil.copytree(source,runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    replacements={} if not args.bindings else json.loads(args.bindings.read_text())
    for path in runtime.rglob('*'):
        if path.suffix not in {'.py','.json','.yaml','.yml'}: continue
        text=path.read_text()
        for old,new in sorted(replacements.items(),key=lambda x:-len(x[0])): text=text.replace(old,new)
        path.write_text(text)
    protocol_path=runtime/PROTOCOLS[args.kind]
    if args.protocol: shutil.copyfile(args.protocol,protocol_path)
    p=json.loads(protocol_path.read_text())
    manifest={str(f.relative_to(runtime)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(runtime.rglob('*')) if f.is_file()}
    manifest_path=output/'PUBLIC_RUNTIME_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    binding=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    os.chdir(runtime)
    sys.path.insert(0,str(runtime));sys.path.insert(0,str(runtime/'model_zoo/UnifiedBackbone'))
    from frequency_common import install_guard
    install_guard()
    import torch
    torch.set_num_threads(2)
    if not args.synthetic and not torch.cuda.is_available(): raise RuntimeError('CUDA required for real diagnostics')
    if args.synthetic and args.kind == 'dense_trajectory' and not torch.cuda.is_available():
        result = dict(status='skipped', reason='Frozen dense-trajectory synthetic evaluator requires CUDA',
                      datasets_loaded=[])
        (output / 'synthetic_check.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    module=importlib.import_module({'frequency':'frequency_runtime','support':'support_runtime',
                                  'continuation':'continuation_runtime','dense_trajectory':'continuation_runtime'}[args.kind])
    if args.kind == 'support': module.guard()
    if args.synthetic:
        if args.kind=='support': module.check(p,binding,output,'cpu')
        elif args.kind=='continuation':
            (output/'synthetic_check.json').write_text(json.dumps(module.synthetic_check('cpu'),indent=2))
        elif args.kind=='frequency': module.regression(p,output,binding,'cpu')
        else:
            import frequency_runtime
            result=module.synthetic_evaluation(frequency_runtime.bridge_module())
            (output/'synthetic_check.json').write_text(json.dumps(result,indent=2))
    elif args.kind=='support': module.run(p,args.seed,binding,output)
    elif args.kind=='continuation': module.run(p,args.seed,'paired',binding,output)
    elif args.kind=='frequency': module.formal(p,f'{args.carrier}_s{args.seed}',output,binding)
    else: module.formal(p,args.seed,output,binding)
    print('Diagnostic finished. Public runtime hashes are separate from historical attestations.')


if __name__=='__main__': main()
