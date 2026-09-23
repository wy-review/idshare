"""No-data CUDA check of the frozen optimizer at newly authorized alphas."""

import argparse
import json
from pathlib import Path
import torch
from strong_runtime import verify
from adamar_frozen_compat import install


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    protocol = verify(args.manifest_sha)
    install()
    from recscale.utils.adamar import RowwiseAdamAR
    assert torch.cuda.is_available()
    checks = []
    for alpha in protocol['new_alphas']:
        torch.manual_seed(42)
        w = torch.nn.Parameter(torch.randn(64, 16, device='cuda') * .01)
        opt = RowwiseAdamAR([{'params': [w], 'adamar': True, 'first_regularized_row': 4}],
                           lr=.001, alpha=alpha, eps_placement='pytorch_outside_sqrt')
        for step in range(64):
            opt.zero_grad()
            w.grad = torch.zeros_like(w)
            w.grad[4 + step % 60] = .1
            opt.step()
            assert torch.isfinite(w).all()
        assert opt.state[w]['last_valid_step'].dtype == torch.int64
        checks.append({'alpha': alpha, 'steps': 64, 'finite': True})
    report = {'status': 'passed', 'mode': 'extension_check', 'setting': protocol['setting'],
              'source_manifest_sha256': args.manifest_sha, 'datasets_loaded': [],
              'auc_computed': False, 'cuda_verified': True, 'checks': checks,
              'uses_test_dataset': False, 'uses_test_labels': False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
