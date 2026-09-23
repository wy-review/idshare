# IDShare

Anonymous implementation of IDShare.

`PAPER_VERSION.json` records the manuscript hashes associated with this release.

**IDShare: Learned Sharing as Regularization for Long-Tailed ID Embeddings**

IDShare learns which IDs share an embedding, providing structural regularization
for long-tailed ID features. This repository contains training, data processing,
experiment configurations, and result analysis for TAAC and KuaiRand-27K.

## Setup

Use Python 3.9-3.11. Full experiments require Linux and an NVIDIA CUDA GPU;
install the appropriate PyTorch build for your CUDA driver.

```bash
python -m pip install -r requirements.txt
```

## Data and Training

Follow [Data Preparation](docs/DATA.md) for official download links and processing
commands. Raw data are not bundled. Set your paths in `configs/paths.local.json`
using `configs/paths.example.json`, then run one configuration:

```bash
python scripts/run.py --list
CUDA_VISIBLE_DEVICES=0 python -B scripts/run.py \
  --experiment main_taac_rankmixer_idshare_s42 \
  --paths configs/paths.local.json --output outputs/taac_idshare_s42
```

[Experiments](docs/EXPERIMENTS.md) describes the five methods, L2 grids, additional
backbones, codebook sizes, and depths. [Diagnostics](docs/DIAGNOSTICS.md) covers
frequency and sharing analyses. Output directories must be new. Catalog entries
marked `reuse` refer to existing configurations, not additional replicates.

## Checks and Analysis

```bash
python -B scripts/validate.py --output /tmp/idshare-validation
python -B analysis/reproduce.py --output outputs/reference_analysis --figures
```

The data-free checks cover preprocessing fixtures, CPU model updates, source
integrity, and accepted-result recomputation. The dense-trajectory check requires
CUDA and is skipped on CPU. Full-data retraining has not been revalidated in this
portable release. Analysis plots reproduce the quantities, not the paper layout.

## Code Layout

- `scripts/`: training, diagnostic, and validation entry points.
- `preprocessing/`: dataset preparation and training-lookup counts.
- `configs/`: experiment catalog and local-path example.
- `snapshots/`: versioned implementations used by the entry points. Versions are
  kept separate to preserve the behavior of the original experiments.
- `analysis/`, `reference/`: result readers and accepted numerical records.
- `tests/`, `provenance/`: synthetic tests and source hashes.

## License

Original IDShare contributions are released under [Apache-2.0](LICENSE).
Third-party code retains its original licenses and copyright notices; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md). Data, weights, predictions,
credentials, and remote schedulers are not included.
