# Experiment map

## Included experiment families

| Family | Code/configuration | Evidence or output |
|---|---|---|
| Five-method main comparison | `main_*` catalog cells; `snapshots/baselines`, `adamar_extension`, `frozenhash` | `reference/generated_regularization_baselines.json` |
| Global L2 response and transfer | `l2_*` cells; `snapshots/baselines` | `reference/taac_main.json`, `kuairand_main.json` |
| DIN and TokenMixer-Large | `additional_backbone_*`; `snapshots/legacy` | `reference/taac_din.json`, `kuairand_tokenmixer.json` |
| AdamAR development grid | `adamar_development_*`; `snapshots/adamar_extension` | Selected coefficients in the main table audit |
| Codebook sizes | `capacity_*`; `snapshots/capacity` | `reference/generated_taac_robustness.json` |
| Backbone depths | `depth_*`; `snapshots/depth` | `reference/generated_taac_robustness.json` |
| Frequency-bucket diagnostic reruns | `scripts/diagnostic.py --kind frequency` | `reference/frequency.json` |
| Read-only sharing-group support analysis | `--kind support` | `reference/support.json` |
| Fixed-route shared versus untied continuation | `--kind continuation` | `reference/continuation.json` |
| Dense pre/post-checkpoint trajectories | `--kind dense_trajectory` | Code and frozen protocol; not treated as accepted paper evidence here |

KuaiRand capacity and depth configurations are retained as separate diagnostic
coverage. The paper's main-backbone capacity/depth result is the TAAC study;
do not label the KuaiRand configuration list as a paper claim or a completed
result table. Older abandoned exploratory variants are not a prescribed
reproduction target.

`scripts/make_catalog.py` generates 270 analysis cells, corresponding to 238
distinct configurations after explicit anchor reuse. These are configuration
counts, not the number of historical submitted jobs. The 168 L2-grid cells
comprise 42 TAAC main + 42 KuaiRand main + 42 DIN + 42 TokenMixer-Large cells.

## Matched main settings

| Setting | TAAC | KuaiRand-27K |
|---|---|---|
| Backbone implementation | `IDShareUnifiedMixer` | `S2DRecModel` with `rankmixer_v2` |
| Backbone family | RankMixer-based, with history processing | Dense RankMixer-based, feature fields only |
| Layers / width | 2 / 64 | 2 / 74 |
| Identity dimension | 16 | 16 |
| IDShare K | 20,000 | 40,000 |
| Seeds | 2021, 42, 2024 | 42, 2024, 933888 |
| Batch size / learning rate | 2048 / 0.001 | 4096 / 0.002 |
| Optimizer / gradient clip | Adam / 1.0 | Adam / 1.0 |
| Schedule | One epoch | Up to 10 epochs; original early stopping and selection |

KuaiRand retains minimum 4 epochs, patience 3, and minimum improvement 0.0001.
Training order uses data seed 20260724 with physically shuffled shards and no
additional DataLoader shuffle. Keep `optimizer_foreach=false`. The exact
resolved configuration written by `scripts/run.py` is the executable record.

The additional TAAC backbone is DIN (attention dimension 32, MLP 128/64).
The additional KuaiRand backbone is the two-layer dense TokenMixer-Large
implementation with 37 tokens, width 74, and feed-forward width 256. Its
per-token SwiGLU differs from the main KuaiRand RankMixer implementation's
GELU feed-forward network. Both KuaiRand backbones use seeds 42, 2024, and
933888 in the public catalog, matching the paper. The unchanged historical
`reference/kuairand_tokenmixer.json` also retains two earlier seeds (641812 and
514571); its five-seed aggregates are not the current paper comparison.

## Methods

- **Continuous:** one independent real-ID embedding, with the frozen special-row
  handling. The main regularizer table uses zero global L2.
- **IDShare:** hard selected shared output, soft straight-through task gradient,
  codebook fitting weight 1, commitment weight 0. Private routing states remain
  trainable but are not sent directly to the backbone. Initial routing and
  zero/OOV behavior are preserved from the corresponding frozen source.
- **FrozenHash:** fixed deterministic ID-to-code assignments at the same K;
  shared embeddings remain trainable. Keep its dedicated initialization and
  backward implementation rather than replacing it with a new hash baseline.
- **TailShare:** pool real IDs with 1-5 training lookups into one trainable
  embedding. Other real IDs keep private embeddings. This requires train-only
  lookup counts and does not mean emitting a permanently zero vector.
- **AdamAR:** the experiment's adaptive-regularization implementation with
  per-ID lazy-update clocks, retained in `adamar_frozen_compat.py`. This is the
  implementation used for the comparison, not a claim to redistribute an
  official author repository. The main selected coefficients are 1e-6 on TAAC
  and 0.1 on KuaiRand. Development uses seed 42; other seeds reuse the selected
  coefficient. Original four-point grids and two-point extensions are listed
  separately through the catalog and frozen protocols.

Core quantization code is in
`snapshots/*/*/recscale/models/zero_anchor_identity_quantizer.py`.
TAAC's shared target/history adapter is in
`model_zoo/UnifiedBackbone/src/idshare_unified_mixer.py` within its snapshots.
TailShare and AdamAR adapters are in `strong_common.py` and
`adamar_frozen_compat.py`.

## Global L2

The seven measured coefficients are
`0, 1e-11, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8`.
The penalty acts on all real-ID rows: Continuous's output table or IDShare's
private routing table, not the shared codebook. The implementation adds the
coupled L2 gradient **after global task-gradient clipping and before Adam**.
Replacing it with optimizer `weight_decay`, AdamW, or minibatch-row-only L2
would change the experiment.

## K and depth

The catalog retains K values 10,000 / 20,000 / 40,000 / 80,000 and depths 2 / 4 / 8.
Each family varies one factor at a time; it is not their Cartesian product.
The two-layer/default-K cells reuse main-comparison anchors. Report all measured
values; do not select a best K or depth and replace the baseline.

## Result conventions

Report equal-weight seed means and sample SD (ddof=1). Paired differences use
the same seed on both methods. Store raw AUC on [0,1]; display AUC percentage
as `100*AUC`, and percentage-point difference as `100*(AUC_A-AUC_B)`.
Compute differences before rounding. Never pool examples across seeds to form
the reported seed mean.

Checkpoint/coefficient selection uses the original validation protocol. All
training entry points in this release load train and validation only. Independent
frequency reruns are not additional seeds for the main comparison. Bucket AUC
does not decompose overall AUC, which includes cross-bucket pairs.

## Version and provenance policy

`provenance/SOURCES.json` retains original and public file hashes and archive
hashes. Private source-tree locations are omitted. Path sanitation and the
removal of unused registrations can change public source bytes without changing
the experiment's mathematics. Original package hashes remain historical
identifiers; do not rewrite them to claim an identical new run.

The portable adapter selects a versioned runtime and applies the catalog's
method/L2/K/depth settings. Unused parent-package copies are omitted. Source
hashes and synthetic checks do not replace full-data numerical reproduction.
