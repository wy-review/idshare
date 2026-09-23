# Independent diagnostic experiments

These experiments are separate from the main regularizer table. Source
snapshots are preserved even where a diagnostic is not in the current
manuscript. They are not automatically launched by the main experiment runner.

## Dependency order

1. Prepare TAAC, feature maps, train-seen mask, and train-only lookup counts.
2. Run `frequency` for Continuous and IDShare on seeds 2021, 42, 2024. Each
   rerun retains `selected_model.pt` and validation predictions. These are new
   diagnostic checkpoints, not the original main-table jobs' checkpoints.
3. Run `support` for each IDShare checkpoint. It exports `id_to_code.npy` and
   `shared_outputs.npy`, verifies exported outputs against the native model,
   and analyzes the final sharing groups.
4. Run `continuation` for each seed, explicitly binding the frequency artifacts
   and support exports. It compares shared and untied outputs from one identical
   checkpoint, with the rest of the model frozen.
5. `dense_trajectory` is a distinct protocol that trains an instrumented first
   phase and records denser continuation snapshots. It cannot reconstruct the
   missing pre-checkpoint trajectory from a final checkpoint alone.

## Synthetic checks

```bash
python -B scripts/diagnostic.py --kind frequency --synthetic --output outputs/frequency_check
python -B scripts/diagnostic.py --kind support --synthetic --output outputs/support_check
python -B scripts/diagnostic.py --kind continuation --synthetic --output outputs/continuation_check
python -B scripts/diagnostic.py --kind dense_trajectory --synthetic --output outputs/trajectory_check
```

The dense-trajectory evaluator is CUDA-specific. On a CPU-only environment its
check writes `status=skipped`, not a passing result. Other checks use small,
synthetic arrays. No real datasets are loaded.

## Bind local artifacts explicitly

The original protocol templates reside in:

```text
snapshots/frequency/taac/FREQUENCY_PROTOCOL.json
snapshots/support/taac/SUPPORT_PROTOCOL.json
snapshots/continuation/taac/CONTINUATION_PROTOCOL.json
snapshots/dense_trajectory/taac/CONTINUATION_PROTOCOL.json
```

Make a local protocol outside `snapshots/`. Retain the scientific variables,
seeds, data contract, readout times, and primary endpoint. Replace artifact
locations with the files produced by your new runs and their **actual hashes**.
Do not use historical checkpoint hashes for newly trained checkpoints. The
following fields matter:

| Stage | Artifact fields to bind |
|---|---|
| Frequency | `counts_path`, `counts_metadata`, `retention_root`; paths inside each execution contract |
| Support | Counts; per-seed Continuous/IDShare prediction/checkpoint records in `sources`; output root |
| Continuation | Counts; IDShare checkpoint/prediction records in `sources`; per-seed `support_exports`; output root |
| Dense trajectory | Local data/feature-map/count paths and output roots in its protocol and runtime bindings |

`--bindings` is an explicit JSON mapping from exact archived path prefixes to
your local paths. It relocates paths in a private copy of the runtime, YAMLs,
manifests, and protocols, not the source snapshot. For example:

```json
{
  "/data/taac2025/10m-time-split-train-val-test-20260531": "/data/taac/prepared"
}
```

Additional paths, such as the feature-map directory and retained artifact
root, must be mapped too. Inspect the chosen snapshot's YAML/JSON templates;
the example above is not a complete binding for every stage.

```bash
CUDA_VISIBLE_DEVICES=0 python -B scripts/diagnostic.py \
  --kind continuation --seed 42 \
  --protocol configs/local_continuation.json \
  --bindings configs/local_bindings.json \
  --output outputs/continuation_s42
```

The adapter stages a runtime under the new output directory and writes
`PUBLIC_RUNTIME_MANIFEST.json`. Its hash is distinct from the original remote
package attestation. Historical remote-manifest CLI checks are not reused as
proof for the public runtime. Data/checkpoint/initial-output consistency checks
inside the scientific routines remain active. The complete GPU artifact chain
has not been rerun during release assembly.

## Interpretation and readout

The original paired continuation freezes the backbone, routing assignments,
and other features. Only directly parameterized identity outputs are trained.
Both branches use identical batches and initial predictions, Adam lr 0.001,
batch size 2048, gradient clipping 1, and 4096 updates. Readouts are fixed at
0, 1024, and 4096, with 4096 as the primary endpoint. This is a local fixed-route
intervention, not a second epoch of the original IDShare algorithm.

Use `analysis/readouts.py` to check structural invariants before aggregating
effect metrics. It retains both AUC and logloss, all frequency buckets, and all
fixed times. The accepted original experiment has **opposite AUC and logloss
arm orderings**. Do not relabel the frozen primary logloss criterion after
seeing AUC, select the best readout time, or report only the favorable metric.

The support analysis measures final-group partner lookup coverage. It is not a
measurement of how many historical updates were actually transferred through
those groups; partner-count correlations alone do not establish causation.

`analysis/reproduce.py` recomputes the accepted three-seed continuation summary
offline. New diagnostic reports should remain separate from `reference/` and
from the original paper's main-table seeds.
