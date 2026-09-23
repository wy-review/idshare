# Data sources and preparation

Raw data are not redistributed. Obtain each release from its official owner
and follow its terms. Use **KuaiRand-27K**, not KuaiRand-1K or KuaiRand-Pure.

## Official links

| Dataset | Source |
|---|---|
| TAAC 2025 / TencentGR-10M | [Official dataset](https://huggingface.co/datasets/TAAC2025/TencentGR-10M) |
| TAAC preprocessing/baseline reference | [Competition repository](https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025) |
| TAAC competition | [Official competition](https://algo.qq.com/2025) |
| KuaiRand | [Project website](https://kuairand.com/) and [official repository](https://github.com/chongminggao/KuaiRand) |
| KuaiRand-27K archive | [Zenodo record](https://zenodo.org/records/10439422) and [27K archive](https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz) |

The official KuaiRand repository lists MD5
`3e3c799a24e2d23a4d2c757fbf9adf59` for `KuaiRand-27K.tar.gz`.
Check the archive before extraction. Check the TAAC dataset card for its release
layout and terms; the indexed schema consumed below is the one retained in the
original preprocessing scripts.

## TAAC

Input layout:

```text
/data/taac/raw/
  seq/*.parquet       # user_id and seq (item_id, action_type, timestamp)
  item_feat/
  user_feat/
  indexer.pkl
```

Only load `indexer.pkl` from a trusted official/local preprocessing source;
Python pickle can execute code. Input IDs and feature indices must agree with
this indexer. Do not replace it with a vocabulary fitted on validation data.

```bash
python -B preprocessing/taac/prepare.py split \
  --raw /data/taac/raw --output /data/taac/prepared
python -B preprocessing/taac/prepare.py metadata \
  --prepared /data/taac/prepared --feature-root /data/taac/features
```

The temporal split is May 23-28, 2025 for training and May 29 for validation,
using UTC+8 day boundaries. An example's history is the sequence prefix before
its target. The model truncates it to 100 items. Positive actions are 1 or 2;
samples require a preceding event and a nonmissing action.

Canonical sizes: **131,222,882 training samples; 21,951,746 validation samples**.
The shared item identity table has 17,487,680 rows including reserved rows.
Padding/OOV/missing/reserved are 0/1/2/3; real raw item indices use offset 3.
Target and historical items share one identity table. The train-seen mask has
SHA256 `28c508568071ff0f963f4bbe1b8d34c62a591e64716a1566169ca11383e58406`.
The public main training adapter rejects a different mask.

The metadata stage writes the feature map, sequence cache, and train-input seen
mask. Side fields 102, 115, 119, and 120 match the frozen TAAC backbone. The
original preprocessing scripts are retained alongside the public wrapper for
audit. Their historical full-split CLI is not the recommended entry point.

## KuaiRand-27K

Expected raw directory: the extracted `KuaiRand-27K/data`, with four
`log_standard*.csv` files, `user_features_27k.csv`, and
`video_features_basic_27k.csv`. Random-exposure logs are excluded from this
experiment. This setting uses 37 feature fields and no history sequences.

```bash
python -B preprocessing/kuairand/prepare.py \
  --raw /data/kuairand/KuaiRand-27K/data \
  --output /data/kuairand/processed
```

The frozen dates and counts are in
`preprocessing/kuairand/SHRED_ZA_P37_ROLLING_BACKTEST_SPLIT.json`.
The public wrapper uses the original static-feature encoders and train-only
vocabulary/numeric transforms, but emits only train and validation shards.
Rows outside those dates are skipped before accessing their labels. It does
not compute held-out metrics or build held-out interaction shards. It retains
the original deterministic in-shard and shard-order shuffle (data seed
20260724), so training must keep `shuffle=false`.

Canonical sizes: **243,480,365 training samples; 12,080,526 validation samples**.
The video identity table has **25,020,526 rows**. All private vocabulary rows,
frequency counts, category vocabularies, and numeric bins are fitted using
training support. Unseen identity values use the shared OOV row. Static
metadata may cover validation-only items; this does not create private
training identity rows for them.

The source `prepare_k1_nonseq_mmap.py` keeps its historical filename; the
wrapper explicitly supplies the **P37 rolling-backtest split**, not that
module's older default split. The wrapper imports feature definitions from
`audit_kuairand27k_nonseq.py`; unused historical preprocessing entry points
are omitted.

The public manifest has new path and artifact hashes. It must not claim to be
byte-identical to the original internal manifest. The training reader verifies
the hashes of the newly generated static tables, counts, and train/validation
shards. Full-size equivalence of this portable wrapper remains to be checked
on a clean Linux machine; the release checks use synthetic fixtures only.

## Lookup counts for TailShare and frequency analysis

Set `configs/paths.local.json` first, then run:

```bash
python -B preprocessing/count_lookups.py --setting taac \
  --paths configs/paths.local.json --output outputs/taac_counts
python -B preprocessing/count_lookups.py --setting kuairand \
  --paths configs/paths.local.json --output outputs/kuairand_counts
```

TAAC counts include target lookups and all occurrences in the truncated
training histories; they are not target-example counts or unique-ID counts.
KuaiRand counts video-ID training occurrences. These routines do not use
validation labels or frequencies. The `.npy` cache and adjacent `.json`
metadata are hash-bound to the baseline protocol. TailShare pools real IDs
with **1-5 training lookups** into one **trainable** embedding; it is not a
permanently zero embedding. Special and OOV rows are excluded from that pool.

## Resources and failure checks

Preprocessing uses substantial RAM and disk. The minimum disk check is not a
maximum storage estimate. Full TAAC loading historically used a large-memory
host; do not start multiple loaders on a workstation. Run one configuration
first and measure memory before planning concurrency.

A row-count, mask, vocabulary, or checksum mismatch should stop reproduction.
Do not use `--allow-small` on real data to bypass these checks. This flag exists
solely for synthetic unit tests.
