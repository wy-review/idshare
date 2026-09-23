#!/usr/bin/env python3
"""Build FuxiCTR feature_map.json + strict time-split indices for TAAC2025 10m-time-split.

Usage:
    # baseline (no item_seq side info)
    python build_taac2025_feature_map.py \
        --raw_root /data/taac2025/10m-time-split \
        --out_dir  /path/to/FuxiCTR/data/taac2025_time_split

    # side info: activate 2 ITEM_SPARSE fids as item_seq side sequences
    python build_taac2025_feature_map.py --side-fids 100,101 \
        --raw_root ... --out_dir .../taac2025_time_split_side_100_101

    # side info: all 13 ITEM_SPARSE fids
    python build_taac2025_feature_map.py --side-fids all \
        --raw_root ... --out_dir .../taac2025_time_split_side_all

--side-fids controls which item_seq__feat_<fid> sequence specs are written
into feature_map.json. MUST match model_config's `sequence_side_fields` — a
mismatch causes `self.dnn` input_dim != actual feature_emb dim at DIN forward.

Requires: pyarrow, numpy
"""
import argparse, json, os, pickle, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as pds

USER_SPARSE = [103, 104, 105, 109]
USER_ARRAY  = [106, 107, 108, 110]
ITEM_SPARSE = [100, 101, 102, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122]
MAXLEN              = 50
USER_ARRAY_MAXLEN   = 10

TRAIN_START_TS = 1747929600
VALID_START_TS = 1748448000
VALID_END_TS   = 1748534400

# action_type is an interaction-level feature (per step in user_seqs),
# NOT an item-level feature.  Its vocab is scanned from user_seqs at build
# time; typical values are small integers (e.g. 0..7).
ACTION_TYPE_FID = "action_type"


def build_feature_map(indexer, out_dir, side_fids, maxlen=MAXLEN,
                      action_type_vocab_size=0):
    features = []
    for fid in USER_SPARSE:
        features.append({f"feat_{fid}": {
            "type": "categorical", "source": "",
            "padding_idx": 0, "vocab_size": len(indexer["f"][str(fid)]) + 1,
        }})
    for fid in USER_ARRAY:
        features.append({f"feat_{fid}": {
            "type": "sequence", "source": "",
            "padding_idx": 0, "vocab_size": len(indexer["f"][str(fid)]) + 1,
            "max_len": USER_ARRAY_MAXLEN,
        }})
    for fid in ITEM_SPARSE:
        features.append({f"feat_{fid}": {
            "type": "categorical", "source": "",
            "padding_idx": 0, "vocab_size": len(indexer["f"][str(fid)]) + 1,
        }})
    features.append({"target_item_id": {
        "type": "categorical", "source": "",
        "padding_idx": 0, "vocab_size": len(indexer["i"]) + 1,
    }})
    features.append({"item_seq": {
        "type": "sequence", "source": "",
        "share_embedding": "target_item_id",
        "padding_idx": 0, "vocab_size": len(indexer["i"]) + 1,
        "max_len": maxlen,
    }})
    # item_seq side info: one sequence spec per activated ITEM_SPARSE fid,
    # sharing vocab with target-side feat_<fid>. The set MUST match the
    # model_config's `sequence_side_fields`; extra specs here would inflate
    # sum_emb_out_dim and break DIN.forward (mat1/mat2 shape mismatch).
    for fid in side_fids:
        if fid not in ITEM_SPARSE:
            raise ValueError(f"side fid {fid} is not in ITEM_SPARSE {ITEM_SPARSE}")
        features.append({f"item_seq__feat_{fid}": {
            "type": "sequence", "source": "",
            "share_embedding": f"feat_{fid}",
            "padding_idx": 0, "vocab_size": len(indexer["f"][str(fid)]) + 1,
            "max_len": maxlen,
        }})
    # action_type side info: interaction-level (per step in user_seqs),
    # NOT from item_feat.  No target-side feat_action_type is added —
    # action_type is only meaningful in the history sequence, not for the
    # target item, so it must NOT appear as a non-seq feature that would
    # pollute the model's NS tokens.  item_seq__action_type gets its own
    # embedding (no share_embedding).
    if action_type_vocab_size > 0:
        features.append({f"item_seq__{ACTION_TYPE_FID}": {
            "type": "sequence", "source": "",
            "padding_idx": 0, "vocab_size": action_type_vocab_size,
            "max_len": maxlen,
        }})
    # input_length: categorical=1, sequence=max_len
    input_length = 0
    total_features = 0
    for f in features:
        name, spec = next(iter(f.items()))
        input_length += spec.get("max_len", 1)
        # total_features: 词表累加, share_embedding 不重复计
        if "share_embedding" not in spec:
            total_features += spec["vocab_size"]
    fm = {
        "dataset_id": out_dir.name,
        "num_fields": len(features),
        "total_features": total_features,
        "input_length": input_length,
        "labels": ["label"],
        "features": features,
    }
    with open(out_dir / "feature_map.json", "w") as f:
        json.dump(fm, f, indent=4)
    print(f"[feature_map] fields={fm['num_fields']} total_features={total_features} input_length={input_length}")


def load_user_ts_dict(user_seqs_dir):
    """Return {uid: np.ndarray[int64] of timestamps}."""
    print(f"[user_seqs] scanning {user_seqs_dir} ...", flush=True)
    ds = pds.dataset(str(user_seqs_dir), format="parquet")
    tbl = ds.to_table(columns=["user_id", "seq"])
    print(f"[user_seqs] loaded {tbl.num_rows} users, building ts_dict ...", flush=True)
    ts_dict = {}
    uids = tbl.column("user_id").to_pylist()
    seqs = tbl.column("seq").to_pylist()
    for uid, seq in zip(uids, seqs):
        ts_dict[uid] = np.fromiter(
            (x["timestamp"] for x in seq), dtype=np.int64, count=len(seq))
    print(f"[user_seqs] ts_dict size={len(ts_dict)}", flush=True)
    return ts_dict


def scan_action_type_vocab(user_seqs_dir):
    """Scan user_seqs to find all distinct action_type values and return vocab_size."""
    import pyarrow.dataset as pds_inner
    print(f"[action_type] scanning {user_seqs_dir} for action_type values ...", flush=True)
    ds = pds_inner.dataset(str(user_seqs_dir), format="parquet")
    tbl = ds.to_table(columns=["seq"])
    vals = set()
    n_entries = 0
    for seq in tbl.column("seq").to_pylist():
        for entry in seq:
            at = entry.get(ACTION_TYPE_FID)
            if at is not None:
                vals.add(int(at))
                n_entries += 1
    if not vals:
        raise ValueError(f"no '{ACTION_TYPE_FID}' found in user_seqs entries; "
                         f"checked {n_entries} entries across {tbl.num_rows} users")
    max_val = max(vals)
    vocab_size = max_val + 2  # +1 for padding_idx=0, +1 because max_val itself is valid
    print(f"[action_type] distinct values={sorted(vals)} vocab_size={vocab_size} "
          f"(scanned {n_entries} entries)", flush=True)
    return vocab_size


def build_time_split(raw_root, out_dir, ts_dict):
    samples_path = raw_root / "train" / "samples.parquet"
    print(f"[samples] reading {samples_path} ...", flush=True)
    t = pq.read_table(str(samples_path)).to_pydict()
    uid_arr    = np.asarray(t["user_id"],        dtype=np.int64)
    cutoff_arr = np.asarray(t["seq_cutoff_pos"], dtype=np.int32)
    N = len(uid_arr)
    print(f"[samples] train rows={N}", flush=True)

    ts_per_row = np.full(N, -1, dtype=np.int64)
    missing = 0
    # 主循环：用纯 Python,10M 量级可接受;若需加速再上 numba
    for i in range(N):
        arr = ts_dict.get(int(uid_arr[i]))
        pos = int(cutoff_arr[i])
        if arr is None or pos < 0 or pos >= len(arr):
            missing += 1
            continue
        ts_per_row[i] = arr[pos]
    print(f"[split] missing ts: {missing}/{N}", flush=True)

    train_mask = (ts_per_row >= TRAIN_START_TS) & (ts_per_row < VALID_START_TS)
    valid_mask = (ts_per_row >= VALID_START_TS) & (ts_per_row < VALID_END_TS)
    train_idx = np.where(train_mask)[0].astype(np.int64)
    valid_idx = np.where(valid_mask)[0].astype(np.int64)
    out_of_range = N - missing - len(train_idx) - len(valid_idx)
    print(f"[split] train={len(train_idx)} valid={len(valid_idx)} out_of_range={out_of_range}", flush=True)

    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "valid_idx.npy", valid_idx)
    return samples_path


def write_manifest(out_dir, split, samples_path, raw_root, idx_path):
    meta = {
        "samples_path": str(samples_path),
        "raw_data_root": str(raw_root),
        "split": split,
        "index_array": str(idx_path) if idx_path else None,
    }
    with open(out_dir / f"{split}_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)


def parse_side_fids(s):
    """Accept '', 'all', or comma-separated list. Returns a sorted list of ints."""
    s = (s or "").strip()
    if not s:
        return []
    if s.lower() == "all":
        return list(ITEM_SPARSE)
    fids = [int(x.strip()) for x in s.split(",") if x.strip()]
    return sorted(set(fids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--maxlen", type=int, default=MAXLEN,
                    help="max sequence length for item_seq (default: {})".format(MAXLEN))
    ap.add_argument("--side-fids", default="",
                    help="comma-separated ITEM_SPARSE fids activated as item_seq side info "
                         "(e.g. '100,101'); or 'all' for full set; empty = none.")
    ap.add_argument("--side-action-type", action="store_true",
                    help="add action_type as item_seq side info (interaction-level, "
                         "not item-level; scanned from user_seqs)")
    ap.add_argument("--skip_time_split", action="store_true",
                    help="only generate feature_map.json (for dry-run)")
    ap.add_argument("--combine-train-valid", action="store_true",
                    help=                         "merge valid_idx into train_idx so training uses train+valid; "
                         "valid_manifest is skipped (set valid_data='' in dataset_config.yaml)")
    args = ap.parse_args()

    side_fids = parse_side_fids(args.side_fids)
    side_action_type = args.side_action_type
    maxlen = args.maxlen
    raw_root = Path(args.raw_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] side_fids={side_fids} side_action_type={side_action_type} "
          f"maxlen={maxlen} out_dir={out_dir}")

    # 1) feature_map
    print(f"[indexer] loading {raw_root/'indexer.pkl'} ...", flush=True)
    with open(raw_root / "indexer.pkl", "rb") as f:
        indexer = pickle.load(f)
    action_type_vocab_size = 0
    if side_action_type:
        action_type_vocab_size = scan_action_type_vocab(raw_root / "user_seqs")
    build_feature_map(indexer, out_dir, side_fids, maxlen=maxlen,
                      action_type_vocab_size=action_type_vocab_size)

    if args.skip_time_split:
        print("[done] skipped time split")
        return

    # 2) time split
    ts_dict = load_user_ts_dict(raw_root / "user_seqs")
    train_samples = build_time_split(raw_root, out_dir, ts_dict)
    del ts_dict  # free memory

    # 2b) optionally merge train+valid
    if args.combine_train_valid:
        train_idx = np.load(out_dir / "train_idx.npy")
        valid_idx = np.load(out_dir / "valid_idx.npy")
        train_idx = np.sort(np.concatenate([train_idx, valid_idx]))
        np.save(out_dir / "train_idx.npy", train_idx)
        os.remove(out_dir / "valid_idx.npy")  # merged, no longer needed
        print(f"[combine] merged train+valid: train={len(train_idx):,} (was {len(train_idx)-len(valid_idx):,}+{len(valid_idx):,})")

    # 3) manifests (train/valid share the same samples file)
    write_manifest(out_dir, "train", train_samples, raw_root, out_dir / "train_idx.npy")
    if not args.combine_train_valid:
        write_manifest(out_dir, "valid", train_samples, raw_root, out_dir / "valid_idx.npy")
    # else: skip valid_manifest; set valid_data="" in dataset_config.yaml
    write_manifest(out_dir, "test",  raw_root / "test" / "samples.parquet", raw_root, None)
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
