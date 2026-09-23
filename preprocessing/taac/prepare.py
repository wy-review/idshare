#!/usr/bin/env python3
"""Build train/validation TAAC inputs; never materialize a held-out split.

The original preprocessing scripts are retained alongside this portable wrapper.
Use only the official, trusted indexer.pkl; pickle is not a safe interchange format.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]


def split_sequence(sequence):
    # Same chronological sample rule as the frozen time split, restricted to May 29.
    start, val_start, end = 1747929600, 1748448000, 1748534400
    seq = [x for x in (sequence or []) if x.get('timestamp', 0) < end]
    samples = {'train': [], 'val': []}
    if len(seq) < 2: return seq, samples
    for pos, item in enumerate(seq):
        ts, action = item.get('timestamp', 0), item.get('action_type')
        if pos == 0 or not start <= ts < end or action is None: continue
        split = 'train' if ts < val_start else 'val'
        samples[split].append((item['item_id'], int(action in (1, 2)), pos))
    return seq, samples


def prepare(raw, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import prepare_time_split as original
    out.mkdir(parents=True, exist_ok=False)
    (out / 'user_seqs').mkdir()
    for name in ['item_feat', 'user_feat', 'indexer.pkl']:
        if not (raw / name).exists(): raise FileNotFoundError(raw / name)
        (out / name).symlink_to((raw / name).resolve())
    writers, counts = {}, {'train': 0, 'val': 0}
    try:
        for split in counts:
            (out / split).mkdir()
            writers[split] = pq.ParquetWriter(out / split / 'samples.parquet', original.SAMPLES_SCHEMA)
        index = 0
        for file in sorted((raw / 'seq').glob('*.parquet')):
            if not file.stat().st_size: continue
            for batch in pq.ParquetFile(file).iter_batches(batch_size=20000, columns=['user_id','seq']):
                users, sequences = [], []
                rows = {s: [] for s in counts}
                payload = batch.to_pydict()
                for uid, sequence in zip(payload['user_id'], payload['seq']):
                    seq, selected = split_sequence(sequence)
                    if not any(selected.values()): continue
                    users.append(uid); sequences.append(seq)
                    for split, samples in selected.items():
                        rows[split].extend((uid, *s) for s in samples)
                if users:
                    pq.write_table(pa.table(dict(user_id=users, seq=sequences), schema=original.USER_SEQ_SCHEMA),
                                   out / 'user_seqs' / f'part-{index:05d}.parquet')
                    index += 1
                for split, data in rows.items():
                    if not data: continue
                    cols = dict(zip(original.SAMPLES_SCHEMA.names, zip(*data)))
                    writers[split].write_table(pa.table(cols, schema=original.SAMPLES_SCHEMA))
                    counts[split] += len(data)
    finally:
        for writer in writers.values(): writer.close()
    (out / 'split_counts.json').write_text(json.dumps(counts, indent=2))
    return counts


def metadata(prepared, feature_root, allow_small=False):
    import numpy as np
    import build_taac2025_feature_map as fm
    import build_train_seen_mask as seen
    import pyarrow.parquet as pq
    sys.path.insert(0, str(ROOT / 'snapshots/baselines/taac'))
    from fuxictr_ext.taac2025.taac_dataloader import _load_user_seqs
    with (prepared / 'indexer.pkl').open('rb') as f: indexer = pickle.load(f)
    destination = feature_root / 'taac2025_time_split_side_top4'
    destination.mkdir(parents=True, exist_ok=False)
    fm.build_feature_map(indexer, destination, [102,115,119,120], maxlen=100)
    seqs = _load_user_seqs(prepared / 'user_seqs')
    cache = prepared / 'user_seqs_numpy'
    cache.mkdir(exist_ok=False)
    with (cache / 'user_seqs.pkl').open('wb') as f: pickle.dump(seqs, f, protocol=4)
    counts = {s: pq.ParquetFile(prepared / s / 'samples.parquet').metadata.num_rows for s in ['train','val']}
    if not allow_small and counts != {'train':131222882, 'val':21951746}:
        raise ValueError(f'Dataset version or split mismatch: {counts}')
    p = dict(protocol=seen.PROTOCOL, allowed_splits=['train','val'], uses_test_dataset=False,
             uses_test_labels=False, overwrite_existing_cache=False, real_id_offset=3,
             special_token_contract=dict(padding=0,oov=1,missing=2,reserved=3),
             data_root=str(prepared), expected_rows=counts,
             cache_output=str(prepared / 'north_star_identity_cache/taac_train_input_seen_mask_r01.npy'))
    report = seen.run_build(p)
    (prepared / 'seen_mask_report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    split = sub.add_parser('split'); split.add_argument('--raw', type=Path, required=True); split.add_argument('--output', type=Path, required=True)
    meta = sub.add_parser('metadata'); meta.add_argument('--prepared', type=Path, required=True); meta.add_argument('--feature-root', type=Path, required=True)
    meta.add_argument('--allow-small', action='store_true', help='Synthetic fixtures only; not paper reproduction')
    args = parser.parse_args()
    result = prepare(args.raw.resolve(), args.output.resolve()) if args.stage == 'split' else metadata(args.prepared.resolve(), args.feature_root.resolve(), args.allow_small)
    print(json.dumps(result))
