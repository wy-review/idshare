#!/usr/bin/env python3
"""Portable KuaiRand-27K train/validation preparation using frozen encoders.

Dates are inspected to select rows; labels outside train/validation are not used.
The historical full preprocessing pipeline remains available for provenance.
"""
from pathlib import Path
from datetime import date
import argparse
import json
import shutil
import sqlite3

import audit_kuairand27k_nonseq as audit
import prepare_k1_nonseq_mmap as original

SPEC = Path(__file__).with_name('SHRED_ZA_P37_ROLLING_BACKTEST_SPLIT.json')


def write_interactions(logs, users, videos, out, expected, shard_rows, allow_small):
    writers = {s: original.InteractionShardWriter(out, s, shard_rows=shard_rows, data_seed=20260724)
               for s in ['train', 'val']}
    days = {day: split for split in writers for day in expected[f'{split}_days']}
    for path in logs:
        for row in original._read_csv_rows(path):
            split = days.get(original.day_from_time_ms(row['time_ms']))
            if split is None:
                continue
            label = int(row['is_click'])
            if label not in [0, 1]:
                raise ValueError('Non-binary label')
            writers[split].add(original._find_row(users, int(row['user_id']), 'user_id'),
                               original._find_row(videos, int(row['video_id']), 'video_id'), label)
    result = {s: writer.finalize() for s, writer in writers.items()}
    if not allow_small:
        for split, record in result.items():
            if (record['rows'], record['positives']) != (expected[f'{split}_rows'], expected[f'{split}_positives']):
                raise ValueError(f'{split} does not match the frozen split')
    return result


def prepare(raw, out, *, min_free_gb=100, shard_rows=5000000, allow_small=False):
    expected = original.load_expected_split(str(SPEC))
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if shutil.disk_usage(out).free < min_free_gb * 1024**3:
        raise RuntimeError('Insufficient free disk for preprocessing')
    logs = sorted(raw.glob('log_standard*.csv'))
    if not logs or (not allow_small and len(logs) != 4):
        raise ValueError('Expected the four standard KuaiRand-27K log files')
    user_file, video_file = raw / 'user_features_27k.csv', raw / 'video_features_basic_27k.csv'
    db = out / 'train_aggregate.sqlite'
    connection = sqlite3.connect(db)
    try:
        audit._configure_database(connection)
        audit._load_user_features(connection, [user_file])
        audit._load_video_basic(connection, [video_file])
        train_days = [(date.fromisoformat(d) - date(1970, 1, 1)).days for d in expected['train_days']]
        aggregate = audit._accumulate_interactions(connection, logs,
                     {'train': train_days, 'test': []}, 1000000)
        if not allow_small and aggregate['train_test_rows'] != expected['train_rows']:
            raise ValueError('Train aggregate row count differs from the frozen split')
        users, user_cards, user_meta = original._build_user_table(connection, user_file, out / 'user_sparse.npy')
        videos, video_cards, video_meta, counts = original._build_video_table(
            connection, video_file, out / 'video_sparse.npy', out / 'video_eval_meta.npy', out / '.video_numeric.tmp.npy')
        cards = user_cards + video_cards
        if len(cards) != 37:
            raise ValueError('Expected 37 fields')
        if not allow_small and cards[list(original.FIELD_NAMES).index('video_id')] != 25020526:
            raise ValueError('Training vocabulary differs from the frozen identity table')
        cache = original._write_frequency_cache(out / 'identity_frequency_cache.npz', counts,
                       dict(zip(original.FIELD_NAMES, cards)), split_sha256=expected['split_sha256'])
    finally:
        connection.close()
    splits = write_interactions(logs, users, videos, out, expected, shard_rows, allow_small)
    manifest = dict(format='kuairand27k_k1_mmap_v1', field_names=list(original.FIELD_NAMES),
                    cardinalities=cards, splits=splits,
                    protocol=dict(task='pointwise_nonseq_is_click', train_only_vocab=True,
                                  random_logs_used=False, sequence_used=False, data_seed=20260724,
                                  split_sha256=expected['split_sha256'], test_statistics_used_for_training=False,
                                  public_train_validation_only=True),
                    tables={name: original.array_artifact(out / (name + '.npy'), out)
                            for name in ['user_sparse','video_sparse','video_eval_meta']},
                    auxiliary_artifacts=[cache],
                    invariants=dict(user=user_meta, video=video_meta, special_token_ids=dict(original.SPECIAL_ID)),
                    source_manifest=[dict(path=p.name, bytes=p.stat().st_size, sha256=original.sha256_file(p))
                                     for p in [user_file, video_file, *logs]])
    path = out / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-free-gb', type=float, default=100)
    parser.add_argument('--allow-small', action='store_true', help='Synthetic fixtures only')
    args = parser.parse_args()
    print(prepare(args.raw.resolve(), args.output.resolve(), min_free_gb=args.min_free_gb,
                  allow_small=args.allow_small))
