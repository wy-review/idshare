"""Fixed train-frequency buckets and validation-only artifact helpers."""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

BUCKETS = ['0', '1', '2-5', '6-10', '11-50', '51-100', '101-500', '501+', 'special']
DTYPE = np.dtype([('target_id', '<i8'), ('lookup_count', '<i8'), ('bucket', 'u1'),
                  ('label', 'u1'), ('prediction', '<f4')])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def milestone(name, **values):
    print(json.dumps({'milestone': name, **values}), flush=True)


def classify(raw_ids, counts):
    raw = np.asarray(raw_ids, dtype=np.int64)
    assert raw.ndim == 1
    frequency = np.zeros(raw.shape, dtype=np.int64)
    known = (raw > 0) & (raw < len(counts) - 3)
    frequency[known] = counts[raw[known] + 3]
    assert (frequency >= 0).all()
    buckets = np.searchsorted([0, 1, 5, 10, 50, 100, 500], frequency, side='left').astype(np.uint8)
    buckets[raw <= 0] = 8
    return frequency, buckets


def metrics(labels, predictions):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(labels, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    assert y.shape == pred.shape and y.ndim == 1
    assert np.isin(y, [0, 1]).all() and np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()
    n, pos = len(y), int(y.sum())
    p = np.clip(pred, 1e-15, 1 - 1e-15)
    return {'n': n, 'positive': pos, 'negative': n - pos,
            'auc': float(roc_auc_score(y, pred)) if 0 < pos < n else None,
            'logloss': float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()) if n else None}


def summarize_predictions(path):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    assert values.dtype == DTYPE
    groups = {}
    for index, name in enumerate(BUCKETS):
        use = values['bucket'] == index
        groups[name] = metrics(values['label'][use], values['prediction'][use])
    return {'overall': metrics(values['label'], values['prediction']), 'buckets': groups}


def install_guard():
    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            for candidate in (os.fsdecode(path), os.path.realpath(path)):
                p = Path(candidate)
                blocked = any(part.lower() in {'test', 'holdout'} for part in p.parts)
                if p.suffix.lower() in {'.npy', '.npz', '.parquet', '.pkl', '.csv', '.tsv'}:
                    blocked |= any(p.name.lower().startswith(k + sep) for k in ('test', 'holdout') for sep in ('_', '.'))
                if blocked:
                    raise RuntimeError('forbidden dataset path')
        elif isinstance(path, (list, tuple)):
            for item in path:
                check(item)

    def audit(event, args):
        if event == 'open' and args:
            check(args[0])

    sys.addaudithook(audit)
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    def wrap(fn):
        def guarded(path, *args, **kwargs):
            check(path)
            return fn(path, *args, **kwargs)
        return guarded

    ds.dataset = wrap(ds.dataset)
    pq.ParquetFile = wrap(pq.ParquetFile)
    pq.read_table = wrap(pq.read_table)
    np.load = wrap(np.load)
