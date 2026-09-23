#!/usr/bin/env python3
"""
prepare_dataset_v3.py — TAAC2025 10M 按日期切分 + 滑动窗口

核心设计:
  - 测试 label: 5/30 的 item
  - 训练 label: 最近 TRAIN_DAYS 天 (默认 7 天, 5/23~5/29) 的 item
  - 每用户存一份完整序列, 训练/测试通过 seq_cutoff_pos 动态截断
  - 序列输入: seq[:pos][-maxlen:] 固定长度滑动窗口

输入: TencentGR-10M/seq/ (原始序列)
输出: 10m-time-split/
        ├── user_seqs/              每用户一行 (user_id, seq)
        │   ├── part-00000.parquet
        │   └── ...
        ├── train/samples.parquet   每行: user_id, target_item_id, label, seq_cutoff_pos
        ├── test/samples.parquet    每行: user_id, target_item_id, label, seq_cutoff_pos
        ├── indexer.pkl  → symlink
        ├── item_feat/   → symlink
        ├── user_feat/   → symlink
        └── mm_emb/      → symlink
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
SRC_DIR = Path(os.environ.get(
    "SRC_DATA_PATH",
    "/data/taac2025/TencentGR-10M",
))
DST_DIR = Path(os.environ.get(
    "DST_PATH",
    "/data/taac2025/10m-time-split",
))

# 截断: 5/31 及之后的 item 全部去掉 (等待区间, action_type 全为 0)
CUTOFF_TIMESTAMP = 1748620800  # 2025-05-31 00:00:00 CST

# 测试日期: 5/30
# 训练日期: 5/23~5/29 (TRAIN_DAYS=7)
TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", "7"))

# CST = UTC+8
CST = timezone(timedelta(hours=8))

# 5/30 00:00:00 CST
TEST_DATE_START = int(datetime(2025, 5, 30, 0, 0, 0, tzinfo=CST).timestamp())
# 5/31 00:00:00 CST = CUTOFF_TIMESTAMP
TEST_DATE_END = CUTOFF_TIMESTAMP
# 训练 label 开始时间
TRAIN_DATE_START = int(datetime(2025, 5, 30 - TRAIN_DAYS, 0, 0, 0, tzinfo=CST).timestamp())
TRAIN_DATE_END = TEST_DATE_START

PARQUET_ROWS_PER_FILE = 200_000

# ============================================================
# Schema
# ============================================================
SEQ_INNER_TYPE = pa.struct([
    pa.field("item_id", pa.int64()),
    pa.field("action_type", pa.int32()),
    pa.field("timestamp", pa.int64()),
])
USER_SEQ_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("seq", pa.list_(SEQ_INNER_TYPE)),
])
SAMPLES_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("target_item_id", pa.int64()),
    pa.field("label", pa.int32()),
    pa.field("seq_cutoff_pos", pa.int32()),
])


def process_sequences():
    """
    读取原始序列, 截断 5/31+, 按日期划分训练/测试 sample.
    每个用户存一份完整序列 (截止到 CUTOFF_TIMESTAMP 之前).
    """
    seq_dir = SRC_DIR / "seq"
    print(f"Source: {seq_dir}")
    print(f"Cutoff: {datetime.fromtimestamp(CUTOFF_TIMESTAMP, tz=CST)}")
    print(f"Test date: 5/30 (ts {TEST_DATE_START} ~ {TEST_DATE_END})")
    print(f"Train label: last {TRAIN_DAYS} days (ts {TRAIN_DATE_START} ~ {TRAIN_DATE_END})")

    seq_files = sorted(seq_dir.glob("*.parquet"))
    seq_files = [f for f in seq_files if f.stat().st_size > 0]
    print(f"Found {len(seq_files)} parquet files")

    # 收集结果
    all_user_seqs = []       # (user_id, seq_list)
    train_samples = []       # (user_id, target_item_id, label, seq_cutoff_pos)
    test_samples = []

    stats = defaultdict(int)

    for fi, fpath in enumerate(seq_files):
        table = pq.read_table(fpath, columns=["user_id", "seq"])
        user_ids = table.column("user_id").to_pylist()
        seqs = table.column("seq").to_pylist()
        stats['total_users'] += len(user_ids)

        for uid, seq in zip(user_ids, seqs):
            if seq is None or len(seq) < 2:
                stats['skipped_short'] += 1
                continue

            # 截掉 5/31 及之后
            truncated = [item for item in seq if item.get('timestamp', 0) < CUTOFF_TIMESTAMP]
            if len(truncated) < 2:
                stats['skipped_after_cutoff'] += 1
                continue

            # 遍历每个 item, 按日期分类
            user_train_items = []  # (index_in_seq, item)
            user_test_items = []

            for i, item in enumerate(truncated):
                ts = item.get('timestamp', 0)
                if TRAIN_DATE_START <= ts < TRAIN_DATE_END:
                    # 训练 label: 需要该 item 之前至少有 1 个 item 作为序列
                    if i >= 1:
                        user_train_items.append((i, item))
                elif TEST_DATE_START <= ts < TEST_DATE_END:
                    if i >= 1:
                        user_test_items.append((i, item))

            if not user_train_items and not user_test_items:
                stats['no_label_items'] += 1
                continue

            # 存完整序列
            all_user_seqs.append((uid, truncated))
            stats['valid_users'] += 1

            # 生成 samples
            for idx_in_seq, item in user_train_items:
                act = item.get('action_type')
                if act is None:
                    continue
                label = 1 if act in (1, 2) else 0
                train_samples.append((uid, item['item_id'], label, idx_in_seq))
                stats['train_samples'] += 1
                if label == 1:
                    stats['train_pos'] += 1

            for idx_in_seq, item in user_test_items:
                act = item.get('action_type')
                if act is None:
                    continue
                label = 1 if act in (1, 2) else 0
                test_samples.append((uid, item['item_id'], label, idx_in_seq))
                stats['test_samples'] += 1
                if label == 1:
                    stats['test_pos'] += 1

        print(f"  File {fi+1}/{len(seq_files)}: {stats['total_users']:,} users, "
              f"{stats['valid_users']:,} valid, "
              f"{stats['train_samples']:,} train, {stats['test_samples']:,} test")

    # Print stats
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"  Total users: {stats['total_users']:,}")
    print(f"  Skipped (short): {stats['skipped_short']:,}")
    print(f"  Skipped (after cutoff): {stats['skipped_after_cutoff']:,}")
    print(f"  No label items: {stats['no_label_items']:,}")
    print(f"  Valid users (with seq): {stats['valid_users']:,}")
    print(f"  Train samples: {stats['train_samples']:,} "
          f"(pos={stats['train_pos']:,}, rate={stats['train_pos']/max(stats['train_samples'],1)*100:.2f}%)")
    print(f"  Test samples: {stats['test_samples']:,} "
          f"(pos={stats['test_pos']:,}, rate={stats['test_pos']/max(stats['test_samples'],1)*100:.2f}%)")
    print(f"{'='*60}\n")

    return all_user_seqs, train_samples, test_samples


def write_user_seqs(user_seqs, out_dir):
    """写出 user_seqs/ 目录, 每个文件 PARQUET_ROWS_PER_FILE 行"""
    seq_dir = out_dir / "user_seqs"
    seq_dir.mkdir(parents=True, exist_ok=True)

    part_idx = 0
    buf_uids, buf_seqs = [], []

    def _flush():
        nonlocal part_idx, buf_uids, buf_seqs
        if not buf_uids:
            return
        tbl = pa.table({"user_id": buf_uids, "seq": buf_seqs}, schema=USER_SEQ_SCHEMA)
        pq.write_table(tbl, seq_dir / f"part-{part_idx:05d}.parquet")
        part_idx += 1
        buf_uids, buf_seqs = [], []

    for uid, seq in tqdm(user_seqs, desc="Writing user_seqs"):
        buf_uids.append(uid)
        buf_seqs.append(seq)
        if len(buf_uids) >= PARQUET_ROWS_PER_FILE:
            _flush()
    _flush()

    print(f"  Wrote {len(user_seqs):,} user sequences to {seq_dir} ({part_idx} files)")


def write_samples(samples, out_path):
    """写出 samples.parquet"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    uids = [s[0] for s in samples]
    items = [s[1] for s in samples]
    labels = [s[2] for s in samples]
    positions = [s[3] for s in samples]

    tbl = pa.table({
        "user_id": pa.array(uids, type=pa.int64()),
        "target_item_id": pa.array(items, type=pa.int64()),
        "label": pa.array(labels, type=pa.int32()),
        "seq_cutoff_pos": pa.array(positions, type=pa.int32()),
    }, schema=SAMPLES_SCHEMA)
    pq.write_table(tbl, out_path)

    n_pos = sum(labels)
    print(f"  Wrote {len(samples):,} samples to {out_path} "
          f"(pos={n_pos:,}, rate={n_pos/max(len(samples),1)*100:.2f}%)")


def create_symlinks():
    """创建软链接到原始数据"""
    DST_DIR.mkdir(parents=True, exist_ok=True)
    links = {
        "indexer.pkl": SRC_DIR / "indexer.pkl",
        "item_feat": SRC_DIR / "item_feat",
        "user_feat": SRC_DIR / "user_feat",
        "mm_emb": SRC_DIR / "mm_emb",
    }
    for name, target in links.items():
        link_path = DST_DIR / name
        if link_path.exists() or link_path.is_symlink():
            continue
        os.symlink(str(target), str(link_path))
        print(f"  Symlink: {link_path} -> {target}")


def main():
    t0 = time.time()
    print(f"{'='*60}")
    print(f"TAAC2025-10M V3: 按日期切分 + 滑动窗口")
    print(f"  Source: {SRC_DIR}")
    print(f"  Output: {DST_DIR}")
    print(f"  Train days: {TRAIN_DAYS}")
    print(f"{'='*60}\n")

    # Symlinks
    print("Creating symlinks...")
    create_symlinks()
    print()

    # Process
    user_seqs, train_samples, test_samples = process_sequences()

    # Write
    print("Writing user sequences...")
    write_user_seqs(user_seqs, DST_DIR)
    print()

    print("Writing train samples...")
    write_samples(train_samples, DST_DIR / "train" / "samples.parquet")
    print()

    print("Writing test samples...")
    write_samples(test_samples, DST_DIR / "test" / "samples.parquet")
    print()

    elapsed = time.time() - t0
    print(f"{'='*60}")
    print(f"Done! Time: {elapsed:.1f}s")
    print(f"  Users with seq: {len(user_seqs):,}")
    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Test samples: {len(test_samples):,}")
    print(f"  Output: {DST_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
