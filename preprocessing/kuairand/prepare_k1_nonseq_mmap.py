#!/usr/bin/env python3
"""Build the frozen train-only KuaiRand-27K K1 mmap dataset.

This is a CPU-only preprocessing job. It reads the K0-P3 SQLite database in
immutable mode, reads only standard interaction logs and static feature files,
and writes static user/video tables plus compact interaction shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


UTC8_MS = 8 * 60 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000
SPECIAL_ID = {"padding": 0, "oov": 1, "missing": 2, "reserved_zero": 3}
GENERAL_SPECIAL_ID = {"padding": 0, "oov": 1, "missing": 2}
IDENTITY_FIELDS = ("video_id", "author_id", "music_id")
USER_PROFILE_FIELDS = (
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    *(f"onehot_feat{i}" for i in range(18)),
)
VIDEO_CATEGORY_FIELDS = (
    "video_type",
    "upload_type",
    "visible_status",
    "music_type",
)
VIDEO_NUMERIC_FIELDS = ("video_duration", "server_width", "server_height")
USER_FIELD_NAMES = ("user_id", *USER_PROFILE_FIELDS)
VIDEO_FIELD_NAMES = (
    "video_id",
    "author_id",
    "music_id",
    *VIDEO_CATEGORY_FIELDS,
    *(f"{name}_bucket" for name in VIDEO_NUMERIC_FIELDS),
)
FIELD_NAMES = (*USER_FIELD_NAMES, *VIDEO_FIELD_NAMES)
INTERACTION_DTYPE = np.dtype(
    [("user_index", "<u4"), ("video_index", "<u4"), ("label", "u1")]
)
EXPECTED = {
    "all_rows": 322_278_385,
    "train_rows": 289_580_429,
    "val_rows": 10_550_484,
    "test_rows": 22_147_472,
    "train_positives": 109_786_197,
    "val_positives": 3_880_240,
    "test_positives": 8_386_105,
    "excluded_rows": 0,
    "excluded_positives": 0,
    "split_sha256": "e48a7ece7ee7426945cd291f584ded152380be3f8429a7ca886472092981ee48",
    "train_days": [
        "2022-04-07",
        "2022-04-08",
        "2022-04-09",
        "2022-04-10",
        "2022-04-11",
        "2022-04-12",
        "2022-04-13",
        "2022-04-14",
        "2022-04-15",
        "2022-04-16",
        "2022-04-17",
        "2022-04-18",
        "2022-04-19",
        "2022-04-20",
        "2022-04-21",
        "2022-04-22",
        "2022-04-23",
        "2022-04-24",
        "2022-04-25",
        "2022-04-26",
        "2022-04-27",
        "2022-04-28",
        "2022-04-29",
        "2022-04-30",
        "2022-05-01",
        "2022-05-02",
        "2022-05-03",
        "2022-05-04",
        "2022-05-05",
    ],
    "val_days": ["2022-05-06"],
    "test_days": ["2022-05-07", "2022-05-08"],
    "excluded_days": [],
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def day_from_time_ms(value: str | int) -> str:
    day_index = (int(value) + UTC8_MS) // DAY_MS
    return time.strftime("%Y-%m-%d", time.gmtime(day_index * 86400))


def split_for_day(day: str, expected: dict | None = None) -> str | None:
    expected = EXPECTED if expected is None else expected
    if day in expected["train_days"]:
        return "train"
    if day in expected["val_days"]:
        return "val"
    if day in expected["test_days"]:
        return "test"
    return None


def load_expected_split(path: str | None) -> dict:
    if path is None:
        return dict(EXPECTED)
    payload = json.loads(Path(path).read_text())
    required = {
        "all_rows",
        "train_rows",
        "val_rows",
        "test_rows",
        "train_positives",
        "val_positives",
        "test_positives",
        "excluded_rows",
        "excluded_positives",
        "split_sha256",
        "train_days",
        "val_days",
        "test_days",
        "excluded_days",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"split spec missing keys: {missing}")
    split_definition = {
        "train": payload["train_days"],
        "val": payload["val_days"],
        "test": payload["test_days"],
        "excluded": payload["excluded_days"],
    }
    observed_sha = canonical_sha256(split_definition)
    if observed_sha != payload["split_sha256"]:
        raise ValueError(
            "split spec SHA256 mismatch: "
            f"expected={payload['split_sha256']} observed={observed_sha}"
        )
    all_days = [
        *payload["train_days"],
        *payload["val_days"],
        *payload["test_days"],
        *payload["excluded_days"],
    ]
    if len(all_days) != len(set(all_days)):
        raise ValueError("split spec day lists overlap")
    return payload


def open_sqlite_immutable(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    result = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise RuntimeError(f"source SQLite integrity_check failed: {result}")
    return connection


def sqlite_rows(
    connection: sqlite3.Connection,
    query: str,
    parameters: Sequence = (),
    fetch_size: int = 100_000,
) -> Iterator[list[tuple]]:
    cursor = connection.execute(query, tuple(parameters))
    while rows := cursor.fetchmany(fetch_size):
        yield rows


def load_integer_count_pairs(
    connection: sqlite3.Connection, field: str
) -> tuple[np.ndarray, np.ndarray]:
    if field == "video_id":
        query = """
            SELECT video_id, train_count
            FROM video_agg
            WHERE train_count>0 AND video_id!=0
            ORDER BY video_id
        """
    else:
        query = f"""
            SELECT CAST(b."{field}" AS INTEGER) AS raw_id,
                   SUM(v.train_count) AS train_count
            FROM video_basic b
            JOIN video_agg v ON v.video_id=b.video_id
            WHERE v.train_count>0
              AND b."{field}" IS NOT NULL
              AND TRIM(b."{field}")!=''
              AND CAST(b."{field}" AS INTEGER)!=0
            GROUP BY raw_id
            ORDER BY raw_id
        """
    raw_parts, count_parts = [], []
    for rows in sqlite_rows(connection, query):
        raw_parts.append(np.fromiter((int(row[0]) for row in rows), dtype=np.int64))
        count_parts.append(np.fromiter((int(row[1]) for row in rows), dtype=np.int64))
    raw = np.concatenate(raw_parts) if raw_parts else np.empty(0, dtype=np.int64)
    counts = (
        np.concatenate(count_parts) if count_parts else np.empty(0, dtype=np.int64)
    )
    if raw.size and (np.any(np.diff(raw) <= 0) or np.any(counts <= 0)):
        raise RuntimeError(f"invalid sorted train counts for {field}")
    return raw, counts


def encode_identity_values(
    values: np.ndarray,
    train_raw_ids: np.ndarray,
    *,
    missing_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    missing_mask = (
        np.zeros(values.shape, dtype=bool)
        if missing_mask is None
        else np.asarray(missing_mask, dtype=bool)
    )
    positions = np.searchsorted(train_raw_ids, values)
    in_range = positions < train_raw_ids.size
    found = np.zeros(values.shape, dtype=bool)
    found[in_range] = train_raw_ids[positions[in_range]] == values[in_range]
    encoded = np.full(values.shape, SPECIAL_ID["oov"], dtype=np.int32)
    encoded[found] = positions[found].astype(np.int32) + len(SPECIAL_ID)
    encoded[values == 0] = SPECIAL_ID["reserved_zero"]
    encoded[missing_mask] = SPECIAL_ID["missing"]
    return encoded


def build_category_vocab(values: Iterable[str]) -> dict[str, int]:
    unique = sorted({str(value) for value in values if str(value).strip() != ""})
    return {
        value: index + len(GENERAL_SPECIAL_ID)
        for index, value in enumerate(unique)
    }


def encode_category(value: str | None, vocab: dict[str, int]) -> int:
    if value is None or str(value).strip() == "":
        return GENERAL_SPECIAL_ID["missing"]
    return vocab.get(str(value), GENERAL_SPECIAL_ID["oov"])


def video_band(counts: np.ndarray) -> np.ndarray:
    result = np.zeros(counts.shape, dtype=np.uint8)
    result[(counts >= 1) & (counts <= 5)] = 1
    result[(counts >= 6) & (counts <= 10)] = 2
    result[(counts >= 11) & (counts <= 20)] = 3
    result[(counts >= 21) & (counts <= 50)] = 4
    result[counts > 50] = 5
    return result


def array_artifact(path: Path, root: Path) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": str(path.relative_to(root)),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class PeakDiskTracker:
    def __init__(self, root: Path):
        self.root = root
        self.peak_output_bytes = 0
        self.min_free_bytes = shutil.disk_usage(root).free

    def sample(self) -> None:
        output_bytes = sum(
            path.stat().st_size for path in self.root.rglob("*") if path.is_file()
        )
        self.peak_output_bytes = max(self.peak_output_bytes, output_bytes)
        self.min_free_bytes = min(self.min_free_bytes, shutil.disk_usage(self.root).free)


class InteractionShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        *,
        shard_rows: int,
        data_seed: int,
    ):
        self.output_dir = output_dir
        self.split = split
        self.shard_rows = int(shard_rows)
        self.data_seed = int(data_seed)
        self.buffer = np.empty(self.shard_rows, dtype=INTERACTION_DTYPE)
        self.buffer_size = 0
        self.shards: list[dict] = []
        self.rows = 0
        self.positives = 0

    def add(self, user_index: int, video_index: int, label: int) -> None:
        self.buffer[self.buffer_size] = (user_index, video_index, label)
        self.buffer_size += 1
        self.rows += 1
        self.positives += int(label)
        if self.buffer_size == self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if self.buffer_size == 0:
            return
        shard_index = len(self.shards)
        data = self.buffer[: self.buffer_size].copy()
        if self.split == "train":
            rng = np.random.default_rng(self.data_seed + shard_index)
            data = data[rng.permutation(len(data))]
        path = self.output_dir / f"{self.split}-{shard_index:05d}.npy"
        np.save(path, data, allow_pickle=False)
        self.shards.append(
            {
                "path": str(path.name),
                "rows": int(len(data)),
                "positives": int(data["label"].sum()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        self.buffer_size = 0

    def finalize(self) -> dict:
        self.flush()
        if self.split == "train":
            rng = np.random.default_rng(self.data_seed)
            order = rng.permutation(len(self.shards)).tolist()
            self.shards = [self.shards[index] for index in order]
        if sum(int(shard["rows"]) for shard in self.shards) != self.rows:
            raise RuntimeError(f"{self.split} shard rows do not close")
        return {
            "rows": self.rows,
            "positives": self.positives,
            "fixed_order": self.split == "train",
            "shuffle_policy": (
                "deterministic_in_shard_and_shard_order"
                if self.split == "train"
                else "source_order"
            ),
            "shards": self.shards,
        }


def _load_all_video_ids_and_counts(
    connection: sqlite3.Connection,
) -> tuple[np.ndarray, np.ndarray]:
    # P3 intentionally aggregates train+test only. Build the static row universe
    # from video_basic so validation-only videos retain side metadata while their
    # train count remains zero and their video identity routes to OOV.
    size = int(connection.execute("SELECT COUNT(*) FROM video_basic").fetchone()[0])
    raw_ids = np.empty(size, dtype=np.int64)
    counts = np.empty(size, dtype=np.int64)
    offset = 0
    for rows in sqlite_rows(
        connection,
        """
        SELECT b.video_id, COALESCE(v.train_count, 0)
        FROM video_basic b
        LEFT JOIN video_agg v ON v.video_id=b.video_id
        ORDER BY b.video_id
        """,
    ):
        length = len(rows)
        raw_ids[offset : offset + length] = [int(row[0]) for row in rows]
        counts[offset : offset + length] = [int(row[1]) for row in rows]
        offset += length
    if offset != size or (size and np.any(np.diff(raw_ids) <= 0)):
        raise RuntimeError("video_basic IDs are not a complete sorted unique array")
    return raw_ids, counts


def _load_all_user_ids_and_counts(
    connection: sqlite3.Connection,
) -> tuple[np.ndarray, np.ndarray]:
    rows = connection.execute(
        "SELECT user_id, train_count FROM user_agg ORDER BY user_id"
    ).fetchall()
    raw_ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    counts = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    if raw_ids.size and np.any(np.diff(raw_ids) <= 0):
        raise RuntimeError("user_agg IDs are not sorted unique")
    return raw_ids, counts


def _read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def _build_user_table(
    connection: sqlite3.Connection,
    user_feature_path: Path,
    output_path: Path,
) -> tuple[np.ndarray, list[int], dict]:
    aggregate_user_ids, aggregate_user_counts = _load_all_user_ids_and_counts(
        connection
    )
    rows_by_id = {
        int(row["user_id"]): row for row in _read_csv_rows(user_feature_path)
    }
    feature_user_ids = np.asarray(sorted(rows_by_id), dtype=np.int64)
    all_user_ids = np.union1d(aggregate_user_ids, feature_user_ids)
    user_counts = np.zeros(all_user_ids.shape, dtype=np.int64)
    aggregate_positions = np.searchsorted(all_user_ids, aggregate_user_ids)
    if not np.array_equal(all_user_ids[aggregate_positions], aggregate_user_ids):
        raise RuntimeError("user feature/aggregate union lost an aggregate ID")
    user_counts[aggregate_positions] = aggregate_user_counts
    train_user_ids = all_user_ids[user_counts > 0]
    category_vocabs = {}
    train_set = set(train_user_ids.tolist())
    for field in USER_PROFILE_FIELDS:
        category_vocabs[field] = build_category_vocab(
            row.get(field, "")
            for user_id, row in rows_by_id.items()
            if user_id in train_set
        )

    table = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(all_user_ids), len(USER_FIELD_NAMES)),
    )
    train_positions = np.searchsorted(train_user_ids, all_user_ids)
    found = np.zeros(all_user_ids.shape, dtype=bool)
    in_range = train_positions < len(train_user_ids)
    found[in_range] = train_user_ids[train_positions[in_range]] == all_user_ids[in_range]
    table[:, 0] = GENERAL_SPECIAL_ID["oov"]
    table[found, 0] = train_positions[found] + len(GENERAL_SPECIAL_ID)
    for row_index, user_id in enumerate(all_user_ids.tolist()):
        row = rows_by_id.get(user_id)
        for field_offset, field in enumerate(USER_PROFILE_FIELDS, start=1):
            table[row_index, field_offset] = encode_category(
                None if row is None else row.get(field),
                category_vocabs[field],
            )
    table.flush()
    cardinalities = [
        len(train_user_ids) + len(GENERAL_SPECIAL_ID),
        *[
            len(category_vocabs[field]) + len(GENERAL_SPECIAL_ID)
            for field in USER_PROFILE_FIELDS
        ],
    ]
    metadata = {
        "all_user_rows": int(len(all_user_ids)),
        "static_universe": "user_features_union_p3_train_test_aggregate",
        "feature_user_rows": int(len(feature_user_ids)),
        "p3_aggregate_user_rows": int(len(aggregate_user_ids)),
        "feature_only_user_rows": int(
            len(np.setdiff1d(feature_user_ids, aggregate_user_ids, assume_unique=True))
        ),
        "train_user_private_rows": int(len(train_user_ids)),
        "test_only_ids_have_private_rows": False,
        "category_vocab_sizes": {
            field: len(vocab) for field, vocab in category_vocabs.items()
        },
    }
    return all_user_ids, cardinalities, metadata


def _train_category_vocabs(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    vocabs = {}
    for field in VIDEO_CATEGORY_FIELDS:
        rows = connection.execute(
            f"""
            SELECT DISTINCT b."{field}"
            FROM video_basic b
            JOIN video_agg v ON v.video_id=b.video_id
            WHERE v.train_count>0
              AND b."{field}" IS NOT NULL
              AND TRIM(b."{field}")!=''
            ORDER BY b."{field}"
            """
        ).fetchall()
        vocabs[field] = build_category_vocab(row[0] for row in rows)
    return vocabs


def _build_numeric_memmap(
    video_feature_path: Path,
    all_video_ids: np.ndarray,
    output_path: Path,
) -> np.memmap:
    numeric = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(all_video_ids), len(VIDEO_NUMERIC_FIELDS)),
    )
    numeric[:] = np.nan
    for row in _read_csv_rows(video_feature_path):
        video_id = int(float(row["video_id"]))
        position = int(np.searchsorted(all_video_ids, video_id))
        if position >= len(all_video_ids) or all_video_ids[position] != video_id:
            continue
        for field_index, field in enumerate(VIDEO_NUMERIC_FIELDS):
            value = row.get(field, "")
            if value not in ("", None):
                numeric[position, field_index] = float(value)
    numeric.flush()
    return numeric


def _quantile_edges(
    numeric: np.ndarray,
    train_mask: np.ndarray,
    *,
    buckets: int = 100,
) -> list[np.ndarray]:
    edges = []
    quantiles = np.linspace(0.0, 1.0, buckets + 1)[1:-1]
    for field_index in range(numeric.shape[1]):
        values = np.asarray(numeric[:, field_index])
        selected = values[train_mask & np.isfinite(values)]
        if selected.size == 0:
            raise RuntimeError(
                f"numeric field {VIDEO_NUMERIC_FIELDS[field_index]} has no train values"
            )
        edges.append(np.unique(np.quantile(selected, quantiles)).astype(np.float32))
    return edges


def _build_video_table(
    connection: sqlite3.Connection,
    video_feature_path: Path,
    output_path: Path,
    eval_meta_path: Path,
    numeric_temp_path: Path,
) -> tuple[np.ndarray, list[int], dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    all_video_ids, all_video_counts = _load_all_video_ids_and_counts(connection)
    identity_counts = {
        field: load_integer_count_pairs(connection, field) for field in IDENTITY_FIELDS
    }
    category_vocabs = _train_category_vocabs(connection)
    numeric = _build_numeric_memmap(
        video_feature_path, all_video_ids, numeric_temp_path
    )
    quantile_edges = _quantile_edges(numeric, all_video_counts > 0)

    table = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(all_video_ids), len(VIDEO_FIELD_NAMES)),
    )
    eval_meta = np.lib.format.open_memmap(
        eval_meta_path, mode="w+", dtype=np.uint8, shape=(len(all_video_ids), 2)
    )
    table[:] = GENERAL_SPECIAL_ID["missing"]
    eval_meta[:, 0] = video_band(all_video_counts)
    eval_meta[:, 1] = 0

    offset = 0
    query = """
        SELECT b.video_id, b.author_id, b.music_id,
               b.video_type, b.upload_type, b.visible_status, b.music_type
        FROM video_basic b
        ORDER BY b.video_id
    """
    for rows in sqlite_rows(connection, query):
        length = len(rows)
        batch_ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        expected_ids = all_video_ids[offset : offset + length]
        if not np.array_equal(batch_ids, expected_ids):
            raise RuntimeError("video_basic static-universe IDs drift")

        video_train_ids, _ = identity_counts["video_id"]
        table[offset : offset + length, 0] = encode_identity_values(
            batch_ids, video_train_ids
        )
        for column, field in ((1, "author_id"), (2, "music_id")):
            missing = np.asarray(
                [row[column] is None or str(row[column]).strip() == "" for row in rows]
            )
            raw = np.asarray(
                [
                    0
                    if missing[index]
                    else int(float(rows[index][column]))
                    for index in range(length)
                ],
                dtype=np.int64,
            )
            train_ids, _ = identity_counts[field]
            encoded = encode_identity_values(raw, train_ids, missing_mask=missing)
            table[offset : offset + length, column] = encoded
            seen = encoded >= len(SPECIAL_ID)
            if field == "author_id":
                eval_meta[offset : offset + length, 1] |= seen.astype(np.uint8)
            else:
                eval_meta[offset : offset + length, 1] |= (
                    seen.astype(np.uint8) * np.uint8(2)
                )

        for category_offset, field in enumerate(VIDEO_CATEGORY_FIELDS, start=3):
            source_column = 3 + (category_offset - 3)
            table[offset : offset + length, category_offset] = [
                encode_category(row[source_column], category_vocabs[field])
                for row in rows
            ]
        offset += length
    if offset != len(all_video_ids):
        raise RuntimeError("video table rows do not close")

    for numeric_index, edges in enumerate(quantile_edges):
        values = np.asarray(numeric[:, numeric_index])
        encoded = np.full(len(values), GENERAL_SPECIAL_ID["missing"], dtype=np.int32)
        finite = np.isfinite(values)
        encoded[finite] = (
            np.searchsorted(edges, values[finite], side="right")
            + len(GENERAL_SPECIAL_ID)
        )
        table[:, 3 + len(VIDEO_CATEGORY_FIELDS) + numeric_index] = encoded

    table.flush()
    eval_meta.flush()
    numeric_temp_path.unlink()

    cardinalities = [
        len(identity_counts[field][0]) + len(SPECIAL_ID)
        for field in IDENTITY_FIELDS
    ]
    cardinalities.extend(
        len(category_vocabs[field]) + len(GENERAL_SPECIAL_ID)
        for field in VIDEO_CATEGORY_FIELDS
    )
    cardinalities.extend(
        len(edges) + 1 + len(GENERAL_SPECIAL_ID) for edges in quantile_edges
    )
    zero_scan = {}
    for field in IDENTITY_FIELDS:
        if field == "video_id":
            row = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(v.train_count),0),
                       COALESCE(SUM(v.test_count),0)
                FROM video_basic b
                LEFT JOIN video_agg v ON v.video_id=b.video_id
                WHERE b.video_id=0
                """
            ).fetchone()
        else:
            row = connection.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM(v.train_count),0),
                       COALESCE(SUM(v.test_count),0)
                FROM video_basic b JOIN video_agg v ON v.video_id=b.video_id
                WHERE CAST(b."{field}" AS TEXT)='0'
                """
            ).fetchone()
        zero_scan[field] = {
            "keys_or_metadata_videos": int(row[0]),
            "train_rows": int(row[1]),
            "test_rows": int(row[2]),
            "routing": "reserved_zero",
        }
    metadata = {
        "all_video_rows": int(len(all_video_ids)),
        "static_universe": "video_basic_with_p3_train_count_left_join",
        "p3_aggregate_video_rows": int(
            connection.execute("SELECT COUNT(*) FROM video_agg").fetchone()[0]
        ),
        "video_basic_without_p3_aggregate": int(np.count_nonzero(all_video_counts == 0))
        - int(
            connection.execute(
                "SELECT COUNT(*) FROM video_agg WHERE train_count=0"
            ).fetchone()[0]
        ),
        "identity_private_rows": {
            field: int(len(identity_counts[field][0])) for field in IDENTITY_FIELDS
        },
        "test_only_ids_have_private_rows": False,
        "raw_zero_scan": zero_scan,
        "category_vocab_sizes": {
            field: len(vocab) for field, vocab in category_vocabs.items()
        },
        "numeric_bucket_policy": {
            field: {
                "train_only": True,
                "requested_buckets": 100,
                "actual_nonmissing_buckets": int(len(edges) + 1),
                "edges": edges.tolist(),
            }
            for field, edges in zip(VIDEO_NUMERIC_FIELDS, quantile_edges)
        },
    }
    return all_video_ids, cardinalities, metadata, identity_counts


def _write_frequency_cache(
    path: Path,
    identity_counts: dict[str, tuple[np.ndarray, np.ndarray]],
    cardinalities_by_field: dict[str, int],
    *,
    split_sha256: str,
) -> dict:
    payload = {}
    for field, (_, counts) in identity_counts.items():
        encoded_counts = np.zeros(cardinalities_by_field[field], dtype=np.int64)
        encoded_counts[len(SPECIAL_ID) :] = counts
        payload[f"{field}_counts"] = encoded_counts
    metadata = {
        "format": "kuairand27k_k1_frequency_cache_v1",
        "split_sha256": split_sha256,
        "identity_fields": list(IDENTITY_FIELDS),
        "special_token_ids": dict(SPECIAL_ID),
        "train_only": True,
        "test_statistics_used": False,
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(path, **payload)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "metadata": metadata,
    }


def _find_row(sorted_ids: np.ndarray, raw_id: int, field: str) -> int:
    position = int(np.searchsorted(sorted_ids, raw_id))
    if position >= len(sorted_ids) or sorted_ids[position] != raw_id:
        raise RuntimeError(f"interaction {field}={raw_id} lacks a static row")
    return position


def _write_interactions(
    standard_paths: Sequence[Path],
    all_user_ids: np.ndarray,
    all_video_ids: np.ndarray,
    output_dir: Path,
    *,
    shard_rows: int,
    data_seed: int,
    allow_noncanonical_counts: bool,
    disk_tracker: PeakDiskTracker,
    expected: dict,
) -> dict:
    writers = {
        split: InteractionShardWriter(
            output_dir, split, shard_rows=shard_rows, data_seed=data_seed
        )
        for split in ("train", "val", "test")
    }
    all_rows = 0
    excluded_rows = 0
    excluded_positives = 0
    for path in standard_paths:
        for row in _read_csv_rows(path):
            all_rows += 1
            label = int(row["is_click"])
            if label not in (0, 1):
                raise RuntimeError(f"non-binary is_click={label}")
            split = split_for_day(day_from_time_ms(row["time_ms"]), expected)
            if split is None:
                excluded_rows += 1
                excluded_positives += label
                continue
            user_index = _find_row(all_user_ids, int(row["user_id"]), "user_id")
            video_index = _find_row(all_video_ids, int(row["video_id"]), "video_id")
            writers[split].add(user_index, video_index, label)
            if all_rows % 10_000_000 == 0:
                disk_tracker.sample()
    splits = {split: writer.finalize() for split, writer in writers.items()}
    observed = {
        "all_rows": all_rows,
        "train_rows": splits["train"]["rows"],
        "val_rows": splits["val"]["rows"],
        "test_rows": splits["test"]["rows"],
    }
    if not allow_noncanonical_counts:
        expected_rows = {key: expected[key] for key in observed}
        observed_positives = {
            f"{split}_positives": splits[split]["positives"]
            for split in ("train", "val", "test")
        }
        expected_positives = {
            key: expected[key] for key in observed_positives
        }
        if (
            observed != expected_rows
            or observed_positives != expected_positives
            or excluded_rows != int(expected["excluded_rows"])
            or excluded_positives != int(expected["excluded_positives"])
        ):
            raise RuntimeError(
                "K1 split invariants drift: "
                f"rows={observed}, positives={observed_positives}, "
                f"excluded_rows={excluded_rows}, "
                f"excluded_positives={excluded_positives}"
            )
    return {
        "observed": observed,
        "excluded_rows": excluded_rows,
        "excluded_positives": excluded_positives,
        "splits": splits,
    }


def prepare(args: argparse.Namespace) -> Path:
    expected = load_expected_split(getattr(args, "split_spec", None))
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(output_dir).free
    if free_before < int(float(args.min_free_gb) * 1024**3):
        raise RuntimeError(
            f"insufficient free disk: {free_before / 1024**3:.1f} GiB "
            f"< {args.min_free_gb} GiB"
        )
    tracker = PeakDiskTracker(output_dir)

    sqlite_path = Path(args.p3_sqlite).resolve()
    source_db_sha_before = sha256_file(sqlite_path)
    connection = open_sqlite_immutable(sqlite_path)
    try:
        user_path = Path(args.user_features).resolve()
        video_path = Path(args.video_features).resolve()
        standard_paths = [Path(path).resolve() for path in args.standard_logs]
        source_files = [sqlite_path, user_path, video_path, *standard_paths]
        source_manifest = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ]

        user_table_path = output_dir / "user_sparse.npy"
        all_user_ids, user_cards, user_meta = _build_user_table(
            connection, user_path, user_table_path
        )
        tracker.sample()

        video_table_path = output_dir / "video_sparse.npy"
        eval_meta_path = output_dir / "video_eval_meta.npy"
        numeric_temp_path = output_dir / ".video_numeric.tmp.npy"
        all_video_ids, video_cards, video_meta, identity_counts = _build_video_table(
            connection,
            video_path,
            video_table_path,
            eval_meta_path,
            numeric_temp_path,
        )
        tracker.sample()

        cardinalities = [*user_cards, *video_cards]
        if len(cardinalities) != len(FIELD_NAMES):
            raise RuntimeError("K1 cardinalities do not close to 37 fields")
        cards_by_field = dict(zip(FIELD_NAMES, cardinalities))
        cache_path = output_dir / "identity_frequency_cache.npz"
        cache_meta = _write_frequency_cache(
            cache_path,
            identity_counts,
            cards_by_field,
            split_sha256=expected["split_sha256"],
        )
        tracker.sample()

        interaction_result = _write_interactions(
            standard_paths,
            all_user_ids,
            all_video_ids,
            output_dir,
            shard_rows=args.shard_rows,
            data_seed=args.data_seed,
            allow_noncanonical_counts=args.allow_noncanonical_counts,
            disk_tracker=tracker,
            expected=expected,
        )
        tracker.sample()

        split_definition = {
            "train": expected["train_days"],
            "val": expected["val_days"],
            "test": expected["test_days"],
            "excluded": expected["excluded_days"],
        }
        split_definition_sha = canonical_sha256(split_definition)

        source_db_sha_after = sha256_file(sqlite_path)
        if source_db_sha_after != source_db_sha_before:
            raise RuntimeError("P3 source SQLite changed during K1 preprocessing")
        manifest = {
            "format": "kuairand27k_k1_mmap_v1",
            "created_unix": time.time(),
            "protocol": {
                "task": "pointwise_nonseq_is_click",
                "random_logs_used": False,
                "sequence_used": False,
                "train_only_vocab": True,
                "test_statistics_used_for_training": False,
                "tail_cutoff": 5,
                "split_sha256": expected["split_sha256"],
                "local_split_definition_sha256": split_definition_sha,
                "data_seed": args.data_seed,
            },
            "field_names": list(FIELD_NAMES),
            "cardinalities": cardinalities,
            "tables": {
                "user_sparse": array_artifact(user_table_path, output_dir),
                "video_sparse": array_artifact(video_table_path, output_dir),
                "video_eval_meta": array_artifact(eval_meta_path, output_dir),
            },
            "splits": interaction_result["splits"],
            "auxiliary_artifacts": [cache_meta],
            "invariants": {
                "split": interaction_result["observed"],
                "split_definition": split_definition,
                "excluded_rows": interaction_result["excluded_rows"],
                "excluded_positives": interaction_result["excluded_positives"],
                "user": user_meta,
                "video": video_meta,
                "special_token_ids": dict(SPECIAL_ID),
                "reserved_zero_missing_oov_mutually_exclusive": True,
                "source_sqlite_open_mode": "sqlite_uri_mode_ro_immutable",
                "source_sqlite_sha256_before": source_db_sha_before,
                "source_sqlite_sha256_after": source_db_sha_after,
                "source_sqlite_bytewise_unchanged": True,
            },
            "source_manifest": source_manifest,
            "disk": {
                "free_before_bytes": free_before,
                "free_after_bytes": shutil.disk_usage(output_dir).free,
                "minimum_free_bytes": tracker.min_free_bytes,
                "peak_output_bytes": tracker.peak_output_bytes,
            },
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        checksums = []
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                checksums.append(f"{sha256_file(path)}  {path.name}")
        (output_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
        return manifest_path
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3-sqlite", required=True)
    parser.add_argument("--user-features", required=True)
    parser.add_argument("--video-features", required=True)
    parser.add_argument("--standard-logs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-seed", type=int, default=20260724)
    parser.add_argument("--shard-rows", type=int, default=5_000_000)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--split-spec")
    parser.add_argument("--allow-noncanonical-counts", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output = prepare(parse_args())
    print(json.dumps({"status": "ok", "manifest": str(output)}, sort_keys=True))
