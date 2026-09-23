#!/usr/bin/env python3
"""Disk-backed audit for KuaiRand-27K non-sequential long-tail fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DAY_MS = 86_400_000
UTC8_MS = 8 * 3_600_000
BINARY_LABELS = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_profile_enter",
    "is_hate",
)
DERIVED_BINARY_LABELS = (
    "play_ge_3s",
    "play_ge_5s",
    "play_ge_7s",
    "play_ge_10s",
    "play_ge_18s",
    "watch_ratio_ge_0p5",
    "watch_ratio_ge_1p0",
    "valid_play_recomputed",
    "long_view_recomputed",
)
AUDIT_LABELS = (*BINARY_LABELS, *DERIVED_BINARY_LABELS)
LABEL_COMPARISONS = (
    ("is_click_vs_valid_play", "is_click", "valid_play_recomputed"),
    ("long_view_vs_recomputed", "long_view", "long_view_recomputed"),
    ("is_click_vs_play_ge_5s", "is_click", "play_ge_5s"),
    ("is_hate_vs_is_click", "is_hate", "is_click"),
    ("is_hate_vs_long_view", "is_hate", "long_view"),
)
CONTINUOUS_SIGNALS = ("play_time_ms", "profile_stay_time", "comment_stay_time")
USER_CATEGORICAL_FIELDS = (
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
USER_CONTINUOUS_FIELDS = (
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
)
VIDEO_CATEGORICAL_FIELDS = (
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "music_id",
    "music_type",
)
VIDEO_CONTINUOUS_FIELDS = ("video_duration", "server_width", "server_height")
FREQUENCY_BANDS = (
    ("unseen", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-500", 101, 500),
    ("501-1000", 501, 1000),
    (">1000", 1001, None),
)
CUMULATIVE_CUTOFFS = (1, 5, 10, 20, 50, 100, 500, 1000)
AGG_COLUMNS = ("train_count", "test_count", *(f"test_{x}" for x in AUDIT_LABELS))
STATISTIC_RISK_TOKENS = (
    "play", "click", "like", "follow", "comment", "forward", "share",
    "count", "cnt", "ratio", "rate", "score", "stat",
)


def _day_index(time_ms: int) -> int:
    return (time_ms + UTC8_MS) // DAY_MS


def _day_iso(day_index: int) -> str:
    return datetime.fromtimestamp(day_index * 86400, tz=timezone.utc).date().isoformat()


def _positive(value: str) -> bool:
    return bool(value) and float(value) > 0.0


def _number(value: str) -> float:
    return float(value) if value else 0.0


def _audit_labels(row: dict[str, str]) -> dict[str, int]:
    labels = {label: int(_positive(row[label])) for label in BINARY_LABELS}
    play_time_ms = max(0.0, _number(row["play_time_ms"]))
    duration_ms = max(0.0, _number(row["duration_ms"]))
    has_duration = duration_ms > 0.0
    labels.update({
        "play_ge_3s": int(play_time_ms >= 3_000.0),
        "play_ge_5s": int(play_time_ms >= 5_000.0),
        "play_ge_7s": int(play_time_ms >= 7_000.0),
        "play_ge_10s": int(play_time_ms >= 10_000.0),
        "play_ge_18s": int(play_time_ms >= 18_000.0),
        "watch_ratio_ge_0p5": int(has_duration and play_time_ms >= 0.5 * duration_ms),
        "watch_ratio_ge_1p0": int(has_duration and play_time_ms >= duration_ms),
        "valid_play_recomputed": int(
            has_duration
            and (
                (duration_ms <= 7_000.0 and play_time_ms >= duration_ms)
                or (duration_ms > 7_000.0 and play_time_ms > 7_000.0)
            )
        ),
        "long_view_recomputed": int(
            has_duration
            and (
                (duration_ms <= 18_000.0 and play_time_ms >= duration_ms)
                or (duration_ms > 18_000.0 and play_time_ms >= 18_000.0)
            )
        ),
    })
    return labels


def _comparison_bucket(left: int, right: int) -> str:
    return f"{left}{right}"


def _comparison_report(counter: Counter[str]) -> dict:
    counts = {bucket: int(counter[bucket]) for bucket in ("00", "01", "10", "11")}
    rows = sum(counts.values())
    return {
        "counts": counts,
        "rows": rows,
        "agreement_rate": (counts["00"] + counts["11"]) / max(rows, 1),
        "left_positive_right_negative": counts["10"],
        "left_negative_right_positive": counts["01"],
    }


def _scan_daily_file(path_string: str) -> dict:
    path = Path(path_string)
    day_rows: Counter[int] = Counter()
    day_labels = {label: Counter() for label in AUDIT_LABELS}
    day_continuous_nonzero = {name: Counter() for name in CONTINUOUS_SIGNALS}
    day_continuous_sum = {name: Counter() for name in CONTINUOUS_SIGNALS}
    tab_rows: Counter[str] = Counter()
    tab_labels = {label: Counter() for label in AUDIT_LABELS}
    day_tab_rows = defaultdict(Counter)
    day_tab_labels = {
        label: defaultdict(Counter) for label in AUDIT_LABELS
    }
    day_comparisons = {
        name: defaultdict(Counter) for name, _, _ in LABEL_COMPARISONS
    }
    tab_comparisons = {
        name: defaultdict(Counter) for name, _, _ in LABEL_COMPARISONS
    }

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        required = {
            "user_id", "video_id", "time_ms", "tab", "duration_ms",
            *BINARY_LABELS, *CONTINUOUS_SIGNALS,
        }
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            day = _day_index(int(row["time_ms"]))
            tab = row.get("tab", "") or "MISSING"
            day_rows[day] += 1
            tab_rows[tab] += 1
            day_tab_rows[day][tab] += 1
            labels = _audit_labels(row)
            for label in AUDIT_LABELS:
                if labels[label]:
                    day_labels[label][day] += 1
                    tab_labels[label][tab] += 1
                    day_tab_labels[label][day][tab] += 1
            for name, left, right in LABEL_COMPARISONS:
                bucket = _comparison_bucket(labels[left], labels[right])
                day_comparisons[name][day][bucket] += 1
                tab_comparisons[name][tab][bucket] += 1
            for signal in CONTINUOUS_SIGNALS:
                value = _number(row[signal])
                day_continuous_sum[signal][day] += value
                if value > 0:
                    day_continuous_nonzero[signal][day] += 1
    return {
        "path": str(path),
        "header": header,
        "day_rows": day_rows,
        "day_labels": day_labels,
        "day_continuous_nonzero": day_continuous_nonzero,
        "day_continuous_sum": day_continuous_sum,
        "tab_rows": tab_rows,
        "tab_labels": tab_labels,
        "day_tab_rows": day_tab_rows,
        "day_tab_labels": day_tab_labels,
        "day_comparisons": day_comparisons,
        "tab_comparisons": tab_comparisons,
    }


def _merge_daily_scans(scans: Iterable[dict]) -> dict:
    merged = {
        "files": [],
        "headers": [],
        "day_rows": Counter(),
        "day_labels": {label: Counter() for label in AUDIT_LABELS},
        "day_continuous_nonzero": {name: Counter() for name in CONTINUOUS_SIGNALS},
        "day_continuous_sum": {name: Counter() for name in CONTINUOUS_SIGNALS},
        "tab_rows": Counter(),
        "tab_labels": {label: Counter() for label in AUDIT_LABELS},
        "day_tab_rows": defaultdict(Counter),
        "day_tab_labels": {
            label: defaultdict(Counter) for label in AUDIT_LABELS
        },
        "day_comparisons": {
            name: defaultdict(Counter) for name, _, _ in LABEL_COMPARISONS
        },
        "tab_comparisons": {
            name: defaultdict(Counter) for name, _, _ in LABEL_COMPARISONS
        },
    }
    for scan in scans:
        merged["files"].append(scan["path"])
        merged["headers"].append(scan["header"])
        merged["day_rows"].update(scan["day_rows"])
        merged["tab_rows"].update(scan["tab_rows"])
        for day, counts in scan["day_tab_rows"].items():
            merged["day_tab_rows"][day].update(counts)
        for label in AUDIT_LABELS:
            merged["day_labels"][label].update(scan["day_labels"][label])
            merged["tab_labels"][label].update(scan["tab_labels"][label])
            for day, counts in scan["day_tab_labels"][label].items():
                merged["day_tab_labels"][label][day].update(counts)
        for name, _, _ in LABEL_COMPARISONS:
            for day, counts in scan["day_comparisons"][name].items():
                merged["day_comparisons"][name][day].update(counts)
            for tab, counts in scan["tab_comparisons"][name].items():
                merged["tab_comparisons"][name][tab].update(counts)
        for signal in CONTINUOUS_SIGNALS:
            merged["day_continuous_nonzero"][signal].update(
                scan["day_continuous_nonzero"][signal]
            )
            merged["day_continuous_sum"][signal].update(scan["day_continuous_sum"][signal])
    return merged


def select_temporal_split(
    day_rows: Counter[int],
    day_clicks: Counter[int],
    test_days: int = 2,
    completeness_ratio: float = 0.5,
) -> dict:
    days = sorted(day_rows)
    if len(days) < test_days + 2:
        raise ValueError(f"Need at least {test_days + 2} dates, found {len(days)}")
    median_rows = float(statistics.median(day_rows[d] for d in days))
    threshold = median_rows * completeness_ratio
    eligible = [d for d in days if day_rows[d] >= threshold and day_clicks[d] > 0]
    if len(eligible) < test_days + 2:
        raise ValueError(f"Insufficient complete dates: eligible={len(eligible)}")
    val_day = eligible[-test_days - 1]
    test = eligible[-test_days:]
    train = [d for d in days if d < val_day]
    if not train:
        raise ValueError("Temporal split produced no training dates")
    selected = set(train + [val_day] + test)
    return {
        "train": train,
        "val": [val_day],
        "test": test,
        "excluded": [d for d in days if d not in selected],
        "median_daily_rows": median_rows,
        "completeness_threshold_rows": threshold,
        "completeness_ratio": completeness_ratio,
    }


def _configure_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-524288;
        PRAGMA locking_mode=EXCLUSIVE;
        """
    )


def _create_aggregate_table(conn: sqlite3.Connection, table: str, key: str) -> None:
    columns = ", ".join(f'"{name}" INTEGER NOT NULL DEFAULT 0' for name in AGG_COLUMNS)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ("{key}" INTEGER PRIMARY KEY, {columns})')


def _empty_vector() -> list[int]:
    return [0] * len(AGG_COLUMNS)


def _flush_aggregate(
    conn: sqlite3.Connection,
    table: str,
    key: str,
    cache: dict[int, list[int]],
) -> int:
    if not cache:
        return 0
    row_count = len(cache)
    names = (key, *AGG_COLUMNS)
    quoted = ", ".join(f'"{name}"' for name in names)
    placeholders = ", ".join("?" for _ in names)
    updates = ", ".join(
        f'"{name}"="{table}"."{name}"+excluded."{name}"' for name in AGG_COLUMNS
    )
    sql = (
        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders}) '
        f'ON CONFLICT("{key}") DO UPDATE SET {updates}'
    )
    conn.executemany(sql, ((entity, *values) for entity, values in cache.items()))
    conn.commit()
    cache.clear()
    return row_count


def _sqlite_bytes(conn: sqlite3.Connection) -> int:
    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    return page_count * page_size


def _accumulate_interactions(
    conn: sqlite3.Connection,
    standard_files: Sequence[Path],
    split: dict,
    flush_unique: int,
) -> dict:
    _create_aggregate_table(conn, "video_agg", "video_id")
    _create_aggregate_table(conn, "user_agg", "user_id")
    train_days, test_days = set(split["train"]), set(split["test"])
    video_cache: dict[int, list[int]] = {}
    user_cache: dict[int, list[int]] = {}
    tab_cache: dict[str, list[int]] = {}
    processed = kept = 0
    flush_events = Counter()
    flushed_unique_rows = Counter()
    peak_database_bytes = _sqlite_bytes(conn)

    for path in standard_files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                processed += 1
                day = _day_index(int(row["time_ms"]))
                if day in train_days:
                    split_index = 0
                elif day in test_days:
                    split_index = 1
                else:
                    continue
                kept += 1
                row_labels = _audit_labels(row)
                labels = [row_labels[label] for label in AUDIT_LABELS]
                for cache, entity in (
                    (video_cache, int(row["video_id"])),
                    (user_cache, int(row["user_id"])),
                ):
                    values = cache.setdefault(entity, _empty_vector())
                    values[split_index] += 1
                    if split_index == 1:
                        for index, value in enumerate(labels, start=2):
                            values[index] += value
                tab = row.get("tab", "") or "MISSING"
                values = tab_cache.setdefault(tab, _empty_vector())
                values[split_index] += 1
                if split_index == 1:
                    for index, value in enumerate(labels, start=2):
                        values[index] += value

                if len(video_cache) >= flush_unique:
                    flushed_unique_rows["video_agg"] += _flush_aggregate(
                        conn, "video_agg", "video_id", video_cache
                    )
                    flush_events["video_agg"] += 1
                    peak_database_bytes = max(peak_database_bytes, _sqlite_bytes(conn))
                if len(user_cache) >= flush_unique:
                    flushed_unique_rows["user_agg"] += _flush_aggregate(
                        conn, "user_agg", "user_id", user_cache
                    )
                    flush_events["user_agg"] += 1
                    peak_database_bytes = max(peak_database_bytes, _sqlite_bytes(conn))

    for table, key, cache in (
        ("video_agg", "video_id", video_cache),
        ("user_agg", "user_id", user_cache),
    ):
        if cache:
            flushed_unique_rows[table] += _flush_aggregate(conn, table, key, cache)
            flush_events[table] += 1
            peak_database_bytes = max(peak_database_bytes, _sqlite_bytes(conn))
    return {
        "processed_rows": processed,
        "train_test_rows": kept,
        "tab_cache": tab_cache,
        "flush_events": dict(flush_events),
        "flushed_unique_rows": dict(flushed_unique_rows),
        "peak_database_bytes": peak_database_bytes,
    }


def _band_for_count(count: int) -> str:
    for name, lower, upper in FREQUENCY_BANDS:
        if count >= lower and (upper is None or count <= upper):
            return name
    raise AssertionError(f"No band for count={count}")


def summarize_aggregate_rows(
    rows: Iterable[Sequence],
    *,
    field: str,
    multi_valued: bool = False,
    top_n: int = 20,
) -> dict:
    train_unique = test_unique = train_occurrences = test_rows = test_oov = 0
    test_label_totals = Counter()
    band_ids = Counter()
    band_train_occurrences = Counter()
    band_test_unique = Counter()
    band_test_rows = Counter()
    band_test_labels = {label: Counter() for label in AUDIT_LABELS}
    cumulative_rows = Counter()
    cumulative_labels = {label: Counter() for label in AUDIT_LABELS}
    seen_rare_rows = Counter()
    seen_rare_labels = {label: Counter() for label in AUDIT_LABELS}
    top: list[tuple[int, str]] = []

    for row in rows:
        key = str(row[0])
        train_count = int(row[1] or 0)
        test_count = int(row[2] or 0)
        labels = {label: int(row[i + 3] or 0) for i, label in enumerate(AUDIT_LABELS)}
        band = _band_for_count(train_count)
        if train_count > 0:
            train_unique += 1
            train_occurrences += train_count
            band_ids[band] += 1
            band_train_occurrences[band] += train_count
            item = (train_count, key)
            if len(top) < top_n:
                heapq.heappush(top, item)
            elif item > top[0]:
                heapq.heapreplace(top, item)
        if test_count <= 0:
            continue
        test_unique += 1
        test_rows += test_count
        band_test_unique[band] += 1
        band_test_rows[band] += test_count
        if train_count == 0:
            test_oov += 1
        for label, value in labels.items():
            test_label_totals[label] += value
            band_test_labels[label][band] += value
        for cutoff in CUMULATIVE_CUTOFFS:
            if train_count == 0 or train_count <= cutoff:
                cumulative_rows[cutoff] += test_count
                for label, value in labels.items():
                    cumulative_labels[label][cutoff] += value
            if 0 < train_count <= cutoff:
                seen_rare_rows[cutoff] += test_count
                for label, value in labels.items():
                    seen_rare_labels[label][cutoff] += value

    bands = []
    for name, _, _ in FREQUENCY_BANDS:
        rows_in_band = int(band_test_rows[name])
        labels = {label: int(band_test_labels[label][name]) for label in AUDIT_LABELS}
        bands.append({
            "band": name,
            "train_unique_keys": int(band_ids[name]),
            "train_occurrences": int(band_train_occurrences[name]),
            "test_unique_keys": int(band_test_unique[name]),
            "test_rows": rows_in_band,
            "test_row_fraction": rows_in_band / max(test_rows, 1),
            "test_label_positives": labels,
            "test_label_rates": {
                label: value / max(rows_in_band, 1) for label, value in labels.items()
            },
        })
    cumulative = []
    seen_rare = []
    for cutoff in CUMULATIVE_CUTOFFS:
        rows_at_cutoff = int(cumulative_rows[cutoff])
        labels = {label: int(cumulative_labels[label][cutoff]) for label in AUDIT_LABELS}
        cumulative.append({
            "cutoff": cutoff,
            "include_unseen_test_rows": rows_at_cutoff,
            "include_unseen_test_row_fraction": rows_at_cutoff / max(test_rows, 1),
            "include_unseen_test_label_positives": labels,
            "include_unseen_test_label_fractions": {
                label: value / max(test_label_totals[label], 1)
                for label, value in labels.items()
            },
        })
        seen_rows_at_cutoff = int(seen_rare_rows[cutoff])
        seen_labels = {
            label: int(seen_rare_labels[label][cutoff]) for label in AUDIT_LABELS
        }
        seen_rare.append({
            "cutoff": cutoff,
            "seen_rare_test_rows": seen_rows_at_cutoff,
            "seen_rare_test_row_fraction": seen_rows_at_cutoff / max(test_rows, 1),
            "seen_rare_test_label_positives": seen_labels,
            "seen_rare_test_label_fractions": {
                label: value / max(test_label_totals[label], 1)
                for label, value in seen_labels.items()
            },
        })
    unseen_band = next(item for item in bands if item["band"] == "unseen")
    return {
        "field": field,
        "multi_valued": multi_valued,
        "coverage_unit": "token_occurrences" if multi_valued else "interactions",
        "train_unique_keys": train_unique,
        "train_occurrences": train_occurrences,
        "test_unique_keys": test_unique,
        "test_rows": test_rows,
        "test_oov_unique_keys": test_oov,
        "unseen_test": {
            "test_unique_keys": unseen_band["test_unique_keys"],
            "test_rows": unseen_band["test_rows"],
            "test_row_fraction": unseen_band["test_row_fraction"],
            "test_label_positives": unseen_band["test_label_positives"],
        },
        "test_label_positives": dict(test_label_totals),
        "top_train_keys": [
            {"key": key, "train_occurrences": count}
            for count, key in sorted(top, reverse=True)
        ],
        "bands": bands,
        "cumulative_cutoffs": cumulative,
        "seen_rare_cutoffs": seen_rare,
    }


def _aggregate_query(table: str, key: str) -> str:
    columns = ", ".join(f'"{name}"' for name in AGG_COLUMNS)
    return f'SELECT CAST("{key}" AS TEXT), {columns} FROM "{table}"'


def _load_user_features(conn: sqlite3.Connection, files: Sequence[Path]) -> list[dict]:
    reports = []
    for file_index, path in enumerate(files):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            fields = [name for name in USER_CATEGORICAL_FIELDS if name in header]
            numeric_fields = [name for name in USER_CONTINUOUS_FIELDS if name in header]
            if "user_id" not in header:
                reports.append({"path": str(path), "error": "missing user_id", "header": header})
                continue
            table = f"user_features_{file_index}"
            definitions = ", ".join(f'"{field}" TEXT' for field in fields)
            suffix = f", {definitions}" if definitions else ""
            conn.execute(f'CREATE TABLE "{table}" ("user_id" INTEGER PRIMARY KEY{suffix})')
            names = ("user_id", *fields)
            quoted = ", ".join(f'"{name}"' for name in names)
            placeholders = ", ".join("?" for _ in names)
            batch = []
            row_count = 0
            numeric_values = {field: [] for field in numeric_fields}
            for row in reader:
                batch.append((int(row["user_id"]), *(row.get(field, "") for field in fields)))
                for field in numeric_fields:
                    if row.get(field, ""):
                        numeric_values[field].append(float(row[field]))
                row_count += 1
                if len(batch) >= 100_000:
                    conn.executemany(f'INSERT OR REPLACE INTO "{table}" ({quoted}) VALUES ({placeholders})', batch)
                    batch.clear()
            if batch:
                conn.executemany(f'INSERT OR REPLACE INTO "{table}" ({quoted}) VALUES ({placeholders})', batch)
            conn.commit()
            reports.append({
                "path": str(path),
                "header": header,
                "rows": row_count,
                "table": table,
                "fields": fields,
                "continuous_profiles": {
                    field: _numeric_profile(values, row_count, exact=True)
                    for field, values in numeric_values.items()
                },
            })
    return reports


def _load_video_basic(conn: sqlite3.Connection, files: Sequence[Path]) -> list[dict]:
    reports = []
    definitions = ", ".join(f'"{field}" TEXT' for field in VIDEO_CATEGORICAL_FIELDS)
    conn.execute(f'CREATE TABLE video_basic (video_id INTEGER PRIMARY KEY, {definitions})')
    conn.execute('CREATE TABLE video_tag (video_id INTEGER NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(video_id, tag))')
    names = ("video_id", *VIDEO_CATEGORICAL_FIELDS)
    quoted = ", ".join(f'"{name}"' for name in names)
    placeholders = ", ".join("?" for _ in names)
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            if "video_id" not in header:
                reports.append({"path": str(path), "error": "missing video_id", "header": header})
                continue
            batch, tag_batch = [], []
            rows = 0
            numeric_fields = [name for name in VIDEO_CONTINUOUS_FIELDS if name in header]
            numeric_values = {field: [] for field in numeric_fields}
            numeric_nonempty = Counter()
            for row in reader:
                video_id = int(float(row["video_id"]))
                batch.append((video_id, *(row.get(field, "") for field in VIDEO_CATEGORICAL_FIELDS)))
                for tag in (row.get("tag", "") or "").split(","):
                    tag = tag.strip()
                    if tag:
                        tag_batch.append((video_id, tag))
                for field in numeric_fields:
                    if row.get(field, ""):
                        numeric_nonempty[field] += 1
                        if video_id % 1000 == 0:
                            numeric_values[field].append(float(row[field]))
                rows += 1
                if len(batch) >= 100_000:
                    conn.executemany(f'INSERT OR REPLACE INTO video_basic ({quoted}) VALUES ({placeholders})', batch)
                    conn.executemany('INSERT OR IGNORE INTO video_tag(video_id, tag) VALUES (?, ?)', tag_batch)
                    batch.clear()
                    tag_batch.clear()
            if batch:
                conn.executemany(f'INSERT OR REPLACE INTO video_basic ({quoted}) VALUES ({placeholders})', batch)
                conn.executemany('INSERT OR IGNORE INTO video_tag(video_id, tag) VALUES (?, ?)', tag_batch)
            conn.commit()
            reports.append({
                "path": str(path),
                "header": header,
                "rows": rows,
                "continuous_profiles": {
                    field: _numeric_profile(
                        values,
                        rows,
                        nonempty=numeric_nonempty[field],
                        exact=False,
                        sample_rule="video_id % 1000 == 0",
                    )
                    for field, values in numeric_values.items()
                },
            })
    return reports


def _numeric_profile(
    values: Sequence[float],
    total_rows: int,
    *,
    nonempty: int | None = None,
    exact: bool,
    sample_rule: str | None = None,
) -> dict:
    ordered = sorted(values)
    present = len(values) if nonempty is None else nonempty

    def quantile(fraction: float) -> float | None:
        if not ordered:
            return None
        index = round((len(ordered) - 1) * fraction)
        return float(ordered[index])

    return {
        "total_rows": total_rows,
        "nonempty_rows": present,
        "missing_fraction": (total_rows - present) / max(total_rows, 1),
        "profile_exact": exact,
        "sample_size": len(ordered),
        "sample_rule": sample_rule,
        "min": float(ordered[0]) if ordered else None,
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p99": quantile(0.99),
        "max": float(ordered[-1]) if ordered else None,
    }


def _side_field_rows(
    conn: sqlite3.Connection,
    side_table: str,
    side_key: str,
    agg_table: str,
    agg_key: str,
    field: str,
) -> Iterator[Sequence]:
    sums = ", ".join(f'SUM(a."{name}")' for name in AGG_COLUMNS)
    query = (
        f'SELECT CAST(s."{field}" AS TEXT), {sums} FROM "{side_table}" s '
        f'JOIN "{agg_table}" a ON s."{side_key}"=a."{agg_key}" '
        f'WHERE s."{field}" IS NOT NULL AND s."{field}" != "" '
        f'GROUP BY s."{field}"'
    )
    yield from conn.execute(query)


def _side_presence(
    conn: sqlite3.Connection,
    side_table: str,
    side_key: str,
    agg_table: str,
    agg_key: str,
    field: str,
) -> tuple[int, int]:
    query = (
        f'SELECT COALESCE(SUM(a.train_count), 0), COALESCE(SUM(a.test_count), 0) '
        f'FROM "{agg_table}" a WHERE EXISTS ('
        f'SELECT 1 FROM "{side_table}" s '
        f'WHERE s."{side_key}"=a."{agg_key}" '
        f'AND s."{field}" IS NOT NULL AND s."{field}" != ""'
        f')'
    )
    train_rows, test_rows = conn.execute(query).fetchone()
    return int(train_rows), int(test_rows)


def _attach_presence(
    summary: dict,
    present: tuple[int, int],
    entity_summary: dict,
) -> dict:
    train_rows, test_rows = present
    summary["interaction_train_rows_with_value"] = train_rows
    summary["interaction_test_rows_with_value"] = test_rows
    summary["interaction_train_row_coverage"] = (
        train_rows / max(entity_summary["train_occurrences"], 1)
    )
    summary["interaction_test_row_coverage"] = (
        test_rows / max(entity_summary["test_rows"], 1)
    )
    return summary


def _file_metadata(path: Path, leakage_risk: bool) -> dict:
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    risk = [
        col for col in header
        if any(token in col.lower() for token in STATISTIC_RISK_TOKENS)
    ]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "header": header,
        "time_leakage_risk": leakage_risk,
        "possible_post_event_statistic_columns": risk,
    }


def _daily_report(merged: dict) -> dict:
    report = {}
    for day in sorted(merged["day_rows"]):
        rows = int(merged["day_rows"][day])
        labels = {label: int(merged["day_labels"][label][day]) for label in AUDIT_LABELS}
        report[_day_iso(day)] = {
            "rows": rows,
            "label_positives": labels,
            "label_rates": {label: value / max(rows, 1) for label, value in labels.items()},
            "continuous_nonzero": {
                name: int(merged["day_continuous_nonzero"][name][day])
                for name in CONTINUOUS_SIGNALS
            },
            "continuous_mean": {
                name: float(merged["day_continuous_sum"][name][day]) / max(rows, 1)
                for name in CONTINUOUS_SIGNALS
            },
            "label_comparisons": {
                name: _comparison_report(merged["day_comparisons"][name][day])
                for name, _, _ in LABEL_COMPARISONS
            },
        }
    return report


def _split_tab_report(merged: dict, days: Sequence[int]) -> dict:
    rows_by_tab = Counter()
    labels_by_tab = {label: Counter() for label in AUDIT_LABELS}
    for day in days:
        rows_by_tab.update(merged["day_tab_rows"][day])
        for label in AUDIT_LABELS:
            labels_by_tab[label].update(merged["day_tab_labels"][label][day])
    total_rows = sum(rows_by_tab.values())
    return {
        str(tab): {
            "rows": int(rows),
            "row_fraction": int(rows) / max(total_rows, 1),
            "is_click_positives": int(labels_by_tab["is_click"][tab]),
            "is_click_rate": int(labels_by_tab["is_click"][tab]) / max(int(rows), 1),
        }
        for tab, rows in sorted(rows_by_tab.items(), key=lambda item: str(item[0]))
    }


def audit_dataset(
    data_dir: Path,
    workers: int = 1,
    test_days: int = 2,
    completeness_ratio: float = 0.5,
    work_dir: Path | None = None,
    flush_unique: int = 1_000_000,
    stage: str = "full",
    expected_split: dict | None = None,
    min_free_gb: float = 100.0,
) -> dict:
    del workers  # The full audit is disk-bound; parallel CSV scans would multiply memory.
    if stage not in {"schema", "daily", "interactions", "full"}:
        raise ValueError(f"Unsupported stage={stage!r}")
    standard_files = sorted(data_dir.glob("log_standard*.csv"))
    if not standard_files:
        raise FileNotFoundError(f"No log_standard*.csv under {data_dir}")
    user_files = sorted(data_dir.glob("user_features*.csv"))
    video_basic_files = sorted(data_dir.glob("video_features_basic*.csv"))
    statistic_files = sorted(data_dir.glob("video_features_statistic*.csv"))
    if stage == "schema":
        standard_metadata = [
            _file_metadata(path, leakage_risk=False) for path in standard_files
        ]
        headers = [metadata["header"] for metadata in standard_metadata]
        headers_equal = all(header == headers[0] for header in headers[1:])
        if not headers_equal:
            raise RuntimeError("Standard log headers are not identical")
        required = {
            "user_id", "video_id", "time_ms", "tab", "duration_ms",
            *BINARY_LABELS, *CONTINUOUS_SIGNALS,
        }
        missing = sorted(required - set(headers[0]))
        if missing:
            raise RuntimeError(f"Standard log schema is missing fields: {missing}")
        disk_target = work_dir if work_dir is not None and work_dir.exists() else data_dir
        disk = shutil.disk_usage(disk_target)
        return {
            "protocol": {
                "dataset": "KuaiRand-27K",
                "task": "pointwise_non_sequential_multi_feedback_audit",
                "stage": "schema",
                "random_logs_used": False,
                "sequence_used": False,
                "source_manifest_sha256": os.environ.get("SOURCE_MANIFEST_SHA256"),
            },
            "source_files": {
                "standard": standard_metadata,
                "random_ignored": [
                    _file_metadata(path, leakage_risk=False)
                    for path in sorted(data_dir.glob("log_random*.csv"))
                ],
                "user_features": [
                    _file_metadata(path, leakage_risk=False) for path in user_files
                ],
                "video_basic": [
                    _file_metadata(path, leakage_risk=False) for path in video_basic_files
                ],
                "video_statistic": [
                    _file_metadata(path, leakage_risk=True) for path in statistic_files
                ],
            },
            "invariants": {
                "standard_headers_equal": True,
                "required_log_fields_present": True,
            },
            "execution": {
                "disk_target": str(disk_target),
                "disk_total_bytes": disk.total,
                "disk_used_bytes": disk.used,
                "disk_free_bytes": disk.free,
            },
        }
    daily_scans = [_scan_daily_file(str(path)) for path in standard_files]
    merged = _merge_daily_scans(daily_scans)
    split = select_temporal_split(
        merged["day_rows"],
        merged["day_labels"]["is_click"],
        test_days=test_days,
        completeness_ratio=completeness_ratio,
    )

    daily_rows = int(sum(merged["day_rows"].values()))
    train_rows_expected = int(sum(merged["day_rows"][day] for day in split["train"]))
    test_rows_expected = int(sum(merged["day_rows"][day] for day in split["test"]))
    selected_rows = train_rows_expected + test_rows_expected
    test_labels_expected = {
        label: int(sum(merged["day_labels"][label][day] for day in split["test"]))
        for label in AUDIT_LABELS
    }
    split_report = {
        "train_dates": [_day_iso(day) for day in split["train"]],
        "val_dates": [_day_iso(day) for day in split["val"]],
        "test_dates": [_day_iso(day) for day in split["test"]],
        "excluded_dates": [_day_iso(day) for day in split["excluded"]],
        "n_train_days": len(split["train"]),
        "n_val_days": len(split["val"]),
        "n_test_days": len(split["test"]),
        "median_daily_rows": split["median_daily_rows"],
        "completeness_threshold_rows": split["completeness_threshold_rows"],
        "completeness_ratio": split["completeness_ratio"],
    }
    split_sha256 = hashlib.sha256(
        json.dumps(split_report, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_split is not None:
        for key in ("train_dates", "val_dates", "test_dates", "excluded_dates"):
            if split_report[key] != expected_split.get(key):
                raise RuntimeError(
                    f"Split drift for {key}: expected={expected_split.get(key)}, "
                    f"actual={split_report[key]}"
                )
    report = {
        "protocol": {
            "dataset": "KuaiRand-27K",
            "task": "pointwise_non_sequential_multi_feedback_audit",
            "stage": stage,
            "raw_binary_labels": list(BINARY_LABELS),
            "derived_binary_labels": list(DERIVED_BINARY_LABELS),
            "audit_labels": list(AUDIT_LABELS),
            "continuous_signals": list(CONTINUOUS_SIGNALS),
            "timezone": "Asia/Shanghai",
            "primary_logs": "standard_only",
            "random_logs_used": False,
            "sequence_used": False,
            "aggregation_backend": (
                "streaming_daily" if stage == "daily" else "sqlite_disk_backed_two_pass"
            ),
            "source_manifest_sha256": os.environ.get("SOURCE_MANIFEST_SHA256"),
        },
        "source_files": {
            "standard": [str(path) for path in standard_files],
            "random_ignored": sorted(str(path) for path in data_dir.glob("log_random*.csv")),
            "user_features": [str(path) for path in user_files],
            "video_basic": [str(path) for path in video_basic_files],
            "video_statistic": [str(path) for path in statistic_files],
            "standard_headers": merged["headers"],
        },
        "split": split_report,
        "split_sha256": split_sha256,
        "expected_split_verified": expected_split is not None,
        "daily": _daily_report(merged),
        "split_tab_diagnostics": {
            name: _split_tab_report(merged, split[name])
            for name in ("train", "val", "test")
        },
        "label_summary_all_dates": {
            label: {
                "positives": int(sum(merged["day_labels"][label].values())),
                "rate": int(sum(merged["day_labels"][label].values())) / max(daily_rows, 1),
            }
            for label in AUDIT_LABELS
        },
        "tab_all_dates": {
            str(tab): {
                "rows": int(rows),
                "label_positives": {
                    label: int(merged["tab_labels"][label][tab]) for label in AUDIT_LABELS
                },
                "label_rates": {
                    label: int(merged["tab_labels"][label][tab]) / max(int(rows), 1)
                    for label in AUDIT_LABELS
                },
                "label_comparisons": {
                    name: _comparison_report(merged["tab_comparisons"][name][tab])
                    for name, _, _ in LABEL_COMPARISONS
                },
            }
            for tab, rows in merged["tab_rows"].items()
        },
        "fields": {},
        "entities": {},
        "side_features": {
            "user_files": [_file_metadata(path, leakage_risk=False) for path in user_files],
            "video_basic_files": [
                _file_metadata(path, leakage_risk=False) for path in video_basic_files
            ],
            "video_statistic_files": [
                _file_metadata(path, leakage_risk=True) for path in statistic_files
            ],
            "video_statistic_primary_model_eligible": False,
            "time_semantics_review_required": True,
        },
        "invariants": {
            "daily_rows_sum": daily_rows,
            "train_rows_expected": train_rows_expected,
            "test_rows_expected": test_rows_expected,
            "test_label_positives_expected": test_labels_expected,
            "selected_train_test_rows_expected": selected_rows,
        },
        "execution": {"daily_rows_scanned": daily_rows},
    }
    if stage == "daily":
        return report

    work_dir = work_dir or data_dir / ".kuairand27k_k0_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    disk_before = shutil.disk_usage(work_dir)
    min_free_bytes = int(min_free_gb * 1024**3)
    if disk_before.free < min_free_bytes:
        raise RuntimeError(
            "Insufficient temporary disk before SQLite aggregation: "
            f"free={disk_before.free} required={min_free_bytes}"
        )
    database_path = work_dir / "kuairand27k_k0.sqlite"
    if database_path.exists():
        database_path.unlink()
    conn = sqlite3.connect(database_path)
    _configure_database(conn)
    interaction_audit = _accumulate_interactions(
        conn, standard_files, split, flush_unique=max(1, flush_unique)
    )
    if interaction_audit["processed_rows"] != daily_rows:
        raise RuntimeError(
            "Two-pass row mismatch: "
            f"daily={daily_rows}, interactions={interaction_audit['processed_rows']}"
        )
    if interaction_audit["train_test_rows"] != selected_rows:
        raise RuntimeError(
            "Selected-row mismatch: "
            f"expected={selected_rows}, interactions={interaction_audit['train_test_rows']}"
        )

    fields = {
        "user_id": summarize_aggregate_rows(
            conn.execute(_aggregate_query("user_agg", "user_id")), field="user_id"
        ),
        "video_id": summarize_aggregate_rows(
            conn.execute(_aggregate_query("video_agg", "video_id")), field="video_id"
        ),
        "tab": summarize_aggregate_rows(
            ((key, *values) for key, values in interaction_audit["tab_cache"].items()),
            field="tab",
        ),
    }
    for entity in ("user_id", "video_id"):
        summary = fields[entity]
        if summary["train_occurrences"] != train_rows_expected:
            raise RuntimeError(
                f"{entity} train rows mismatch: "
                f"expected={train_rows_expected}, actual={summary['train_occurrences']}"
            )
        if summary["test_rows"] != test_rows_expected:
            raise RuntimeError(
                f"{entity} test rows mismatch: "
                f"expected={test_rows_expected}, actual={summary['test_rows']}"
            )
        if summary["test_label_positives"] != test_labels_expected:
            raise RuntimeError(
                f"{entity} test labels mismatch: "
                f"expected={test_labels_expected}, actual={summary['test_label_positives']}"
            )
    report["fields"] = fields
    report["entities"] = {"user_id": fields["user_id"], "video_id": fields["video_id"]}
    report["side_features"].update({
        "interaction_unique_users_train": fields["user_id"]["train_unique_keys"],
        "interaction_unique_videos_train": fields["video_id"]["train_unique_keys"],
    })
    report["invariants"].update({
        "second_pass_rows": interaction_audit["processed_rows"],
        "second_pass_matches_daily": True,
        "selected_train_test_rows_actual": interaction_audit["train_test_rows"],
        "selected_rows_match": True,
        "entity_row_totals_match": True,
        "entity_test_label_totals_match": True,
    })
    report["execution"].update({
        "processed_rows": interaction_audit["processed_rows"],
        "train_test_rows": interaction_audit["train_test_rows"],
        "database_path": str(database_path),
        "flush_unique_limit": flush_unique,
        "flush_events": interaction_audit["flush_events"],
        "flushed_unique_rows": interaction_audit["flushed_unique_rows"],
        "minimum_free_disk_bytes_required": min_free_bytes,
        "free_disk_bytes_before_aggregation": disk_before.free,
        "peak_database_bytes_interactions": interaction_audit["peak_database_bytes"],
        "database_bytes_after_interactions": _sqlite_bytes(conn),
    })
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
    report["invariants"]["sqlite_integrity_check"] = integrity
    if stage == "interactions":
        conn.close()
        disk_after = shutil.disk_usage(work_dir)
        report["execution"].update({
            "database_bytes_final": database_path.stat().st_size,
            "free_disk_bytes_after_aggregation": disk_after.free,
        })
        return report

    user_reports = _load_user_features(conn, user_files)
    for user_file_report in user_reports:
        if "table" not in user_file_report:
            continue
        for field in user_file_report["fields"]:
            summary = summarize_aggregate_rows(
                _side_field_rows(
                    conn,
                    user_file_report["table"],
                    "user_id",
                    "user_agg",
                    "user_id",
                    field,
                ),
                field=f"user__{field}",
            )
            fields[f"user__{field}"] = _attach_presence(
                summary,
                _side_presence(
                    conn,
                    user_file_report["table"],
                    "user_id",
                    "user_agg",
                    "user_id",
                    field,
                ),
                fields["user_id"],
            )

    video_basic_reports = _load_video_basic(conn, video_basic_files)
    for field in VIDEO_CATEGORICAL_FIELDS:
        summary = summarize_aggregate_rows(
            _side_field_rows(conn, "video_basic", "video_id", "video_agg", "video_id", field),
            field=f"video__{field}",
        )
        fields[f"video__{field}"] = _attach_presence(
            summary,
            _side_presence(conn, "video_basic", "video_id", "video_agg", "video_id", field),
            fields["video_id"],
        )
    tag_summary = summarize_aggregate_rows(
        _side_field_rows(conn, "video_tag", "video_id", "video_agg", "video_id", "tag"),
        field="video__tag",
        multi_valued=True,
    )
    fields["video__tag"] = _attach_presence(
        tag_summary,
        _side_presence(conn, "video_tag", "video_id", "video_agg", "video_id", "tag"),
        fields["video_id"],
    )

    statistic_reports = [_file_metadata(path, leakage_risk=True) for path in statistic_files]
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Final SQLite integrity_check failed: {integrity}")
    report["invariants"]["sqlite_integrity_check_after_side_features"] = integrity
    conn.close()
    report["side_features"].update({
        "user_files": user_reports,
        "video_basic_files": video_basic_reports,
        "video_statistic_files": statistic_reports,
    })
    disk_after = shutil.disk_usage(work_dir)
    report["execution"].update({
        "database_bytes_final": database_path.stat().st_size,
        "free_disk_bytes_after_aggregation": disk_after.free,
    })
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--completeness-ratio", type=float, default=0.5)
    parser.add_argument("--flush-unique", type=int, default=1_000_000)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument(
        "--stage", choices=("schema", "daily", "interactions", "full"), default="full"
    )
    parser.add_argument("--expected-split-report", type=Path)
    return parser.parse_args()


def _console_summary(report: dict) -> dict:
    return report.get("split", report["protocol"])


def main() -> None:
    args = parse_args()
    expected_split = None
    if args.expected_split_report:
        expected_split = json.loads(args.expected_split_report.read_text())["split"]
    report = audit_dataset(
        args.data_dir,
        workers=args.workers,
        test_days=args.test_days,
        completeness_ratio=args.completeness_ratio,
        work_dir=args.work_dir,
        flush_unique=args.flush_unique,
        stage=args.stage,
        expected_split=expected_split,
        min_free_gb=args.min_free_gb,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
