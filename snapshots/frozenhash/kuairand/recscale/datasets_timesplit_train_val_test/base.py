"""Shared helpers for deterministic day-level temporal splits.

The helpers in this module are read-only with respect to source datasets. They
convert timestamps to Asia/Shanghai dates, assign last-day/penultimate-day
splits, and accumulate split-report statistics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Iterable, Mapping

SECONDS_PER_DAY = 86_400
MILLISECONDS_PER_DAY = 86_400_000
DEFAULT_TZ = "Asia/Shanghai"


def timestamp_to_date(value, unit: str = "s", tz: str = DEFAULT_TZ) -> str:
    """Convert a unix timestamp to YYYY-MM-DD in the requested timezone.

    Args:
        value: int/float/string timestamp.
        unit: "s" for seconds or "ms" for milliseconds.
        tz: IANA timezone, default Asia/Shanghai.
    """
    if value is None or value == "":
        raise ValueError("empty timestamp")
    ts = float(value)
    if unit == "ms":
        ts /= 1000.0
    elif unit != "s":
        raise ValueError(f"unsupported timestamp unit: {unit}")
    return datetime.fromtimestamp(ts, tz=ZoneInfo(tz)).date().isoformat()


def infer_splits_from_dates(dates: Iterable[str]) -> dict[str, set[str]]:
    """Return train/val/test date sets under penultimate/last-day rule."""
    unique_dates = sorted(set(dates))
    if len(unique_dates) < 3:
        raise ValueError(
            f"need at least 3 distinct dates for train/val/test, got {unique_dates}"
        )
    return {
        "train": set(unique_dates[:-2]),
        "val": {unique_dates[-2]},
        "test": {unique_dates[-1]},
    }


def infer_splits_from_day_stats(
    stats: Mapping[str, DayStats],
    min_samples: int = 0,
    min_users: int = 0,
) -> dict[str, set[str]]:
    """Return train/val/test dates after filtering eligible evaluation days.

    Dates after the selected test date are excluded entirely because including
    them in train would leak future interactions. Dates before val/test remain
    in train, even if they fail the eval-day thresholds. This is useful for
    datasets such as KuaiRec whose literal last calendar days are tiny tail
    artifacts.
    """
    all_dates = sorted(stats)
    eligible = [
        d for d in all_dates
        if stats[d].samples >= min_samples and len(stats[d].users) >= min_users
    ]
    if len(eligible) < 3:
        raise ValueError(
            f"need at least 3 eligible dates, got {eligible}; "
            f"thresholds min_samples={min_samples}, min_users={min_users}"
        )
    val_date, test_date = eligible[-2], eligible[-1]
    return {
        "train": {d for d in all_dates if d < val_date},
        "val": {val_date},
        "test": {test_date},
    }


def split_for_date(date: str, split_dates: Mapping[str, set[str]]) -> str:
    for split, date_set in split_dates.items():
        if date in date_set:
            return split
    raise KeyError(f"date {date} not covered by split dates")


@dataclass
class DayStats:
    samples: int = 0
    positives: int = 0
    users: set = field(default_factory=set)
    items: set = field(default_factory=set)

    def add(self, user, item, label: int) -> None:
        self.samples += 1
        self.positives += int(label)
        self.users.add(user)
        self.items.add(item)

    @property
    def label_rate(self) -> float:
        return self.positives / self.samples if self.samples else 0.0


def new_day_stats() -> defaultdict[str, DayStats]:
    return defaultdict(DayStats)


def render_day_distribution(title: str, stats: Mapping[str, DayStats]) -> str:
    lines = [f"# {title}", "", "| date | samples | positives | label_rate | users | items |", "|---|---:|---:|---:|---:|---:|"]
    for date in sorted(stats):
        st = stats[date]
        lines.append(
            f"| {date} | {st.samples} | {st.positives} | {st.label_rate:.6f} | {len(st.users)} | {len(st.items)} |"
        )
    return "\n".join(lines) + "\n"


def render_split_summary(stats: Mapping[str, DayStats], split_dates: Mapping[str, set[str]]) -> str:
    lines = ["## Proposed split summary", "", "| split | dates | samples | positives | label_rate | users | items |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        merged = DayStats()
        for date in sorted(split_dates[split]):
            st = stats[date]
            merged.samples += st.samples
            merged.positives += st.positives
            merged.users.update(st.users)
            merged.items.update(st.items)
        date_text = f"{min(split_dates[split])}..{max(split_dates[split])}" if len(split_dates[split]) > 1 else next(iter(split_dates[split]))
        lines.append(
            f"| {split} | {date_text} | {merged.samples} | {merged.positives} | {merged.label_rate:.6f} | {len(merged.users)} | {len(merged.items)} |"
        )
    return "\n".join(lines) + "\n"
