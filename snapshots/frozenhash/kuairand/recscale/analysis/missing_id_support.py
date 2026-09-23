"""Audit natural ID=0 support from an existing field-frequency cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _default_field_name(field_index: int) -> str:
    if field_index < 26:
        return f"C{field_index + 1}"
    return f"I{field_index - 25}_bucket"


def summarize_missing_id_support(
    cache_path: str | os.PathLike[str],
    *,
    target_fields: list[int] | None = None,
) -> dict:
    cache_path = str(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        days = [int(day) for day in str(data["days"].item()).split(",") if day]
        available_fields = data["fields"].astype(np.int64).tolist()
        fields = available_fields if target_fields is None else [int(idx) for idx in target_fields]
        missing_fields = sorted(set(fields) - set(available_fields))
        if missing_fields:
            raise ValueError(
                f"target fields {missing_fields} are absent from cache fields={available_fields}"
            )

        summaries = []
        sample_totals = []
        for field_index in fields:
            counts = data[f"counts_{field_index}"].astype(np.uint64)
            if counts.ndim != 1 or counts.size == 0:
                raise ValueError(f"counts_{field_index} must be a non-empty 1D array")
            total = int(counts.sum(dtype=np.uint64))
            missing = int(counts[0])
            sample_totals.append(total)
            summaries.append(
                {
                    "field_index": field_index,
                    "field_name": _default_field_name(field_index),
                    "samples_seen": total,
                    "missing_id_count": missing,
                    "missing_id_rate": (missing / total) if total else 0.0,
                    "observed_id_count": total - missing,
                }
            )

    unique_totals = sorted(set(sample_totals))
    return {
        "analysis": "natural_missing_id_support",
        "cache_path": cache_path,
        "train_days": days,
        "target_fields": fields,
        "samples_seen": unique_totals[0] if len(unique_totals) == 1 else None,
        "sample_totals_consistent": len(unique_totals) == 1,
        "fields": summaries,
    }


def _parse_fields(raw: str) -> list[int] | None:
    if not raw:
        return None
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--fields", default="")
    parser.add_argument(
        "--output",
        default=os.path.join(os.environ.get("JOB_OUTPUT_DIR", "."), "training_report.json"),
    )
    args = parser.parse_args()

    report = summarize_missing_id_support(
        args.cache_path,
        target_fields=_parse_fields(args.fields),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
