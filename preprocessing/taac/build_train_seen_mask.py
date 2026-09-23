#!/usr/bin/env python3
"""Build a train-only target+sequence identity support mask for TAAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path


PROTOCOL = "taac_train_input_seen_mask_r01"
FINGERPRINT_ALGORITHM = "little-endian-int64-user-target-plus-int32-cutoff-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol: dict) -> None:
    if protocol.get("protocol") != PROTOCOL:
        raise RuntimeError("protocol name mismatch")
    if protocol.get("allowed_splits") != ["train", "val"]:
        raise RuntimeError("only train and validation are allowed")
    if protocol.get("uses_test_dataset") is not False or protocol.get("uses_test_labels") is not False:
        raise RuntimeError("test/holdout boundary must remain false")
    if protocol.get("overwrite_existing_cache") is not False:
        raise RuntimeError("cache overwrite must remain disabled")
    if int(protocol.get("real_id_offset", -1)) != 3:
        raise RuntimeError("real item offset must be 3")
    if sorted(int(value) for value in protocol["special_token_contract"].values()) != [0, 1, 2, 3]:
        raise RuntimeError("special token IDs must be contiguous 0..3")


def no_data_report(protocol: dict) -> dict:
    validate_protocol(protocol)
    return {
        "status": "passed",
        "protocol": PROTOCOL,
        "mode": "remote_no_data_regression",
        "datasets_loaded": [],
        "uses_test_dataset": False,
        "uses_test_labels": False,
        "test_or_holdout_read": False,
        "all_execution_checks_pass": True,
    }


def _dataset(path: Path):
    try:
        import pyarrow.dataset as pds
    except ImportError as exc:  # pragma: no cover - remote dependency gate
        raise RuntimeError("pyarrow is required on the RTP worker") from exc
    return pds.dataset(path, format="parquet")


def _scan_train_samples(dataset, *, seen, max_cutoff, expected_rows: int) -> dict:
    import numpy as np

    digest = hashlib.sha256(FINGERPRINT_ALGORITHM.encode("ascii"))
    rows = 0
    target_seen_before_sequences = np.zeros_like(seen)
    for batch in dataset.to_batches(
        columns=["user_id", "target_item_id", "seq_cutoff_pos"],
        batch_size=1_048_576,
    ):
        payload = batch.to_pydict()
        users = np.asarray(payload["user_id"], dtype=np.int64)
        targets = np.asarray(payload["target_item_id"], dtype=np.int64)
        cutoffs = np.asarray(payload["seq_cutoff_pos"], dtype=np.int32)
        if not (users.size == targets.size == cutoffs.size):
            raise RuntimeError("train sample column length mismatch")
        if users.size and (
            int(users.min()) < 0
            or int(users.max()) >= max_cutoff.size
            or int(targets.min()) <= 0
            or int(targets.max()) >= seen.size
            or int(cutoffs.min()) < 0
        ):
            raise RuntimeError("train sample identity/cutoff outside frozen bounds")
        rows += int(users.size)
        unique_targets = np.unique(targets)
        seen[unique_targets] = True
        target_seen_before_sequences[unique_targets] = True
        np.maximum.at(max_cutoff, users, cutoffs)
        digest.update(users.astype("<i8", copy=False).tobytes())
        digest.update(targets.astype("<i8", copy=False).tobytes())
        digest.update(cutoffs.astype("<i4", copy=False).tobytes())
    if rows != expected_rows:
        raise RuntimeError(f"train row mismatch: expected={expected_rows}, observed={rows}")
    return {
        "rows": rows,
        "target_seen_ids": int(np.count_nonzero(target_seen_before_sequences)),
        "users_with_consumed_history": int(np.count_nonzero(max_cutoff)),
        "consumed_columns_sha256": digest.hexdigest(),
    }


def _sequence_item_array(value):
    import numpy as np

    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    if not value:
        return np.zeros(0, dtype=np.int64)
    if isinstance(value[0], dict):
        return np.fromiter(
            (int(item.get("item_id", 0)) for item in value),
            dtype=np.int64,
            count=len(value),
        )
    return np.asarray(value, dtype=np.int64)


def _add_consumed_sequence_prefixes(*, sequence_path: Path, seen, max_cutoff) -> dict:
    import numpy as np

    with sequence_path.open("rb") as handle:
        sequences = pickle.load(handle)
    if not isinstance(sequences, dict):
        raise RuntimeError("user sequence cache is not a mapping")
    users = np.flatnonzero(max_cutoff)
    consumed_occurrences = 0
    missing_users = 0
    short_sequences = 0
    for user_id in users.tolist():
        value = sequences.get(int(user_id))
        if value is None:
            missing_users += 1
            continue
        items = _sequence_item_array(value)
        cutoff = int(max_cutoff[user_id])
        if cutoff > items.size:
            short_sequences += 1
            continue
        prefix = items[:cutoff]
        if prefix.size:
            if int(prefix.min()) < 0 or int(prefix.max()) >= seen.size:
                raise RuntimeError("sequence item outside frozen indexer range")
            positive = prefix[prefix > 0]
            if positive.size:
                seen[np.unique(positive)] = True
                consumed_occurrences += int(positive.size)
    if missing_users or short_sequences:
        raise RuntimeError(
            "training sequence support is incomplete: "
            f"missing_users={missing_users}, short_sequences={short_sequences}"
        )
    return {
        "sequence_cache_entries": len(sequences),
        "users_scanned": int(users.size),
        "consumed_positive_sequence_occurrences": consumed_occurrences,
        "missing_users": missing_users,
        "short_sequences": short_sequences,
    }


def _scan_validation_targets(dataset, *, seen, expected_rows: int) -> dict:
    import numpy as np

    rows = 0
    unseen = 0
    digest = hashlib.sha256(b"little-endian-int64-validation-target-v1")
    for batch in dataset.to_batches(columns=["target_item_id"], batch_size=1_048_576):
        target = np.asarray(batch.to_pydict()["target_item_id"], dtype=np.int64)
        if target.size and (int(target.min()) <= 0 or int(target.max()) >= seen.size):
            raise RuntimeError("validation target outside frozen indexer range")
        rows += int(target.size)
        unseen += int(np.count_nonzero(~seen[target]))
        digest.update(target.astype("<i8", copy=False).tobytes())
    if rows != expected_rows:
        raise RuntimeError(f"validation row mismatch: expected={expected_rows}, observed={rows}")
    return {
        "rows": rows,
        "train_input_unseen_target_occurrences": unseen,
        "train_input_unseen_target_fraction": unseen / max(rows, 1),
        "target_sha256": digest.hexdigest(),
    }


def run_build(protocol: dict) -> dict:
    validate_protocol(protocol)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required on the RTP worker") from exc

    data_root = Path(protocol["data_root"])
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    indexer_path = data_root / "indexer.pkl"
    with indexer_path.open("rb") as handle:
        indexer = pickle.load(handle)
    user_values = [int(value) for value in indexer["u"].values()]
    item_values = [int(value) for value in indexer["i"].values()]
    max_user = max(user_values)
    max_item = max(item_values)
    seen = np.zeros(max_item + 1, dtype=np.bool_)
    max_cutoff = np.zeros(max_user + 1, dtype=np.int32)

    train_dataset = _dataset(data_root / "train/samples.parquet")
    validation_dataset = _dataset(data_root / "val/samples.parquet")
    train = _scan_train_samples(
        train_dataset,
        seen=seen,
        max_cutoff=max_cutoff,
        expected_rows=int(protocol["expected_rows"]["train"]),
    )
    sequences = _add_consumed_sequence_prefixes(
        sequence_path=data_root / "user_seqs_numpy/user_seqs.pkl",
        seen=seen,
        max_cutoff=max_cutoff,
    )
    validation = _scan_validation_targets(
        validation_dataset,
        seen=seen,
        expected_rows=int(protocol["expected_rows"]["val"]),
    )
    seen_count = int(np.count_nonzero(seen))
    cache_path = Path(protocol["cache_output"])
    if cache_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen cache candidate: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, seen, allow_pickle=False)
    os.replace(temporary, cache_path)
    cache_sha = sha256_file(cache_path)

    return {
        "status": "completed",
        "protocol": PROTOCOL,
        "mode": "train_input_seen_mask_build",
        "datasets_loaded": [
            "taac_train",
            "taac_validation",
            "taac_shared_sequence_train_prefixes",
        ],
        "uses_test_dataset": False,
        "uses_test_labels": False,
        "test_or_holdout_read": False,
        "sequence_enabled": True,
        "indexer": {
            "max_user_id": max_user,
            "max_item_id": max_item,
            "item_mapping_size": len(item_values),
        },
        "train": train,
        "sequences": sequences,
        "validation": validation,
        "seen_mask": {
            "path": str(cache_path),
            "sha256": cache_sha,
            "bytes": cache_path.stat().st_size,
            "length": int(seen.size),
            "train_input_seen_ids": seen_count,
            "real_id_offset": int(protocol["real_id_offset"]),
            "first_private_id": 4,
        },
        "decision": "TRAIN_INPUT_SEEN_MASK_BUILT",
        "capacity_training_authorized": False,
        "manuscript_update_authorized": False,
        "all_execution_checks_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--no-data-regression", action="store_true")
    args = parser.parse_args()
    protocol_sha = sha256_file(args.protocol)
    if protocol_sha != args.expected_protocol_sha256:
        raise RuntimeError("protocol SHA256 mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = no_data_report(protocol) if args.no_data_regression else run_build(protocol)
    report.update(
        {
            "protocol_sha256": protocol_sha,
            "package_sha256": args.expected_package_sha256,
            "source_manifest_sha256": args.expected_source_manifest_sha256,
            "freeze_sha256": args.expected_freeze_sha256,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": report["mode"],
                "datasets_loaded": report["datasets_loaded"],
                "uses_test_dataset": False,
                "uses_test_labels": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
