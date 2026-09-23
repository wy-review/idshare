"""Absolute-AUC readout for the pre-registered KuaiRand K1 segments."""

from __future__ import annotations

import numpy as np


VIDEO_BAND = {
    "unseen": 0,
    "1-5": 1,
    "6-10": 2,
    "11-20": 3,
    "21-50": 4,
    ">50": 5,
}
AUTHOR_SEEN_BIT = np.uint8(1)
VALID_MUSIC_SEEN_BIT = np.uint8(2)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC from average ranks, including exact tie handling."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    return float(
        (ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def _segment(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict:
    selected_labels = labels[mask]
    selected_scores = scores[mask]
    rows = int(mask.sum())
    positives = int(selected_labels.sum())
    negatives = rows - positives
    auc = None
    if positives > 0 and negatives > 0:
        auc = _binary_auc(selected_labels, selected_scores)
    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "auc": auc,
    }


def evaluate_kuairand_k1_segments(
    labels,
    scores,
    video_train_band,
    side_bits,
) -> dict:
    """Evaluate the frozen K1 segments using absolute AUC.

    ``video_train_band`` is a mutually exclusive code based only on training
    occurrence count. ``side_bits`` uses bit0=author train-seen and
    bit1=valid-music train-seen; missing and reserved-zero music never set bit1.
    """

    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    bands = np.asarray(video_train_band, dtype=np.uint8)
    bits = np.asarray(side_bits, dtype=np.uint8)
    if not (labels.shape == scores.shape == bands.shape == bits.shape):
        raise ValueError(
            "K1 segmented evaluator inputs must have identical one-dimensional shapes"
        )
    if labels.ndim != 1:
        raise ValueError("K1 segmented evaluator inputs must be one-dimensional")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("K1 labels must be binary")
    if np.any(bands > VIDEO_BAND[">50"]):
        raise ValueError("K1 video train band contains an unknown code")
    if np.any(bits & np.uint8(0xFC)):
        raise ValueError("K1 side bits contain unknown flags")

    all_rows = np.ones(labels.shape, dtype=bool)
    unseen = bands == VIDEO_BAND["unseen"]
    rare_1_5 = bands == VIDEO_BAND["1-5"]
    gt5 = bands >= VIDEO_BAND["6-10"]
    author_seen = (bits & AUTHOR_SEEN_BIT) != 0
    music_seen = (bits & VALID_MUSIC_SEEN_BIT) != 0
    side_available = author_seen | music_seen

    segments = {
        "overall": _segment(labels, scores, all_rows),
        "video_unseen_side_available": _segment(
            labels, scores, unseen & side_available
        ),
        "video_unseen_side_unavailable": _segment(
            labels, scores, unseen & ~side_available
        ),
        "video_seen_rare_1_5_side_available": _segment(
            labels, scores, rare_1_5 & side_available
        ),
        "video_seen_rare_1_5_side_unavailable": _segment(
            labels, scores, rare_1_5 & ~side_available
        ),
        "video_train_count_gt5": _segment(labels, scores, gt5),
    }

    nested_diagnostics = {}
    for cutoff, max_band in ((10, 2), (20, 3), (50, 4)):
        mask = (bands >= VIDEO_BAND["1-5"]) & (bands <= max_band)
        nested_diagnostics[f"video_seen_rare_1_{cutoff}"] = _segment(
            labels, scores, mask
        )

    unseen_side_diagnostics = {
        "author_seen": _segment(labels, scores, unseen & author_seen),
        "valid_music_seen": _segment(labels, scores, unseen & music_seen),
        "author_and_music_seen": _segment(
            labels, scores, unseen & author_seen & music_seen
        ),
        "author_only": _segment(
            labels, scores, unseen & author_seen & ~music_seen
        ),
        "music_only": _segment(
            labels, scores, unseen & ~author_seen & music_seen
        ),
        "neither": _segment(labels, scores, unseen & ~author_seen & ~music_seen),
    }

    checks = {
        "primary_segments_close": (
            segments["video_unseen_side_available"]["rows"]
            + segments["video_unseen_side_unavailable"]["rows"]
            == int(unseen.sum())
        ),
        "unseen_joint_closes": (
            sum(
                unseen_side_diagnostics[name]["rows"]
                for name in (
                    "author_and_music_seen",
                    "author_only",
                    "music_only",
                    "neither",
                )
            )
            == int(unseen.sum())
        ),
        "seen_rare_1_5_closes": (
            segments["video_seen_rare_1_5_side_available"]["rows"]
            + segments["video_seen_rare_1_5_side_unavailable"]["rows"]
            == int(rare_1_5.sum())
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"K1 segmented evaluation does not close: {checks}")

    return {
        "definitions": {
            "video_train_band": dict(VIDEO_BAND),
            "side_bits": {
                "author_train_seen": int(AUTHOR_SEEN_BIT),
                "valid_music_train_seen": int(VALID_MUSIC_SEEN_BIT),
            },
            "selection_policy": "report_only_no_model_or_cutoff_selection",
        },
        "segments": segments,
        "nested_seen_rare_diagnostics": nested_diagnostics,
        "unseen_side_diagnostics": unseen_side_diagnostics,
        "checks": checks,
    }


def evaluate_kuairand_k1_exact_counts(
    labels,
    scores,
    video_ids,
    video_train_counts,
    author_train_counts,
    music_train_counts,
    *,
    tail_cutoff: int = 5,
    total_test_rows: int | None = None,
) -> dict:
    """Split the frozen seen-rare segment into exact train-count buckets.

    Counts must come from the frozen train-only frequency cache. Special,
    missing, reserved-zero, and OOV identities have count zero and therefore
    never enter a seen-rare bucket.
    """

    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    video_ids = np.asarray(video_ids, dtype=np.int64)
    video_counts = np.asarray(video_train_counts, dtype=np.int64)
    author_counts = np.asarray(author_train_counts, dtype=np.int64)
    music_counts = np.asarray(music_train_counts, dtype=np.int64)
    arrays = (
        labels,
        scores,
        video_ids,
        video_counts,
        author_counts,
        music_counts,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("K1 exact-count inputs must be one-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("K1 exact-count inputs must have identical shapes")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("K1 exact-count labels must be binary")
    if any(np.any(counts < 0) for counts in (video_counts, author_counts, music_counts)):
        raise ValueError("K1 exact-count train counts must be non-negative")
    if tail_cutoff < 1:
        raise ValueError("K1 exact-count tail_cutoff must be positive")
    if total_test_rows is None:
        total_test_rows = len(labels)
    total_test_rows = int(total_test_rows)
    if total_test_rows < len(labels):
        raise ValueError("K1 exact-count total_test_rows cannot be smaller than inputs")

    rare = (video_counts > 0) & (video_counts <= tail_cutoff)
    author_tail = (author_counts > 0) & (author_counts <= tail_cutoff)
    music_tail = (music_counts > 0) & (music_counts <= tail_cutoff)
    exact = {}
    for count in range(1, tail_cutoff + 1):
        mask = video_counts == count
        rows = int(mask.sum())
        distinct_videos = int(np.unique(video_ids[mask]).size)
        side_masks = {
            "author_tail": mask & author_tail,
            "music_tail": mask & music_tail,
            "any_side_tail": mask & (author_tail | music_tail),
            "both_side_tail": mask & author_tail & music_tail,
            "neither_side_tail": mask & ~(author_tail | music_tail),
        }
        exact[str(count)] = {
            **_segment(labels, scores, mask),
            "distinct_videos": distinct_videos,
            "row_share_of_test": (
                float(rows / total_test_rows) if total_test_rows else 0.0
            ),
            "row_share_of_seen_rare_1_5": (
                float(rows / rare.sum()) if rare.any() else 0.0
            ),
            "distinct_video_share_of_seen_rare_1_5": None,
            "side_tail_routing": {
                name: {
                    "rows": int(side_mask.sum()),
                    "row_share": float(side_mask.sum() / rows) if rows else 0.0,
                }
                for name, side_mask in side_masks.items()
            },
        }

    rare_distinct = int(np.unique(video_ids[rare]).size)
    for bucket in exact.values():
        bucket["distinct_video_share_of_seen_rare_1_5"] = (
            float(bucket["distinct_videos"] / rare_distinct)
            if rare_distinct
            else 0.0
        )

    rare_segment = _segment(labels, scores, rare)
    checks = {
        "exact_rows_close": (
            sum(bucket["rows"] for bucket in exact.values())
            == rare_segment["rows"]
        ),
        "exact_positives_close": (
            sum(bucket["positives"] for bucket in exact.values())
            == rare_segment["positives"]
        ),
        "exact_negatives_close": (
            sum(bucket["negatives"] for bucket in exact.values())
            == rare_segment["negatives"]
        ),
        "exact_distinct_videos_close": (
            sum(bucket["distinct_videos"] for bucket in exact.values())
            == rare_distinct
        ),
        "side_routing_closes_per_bucket": all(
            bucket["side_tail_routing"]["both_side_tail"]["rows"]
            + (
                bucket["side_tail_routing"]["author_tail"]["rows"]
                - bucket["side_tail_routing"]["both_side_tail"]["rows"]
            )
            + (
                bucket["side_tail_routing"]["music_tail"]["rows"]
                - bucket["side_tail_routing"]["both_side_tail"]["rows"]
            )
            + bucket["side_tail_routing"]["neither_side_tail"]["rows"]
            == bucket["rows"]
            for bucket in exact.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"K1 exact-count evaluation does not close: {checks}")

    return {
        "definitions": {
            "video_bucket": "exact_train_occurrence_count",
            "tail_cutoff": int(tail_cutoff),
            "frequency_source": "frozen_train_only_frequency_cache",
            "selection_policy": "diagnostic_only_no_cutoff_or_model_selection",
        },
        "seen_rare_1_5": {
            **rare_segment,
            "distinct_videos": rare_distinct,
        },
        "exact_counts": exact,
        "checks": checks,
    }
