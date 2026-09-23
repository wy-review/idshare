"""
recscale.datasets.criteo_1tb — Criteo 1TB Click Logs Dataset

数据来源: https://huggingface.co/datasets/criteo/CriteoClickLogs
格式: tab 分隔，40列 (1 label + 13 int dense + 26 hex sparse)，无 header
文件: day_2.gz ~ day_23.gz (共 22 个文件，每文件 ~2000 万行，总计 ~4.4 亿行)
      或已解压版本: day_2.tsv ~ day_23.tsv（读取速度快约 3x）

注意: gzip 文件末尾 stream 不完整 (unexpected end of file)，
      这是 HuggingFace 源文件本身的问题，数据内容完整。
      读取时需忽略末尾错误，本 adapter 已内置处理。

特征说明:
  col 0:    label (0/1)
  col 1-13: dense integer features (I1-I13)，含大量空值
  col 14-39: sparse hex features (C1-C26)，8位十六进制字符串，含空值

切分方式:
  train: day_2 ~ day_22 (21 天)
  test:  day_23 (1 天)
  或自定义 train_days / test_days

内存策略: IterableDataset，流式逐行 yield，无需预分配全量内存
  - Pass-1 (train only): 扫描计数 + reservoir sampling 构建分桶边界
  - 训练/评估时按需流式读取，DataLoader 的 num_workers 控制并行度

文件查找顺序（自动）:
  1. day_N.tsv  （解压版，快 ~3x，推荐）
  2. day_N.gz   （原始压缩版，回退）

YAML 示例:
```yaml
dataset:
  type: criteo_1tb
  path: /data/criteo/Criteo_1TB_tsv  # 解压版目录（推荐）
  # 或 path: /data/criteo/Criteo_1TB  # 原始 gz 目录
  train_days: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
  test_days: [23]
  bucketize_dense: true      # 强烈推荐，分桶后所有模型均提升
  num_buckets: 100
  bucket_method: log         # log / quantile / uniform
  max_rows: 0                # 调试用，限制总行数
```
"""

import gzip
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import IterableDataset

from . import register_dataset


# 列定义
DENSE_COLS  = [f"I{i}" for i in range(1, 14)]   # I1 ~ I13
SPARSE_COLS = [f"C{i}" for i in range(1, 27)]   # C1 ~ C26

# 全量 Criteo 1TB sparse 特征基数（在完整 4.4 亿行上统计，来自 DLRM/Meta 实现）
_DEFAULT_CARDINALITIES = [
    1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
    8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18,
    15, 286181, 105, 142572,
]


def _find_day_file(data_path: str, day: int):
    """
    查找 day_N 文件，优先返回解压的 .tsv，其次回退到 .gz。
    返回 (filepath, is_gz) 或 (None, None)。
    """
    tsv_path = os.path.join(data_path, f"day_{day}.tsv")
    if os.path.exists(tsv_path):
        return tsv_path, False
    gz_path = os.path.join(data_path, f"day_{day}.gz")
    if os.path.exists(gz_path):
        return gz_path, True
    return None, None


def _open_file(path: str, is_gz: bool):
    """返回文本行迭代器，支持 tsv（直接 open）和 gz（gzip.open）。"""
    if is_gz:
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                yield from f
        except (EOFError, OSError):
            pass
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from f


def _cache_days_key(days) -> str:
    return ",".join(str(int(d)) for d in days)


def _load_bucket_cache(cache_path: str, bucket_method: str, num_buckets: int, days):
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_method = str(data["bucket_method"].item())
            cached_buckets = int(data["num_buckets"].item())
            cached_days = str(data["days"].item())
            if (
                cached_method != bucket_method
                or cached_buckets != int(num_buckets)
                or cached_days != _cache_days_key(days)
            ):
                return None
            boundaries = {}
            for i in range(13):
                boundaries[i] = {
                    "method": bucket_method,
                    "edges": data[f"edges_{i}"].astype(np.float32),
                }
            n_total = int(data["n_total"].item())
            return n_total, boundaries
    except Exception as exc:
        print(f"[Criteo1TB] WARNING: failed to load bucket cache {cache_path}: {exc}")
        return None


def _save_bucket_cache(
    cache_path: str,
    boundaries: dict,
    n_total: int,
    bucket_method: str,
    num_buckets: int,
    days,
) -> None:
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}.npz"
    payload = {
        "n_total": np.array(n_total, dtype=np.int64),
        "bucket_method": np.array(bucket_method),
        "num_buckets": np.array(num_buckets, dtype=np.int64),
        "days": np.array(_cache_days_key(days)),
    }
    for i in range(13):
        payload[f"edges_{i}"] = np.asarray(boundaries[i]["edges"], dtype=np.float32)
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, cache_path)
    print(f"[Criteo1TB] Saved bucket cache: {cache_path}")


def _try_acquire_cache_lock(lock_path: str) -> bool:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()} time={time.time()}\n")
        return True
    except FileExistsError:
        return False


def _wait_for_bucket_cache(
    cache_path: str,
    bucket_method: str,
    num_buckets: int,
    days,
    lock_path: str,
):
    stale_after = 6 * 60 * 60
    while True:
        loaded = _load_bucket_cache(cache_path, bucket_method, num_buckets, days)
        if loaded is not None:
            return loaded
        try:
            age = time.time() - os.path.getmtime(lock_path)
            if age > stale_after:
                print(f"[Criteo1TB] Removing stale bucket-cache lock: {lock_path}")
                os.unlink(lock_path)
                return None
        except FileNotFoundError:
            return None
        print(f"[Criteo1TB] Waiting for bucket cache: {cache_path}")
        time.sleep(30)


def _load_field_frequency_cache(
    cache_path: str,
    *,
    days,
    fields: list[int],
    cards: list[int],
):
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_days = str(data["days"].item())
            cached_fields = data["fields"].astype(np.int64).tolist()
            if cached_days != _cache_days_key(days) or cached_fields != list(fields):
                return None

            counts = {}
            for idx in fields:
                arr = data[f"counts_{idx}"].astype(np.uint32)
                if len(arr) != int(cards[idx]):
                    return None
                counts[int(idx)] = arr
            return counts
    except Exception as exc:
        print(f"[Criteo1TB] WARNING: failed to load field-frequency cache {cache_path}: {exc}")
        return None


def _save_field_frequency_cache(
    cache_path: str,
    *,
    days,
    fields: list[int],
    counts: dict[int, np.ndarray],
) -> None:
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}.npz"
    payload = {
        "days": np.array(_cache_days_key(days)),
        "fields": np.asarray(fields, dtype=np.int64),
    }
    for idx in fields:
        payload[f"counts_{idx}"] = np.asarray(counts[int(idx)], dtype=np.uint32)
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, cache_path)
    print(f"[Criteo1TB] Saved field-frequency cache: {cache_path}")


def _wait_for_field_frequency_cache(
    cache_path: str,
    *,
    days,
    fields: list[int],
    cards: list[int],
    lock_path: str,
):
    stale_after = 6 * 60 * 60
    while True:
        loaded = _load_field_frequency_cache(
            cache_path, days=days, fields=fields, cards=cards
        )
        if loaded is not None:
            return loaded
        try:
            age = time.time() - os.path.getmtime(lock_path)
            if age > stale_after:
                print(f"[Criteo1TB] Removing stale field-frequency lock: {lock_path}")
                os.unlink(lock_path)
                return None
        except FileNotFoundError:
            return None
        print(f"[Criteo1TB] Waiting for field-frequency cache: {cache_path}")
        time.sleep(30)


def _parse_line(line):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 40:
        return None
    try:
        label = int(parts[0])
    except ValueError:
        return None
    dense = []
    for v in parts[1:14]:
        try:
            dense.append(int(v) if v else 0)
        except ValueError:
            dense.append(0)
    sparse = []
    for v in parts[14:40]:
        if v:
            try:
                sparse.append(int(v, 16) & 0xFFFFFFFF)
            except ValueError:
                sparse.append(0)
        else:
            sparse.append(0)
    return label, dense, sparse


def _apply_sparse_noise(
    sparse_arr: np.ndarray,
    cards: list,
    *,
    mode: str,
    random_id_prob: float,
    rng: random.Random,
    num_raw_sparse: int = 26,
) -> np.ndarray:
    """Apply optional field-level noise for T2 robustness experiments."""
    if mode == "none":
        return sparse_arr

    out = sparse_arr.copy()
    n = min(num_raw_sparse, len(out))

    if mode == "random_ids":
        for i in range(n):
            if rng.random() < random_id_prob:
                card = int(cards[i])
                out[i] = rng.randint(1, max(card - 1, 1))
        return out

    if mode == "shuffle_fields":
        perm = list(range(n))
        rng.shuffle(perm)
        raw = out[:n].copy()
        for dst, src in enumerate(perm):
            out[dst] = raw[src]
        return out

    raise ValueError(f"Unknown noise mode: {mode}")


def _append_synthetic_sparse(
    sparse_arr: np.ndarray,
    *,
    num_fields: int,
    cardinality: int,
    mode: str,
    rng: random.Random,
) -> np.ndarray:
    """Append label-independent synthetic sparse fields for H1 robustness tests."""
    if num_fields <= 0:
        return sparse_arr

    if mode == "zero":
        extra = np.zeros(num_fields, dtype=np.int64)
    elif mode == "random":
        extra = np.array(
            [rng.randint(1, cardinality - 1) for _ in range(num_fields)],
            dtype=np.int64,
        )
    else:
        raise ValueError(f"Unknown synthetic_sparse mode: {mode}")

    return np.concatenate([sparse_arr, extra])


def _normalize_info_funnel_fields(info_cfg: dict, base_seed: int) -> list[dict]:
    """Validate dataset.info_funnel synthetic diagnostic fields."""
    if not info_cfg or not bool(info_cfg.get("enabled", False)):
        return []

    fields = info_cfg.get("fields") or []
    if not isinstance(fields, list):
        raise ValueError("dataset.info_funnel.fields must be a list")

    normalized = []
    for idx, raw in enumerate(fields):
        if not isinstance(raw, dict):
            raise ValueError("dataset.info_funnel.fields entries must be dicts")

        kind = str(raw.get("type", "")).lower()
        if kind not in ("pure_noise", "rare_noise", "spurious"):
            raise ValueError(
                "dataset.info_funnel field type must be "
                f"pure_noise|rare_noise|spurious, got {kind!r}"
            )

        num_fields = int(raw.get("num_fields", 1))
        cardinality = int(raw.get("cardinality", 10000))
        if num_fields < 0:
            raise ValueError("dataset.info_funnel num_fields must be >= 0")
        if cardinality <= 1:
            raise ValueError("dataset.info_funnel cardinality must be > 1")
        if kind == "spurious" and cardinality <= 3:
            raise ValueError("dataset.info_funnel spurious cardinality must be > 3")

        spec = {
            "type": kind,
            "name": str(raw.get("name", f"{kind}_{idx}")),
            "num_fields": num_fields,
            "cardinality": cardinality,
            "seed": int(raw.get("seed", base_seed + idx * 1009)),
        }

        if kind == "rare_noise":
            active_prob = float(raw.get("active_prob", 0.05))
            if active_prob < 0.0 or active_prob > 1.0:
                raise ValueError("dataset.info_funnel rare_noise active_prob must be in [0, 1]")
            spec["active_prob"] = active_prob

        if kind == "spurious":
            train_correlation = float(raw.get("train_correlation", 0.95))
            test_mode = str(raw.get("test_mode", "independent")).lower()
            if train_correlation < 0.0 or train_correlation > 1.0:
                raise ValueError(
                    "dataset.info_funnel spurious train_correlation must be in [0, 1]"
                )
            if test_mode not in ("same", "independent", "reverse"):
                raise ValueError(
                    "dataset.info_funnel spurious test_mode must be "
                    f"same|independent|reverse, got {test_mode!r}"
                )
            spec["train_correlation"] = train_correlation
            spec["test_mode"] = test_mode

        normalized.append(spec)

    return normalized


def _info_funnel_metadata(fields: list[dict]) -> tuple[list[str], list[int]]:
    cols = []
    cards = []
    for spec in fields:
        for i in range(spec["num_fields"]):
            cols.append(f"if_{spec['name']}_{i}")
            cards.append(spec["cardinality"])
    return cols, cards


def _draw_spurious_id(
    *,
    label: int,
    split: str,
    cardinality: int,
    train_correlation: float,
    test_mode: str,
    rng: random.Random,
) -> int:
    """Draw an ID whose label correlation exists only under chosen split semantics."""
    y = 1 if int(label) == 1 else 0

    if split == "train" or test_mode == "same":
        use_label_class = rng.random() < train_correlation
        class_id = y if use_label_class else 1 - y
    elif test_mode == "reverse":
        use_opposite_class = rng.random() < train_correlation
        class_id = 1 - y if use_opposite_class else y
    else:
        class_id = rng.randint(0, 1)

    max_id = cardinality - 1
    split_point = max(1, max_id // 2)
    if class_id == 0:
        return rng.randint(1, split_point)
    return rng.randint(split_point + 1, max_id)


def _append_info_funnel_sparse(
    sparse_arr: np.ndarray,
    *,
    label: int,
    split: str,
    fields: list[dict],
    rng: random.Random,
) -> np.ndarray:
    """Append controlled fields for information-funnel diagnostics."""
    if not fields:
        return sparse_arr

    extra = []
    for spec in fields:
        kind = spec["type"]
        cardinality = spec["cardinality"]
        for _ in range(spec["num_fields"]):
            if kind == "pure_noise":
                extra.append(rng.randint(1, cardinality - 1))
            elif kind == "rare_noise":
                if rng.random() < spec["active_prob"]:
                    extra.append(rng.randint(1, cardinality - 1))
                else:
                    extra.append(0)
            elif kind == "spurious":
                extra.append(
                    _draw_spurious_id(
                        label=label,
                        split=split,
                        cardinality=cardinality,
                        train_correlation=spec["train_correlation"],
                        test_mode=spec["test_mode"],
                        rng=rng,
                    )
                )
            else:
                raise ValueError(f"Unknown info_funnel field type: {kind}")

    return np.concatenate([sparse_arr, np.asarray(extra, dtype=np.int64)])


def _normalize_field_perturbation(
    perturb_cfg: dict,
    base_seed: int,
    default_train_days=None,
) -> dict:
    """Validate dataset.field_perturbation config for real-field diagnostics."""
    if not perturb_cfg or not bool(perturb_cfg.get("enabled", False)):
        return {"enabled": False}

    mode = str(perturb_cfg.get("mode", "mask")).lower()
    split = str(perturb_cfg.get("split", "test")).lower()
    fields = [int(i) for i in (perturb_cfg.get("fields") or [])]
    prob = float(perturb_cfg.get("prob", 1.0))

    if mode not in ("mask", "random_ids", "freq_mask", "freq_random_ids"):
        raise ValueError(
            "dataset.field_perturbation.mode must be "
            f"mask|random_ids|freq_mask|freq_random_ids, got {mode!r}"
        )
    if split not in ("train", "test", "both"):
        raise ValueError(
            f"dataset.field_perturbation.split must be train|test|both, got {split!r}"
        )
    if not fields:
        raise ValueError("dataset.field_perturbation.fields must be non-empty when enabled")
    if any(i < 0 for i in fields):
        raise ValueError("dataset.field_perturbation.fields must be non-negative indices")
    if prob < 0.0 or prob > 1.0:
        raise ValueError("dataset.field_perturbation.prob must be in [0, 1]")

    normalized = {
        "enabled": True,
        "mode": mode,
        "split": split,
        "fields": fields,
        "prob": prob,
        "seed": int(perturb_cfg.get("seed", base_seed)),
    }

    if mode.startswith("freq_"):
        freq_cfg = perturb_cfg.get("frequency") or {}
        target = str(freq_cfg.get("target", "tail")).lower()
        if target not in ("tail", "head"):
            raise ValueError(
                "dataset.field_perturbation.frequency.target must be tail|head, "
                f"got {target!r}"
            )
        max_count = int(freq_cfg.get("max_count", 5))
        min_count = int(freq_cfg.get("min_count", 100))
        if max_count < 0:
            raise ValueError("dataset.field_perturbation.frequency.max_count must be >= 0")
        if min_count < 1:
            raise ValueError("dataset.field_perturbation.frequency.min_count must be >= 1")
        train_days = freq_cfg.get("train_days", default_train_days)
        if not train_days:
            raise ValueError(
                "dataset.field_perturbation.frequency.train_days must be provided"
            )
        train_days = [int(d) for d in train_days]
        cache_path = freq_cfg.get("cache_path")
        if not cache_path:
            day_key = "_".join(str(d) for d in train_days)
            field_key = "_".join(str(i) for i in fields)
            cache_path = f"/tmp/recscale_criteo1tb_fieldfreq_d{day_key}_f{field_key}.npz"

        normalized["frequency"] = {
            "enabled": True,
            "target": target,
            "max_count": max_count,
            "min_count": min_count,
            "train_days": train_days,
            "cache_path": str(cache_path),
            "include_zero_id": bool(freq_cfg.get("include_zero_id", False)),
            "max_rows": int(freq_cfg.get("max_rows", 0)),
        }

    return normalized


def _should_apply_field_perturbation(split: str, target_split: str) -> bool:
    if target_split == "both":
        return True
    if target_split == "train":
        return split == "train"
    return split != "train"


def _apply_field_perturbation(
    sparse_arr: np.ndarray,
    cards: list[int],
    *,
    fields: list[int],
    mode: str,
    prob: float,
    rng: random.Random,
    frequency_state=None,
) -> np.ndarray:
    """Perturb selected model-field indices after Criteo preprocessing."""
    out = sparse_arr.copy()
    n = len(out)
    for idx in fields:
        if idx >= n:
            raise ValueError(
                f"dataset.field_perturbation field index {idx} out of range for sparse len {n}"
            )
        if idx >= len(cards):
            raise ValueError(
                f"dataset.field_perturbation field index {idx} lacks cardinality metadata"
            )
        if rng.random() > prob:
            continue
        if mode == "mask":
            out[idx] = 0
        elif mode == "random_ids":
            card = int(cards[idx])
            out[idx] = rng.randint(1, max(card - 1, 1))
        elif mode in ("freq_mask", "freq_random_ids"):
            if frequency_state is None:
                raise ValueError("frequency_state is required for frequency perturbation")
            if not _matches_frequency_target(int(out[idx]), idx, frequency_state):
                continue
            if mode == "freq_mask":
                out[idx] = 0
            else:
                out[idx] = _draw_frequency_candidate(idx, int(out[idx]), frequency_state, rng)
        else:
            raise ValueError(f"Unknown field_perturbation mode: {mode}")
    return out


def _matches_frequency_target(value: int, idx: int, state: dict) -> bool:
    if value == 0 and not state["include_zero_id"]:
        return False
    counts = state["counts"][int(idx)]
    count = int(counts[value]) if 0 <= value < len(counts) else 0
    if state["target"] == "tail":
        return count <= state["max_count"]
    return count >= state["min_count"]


def _draw_frequency_candidate(
    idx: int,
    original: int,
    state: dict,
    rng: random.Random,
) -> int:
    candidates = state["candidates"].get(int(idx))
    if candidates is None or len(candidates) == 0:
        card = int(state["cards"][int(idx)])
        return rng.randint(1, max(card - 1, 1))
    if len(candidates) == 1:
        return int(candidates[0])
    for _ in range(8):
        value = int(candidates[rng.randrange(len(candidates))])
        if value != original:
            return value
    return int(candidates[rng.randrange(len(candidates))])


def _build_frequency_state(
    counts: dict[int, np.ndarray],
    cards: list[int],
    freq_cfg: dict,
) -> dict:
    target = freq_cfg["target"]
    include_zero_id = bool(freq_cfg.get("include_zero_id", False))
    candidates = {}
    for idx, arr in counts.items():
        if target == "tail":
            mask = arr <= int(freq_cfg["max_count"])
        else:
            mask = arr >= int(freq_cfg["min_count"])
        if not include_zero_id and len(mask) > 0:
            mask = mask.copy()
            mask[0] = False
        candidates[int(idx)] = np.flatnonzero(mask).astype(np.int64)

    return {
        "counts": counts,
        "cards": cards,
        "target": target,
        "max_count": int(freq_cfg["max_count"]),
        "min_count": int(freq_cfg["min_count"]),
        "include_zero_id": include_zero_id,
        "candidates": candidates,
    }


def _build_field_frequency_counts(
    *,
    data_path: str,
    train_days: list[int],
    fields: list[int],
    cards: list[int],
    boundaries,
    bucketize: bool,
    num_buckets: int,
    max_rows: int = 0,
) -> dict[int, np.ndarray]:
    counts = {
        int(idx): np.zeros(int(cards[int(idx)]), dtype=np.uint32)
        for idx in fields
    }
    dense_fields = [idx for idx in fields if int(idx) >= 26]
    if dense_fields and (not bucketize or boundaries is None):
        raise ValueError(
            "frequency perturbation for dense bucket fields requires bucketized boundaries"
        )

    files = []
    for day in train_days:
        fpath, is_gz = _find_day_file(data_path, day)
        if fpath is None:
            print(
                f"[Criteo1TB] WARNING: day_{day}.tsv / day_{day}.gz not found "
                f"for field-frequency cache, skipping"
            )
            continue
        files.append((fpath, is_gz))
    if not files:
        raise FileNotFoundError(
            f"No train day files found for field-frequency days={train_days}"
        )

    n_rows = 0
    for fpath, is_gz in files:
        cnt = 0
        for line in _open_file(fpath, is_gz):
            parsed = _parse_line(line)
            if parsed is None:
                continue
            _, dense, sparse_raw = parsed
            bucket_out = None
            for idx in fields:
                idx = int(idx)
                if idx < 26:
                    value = 0
                    if sparse_raw[idx] != 0:
                        card = int(cards[idx])
                        value = (sparse_raw[idx] % (card - 1)) + 1
                else:
                    if bucket_out is None:
                        dense_arr = np.asarray(dense, dtype=np.float32)
                        bucket_out = _apply_boundaries_row(
                            dense_arr, boundaries, num_buckets
                        )
                    value = int(bucket_out[idx - 26])
                counts[idx][value] += 1

            cnt += 1
            n_rows += 1
            if 0 < max_rows <= n_rows:
                break
        print(f"  fieldfreq {os.path.basename(fpath)}: {cnt:,} rows")
        if 0 < max_rows <= n_rows:
            break

    print(
        f"[Criteo1TB] Built field-frequency counts for fields={fields}, "
        f"rows={n_rows:,}"
    )
    return counts


@register_dataset("criteo_1tb")
class Criteo1TBDataset(IterableDataset):
    """
    Criteo 1TB Click Logs，流式 IterableDataset。

    不预分配内存，逐行 yield dict：
      {"sparse": np.int64[39], "label": float32}

    Pass-1 (train only): 统计行数 + reservoir sampling 构建分桶边界
    训练/评估: DataLoader 流式读取
    """

    # 跨 split 共享（train 时建立，test 时复用）
    _shared_boundaries: dict = None
    _shared_cardinalities: list = None

    def __init__(self, config: dict, split: str = "train"):
        super().__init__()
        dc = config["dataset"]
        data_path    = dc["path"]
        self.data_path = data_path
        self.split    = split
        # max_rows: global limit; test_max_rows: test-only limit (e.g. 500000 for fast eval)
        global_max = dc.get("max_rows", 0)
        test_max   = dc.get("test_max_rows", 0)
        if split != "train" and test_max > 0:
            self.max_rows = test_max
        else:
            self.max_rows = global_max

        default_train = list(range(2, 23))
        default_test  = [23]
        days = dc.get("train_days" if split == "train" else "test_days",
                      default_train if split == "train" else default_test)
        self.train_days = [int(d) for d in dc.get("train_days", default_train)]

        self.bucketize     = dc.get("bucketize_dense", True)
        self.bucket_method = dc.get("bucket_method", "log")
        self.num_buckets   = dc.get("num_buckets", 100)
        self.bucket_cache_path = dc.get("bucket_cache_path")

        noise_cfg = dc.get("noise") or {}
        self.noise_mode = str(noise_cfg.get("mode", "none")).lower()
        self.noise_random_id_prob = float(noise_cfg.get("random_id_prob", 0.1))
        self.noise_seed = int(noise_cfg.get("seed", config.get("seed", 42)))
        if self.noise_mode not in ("none", "random_ids", "shuffle_fields"):
            raise ValueError(
                f"dataset.noise.mode must be none|random_ids|shuffle_fields, got {self.noise_mode!r}"
            )
        if self.noise_mode != "none" and split != "train":
            print(f"[Criteo1TB] noise.mode={self.noise_mode} ignored for split={split} (train only)")
        elif self.noise_mode != "none":
            print(f"[Criteo1TB] noise.mode={self.noise_mode}, "
                  f"random_id_prob={self.noise_random_id_prob}, seed={self.noise_seed}")

        synthetic_cfg = dc.get("synthetic_sparse") or {}
        self.synthetic_num_fields = int(synthetic_cfg.get("num_fields", 0))
        self.synthetic_cardinality = int(synthetic_cfg.get("cardinality", 10000))
        self.synthetic_mode = str(synthetic_cfg.get("mode", "zero")).lower()
        self.synthetic_seed = int(synthetic_cfg.get("seed", config.get("seed", 42)))
        if self.synthetic_num_fields < 0:
            raise ValueError("dataset.synthetic_sparse.num_fields must be >= 0")
        if self.synthetic_num_fields > 0:
            if self.synthetic_mode not in ("zero", "random"):
                raise ValueError(
                    "dataset.synthetic_sparse.mode must be zero|random, "
                    f"got {self.synthetic_mode!r}"
                )
            if self.synthetic_cardinality <= 1:
                raise ValueError("dataset.synthetic_sparse.cardinality must be > 1")
            print(
                f"[Criteo1TB] synthetic_sparse.mode={self.synthetic_mode}, "
                f"num_fields={self.synthetic_num_fields}, "
                f"cardinality={self.synthetic_cardinality}, seed={self.synthetic_seed}"
            )

        info_funnel_cfg = dc.get("info_funnel") or {}
        self.info_funnel_seed = int(info_funnel_cfg.get("seed", config.get("seed", 42)))
        self.info_funnel_fields = _normalize_info_funnel_fields(
            info_funnel_cfg, self.info_funnel_seed
        )
        self.info_funnel_num_fields = sum(
            spec["num_fields"] for spec in self.info_funnel_fields
        )
        if self.info_funnel_num_fields > 0:
            field_desc = ", ".join(
                f"{spec['name']}:{spec['type']}x{spec['num_fields']}"
                for spec in self.info_funnel_fields
            )
            print(
                f"[Criteo1TB] info_funnel fields={self.info_funnel_num_fields}, "
                f"seed={self.info_funnel_seed}, specs=[{field_desc}]"
            )

        perturb_cfg = dc.get("field_perturbation") or {}
        self.field_perturbation = _normalize_field_perturbation(
            perturb_cfg,
            int(perturb_cfg.get("seed", config.get("seed", 42))),
            default_train_days=self.train_days,
        )
        if self.field_perturbation.get("enabled", False):
            print(
                "[Criteo1TB] field_perturbation "
                f"split={self.field_perturbation['split']}, "
                f"mode={self.field_perturbation['mode']}, "
                f"fields={self.field_perturbation['fields']}, "
                f"prob={self.field_perturbation['prob']}, "
                f"seed={self.field_perturbation['seed']}"
            )

        print(f"[Criteo1TB] split={split}, days={days}, "
              f"bucketize={self.bucketize}({self.bucket_method})")

        # 文件列表（优先 .tsv，回退 .gz）
        self.files = []   # list of (filepath, is_gz)
        for d in days:
            fpath, is_gz = _find_day_file(data_path, d)
            if fpath is not None:
                self.files.append((fpath, is_gz))
            else:
                print(f"[Criteo1TB] WARNING: day_{d}.tsv / day_{d}.gz not found in {data_path}, skipping")
        if not self.files:
            raise FileNotFoundError(f"No day_*.tsv or day_*.gz found in {data_path} for days={days}")

        # 文件格式提示
        n_tsv = sum(1 for _, is_gz in self.files if not is_gz)
        n_gz  = sum(1 for _, is_gz in self.files if is_gz)
        if n_tsv > 0 and n_gz == 0:
            print(f"[Criteo1TB] Using pre-decompressed .tsv files (~3x faster)")
        elif n_gz > 0 and n_tsv == 0:
            print(f"[Criteo1TB] Using .gz files (consider pre-decompressing to Criteo_1TB_tsv/)")
        else:
            print(f"[Criteo1TB] Mixed: {n_tsv} tsv + {n_gz} gz")

        # ---- Pass-1: 统计行数 + 分桶边界 ----
        need_boundaries = (split == "train" and self.bucketize
                           and Criteo1TBDataset._shared_boundaries is None)
        cache_enabled = (
            split == "train"
            and self.bucketize
            and self.bucket_cache_path
            and self.max_rows == 0
        )
        cache_builder = False
        cache_lock_path = f"{self.bucket_cache_path}.lock" if cache_enabled else None
        if cache_enabled:
            loaded = _load_bucket_cache(
                self.bucket_cache_path, self.bucket_method, self.num_buckets, days
            )
            if loaded is None:
                if _try_acquire_cache_lock(cache_lock_path):
                    cache_builder = True
                    print(f"[Criteo1TB] Building bucket cache: {self.bucket_cache_path}")
                else:
                    loaded = _wait_for_bucket_cache(
                        self.bucket_cache_path,
                        self.bucket_method,
                        self.num_buckets,
                        days,
                        cache_lock_path,
                    )
                    if loaded is None and _try_acquire_cache_lock(cache_lock_path):
                        cache_builder = True
                        print(
                            "[Criteo1TB] Building bucket cache after stale/missing "
                            f"lock: {self.bucket_cache_path}"
                        )
            if loaded is not None:
                self.n_total, cached_boundaries = loaded
                Criteo1TBDataset._shared_boundaries = cached_boundaries
                need_boundaries = False
                print(f"[Criteo1TB] Loaded bucket cache: {self.bucket_cache_path}")

        if not (cache_enabled and not cache_builder and Criteo1TBDataset._shared_boundaries is not None):
            SAMPLE_SIZE = 2_000_000
            rng_res = random.Random(42)
            reservoirs = [[] for _ in range(13)] if need_boundaries else None

            print(f"[Criteo1TB] Pass-1: counting rows"
                  f"{' + reservoir sampling' if need_boundaries else ''}...")
            n_total = 0
            for fpath, is_gz in self.files:
                cnt = 0
                for line in _open_file(fpath, is_gz):
                    parsed = _parse_line(line)
                    if parsed is None:
                        continue
                    cnt += 1
                    if need_boundaries:
                        _, dense, _ = parsed
                        for i in range(13):
                            if len(reservoirs[i]) < SAMPLE_SIZE:
                                reservoirs[i].append(dense[i])
                            else:
                                j = rng_res.randint(0, n_total + cnt - 1)
                                if j < SAMPLE_SIZE:
                                    reservoirs[i][j] = dense[i]
                    if 0 < self.max_rows <= (n_total + cnt):
                        break
                n_total += cnt
                print(f"  {os.path.basename(fpath)}: {cnt:,} rows")
                if 0 < self.max_rows <= n_total:
                    break

            self.n_total = min(n_total, self.max_rows) if self.max_rows > 0 else n_total
            print(f"[Criteo1TB] Total: {self.n_total:,} rows")

            # 构建分桶边界
            if need_boundaries:
                print(f"[Criteo1TB] Building bucket boundaries from "
                      f"{min(n_total, SAMPLE_SIZE):,} samples...")
                boundaries = _build_boundaries(reservoirs, self.bucket_method,
                                               self.num_buckets)
                Criteo1TBDataset._shared_boundaries = boundaries
                print(f"[Criteo1TB] Bucket boundaries built for {len(DENSE_COLS)} dense cols")
                if cache_builder:
                    _save_bucket_cache(
                        self.bucket_cache_path,
                        boundaries,
                        self.n_total,
                        self.bucket_method,
                        self.num_buckets,
                        days,
                    )
                    try:
                        os.unlink(cache_lock_path)
                    except FileNotFoundError:
                        pass

        # Cardinalities
        if split == "train" and Criteo1TBDataset._shared_cardinalities is None:
            Criteo1TBDataset._shared_cardinalities = [c + 2 for c in _DEFAULT_CARDINALITIES]

        self.boundaries = Criteo1TBDataset._shared_boundaries
        self.cards      = Criteo1TBDataset._shared_cardinalities or \
                          [c + 2 for c in _DEFAULT_CARDINALITIES]
        bucket_card = self.num_buckets + 2
        if self.bucketize and self.boundaries is not None:
            self.output_cards = list(self.cards) + [bucket_card] * 13
        else:
            self.output_cards = list(self.cards)

        # ---- 更新 config（train 时设置 sparse_cols / cardinalities）----
        if split == "train":
            if self.bucketize and self.boundaries is not None:
                all_cards = list(self.cards) + [bucket_card] * 13
                dc["sparse_cols"]   = SPARSE_COLS + [f"{c}_bucket" for c in DENSE_COLS]
                dc["cardinalities"] = all_cards
                dc["num_dense"]     = 0
                dc["dense_cols"]    = []
            else:
                dc["sparse_cols"]   = SPARSE_COLS
                dc["cardinalities"] = list(self.cards)
                dc["num_dense"]     = 13
                dc["dense_cols"]    = DENSE_COLS

            if self.synthetic_num_fields > 0:
                synthetic_cols = [
                    f"synthetic_noise_{i}" for i in range(self.synthetic_num_fields)
                ]
                dc["sparse_cols"] = list(dc["sparse_cols"]) + synthetic_cols
                dc["cardinalities"] = list(dc["cardinalities"]) + [
                    self.synthetic_cardinality
                ] * self.synthetic_num_fields
                self.output_cards = list(self.output_cards) + [
                    self.synthetic_cardinality
                ] * self.synthetic_num_fields

            if self.info_funnel_num_fields > 0:
                info_cols, info_cards = _info_funnel_metadata(self.info_funnel_fields)
                dc["sparse_cols"] = list(dc["sparse_cols"]) + info_cols
                dc["cardinalities"] = list(dc["cardinalities"]) + info_cards
                self.output_cards = list(self.output_cards) + info_cards
        else:
            if self.synthetic_num_fields > 0:
                self.output_cards = list(self.output_cards) + [
                    self.synthetic_cardinality
                ] * self.synthetic_num_fields
            if self.info_funnel_num_fields > 0:
                _, info_cards = _info_funnel_metadata(self.info_funnel_fields)
                self.output_cards = list(self.output_cards) + info_cards

        n_sparse_out = 39 if (self.bucketize and self.boundaries is not None) else 26
        n_sparse_out += self.synthetic_num_fields
        n_sparse_out += self.info_funnel_num_fields
        print(f"[Criteo1TB] {split}: {self.n_total:,} samples, "
              f"sparse_cols={n_sparse_out}, "
              f"dense={'none (bucketized)' if self.bucketize and self.boundaries is not None else 13}")

        self.field_frequency_state = None
        if (
            self.field_perturbation.get("enabled", False)
            and self.field_perturbation["mode"].startswith("freq_")
            and _should_apply_field_perturbation(
                self.split, self.field_perturbation["split"]
            )
        ):
            freq_cfg = self.field_perturbation["frequency"]
            counts = self._load_or_build_field_frequency_counts(
                freq_cfg, fields=self.field_perturbation["fields"]
            )
            self.field_frequency_state = _build_frequency_state(
                counts, self.output_cards, freq_cfg
            )
            cand_desc = ", ".join(
                f"{idx}:{len(candidates)}"
                for idx, candidates in self.field_frequency_state["candidates"].items()
            )
            print(
                "[Criteo1TB] field_frequency "
                f"target={freq_cfg['target']}, "
                f"max_count={freq_cfg['max_count']}, "
                f"min_count={freq_cfg['min_count']}, "
                f"candidates=[{cand_desc}]"
            )

    def get_field_frequency_counts(
        self,
        *,
        fields: list[int],
        train_days: list[int],
        cache_path: str,
        max_rows: int = 0,
    ) -> dict[int, np.ndarray]:
        """Load or build post-hash training counts for analysis-only field panels."""
        normalized_fields = [int(idx) for idx in fields]
        if not normalized_fields:
            raise ValueError("fields must be non-empty for frequency analysis")
        if any(idx < 0 or idx >= len(self.output_cards) for idx in normalized_fields):
            raise ValueError(
                f"frequency analysis fields out of range for {len(self.output_cards)} fields"
            )
        return self._load_or_build_field_frequency_counts(
            {
                "train_days": [int(day) for day in train_days],
                "cache_path": str(cache_path),
                "max_rows": int(max_rows),
            },
            fields=normalized_fields,
        )

    def _load_or_build_field_frequency_counts(
        self,
        freq_cfg: dict,
        *,
        fields: list[int],
    ) -> dict[int, np.ndarray]:
        train_days = freq_cfg["train_days"]
        cache_path = freq_cfg["cache_path"]
        loaded = _load_field_frequency_cache(
            cache_path,
            days=train_days,
            fields=fields,
            cards=self.output_cards,
        )
        if loaded is not None:
            print(f"[Criteo1TB] Loaded field-frequency cache: {cache_path}")
            return loaded

        lock_path = f"{cache_path}.lock"
        if _try_acquire_cache_lock(lock_path):
            print(f"[Criteo1TB] Building field-frequency cache: {cache_path}")
            counts = _build_field_frequency_counts(
                data_path=self.data_path,
                train_days=train_days,
                fields=fields,
                cards=self.output_cards,
                boundaries=self.boundaries,
                bucketize=self.bucketize,
                num_buckets=self.num_buckets,
                max_rows=int(freq_cfg.get("max_rows", 0)),
            )
            _save_field_frequency_cache(
                cache_path,
                days=train_days,
                fields=fields,
                counts=counts,
            )
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass
            return counts

        loaded = _wait_for_field_frequency_cache(
            cache_path,
            days=train_days,
            fields=fields,
            cards=self.output_cards,
            lock_path=lock_path,
        )
        if loaded is not None:
            print(f"[Criteo1TB] Loaded field-frequency cache: {cache_path}")
            return loaded

        if _try_acquire_cache_lock(lock_path):
            print(
                "[Criteo1TB] Building field-frequency cache after stale/missing "
                f"lock: {cache_path}"
            )
            counts = _build_field_frequency_counts(
                data_path=self.data_path,
                train_days=train_days,
                fields=fields,
                cards=self.output_cards,
                boundaries=self.boundaries,
                bucketize=self.bucketize,
                num_buckets=self.num_buckets,
                max_rows=int(freq_cfg.get("max_rows", 0)),
            )
            _save_field_frequency_cache(
                cache_path,
                days=train_days,
                fields=fields,
                counts=counts,
            )
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass
            return counts

        raise RuntimeError(f"Could not build or load field-frequency cache: {cache_path}")

    # ---------------------------------------------------------------
    # IterableDataset 接口
    # ---------------------------------------------------------------

    def __len__(self):
        return self.n_total

    def __iter__(self):
        # 多 worker 分片支持
        worker_info = torch.utils.data.get_worker_info()
        files = self.files  # list of (fpath, is_gz)
        if worker_info is not None:
            # 按文件分片给各 worker
            all_files = self.files
            wid   = worker_info.id
            nw    = worker_info.num_workers
            files = [f for i, f in enumerate(all_files) if i % nw == wid]

        n_yielded = 0
        worker_off = worker_info.id if worker_info is not None else 0
        noise_rng = random.Random(self.noise_seed + worker_off * 1_000_003)
        synthetic_rng = random.Random(self.synthetic_seed + worker_off * 1_000_003)
        info_funnel_rng = random.Random(self.info_funnel_seed + worker_off * 1_000_003)
        field_perturb_rng = random.Random(
            int(self.field_perturbation.get("seed", 42)) + worker_off * 1_000_003
        )
        apply_noise = self.split == "train" and self.noise_mode != "none"
        append_synthetic = self.synthetic_num_fields > 0
        append_info_funnel = self.info_funnel_num_fields > 0
        apply_field_perturbation = (
            self.field_perturbation.get("enabled", False)
            and _should_apply_field_perturbation(
                self.split, self.field_perturbation["split"]
            )
        )

        for fpath, is_gz in files:
            for line in _open_file(fpath, is_gz):
                parsed = _parse_line(line)
                if parsed is None:
                    continue
                label, dense, sparse_raw = parsed

                # --- sparse: hash mod ---
                sparse_out = np.zeros(26, dtype=np.int64)
                for i in range(26):
                    if sparse_raw[i] != 0:
                        card = self.cards[i]
                        sparse_out[i] = (sparse_raw[i] % (card - 1)) + 1

                # --- dense bucketize ---
                if self.bucketize and self.boundaries is not None:
                    dense_arr = np.array(dense, dtype=np.float32)
                    bucket_out = _apply_boundaries_row(dense_arr, self.boundaries,
                                                       self.num_buckets)
                    sparse_final = np.concatenate([sparse_out, bucket_out])
                    if apply_noise:
                        sparse_final = _apply_sparse_noise(
                            sparse_final,
                            self.cards,
                            mode=self.noise_mode,
                            random_id_prob=self.noise_random_id_prob,
                            rng=noise_rng,
                            num_raw_sparse=26,
                        )
                    if append_synthetic:
                        sparse_final = _append_synthetic_sparse(
                            sparse_final,
                            num_fields=self.synthetic_num_fields,
                            cardinality=self.synthetic_cardinality,
                            mode=self.synthetic_mode,
                            rng=synthetic_rng,
                        )
                    if append_info_funnel:
                        sparse_final = _append_info_funnel_sparse(
                            sparse_final,
                            label=label,
                            split=self.split,
                            fields=self.info_funnel_fields,
                            rng=info_funnel_rng,
                        )
                    if apply_field_perturbation:
                        sparse_final = _apply_field_perturbation(
                            sparse_final,
                            self.output_cards,
                            fields=self.field_perturbation["fields"],
                            mode=self.field_perturbation["mode"],
                            prob=self.field_perturbation["prob"],
                            rng=field_perturb_rng,
                            frequency_state=self.field_frequency_state,
                        )
                    yield {"sparse": sparse_final,
                           "label":  np.float32(label)}
                else:
                    dense_arr = np.log1p(np.abs(np.array(dense, dtype=np.float32))) \
                                * np.sign(np.array(dense, dtype=np.float32))
                    if apply_noise:
                        sparse_out = _apply_sparse_noise(
                            sparse_out,
                            self.cards,
                            mode=self.noise_mode,
                            random_id_prob=self.noise_random_id_prob,
                            rng=noise_rng,
                            num_raw_sparse=26,
                        )
                    if append_synthetic:
                        sparse_out = _append_synthetic_sparse(
                            sparse_out,
                            num_fields=self.synthetic_num_fields,
                            cardinality=self.synthetic_cardinality,
                            mode=self.synthetic_mode,
                            rng=synthetic_rng,
                        )
                    if append_info_funnel:
                        sparse_out = _append_info_funnel_sparse(
                            sparse_out,
                            label=label,
                            split=self.split,
                            fields=self.info_funnel_fields,
                            rng=info_funnel_rng,
                        )
                    if apply_field_perturbation:
                        sparse_out = _apply_field_perturbation(
                            sparse_out,
                            self.output_cards,
                            fields=self.field_perturbation["fields"],
                            mode=self.field_perturbation["mode"],
                            prob=self.field_perturbation["prob"],
                            rng=field_perturb_rng,
                            frequency_state=self.field_frequency_state,
                        )
                    yield {"sparse": sparse_out,
                           "dense":  dense_arr,
                           "label":  np.float32(label)}

                n_yielded += 1
                if self.max_rows > 0 and n_yielded >= self.max_rows:
                    return


# ---------------------------------------------------------------
# 分桶工具函数
# ---------------------------------------------------------------

def _build_boundaries(reservoirs, bucket_method, num_buckets):
    boundaries = {}
    for i in range(13):
        arr = np.array(reservoirs[i], dtype=np.float32)
        if len(arr) == 0:
            boundaries[i] = {"method": bucket_method, "edges": np.array([])}
            continue
        if bucket_method == "log":
            transformed = np.log1p(np.abs(arr)) * np.sign(arr)
            vmin, vmax = transformed.min(), transformed.max()
            edges = np.linspace(vmin, vmax, num_buckets + 1)[1:-1] \
                    if vmax - vmin > 1e-8 else np.array([vmin])
        elif bucket_method == "quantile":
            percentiles = np.linspace(0, 100, num_buckets + 1)[1:-1]
            edges = np.unique(np.percentile(arr, percentiles))
        else:  # uniform
            vmin, vmax = arr.min(), arr.max()
            edges = np.linspace(vmin, vmax, num_buckets + 1)[1:-1]
        boundaries[i] = {"method": bucket_method, "edges": edges}
        print(f"  I{i+1}: range [{arr.min():.0f}, {arr.max():.0f}], {len(edges)} edges")
    return boundaries


def _apply_boundaries_row(dense_row, boundaries, num_buckets):
    """单行 dense → bucket indices，返回 np.int64[13]"""
    result = np.zeros(13, dtype=np.int64)
    for i in range(13):
        v = dense_row[i]
        info = boundaries.get(i, {})
        edges = info.get("edges", np.array([]))
        method = info.get("method", "log")
        if method == "log":
            v_t = np.log1p(abs(v)) * np.sign(v)
        else:
            v_t = v
        idx = int(np.searchsorted(edges, v_t)) + 1
        result[i] = min(max(idx, 1), num_buckets)
    return result
