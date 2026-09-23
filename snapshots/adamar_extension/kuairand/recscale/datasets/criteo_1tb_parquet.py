"""
recscale.datasets.criteo_1tb_parquet — Criteo 1TB Click Logs (新版 Parquet 格式)

数据来源: https://huggingface.co/datasets/criteo/CriteoClickLogs
格式: Snappy Parquet，按天分区
文件: data/day=YYYY-MM-DD/part-*.snappy.parquet (共 24 天，每天 ~50 个 part)
      日期范围: 2015-02-15 ~ 2015-03-10

2026-05 Criteo 官方重新上传完整数据，格式从 gzip TSV 改为 Parquet：
  - 旧版 day_N.gz 已删除（数据损坏/截断）
  - 新版 data/day=YYYY-MM-DD/  Snappy Parquet，276GB，完整 ~4.4 亿行

列说明:
  label:                     int32, 0/1
  integer_feature_1 ~ 13:    int32, dense 特征（含 null）
  categorical_feature_1 ~ 26: string, hex 编码 sparse 特征（含 null）

切分方式:
  train_days / test_days 支持两种格式:
    - 日期字符串: ["2015-03-09", "2015-03-10"]
    - 兼容旧整数: [22, 23] → 自动映射 (day_2=2015-02-15 ... day_23=2015-03-10)

内存策略: IterableDataset，基于 pyarrow row-group 流式读取
  - Pass-1 (train only): 逐 row-group 统计行数 + reservoir sampling 构建分桶边界
  - 训练/评估: DataLoader 流式读取，num_workers 控制并行度

YAML 示例:
```yaml
dataset:
  type: criteo_1tb_parquet
  path: /data/Criteo_1TB_new
  train_days: ["2015-03-09"]         # 或兼容旧格式: [22]
  test_days: ["2015-03-10"]          # 或: [23]
  test_max_rows: 500000
  bucketize_dense: true
  num_buckets: 100
  bucket_method: log
```
"""

import glob
import os
import random

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from . import register_dataset


# ---------------------------------------------------------------
# 列定义
# ---------------------------------------------------------------

DENSE_COLS = [f"integer_feature_{i}" for i in range(1, 14)]       # 13 dense
SPARSE_COLS_PARQUET = [f"categorical_feature_{i}" for i in range(1, 27)]  # 26 sparse
# 输出用的列名（与旧版一致）
SPARSE_COLS_OUT = [f"C{i}" for i in range(1, 27)]
DENSE_COLS_OUT = [f"I{i}" for i in range(1, 14)]

# 全量 Criteo 1TB sparse 特征基数（同旧版，来自 DLRM/Meta）
_DEFAULT_CARDINALITIES = [
    1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
    8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18,
    15, 286181, 105, 142572,
]

# ---------------------------------------------------------------
# 旧整数 day_id → 日期 映射
# day_0 = 2015-02-15, day_1 = 2015-02-16, ..., day_23 = 2015-03-10
# 共 24 天
# 旧版 criteo_1tb.py 默认 train=day_2~22, test=day_23
# (day_0/day_1 在旧 gz 版本中损坏，新 parquet 版已修复)
# ---------------------------------------------------------------

_DAY_ID_TO_DATE = {}
from datetime import date, timedelta
_base = date(2015, 2, 15)
for _i in range(24):
    _day_id = _i  # day_0 ~ day_23
    _dt = _base + timedelta(days=_i)
    _DAY_ID_TO_DATE[_day_id] = _dt.isoformat()  # "2015-02-15"

# 反向映射
_DATE_TO_DAY_ID = {v: k for k, v in _DAY_ID_TO_DATE.items()}

# 所有可用日期（排序）
_ALL_DATES = sorted(_DAY_ID_TO_DATE.values())


def _normalize_days(days: list) -> list[str]:
    """
    将 train_days / test_days 统一为日期字符串列表。
    支持输入:
      - [22, 23]  → ["2015-03-09", "2015-03-10"]
      - ["2015-03-09", "2015-03-10"]  → 原样返回
    """
    result = []
    for d in days:
        if isinstance(d, int):
            if d in _DAY_ID_TO_DATE:
                result.append(_DAY_ID_TO_DATE[d])
            else:
                raise ValueError(
                    f"Day ID {d} not in valid range [0, 23]. "
                    f"Use date strings like '2015-02-15' or integer IDs 0~23."
                )
        elif isinstance(d, str):
            result.append(d)
        else:
            raise ValueError(f"Invalid day format: {d} (type={type(d)})")
    return sorted(result)


def _find_parquet_files(data_path: str, day_date: str) -> list[str]:
    """查找某天所有 parquet part 文件，返回排序后的文件路径列表。"""
    day_dir = os.path.join(data_path, "data", f"day={day_date}")
    if not os.path.isdir(day_dir):
        return []
    files = sorted(glob.glob(os.path.join(day_dir, "*.parquet")))
    return files


# ---------------------------------------------------------------
# 分桶工具（从旧版 criteo_1tb.py 复用）
# ---------------------------------------------------------------

def _build_boundaries(reservoirs, bucket_method, num_buckets):
    """从 reservoir sampling 的样本构建分桶边界。"""
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


def _apply_boundaries_batch(dense_batch, boundaries, num_buckets):
    """
    批量 dense → bucket indices，向量化处理。
    dense_batch: np.float32 [N, 13]
    返回 np.int64 [N, 13]
    """
    N = dense_batch.shape[0]
    result = np.zeros((N, 13), dtype=np.int64)
    for i in range(13):
        col = dense_batch[:, i]
        info = boundaries.get(i, {})
        edges = info.get("edges", np.array([]))
        method = info.get("method", "log")
        if method == "log":
            col_t = np.log1p(np.abs(col)) * np.sign(col)
        else:
            col_t = col
        idx = np.searchsorted(edges, col_t).astype(np.int64) + 1
        result[:, i] = np.clip(idx, 1, num_buckets)
    return result


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

@register_dataset("criteo_1tb_parquet")
class Criteo1TBParquetDataset(IterableDataset):
    """
    Criteo 1TB Click Logs (Parquet 格式)，流式 IterableDataset。

    输出与旧版 criteo_1tb 完全一致:
      {"sparse": np.int64[39], "label": float32}  (bucketize=True)
      {"sparse": np.int64[26], "dense": np.float32[13], "label": float32}  (bucketize=False)
    """

    # 跨 split 共享
    _shared_boundaries: dict = None
    _shared_cardinalities: list = None

    def __init__(self, config: dict, split: str = "train"):
        super().__init__()
        dc = config["dataset"]
        data_path = dc["path"]
        self.split = split

        # max_rows 处理
        global_max = dc.get("max_rows", 0)
        test_max = dc.get("test_max_rows", 0)
        if split != "train" and test_max > 0:
            self.max_rows = test_max
        else:
            self.max_rows = global_max

        # 天数配置
        # 新版默认: train=day_0~22 (24天中前23天), test=day_23 (最后1天)
        # 兼容旧版: train=day_2~22, test=day_23 (跳过旧版损坏的 day_0/1)
        default_train_days = list(range(0, 23))    # day_0 ~ day_22
        default_test_days = [23]                    # day_23
        raw_days = dc.get(
            "train_days" if split == "train" else "test_days",
            default_train_days if split == "train" else default_test_days,
        )
        self.days = _normalize_days(raw_days)

        # 分桶配置
        self.bucketize = dc.get("bucketize_dense", True)
        self.bucket_method = dc.get("bucket_method", "log")
        self.num_buckets = dc.get("num_buckets", 100)

        print(f"[Criteo1TB-Parquet] split={split}, days={self.days}, "
              f"bucketize={self.bucketize}({self.bucket_method})")

        # 收集所有 parquet 文件
        self.files = []  # list of parquet file paths
        for day_date in self.days:
            day_files = _find_parquet_files(data_path, day_date)
            if day_files:
                self.files.extend(day_files)
                print(f"  day={day_date}: {len(day_files)} parquet files")
            else:
                print(f"  day={day_date}: WARNING - no parquet files found!")

        if not self.files:
            raise FileNotFoundError(
                f"No parquet files found in {data_path}/data/ for days={self.days}"
            )

        print(f"[Criteo1TB-Parquet] Total: {len(self.files)} parquet files")

        # ---- Pass-1: 统计行数 + 分桶边界 ----
        need_boundaries = (
            split == "train"
            and self.bucketize
            and Criteo1TBParquetDataset._shared_boundaries is None
        )
        SAMPLE_SIZE = 2_000_000

        print(f"[Criteo1TB-Parquet] Pass-1: counting rows"
              f"{' + vectorized sampling' if need_boundaries else ''}...")

        # 快速统计行数（只读 metadata，不读数据）
        n_total = 0
        file_row_counts = []
        for fpath in self.files:
            pf = pq.ParquetFile(fpath)
            rows = pf.metadata.num_rows
            file_row_counts.append(rows)
            n_total += rows
            if 0 < self.max_rows <= n_total:
                n_total = self.max_rows
                break

        self.n_total = n_total
        print(f"[Criteo1TB-Parquet] Total: {self.n_total:,} rows "
              f"({len(file_row_counts)} files scanned)")

        # 向量化采样：从随机子集文件中读取 dense 列，拼接后随机抽样
        if need_boundaries:
            rng = np.random.RandomState(42)
            # 选取足够多的文件凑够 SAMPLE_SIZE 行
            # 每文件平均 ~n_total/n_files 行，选 ceil(SAMPLE_SIZE/avg_rows) 个文件
            n_files = len(file_row_counts)
            avg_rows = n_total / max(n_files, 1)
            n_sample_files = min(n_files, max(1, int(np.ceil(SAMPLE_SIZE / avg_rows)) + 2))
            sample_file_idxs = rng.choice(n_files, size=n_sample_files, replace=False)
            sample_file_idxs.sort()

            print(f"[Criteo1TB-Parquet] Sampling from {n_sample_files}/{n_files} files "
                  f"for bucket boundaries...")

            sampled_dense = []  # list of np.float32 [rows_i, 13]
            total_sampled = 0
            for fidx in sample_file_idxs:
                pf = pq.ParquetFile(self.files[fidx])
                table = pf.read(columns=DENSE_COLS)
                # null → 0, vectorized
                arrays = []
                for col_name in DENSE_COLS:
                    col = table.column(col_name)
                    filled = pc.if_else(pc.is_null(col), 0, col)
                    arrays.append(filled.to_numpy(zero_copy_only=False).astype(np.float32))
                dense_block = np.stack(arrays, axis=1)  # [rows, 13]
                sampled_dense.append(dense_block)
                total_sampled += dense_block.shape[0]
                if total_sampled >= SAMPLE_SIZE * 2:
                    break

            all_dense = np.concatenate(sampled_dense, axis=0)  # [total_sampled, 13]
            # 随机下采样到 SAMPLE_SIZE
            if all_dense.shape[0] > SAMPLE_SIZE:
                idxs = rng.choice(all_dense.shape[0], size=SAMPLE_SIZE, replace=False)
                all_dense = all_dense[idxs]

            print(f"[Criteo1TB-Parquet] Sampled {all_dense.shape[0]:,} rows for boundaries")

            # 转为 reservoir 格式（list of lists）以复用 _build_boundaries
            reservoirs = [all_dense[:, i].tolist() for i in range(13)]
            boundaries = _build_boundaries(reservoirs, self.bucket_method,
                                           self.num_buckets)
            Criteo1TBParquetDataset._shared_boundaries = boundaries
            print(f"[Criteo1TB-Parquet] Bucket boundaries built for 13 dense cols")

        # Cardinalities
        if split == "train" and Criteo1TBParquetDataset._shared_cardinalities is None:
            Criteo1TBParquetDataset._shared_cardinalities = [c + 2 for c in _DEFAULT_CARDINALITIES]

        self.boundaries = Criteo1TBParquetDataset._shared_boundaries
        self.cards = Criteo1TBParquetDataset._shared_cardinalities or \
                     [c + 2 for c in _DEFAULT_CARDINALITIES]

        # ---- 更新 config ----
        if split == "train":
            bucket_card = self.num_buckets + 2
            if self.bucketize and self.boundaries is not None:
                all_cards = list(self.cards) + [bucket_card] * 13
                dc["sparse_cols"] = SPARSE_COLS_OUT + [f"{c}_bucket" for c in DENSE_COLS_OUT]
                dc["cardinalities"] = all_cards
                dc["num_dense"] = 0
                dc["dense_cols"] = []
            else:
                dc["sparse_cols"] = SPARSE_COLS_OUT
                dc["cardinalities"] = list(self.cards)
                dc["num_dense"] = 13
                dc["dense_cols"] = DENSE_COLS_OUT

        n_sparse_out = 39 if (self.bucketize and self.boundaries is not None) else 26
        print(f"[Criteo1TB-Parquet] {split}: {self.n_total:,} samples, "
              f"sparse_cols={n_sparse_out}, "
              f"dense={'none (bucketized)' if self.bucketize and self.boundaries is not None else 13}")

    # ---------------------------------------------------------------
    # IterableDataset 接口
    # ---------------------------------------------------------------

    def __len__(self):
        return self.n_total

    def __iter__(self):
        # 多 worker 分片
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        if worker_info is not None:
            wid = worker_info.id
            nw = worker_info.num_workers
            files = [f for i, f in enumerate(files) if i % nw == wid]

        n_yielded = 0
        for fpath in files:
            pf = pq.ParquetFile(fpath)
            for rg_idx in range(pf.metadata.num_row_groups):
                # 读取整个 row-group（通常几万 ~ 几十万行）
                table = pf.read_row_group(rg_idx)

                # 提取列为 numpy
                labels = table.column("label").to_numpy()
                dense_arrays = []
                for col_name in DENSE_COLS:
                    col = table.column(col_name)
                    # null → 0 (pure pyarrow, no pandas dependency)
                    filled = pc.if_else(pc.is_null(col), 0, col)
                    arr = filled.to_numpy(zero_copy_only=False).astype(np.float32)
                    dense_arrays.append(arr)
                dense_matrix = np.stack(dense_arrays, axis=1)  # [N, 13]

                sparse_arrays = []
                for col_idx, col_name in enumerate(SPARSE_COLS_PARQUET):
                    col = table.column(col_name).to_pylist()
                    card = self.cards[col_idx]
                    arr = np.zeros(len(col), dtype=np.int64)
                    for j, v in enumerate(col):
                        if v is not None and v != "":
                            try:
                                raw = int(v, 16) & 0xFFFFFFFF
                                arr[j] = (raw % (card - 1)) + 1
                            except ValueError:
                                arr[j] = 0
                    sparse_arrays.append(arr)
                sparse_matrix = np.stack(sparse_arrays, axis=1)  # [N, 26]

                # 分桶
                if self.bucketize and self.boundaries is not None:
                    bucket_matrix = _apply_boundaries_batch(
                        dense_matrix, self.boundaries, self.num_buckets
                    )  # [N, 13]
                    sparse_final = np.concatenate(
                        [sparse_matrix, bucket_matrix], axis=1
                    )  # [N, 39]

                    for i in range(len(labels)):
                        yield {
                            "sparse": sparse_final[i],
                            "label": np.float32(labels[i]),
                        }
                        n_yielded += 1
                        if self.max_rows > 0 and n_yielded >= self.max_rows:
                            return
                else:
                    dense_out = np.log1p(np.abs(dense_matrix)) * np.sign(dense_matrix)
                    for i in range(len(labels)):
                        yield {
                            "sparse": sparse_matrix[i],
                            "dense": dense_out[i],
                            "label": np.float32(labels[i]),
                        }
                        n_yielded += 1
                        if self.max_rows > 0 and n_yielded >= self.max_rows:
                            return
