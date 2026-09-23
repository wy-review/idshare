"""
recscale.datasets.csv_dataset — CSV/ZIP 格式数据集加载

支持: Criteo, Avazu, TaobaoAd, KuaiRec, Amazon

Dense 分桶功能 (bucketize_dense):
  将 dense 特征离散化为 sparse 特征，去掉 dense 输入，全部走 embedding。
  支持策略:
  - "log": log1p 后等宽分桶 (适合 Criteo 长尾分布)
  - "quantile": 等频分桶 (每个桶样本数相等)
  - "uniform": 等宽分桶 (按值域均匀切分)
  配置: bucketize_dense: true / "log" / "quantile" / "uniform"
        num_buckets: 100 (默认)
"""

import csv
import io
import os
import zipfile
from pathlib import Path

import numpy as np

from . import register_dataset
from .base import BaseDataset


@register_dataset("csv")
class CSVDataset(BaseDataset):
    """
    从 CSV 或 ZIP 内嵌 CSV 加载数据。

    YAML 配置示例:
    ```yaml
    dataset:
      type: csv
      path: /path/to/data.zip
      label_col: label
      dense_cols: [I1, I2, ..., I13]
      sparse_cols: [C1, C2, ..., C26]
      bucketize_dense: true       # 将 dense 特征分桶转为 sparse (可选)
      num_buckets: 100            # 分桶数 (默认 100)
      bucket_method: log          # 分桶方法: log/quantile/uniform (默认 log)
      split:
        train: train.csv
        test: test.csv
      max_rows: 0
    ```
    """

    # 类级别共享: train 构建的 feature_maps 供 test/valid 复用
    _shared_feature_maps: dict = None
    _shared_cardinalities: list = None
    _shared_bucket_boundaries: dict = None  # dense 分桶边界

    def __init__(self, config: dict, split: str = "train"):
        dc = config["dataset"]
        self.label_col = dc["label_col"]
        # 注意：train 分桶后会修改 dc，所以用 _original_dense_cols 恢复
        self.dense_cols = dc.get("_original_dense_cols") or dc.get("dense_cols") or []
        self.sparse_cols = dc.get("_original_sparse_cols") or dc.get("sparse_cols") or []
        max_rows = dc.get("max_rows", 0)

        # Dense 分桶配置
        bucketize = dc.get("bucketize_dense", False)
        self.bucketize_dense = bool(bucketize)
        self.bucket_method = bucketize if isinstance(bucketize, str) else dc.get("bucket_method", "log")
        self.num_buckets = dc.get("num_buckets", 100)

        # 确定文件路径
        data_path = dc["path"]
        split_file = dc["split"][split]

        # 加载数据
        print(f"[CSVDataset] Loading {split}: {split_file} from {data_path}")
        rows = self._read_csv(data_path, split_file, max_rows)
        print(f"[CSVDataset] Loaded {len(rows):,} rows")

        # 构建或复用特征映射
        if split == "train":
            if "cardinalities" not in dc or not dc["cardinalities"]:
                print("[CSVDataset] Building feature maps from data...")
                self.feature_maps, cardinalities = self._build_feature_maps(rows)
                dc["cardinalities"] = cardinalities
                CSVDataset._shared_feature_maps = self.feature_maps
                CSVDataset._shared_cardinalities = cardinalities
            else:
                self.feature_maps = None
        else:
            self.feature_maps = CSVDataset._shared_feature_maps
            if self.feature_maps is None:
                print("[CSVDataset] Warning: no shared feature maps, using hash fallback")

        # 转为 numpy
        self.labels = np.array(
            [float(r[self.label_col]) for r in rows], dtype=np.float32
        )

        # Dense 特征处理
        self.dense = None
        dense_as_sparse = None
        if self.dense_cols:
            raw_dense = np.array(
                [[self._safe_float(r.get(c, "0")) for c in self.dense_cols] for r in rows],
                dtype=np.float32,
            )

            if self.bucketize_dense:
                # 分桶: dense → sparse indices
                if split == "train":
                    dense_as_sparse, boundaries = self._bucketize_train(raw_dense)
                    CSVDataset._shared_bucket_boundaries = boundaries
                    print(f"[CSVDataset] Bucketized {len(self.dense_cols)} dense cols → "
                          f"sparse ({self.num_buckets} buckets, method={self.bucket_method})")
                else:
                    boundaries = CSVDataset._shared_bucket_boundaries
                    if boundaries is None:
                        print("[CSVDataset] Warning: no bucket boundaries, falling back to raw dense")
                        self.dense = np.log1p(np.abs(raw_dense)) * np.sign(raw_dense)
                    else:
                        dense_as_sparse = self._bucketize_test(raw_dense, boundaries)
                        print(f"[CSVDataset] Bucketized {len(self.dense_cols)} dense cols (test)")
            else:
                # 传统方式: log1p 标准化
                self.dense = np.log1p(np.abs(raw_dense)) * np.sign(raw_dense)

        # Sparse 特征
        self.sparse = np.array(
            [[self._map_sparse(r.get(c, ""), c, col_idx)
              for col_idx, c in enumerate(self.sparse_cols)] for r in rows],
            dtype=np.int64,
        )

        # 如果 dense 分桶了，拼接到 sparse 后面
        if dense_as_sparse is not None:
            self.sparse = np.concatenate([self.sparse, dense_as_sparse], axis=1)
            # 更新 config: 告诉模型 dense_cols 已经没了，sparse_cols 多了
            if split == "train":
                bucket_card = self.num_buckets + 2  # +1 padding, +1 overflow
                new_cards = [bucket_card] * len(self.dense_cols)
                dc["cardinalities"] = dc["cardinalities"] + new_cards
                # 保存原始列名供 test 使用
                dc["_original_dense_cols"] = list(self.dense_cols)
                dc["_original_sparse_cols"] = list(self.sparse_cols)
                dc["sparse_cols"] = self.sparse_cols + [f"{c}_bucket" for c in self.dense_cols]
                dc["dense_cols"] = []  # 清空 dense（模型不再接收 dense input）
                print(f"[CSVDataset] Updated config: {len(dc['sparse_cols'])} sparse cols "
                      f"(was {len(self.sparse_cols)}), 0 dense cols")
            self.dense = None  # 不再有 dense

        # Validate sparse indices
        if hasattr(self, 'sparse') and len(self.sparse) > 0:
            cardinalities = dc.get("cardinalities") or [10000] * self.sparse.shape[1]
            for col_idx, card in enumerate(cardinalities):
                if col_idx < self.sparse.shape[1]:
                    max_idx = self.sparse[:, col_idx].max()
                    if max_idx >= card:
                        print(f"[CSVDataset] WARNING: Column {col_idx} "
                              f"has index {max_idx} >= cardinality {card}")

        n_pos = (self.labels > 0.5).sum()
        print(f"[CSVDataset] {split}: {len(rows):,} samples, "
              f"pos={n_pos:,} ({n_pos/len(rows)*100:.2f}%)")

    def _bucketize_train(self, raw_dense: np.ndarray):
        """在训练集上计算分桶边界并分桶"""
        n_cols = raw_dense.shape[1]
        boundaries = {}
        result = np.zeros_like(raw_dense, dtype=np.int64)

        for i in range(n_cols):
            col = raw_dense[:, i]

            if self.bucket_method == "log":
                # log1p 后等宽分桶
                transformed = np.log1p(np.abs(col)) * np.sign(col)
                vmin, vmax = transformed.min(), transformed.max()
                if vmax - vmin < 1e-8:
                    edges = np.array([vmin])
                else:
                    edges = np.linspace(vmin, vmax, self.num_buckets + 1)[1:-1]
                indices = np.searchsorted(edges, transformed) + 1  # 1-based

            elif self.bucket_method == "quantile":
                # 等频分桶
                percentiles = np.linspace(0, 100, self.num_buckets + 1)[1:-1]
                edges = np.percentile(col, percentiles)
                edges = np.unique(edges)  # 去重
                indices = np.searchsorted(edges, col) + 1

            elif self.bucket_method == "uniform":
                # 等宽分桶
                vmin, vmax = col.min(), col.max()
                if vmax - vmin < 1e-8:
                    edges = np.array([vmin])
                else:
                    edges = np.linspace(vmin, vmax, self.num_buckets + 1)[1:-1]
                indices = np.searchsorted(edges, col) + 1

            else:
                raise ValueError(f"Unknown bucket method: {self.bucket_method}")

            # Clamp to [1, num_buckets]
            indices = np.clip(indices, 1, self.num_buckets)
            result[:, i] = indices
            boundaries[i] = {"method": self.bucket_method, "edges": edges.tolist()}

        return result, boundaries

    def _bucketize_test(self, raw_dense: np.ndarray, boundaries: dict):
        """用训练集的边界对测试集分桶"""
        n_cols = raw_dense.shape[1]
        result = np.zeros_like(raw_dense, dtype=np.int64)

        for i in range(n_cols):
            col = raw_dense[:, i]
            info = boundaries.get(i, {})
            edges = np.array(info.get("edges", []))
            method = info.get("method", "log")

            if method == "log":
                transformed = np.log1p(np.abs(col)) * np.sign(col)
                indices = np.searchsorted(edges, transformed) + 1
            else:
                indices = np.searchsorted(edges, col) + 1

            indices = np.clip(indices, 1, self.num_buckets)
            result[:, i] = indices

        return result

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            "sparse": self.sparse[idx],
            "label": self.labels[idx],
        }
        if self.dense is not None:
            sample["dense"] = self.dense[idx]
        return sample

    def _read_csv(self, data_path: str, csv_name: str, max_rows: int) -> list[dict]:
        """从 ZIP 或目录读取 CSV"""
        rows = []

        if data_path.endswith(".zip"):
            with zipfile.ZipFile(data_path) as z:
                with z.open(csv_name) as f:
                    reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                    for i, row in enumerate(reader):
                        if 0 < max_rows <= i:
                            break
                        rows.append(row)
        else:
            csv_path = os.path.join(data_path, csv_name)
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if 0 < max_rows <= i:
                        break
                    rows.append(row)

        return rows

    def _build_feature_maps(self, rows: list[dict]):
        """构建 sparse 特征映射: value → index (1-based, 0=unknown)"""
        feature_maps = {}
        cardinalities = []
        for col in self.sparse_cols:
            vals = sorted(set(r.get(col, "") for r in rows))
            mapping = {v: i + 1 for i, v in enumerate(vals) if v}
            feature_maps[col] = mapping
            cardinalities.append(len(mapping) + 1)  # +1 for padding
        return feature_maps, cardinalities

    def _map_sparse(self, val: str, col: str, col_idx: int = 0) -> int:
        if self.feature_maps is not None:
            return self.feature_maps.get(col, {}).get(val, 0)
        if not val:
            return 0
        cardinalities = CSVDataset._shared_cardinalities or [10000] * len(self.sparse_cols)
        if col_idx < len(cardinalities):
            card = cardinalities[col_idx]
        else:
            card = 10000
        return hash(val) % (card - 1) + 1 if card > 1 else 0

    @staticmethod
    def _safe_float(s: str) -> float:
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0
