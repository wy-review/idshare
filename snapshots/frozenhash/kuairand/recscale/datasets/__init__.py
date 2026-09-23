"""
recscale.datasets — 数据集注册表
"""

from typing import Type
from .base import BaseDataset

_REGISTRY: dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    """装饰器: 注册数据集类型"""
    def decorator(cls: Type[BaseDataset]) -> Type[BaseDataset]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def create_dataset(config: dict, split: str = "train") -> BaseDataset:
    """根据 config 创建数据集实例"""
    dtype = config["dataset"]["type"]
    if dtype not in _REGISTRY:
        _import_all_datasets()
    if dtype not in _REGISTRY:
        raise ValueError(f"Unknown dataset type: {dtype}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[dtype](config, split=split)


def _import_all_datasets():
    from . import pointwise_csv  # noqa: F401
    try:
        import recscale.datasets_timesplit_train_val_test  # noqa: F401
    except ImportError:
        pass

    for mod in [
        "kuairec", "kuairec_pointwise", "kuairand", "amazon",
        "taobao_ad", "taobao_ad_enhanced", "taobao_mm",
        "taac2025_user", "taac2025_time",
        "openonerec",
        "kuairand27k_k1",
        "criteo_1tb",
        "criteo_1tb_parquet",
    ]:
        try:
            __import__(f"recscale.datasets.{mod}")
        except ImportError:
            try:
                import importlib
                importlib.import_module(f".{mod}", package=__name__)
            except ImportError:
                pass
