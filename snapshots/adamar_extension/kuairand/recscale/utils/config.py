"""
recscale.utils.config — YAML 配置加载、验证、命令行覆盖
"""

import yaml
import copy
from pathlib import Path
from typing import Any


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def merge_cli_overrides(config: dict, overrides: list[str]) -> dict:
    """
    合并命令行参数覆盖, 支持 dotted key:
      --training.batch_size 8192  →  config["training"]["batch_size"] = 8192
      --model.name dcn            →  config["model"]["name"] = "dcn"
    """
    config = copy.deepcopy(config)
    i = 0
    while i < len(overrides):
        key = overrides[i]
        if not key.startswith("--"):
            i += 1
            continue
        key = key.lstrip("-")
        if i + 1 >= len(overrides):
            break
        val = overrides[i + 1]
        i += 2

        # 自动类型推断
        val = _auto_cast(val)

        # dotted key → nested dict
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val

    return config


def _auto_cast(val: str) -> Any:
    """自动类型转换: int > float > bool > str"""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def validate_config(config: dict) -> list[str]:
    """验证配置, 返回错误列表 (空=OK)"""
    errors = []

    for section in ["dataset", "model", "training"]:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    if "dataset" in config:
        ds = config["dataset"]
        if "name" not in ds:
            errors.append("dataset.name is required")
        if "path" not in ds:
            errors.append("dataset.path is required")

    if "model" in config:
        m = config["model"]
        if "name" not in m:
            errors.append("model.name is required")

    if "training" in config:
        t = config["training"]
        if t.get("batch_size", 0) <= 0:
            errors.append("training.batch_size must be > 0")

    return errors


def get_config(config_path: str, cli_overrides: list[str] = None) -> dict:
    """加载 + 覆盖 + 验证"""
    config = load_config(config_path)
    if cli_overrides:
        config = merge_cli_overrides(config, cli_overrides)

    errors = validate_config(config)
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    # 设置默认值
    config.setdefault("seed", 42)
    config.setdefault("distributed", {})
    config["distributed"].setdefault("enabled", False)
    config["distributed"].setdefault("backend", "nccl")
    config["training"].setdefault("num_workers", 4)
    config["training"].setdefault("use_amp", False)
    config["training"].setdefault("log_every", 100)
    config["training"].setdefault("eval_every", 0)
    config["training"].setdefault("save_dir", "./outputs")
    config["training"].setdefault("optimizer", "adam")
    config["training"].setdefault("weight_decay", 0.0)
    config["training"].setdefault("grad_clip", 1.0)
    config["model"].setdefault("dropout", 0.0)

    return config
