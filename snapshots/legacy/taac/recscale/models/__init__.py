# Anonymous release: omitted unused internal adapters.
"""
recscale.models — 模型注册表
"""

from typing import Type
from .base import RecModel

_REGISTRY: dict[str, Type[RecModel]] = {}


def register_model(cls: Type[RecModel]) -> Type[RecModel]:
    """装饰器: 注册模型到全局 registry"""
    _REGISTRY[cls.model_name] = cls
    return cls


def create_model(config: dict) -> RecModel:
    """根据 config 创建模型实例"""
    name = config["model"]["name"]
    if name not in _REGISTRY:
        # 触发所有模型模块的 import (lazy load)
        _import_all_models()
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return _REGISTRY[name](config)


def _import_all_models():
    """Import 所有模型子模块, 触发 @register_model"""
    from . import mlp, dcn, deepfm, din, din_enhanced  # noqa: F401
    from . import wukong, rankmixer, onetrans, hyformer  # noqa: F401
    from . import mixer, unimixer, hybrid  # noqa: F401
    from . import rankmixer_v2, unimixer_v2, tokenmixer_large_v2  # noqa: F401
    from . import tokenmixer_large_v3  # noqa: F401
    from . import tokenmixer_large_aux  # noqa: F401
    from . import tokenmixer_large_moe  # noqa: F401
    from . import onetrans_v2, hyformer_v2, hyformer_v3  # noqa: F401
    from . import hyformer_v2_propagate_nstokens  # noqa: F401
    from . import hyformer_v2_user_condition  # noqa: F401
    from . import mixformer  # noqa: F401
    from . import projected_controls  # noqa: F401
    from . import s2drec  # noqa: F401
    try:
        from . import sasrec  # noqa: F401
    except ImportError:
        pass
