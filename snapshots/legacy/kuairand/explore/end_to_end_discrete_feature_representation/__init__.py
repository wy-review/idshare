"""Analysis tools for the frozen end-to-end discrete representation study.

The exports below preserve compatibility for the archived synthetic prototype.
They are not part of the current F4 model or result claims.
"""

__all__ = [
    "EndToEndDiscreteClassifier",
    "HierarchicalQuantizerOutput",
    "ZeroAnchoredHierarchicalQuantizer",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from explore._archive_invalid_protocol.end_to_end_discrete_feature_representation_pre_f4.scripts import (
        hierarchical_discrete,
    )

    return getattr(hierarchical_discrete, name)
