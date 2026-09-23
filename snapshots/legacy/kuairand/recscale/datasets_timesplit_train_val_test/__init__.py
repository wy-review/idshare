"""Temporal train/val/test split utilities and experimental readers.

This package is intentionally separate from recscale.datasets so the new
last-day/penultimate-day split work can be developed without changing existing
readers or mutating remote data.
"""

# Import implemented readers so they register with recscale.datasets when this
# package is imported. These readers are non-destructive: TAAC/Taobao temporal
# readers expect already prepared train/val/test files and do not create them.
for _mod in ("kuairec_temporal", "kuairand_temporal", "taac_temporal", "taobao_temporal"):
    try:
        __import__(f"{__name__}.{_mod}")
    except Exception:
        # Keep package import lightweight for planning/report tooling. Some
        # readers require optional dependencies such as pyarrow.
        pass
