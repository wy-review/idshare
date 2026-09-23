"""training_report.json writer for FuxiCTR.

Generates a structured JSON report after training, compatible with
the remote-training-platform ``get_job_report`` API.  The report
file is written to the job working directory root so that
``worker_agent._build_report_summary()`` picks it up automatically.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _total_seconds(started_at: Optional[str],
                   finished_at: Optional[str]) -> Optional[float]:
    t0, t1 = _parse_iso(started_at), _parse_iso(finished_at)
    if t0 and t1:
        return round((t1 - t0).total_seconds(), 2)
    return None


def _gpu_name(model) -> str:
    """Best-effort GPU name from the model device."""
    try:
        if model.device.type == "cuda":
            return torch.cuda.get_device_name(model.device)
    except Exception:
        pass
    return "cpu"


def _param_counts(model) -> Dict[str, int]:
    """Return total and trainable param counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_params": total, "n_trainable_params": trainable}


def _best_epoch_from_history(
    history: List[Dict[str, Any]],
    monitor_metric: str,
    monitor_mode: str,
) -> Optional[int]:
    """Derive the best epoch from accumulated history."""
    if not history or monitor_metric not in history[0]:
        return None
    values = [(h.get("epoch"), h.get(monitor_metric)) for h in history]
    values = [(e, v) for e, v in values if v is not None]
    if not values:
        return None
    if monitor_mode == "max":
        return max(values, key=lambda x: x[1])[0]
    else:
        return min(values, key=lambda x: x[1])[0]


def build_training_report(
    params: Dict[str, Any],
    model: Any,
    valid_result: Dict[str, Any],
    test_result: Dict[str, Any],
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the ``training_report.json`` payload as a dict.

    Args:
        params: The full param dict from ``load_config()``.
        model: The trained BaseModel instance (after ``fit()``).
        valid_result: Dict returned by ``model.evaluate(valid_gen)``.
        test_result: Dict returned by ``model.evaluate(test_gen)``.
        started_at: ISO timestamp from ``model._started_at``.
        finished_at: ISO timestamp from ``model._finished_at``.
    """
    started = started_at or _iso_now()
    finished = finished_at or _iso_now()

    history = getattr(model, "_history", [])
    monitor = getattr(model, "_monitor", None)
    # Extract metric name from Monitor object (kv_pairs is a dict like {"AUC": 1})
    if monitor is not None and hasattr(monitor, "kv_pairs"):
        monitor_metric = ",".join(monitor.kv_pairs.keys())
    else:
        monitor_metric = str(monitor)
    monitor_mode = getattr(model, "_monitor_mode", "max")

    # Best epoch / metric
    best_epoch = _best_epoch_from_history(history, monitor_metric, monitor_mode)
    best_valid_metric = getattr(model, "_best_metric", None)

    # Param counts
    pc = _param_counts(model)
    # Detailed param breakdown from count_parameters()
    pc["sparse_params"] = getattr(model, "_sparse_params", None)
    pc["dense_params"] = getattr(model, "_dense_params", None)
    gflops = getattr(model, "_gflops", None)

    # Optimizer info
    optimizer_info: Dict[str, Any] = {}
    opt = getattr(model, "optimizer", None)
    if opt is not None:
        optimizer_info["type"] = type(opt).__name__
        optimizer_info["lr"] = opt.param_groups[0].get("lr")

    # GPU
    gpu = _gpu_name(model)

    report = {
        "experiment_id": params.get("experiment_id",
                                    params.get("model_id",
                                               getattr(model, "model_id", ""))),
        "model_id": params.get("model_id", getattr(model, "model_id", "")),
        "model_class": f"{type(model).__module__}:{type(model).__name__}",
        "model_type": params.get("model", ""),
        "dataset_id": params.get("dataset_id", ""),
        "started_at": started,
        "finished_at": finished,
        "total_time_sec": _total_seconds(started, finished),
        "gpu": gpu,
        "host": socket.gethostname(),
        "epochs_planned": getattr(model, "_epochs_planned", None),
        "early_stop": getattr(model, "_stop_training", False),
        "best_epoch": best_epoch,
        "best_valid_metric": best_valid_metric,
        "best_valid_metric_name": monitor_metric,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "valid_metrics": dict(valid_result) if valid_result else {},
        "test_metrics": dict(test_result) if test_result else {},
        "history": list(history),
        **pc,
        "optimizer": optimizer_info,
        "batch_size": params.get("batch_size"),
        "embedding_regularizer": params.get("embedding_regularizer"),
        "net_regularizer": params.get("net_regularizer"),
        "gflops_per_batch": gflops,
    }
    return report


def write_training_report(report: Dict[str, Any], output_dir: str,
                          filename: str = "training_report.json") -> str:
    """Write ``report`` as JSON under ``output_dir`` and return the path.

    Pretty-printed so human review is bearable; ``ensure_ascii=False``
    preserves any cjk content in model_id / experiment_id etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    return path
