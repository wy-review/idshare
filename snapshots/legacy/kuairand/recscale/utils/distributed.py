"""
recscale.utils.distributed — DDP 分布式训练工具
"""

import os
import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    """
    初始化 DDP。通过 torchrun 环境变量自动检测。

    Returns: (rank, world_size, local_rank)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = os.environ.get("DIST_BACKEND", "nccl")
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def cleanup_distributed():
    """销毁进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """是否主进程 (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def gather_predictions(preds: list, labels: list, world_size: int, device: torch.device):
    """
    DDP 环境下收集所有 GPU 的预测结果。
    单卡时直接返回。
    """
    if world_size <= 1:
        return preds, labels

    import numpy as np

    preds_tensor = torch.tensor(preds, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)

    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)

    all_preds = torch.cat(gathered_preds).cpu().numpy().tolist()
    all_labels = torch.cat(gathered_labels).cpu().numpy().tolist()
    return all_preds, all_labels
