"""
recscale.evaluator — CTR 评估指标
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


class CTREvaluator:
    """CTR 判别式任务评估: AUC + LogLoss"""

    @staticmethod
    def compute(labels: list, scores: list) -> dict:
        labels = np.array(labels)
        scores = np.array(scores)

        # Clip scores to avoid log(0)
        scores = np.clip(scores, 1e-7, 1 - 1e-7)

        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.0

        try:
            logloss = log_loss(labels, scores)
        except ValueError:
            logloss = float("inf")

        return {
            "auc": auc,
            "logloss": logloss,
            "num_samples": len(labels),
            "num_pos": int((labels > 0.5).sum()),
            "pos_rate": float((labels > 0.5).mean()),
        }
