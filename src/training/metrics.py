from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique_values = np.unique(y_true)
    if len(unique_values) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def multilabel_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    results = {}

    for i, label in enumerate(label_names):
        results[label] = safe_roc_auc(y_true[:, i], y_prob[:, i])

    valid_scores = [score for score in results.values() if not np.isnan(score)]
    results["mean_auroc"] = float(np.mean(valid_scores)) if valid_scores else float("nan")

    return results