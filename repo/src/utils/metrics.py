"""Metric computation used identically by every experiment (E01-E07) so
that numbers are comparable across representations / models / feature sets,
per spec sections 14-18.

No metric here is invented -- everything is a thin, labeled wrapper around
sklearn so the exact formula is traceable.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    scores = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
    return dict(zip(class_names, scores.tolist()))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def top_confusion_pairs(cm: np.ndarray, class_names: List[str], top_k: int = 10):
    """Return the top-k (true, pred, count) off-diagonal confusion pairs,
    for src/experiments/error_analysis.py. Purely descriptive -- no cause
    is attributed here, that happens only against real inspected samples.
    """
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def confidence_interval_binomial(accuracy: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wald interval for a proportion (accuracy). Adequate for reporting
    a CI on test accuracy without pulling in extra dependencies; swap for
    a bootstrap CI in evaluate.py if n is small (<100) or accuracy is
    near 0/1, where Wald is known to be a poor approximation.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    se = (accuracy * (1 - accuracy) / n) ** 0.5
    return (max(0.0, accuracy - z * se), min(1.0, accuracy + z * se))
