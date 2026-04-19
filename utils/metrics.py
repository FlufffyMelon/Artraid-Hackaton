"""Метрики классификации и подбор порога решения."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(
    y_true,
    y_proba,
    lo: float = 0.05,
    hi: float = 0.81,
    step: float = 0.01,
) -> tuple[float, float]:
    """Найти порог, максимизирующий F1 на диапазоне вероятностей.

    Возвращает ``(best_f1, best_threshold)``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(lo, hi, step):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_f1, best_t


def compute_classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Полный набор метрик качества бинарного классификатора.

    Требуется ≥2 класса в ``y_true`` — иначе ROC/PR AUC недоступны и
    возвращаются как ``None``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    multiclass = len(np.unique(y_true)) > 1
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if multiclass else None,
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if multiclass else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if multiclass else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }
