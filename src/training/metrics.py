"""
Evaluation metrics for binary classification.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Core metrics
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

    # Ranking metrics (handle edge cases)
    try:
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    except ValueError:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0

    # Confusion matrix components
    if len(y_true) > 0:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)

        # Specificity (true negative rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['tp'] = 0
        metrics['fp'] = 0
        metrics['tn'] = 0
        metrics['fn'] = 0
        metrics['specificity'] = 0.0

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Find the optimal probability threshold for classification.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        thresholds: Array of thresholds to try (default: 0.01 to 0.99)

    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for each line
    """
    if prefix:
        prefix = f"{prefix} "

    print(f"{prefix}F1: {metrics['f1']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall: {metrics['recall']:.4f}")

    if 'roc_auc' in metrics:
        print(f"{prefix}ROC-AUC: {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"{prefix}PR-AUC: {metrics['pr_auc']:.4f}")

    if 'tp' in metrics:
        print(f"{prefix}TP: {metrics['tp']}, FP: {metrics['fp']}, "
              f"TN: {metrics['tn']}, FN: {metrics['fn']}")
