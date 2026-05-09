from pathlib import Path
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.models.base import BaseModel


def evaluate(
    model: BaseModel,
    loader: DataLoader,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Run full evaluation on a DataLoader and return classification metrics.

    Metrics computed:
    - Accuracy
    - F1 score (binary, positive class = PNEUMONIA)
    - AUC-ROC
    - Precision
    - Recall (sensitivity)
    - Specificity (true negative rate)
    - Confusion matrix

    Args:
        model: Trained BaseModel subclass.
        loader: DataLoader for the evaluation set (val or test).
        device: torch device. Auto-detected if None.

    Returns:
        Dict of metric name -> value.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]  # prob of PNEUMONIA
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy":    accuracy_score(all_labels, all_preds),
        "f1":          f1_score(all_labels, all_preds, pos_label=1),
        "auc_roc":     roc_auc_score(all_labels, all_probs),
        "precision":   precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "recall":      recall_score(all_labels, all_preds, pos_label=1),
        "specificity": specificity,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = ""):
    """Pretty-print evaluation metrics."""
    header = f"Evaluation Results — {model_name}" if model_name else "Evaluation Results"
    print(f"\n{header}")
    print("-" * 45)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Confusion Matrix: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")