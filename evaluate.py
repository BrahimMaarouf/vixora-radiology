"""
evaluate.py — Comprehensive evaluation on the held-out test set.

Outputs:
  • Classification report (precision, recall, F1 per class)
  • Macro / weighted F1
  • Confusion matrix (saved as PNG)
  • Per-class ROC-AUC (one-vs-rest)
  • All results written to results/
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

import config
from dataset import build_dataloaders
from model import load_model

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def collect_predictions(model, loader) -> tuple:
    """
    Run the model on a DataLoader and collect predictions + probabilities.

    Returns:
        (all_labels, all_preds, all_probs)
        Shapes: (N,), (N,), (N, C)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─── Confusion Matrix Plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path: str) -> None:
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASSES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"[eval] Confusion matrix saved → {save_path}")


# ─── ROC-AUC ──────────────────────────────────────────────────────────────────

def compute_roc_auc(labels, probs) -> dict:
    """One-vs-rest ROC-AUC for each class."""
    auc_scores = {}
    n_classes  = probs.shape[1]
    for i in range(n_classes):
        binary_labels = (labels == i).astype(int)
        try:
            auc = roc_auc_score(binary_labels, probs[:, i])
        except ValueError:
            auc = float("nan")
        auc_scores[config.IDX_TO_CLASS[i]] = round(auc, 4)
    return auc_scores


# ─── Per-Class Recall ─────────────────────────────────────────────────────────

def compute_per_class_recall(labels, preds) -> dict:
    from sklearn.metrics import recall_score
    recall = recall_score(labels, preds, average=None, zero_division=0)
    return {config.IDX_TO_CLASS[i]: round(float(r), 4) for i, r in enumerate(recall)}


# ─── Main Evaluation ─────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str = config.BEST_CKPT) -> dict:
    logger.info("=" * 70)
    logger.info("Chest X-ray Diagnostic System — Test Set Evaluation")
    logger.info("=" * 70)

    # ── Load model + data ──────────────────────────────────────────────────────
    model = load_model(checkpoint_path)
    _, _, test_loader = build_dataloaders()

    # ── Collect predictions ────────────────────────────────────────────────────
    labels, preds, probs = collect_predictions(model, test_loader)

    # ── Metrics ───────────────────────────────────────────────────────────────
    report     = classification_report(labels, preds, target_names=config.CLASSES, zero_division=0)
    macro_f1   = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1= f1_score(labels, preds, average="weighted", zero_division=0)
    auc_scores = compute_roc_auc(labels, probs)
    recall_map = compute_per_class_recall(labels, preds)

    logger.info(f"\n{report}")
    logger.info(f"Macro F1   : {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"ROC-AUC    : {auc_scores}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, cm_path)

    # ── Serialize results ──────────────────────────────────────────────────────
    results = {
        "macro_f1"   : round(macro_f1,    4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc"    : auc_scores,
        "recall"     : recall_map,
        "report"     : report,
    }
    json_path = os.path.join(config.RESULTS_DIR, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[eval] Results saved → {json_path}")

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate Chest X-ray Classifier on test set")
    parser.add_argument("--ckpt", type=str, default=config.BEST_CKPT, help="Path to model checkpoint")
    args = parser.parse_args()

    results = evaluate(args.ckpt)
    print(f"\nMacro F1: {results['macro_f1']:.4f}")
