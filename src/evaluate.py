"""
Evaluation utilities:
  - Runs inference on a DataLoader
  - Computes accuracy, precision, recall, F1, AUC
  - Saves confusion matrix, ROC curve, and training curves
"""
import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        labels   – ground-truth labels
        preds    – predicted class indices
        probs    – softmax probability of class 1 (fake)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            with autocast(device.type if device.type == "cuda" else "cpu", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> dict:
    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auc":       roc_auc_score(labels, probs),
    }
    logger.info("─" * 40)
    logger.info("Evaluation Results")
    logger.info("─" * 40)
    for k, v in metrics.items():
        logger.info(f"  {k:12s}: {v:.4f}")
    logger.info("\n" + classification_report(labels, preds, target_names=["Real", "Fake"]))
    return metrics


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: str,
) -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    auc: float,
    save_path: str,
) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved → {save_path}")


def plot_training_curves(history: dict, save_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved → {path}")


def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    results_dir: str,
    history: dict | None = None,
) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    labels, preds, probs = predict_loader(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)

    plot_confusion_matrix(labels, preds, os.path.join(plots_dir, "confusion_matrix.png"))
    plot_roc_curve(labels, probs, metrics["auc"], os.path.join(plots_dir, "roc_curve.png"))

    if history:
        plot_training_curves(history, plots_dir)

    # Save raw metrics
    import json
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)

    return metrics
