"""Evaluate a trained model and generate plots.

Can be run standalone (after training):
    python -m src.evaluate

It reproduces the exact deterministic test split used during training,
computes the standard classification metrics and saves:
    metrics.json, confusion_matrix.png, roc_curve.png, training_curves.png
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .data import Dataset, prepare_data
from .model import HeartMLP


def _probs(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def compute_and_save_metrics(model, data: Dataset, device, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    y_true = data.y_test.astype(int)
    y_prob = _probs(model, data.X_test, device)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "n_test": int(len(y_true)),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(outdir, "classification_report.txt"), "w") as f:
        f.write(
            classification_report(
                y_true, y_pred, target_names=["No disease", "Disease"], zero_division=0
            )
        )

    _plot_confusion(y_true, y_pred, outdir)
    _plot_roc(y_true, y_prob, metrics["roc_auc"], outdir)
    _plot_training_curves(outdir)
    return metrics


def _plot_confusion(y_true, y_pred, outdir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No disease", "Disease"])
    ax.set_yticks([0, 1], labels=["No disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=14,
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def _plot_roc(y_true, y_prob, auc: float, outdir: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_curve.png"), dpi=120)
    plt.close(fig)


def _plot_training_curves(outdir: str) -> None:
    path = os.path.join(outdir, "history.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(epochs, h["train_loss"], label="train")
    ax1.plot(epochs, h["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, h["train_acc"], label="train")
    ax2.plot(epochs, h["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "training_curves.png"), dpi=120)
    plt.close(fig)


def load_model(outdir: str, device) -> HeartMLP:
    with open(os.path.join(outdir, "config.json")) as f:
        cfg = json.load(f)
    model = HeartMLP(
        cfg["input_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(outdir, "model.pt"), map_location=device))
    model.eval()
    return model


def main(outdir: str = "results") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(outdir, "config.json")) as f:
        seed = json.load(f).get("seed", 42)
    data = prepare_data(seed=seed)
    model = load_model(outdir, device)
    metrics = compute_and_save_metrics(model, data, device, outdir)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}" if isinstance(v, float) else f"  {k:10s}: {v}")
    print(f"\nPlots and reports written to {outdir}/")


if __name__ == "__main__":
    main()
