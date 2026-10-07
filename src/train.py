"""Train the heart-disease MLP.

Usage:
    python -m src.train                       # sensible defaults
    python -m src.train --epochs 300 --lr 1e-3

Artifacts written to ``--outdir`` (default ``results/``):
    model.pt            best model weights (lowest validation loss)
    preprocessor.joblib fitted preprocessing pipeline
    config.json         architecture + training configuration
    history.json        per-epoch train/val loss and accuracy
    metrics.json        final test-set metrics
"""

from __future__ import annotations

import argparse
import json
import os
import random

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import prepare_data
from .model import HeartMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    data = prepare_data(path=args.data, seed=args.seed)
    train_loader = _loader(data.X_train, data.y_train, args.batch_size, shuffle=True)
    val_loader = _loader(data.X_val, data.y_val, args.batch_size, shuffle=False)
    test_loader = _loader(data.X_test, data.y_test, args.batch_size, shuffle=False)

    hidden = tuple(int(h) for h in args.hidden.split(",") if h)
    model = HeartMLP(data.input_dim, hidden_dims=hidden, dropout=args.dropout).to(device)

    # Slight class weighting in case the split is mildly imbalanced.
    pos = float(data.y_train.sum())
    neg = float(len(data.y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, patience_left = float("inf"), None, args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        tr_loss, tr_acc = _evaluate(model, train_loader, criterion, device)
        va_loss, va_acc = _evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d}/{args.epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.3f}"
            )

        if patience_left <= 0:
            print(f"early stopping at epoch {epoch} (best val loss {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- persist artifacts -------------------------------------------------
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    joblib.dump(data.preprocessor, os.path.join(args.outdir, "preprocessor.joblib"))
    config = {
        "input_dim": data.input_dim,
        "hidden_dims": list(hidden),
        "dropout": args.dropout,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_run": len(history["train_loss"]),
        "feature_names": data.feature_names,
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- final test metrics (delegated to evaluate.py) ---------------------
    from .evaluate import compute_and_save_metrics

    metrics = compute_and_save_metrics(model, data, device, args.outdir)
    print("\nTest metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:.4f}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the heart-disease MLP")
    p.add_argument("--data", default="data/heart.csv")
    p.add_argument("--outdir", default="results")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", default="64,32", help="comma-separated hidden sizes")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=25, help="early-stopping patience")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
