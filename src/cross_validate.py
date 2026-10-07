"""Stratified k-fold cross-validation of the MLP.

A single train/test split on ~300 rows is noisy, so this script estimates
performance with k-fold cross-validation and reports mean +/- std for each
metric. The preprocessor is re-fit *inside* every fold (fit on the training
folds only) to avoid any information leaking from the held-out fold.

Usage:
    python -m src.cross_validate            # 5 folds, defaults
    python -m src.cross_validate --folds 10 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor, load_raw
from .model import HeartMLP
from .train import set_seed


def _train_one_fold(X_tr, y_tr, X_va, y_va, args, device) -> dict:
    model = HeartMLP(
        X_tr.shape[1], hidden_dims=tuple(int(h) for h in args.hidden.split(",") if h),
        dropout=args.dropout,
    ).to(device)
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True,
    )
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X_va).to(device))).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_va.astype(int), pred)),
        "f1": float(f1_score(y_va.astype(int), pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_va.astype(int), prob)),
    }


def cross_validate(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_raw(args.data)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    per_fold: list[dict] = []
    for i, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        pre = build_preprocessor()
        X_tr = pre.fit_transform(X.iloc[tr_idx]).astype(np.float32)
        X_va = pre.transform(X.iloc[va_idx]).astype(np.float32)
        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[va_idx].astype(np.float32)
        m = _train_one_fold(X_tr, y_tr, X_va, y_va, args, device)
        per_fold.append(m)
        print(f"fold {i}/{args.folds}: acc {m['accuracy']:.3f}  f1 {m['f1']:.3f}  auc {m['roc_auc']:.3f}")

    summary = {}
    for metric in ("accuracy", "f1", "roc_auc"):
        vals = np.array([f[metric] for f in per_fold])
        summary[metric] = {"mean": float(vals.mean()), "std": float(vals.std())}

    result = {"folds": args.folds, "per_fold": per_fold, "summary": summary}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "cv_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{args.folds}-fold cross-validation (mean +/- std):")
    for metric, s in summary.items():
        print(f"  {metric:9s}: {s['mean']:.3f} +/- {s['std']:.3f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-validate the heart-disease MLP")
    p.add_argument("--data", default="data/heart.csv")
    p.add_argument("--outdir", default="results")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", default="64,32")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    cross_validate(build_parser().parse_args())
