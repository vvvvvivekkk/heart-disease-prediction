"""Classical ML baselines for comparison with the neural network.

Trains Logistic Regression, Random Forest and Gradient Boosting on the *same*
deterministic split used by the MLP, so the comparison in the README is fair.

Usage:
    python -m src.baselines
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .data import prepare_data


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
}


def run(outdir: str = "results", seed: int = 42) -> dict:
    data = prepare_data(seed=seed)
    # Baselines don't need a separate val set; train on train+val, test on test.
    X_tr = np.vstack([data.X_train, data.X_val])
    y_tr = np.concatenate([data.y_train, data.y_val]).astype(int)
    y_te = data.y_test.astype(int)

    results = {}
    for name, factory in MODELS.items():
        clf = factory()
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(data.X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results[name] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_te, prob)),
        }
        print(f"{name:20s} acc {results[name]['accuracy']:.3f}  "
              f"f1 {results[name]['f1']:.3f}  auc {results[name]['roc_auc']:.3f}")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "baselines.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run()
