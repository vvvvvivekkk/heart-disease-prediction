"""End-to-end smoke test: a tiny training run must produce valid artifacts,
and single-patient inference must work off those artifacts."""
import argparse
import json
import os

from src.data import FEATURE_COLUMNS
from src.predict import predict_one
from src.train import train


def _args(outdir: str) -> argparse.Namespace:
    return argparse.Namespace(
        data="data/heart.csv", outdir=outdir, epochs=2, batch_size=32,
        lr=1e-3, weight_decay=1e-4, hidden="16", dropout=0.3, patience=50, seed=42,
    )


def test_train_produces_valid_artifacts_and_metrics(tmp_path):
    outdir = str(tmp_path)
    metrics = train(_args(outdir))

    for fname in ("model.pt", "preprocessor.joblib", "config.json",
                  "history.json", "metrics.json"):
        assert os.path.exists(os.path.join(outdir, fname)), f"missing {fname}"

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0

    saved = json.load(open(os.path.join(outdir, "metrics.json")))
    assert saved["n_test"] == 46


def test_predict_one_after_training(tmp_path):
    outdir = str(tmp_path)
    train(_args(outdir))

    sample = {"age": 57, "sex": 1, "cp": 0, "trestbps": 130, "chol": 236,
              "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0, "oldpeak": 0.0,
              "slope": 1, "ca": 1, "thal": 2}
    assert set(sample) == set(FEATURE_COLUMNS)

    label, prob = predict_one(sample, outdir=outdir)
    assert label in (0, 1)
    assert 0.0 <= prob <= 1.0
