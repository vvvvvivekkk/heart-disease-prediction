"""Single-patient inference demo.

Example:
    python -m src.predict

Edit the ``sample`` dictionary below (or import ``predict_one`` elsewhere) to
score a new patient record. All 13 input features must be provided.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import torch

from .data import FEATURE_COLUMNS
from .evaluate import load_model


def predict_one(sample: dict, outdir: str = "results") -> tuple[int, float]:
    """Return (predicted_label, probability_of_disease) for one patient."""
    device = torch.device("cpu")
    preprocessor = joblib.load(os.path.join(outdir, "preprocessor.joblib"))
    model = load_model(outdir, device)

    row = pd.DataFrame([{c: sample[c] for c in FEATURE_COLUMNS}])
    X = preprocessor.transform(row).astype("float32")
    prob = float(torch.sigmoid(model(torch.from_numpy(X))).item())
    return int(prob >= 0.5), prob


if __name__ == "__main__":
    # A sample patient record (values follow the UCI Cleveland encoding).
    sample = {
        "age": 57, "sex": 1, "cp": 0, "trestbps": 130, "chol": 236,
        "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0, "oldpeak": 0.0,
        "slope": 1, "ca": 1, "thal": 2,
    }
    label, prob = predict_one(sample)
    verdict = "HEART DISEASE likely" if label == 1 else "No heart disease"
    print(f"P(disease) = {prob:.3f}  ->  {verdict}")
    print(json.dumps(sample, indent=2))
