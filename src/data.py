"""Data loading and preprocessing for the Heart Disease Prediction project.

The pipeline is deliberately explicit so it is easy to explain in a report:

* continuous features  -> standardised (zero mean, unit variance)
* multi-class features  -> one-hot encoded
* binary features       -> passed through unchanged

Everything is wired through a single scikit-learn ``ColumnTransformer`` so the
exact same transformation can be re-applied at evaluation / inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column groups (based on the standard UCI Cleveland schema)
# ---------------------------------------------------------------------------
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]
BINARY = ["sex", "fbs", "exang"]

FEATURE_COLUMNS = CONTINUOUS + CATEGORICAL + BINARY
TARGET_COLUMN = "target"

# A human-readable description of every column, handy for the README / notebook.
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)",
    "target": "Diagnosis of heart disease (1 = present, 0 = absent)",
}


@dataclass
class Dataset:
    """A single train / val / test split with its fitted preprocessor."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]


def load_raw(path: str = "data/heart.csv") -> pd.DataFrame:
    """Load the raw CSV, strip a possible BOM and drop exact duplicate rows."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Column-wise preprocessing: scale continuous, one-hot categorical."""
    return ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), CONTINUOUS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("bin", "passthrough", BINARY),
        ]
    )


def _expanded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - defensive, older sklearn
        return [f"f{i}" for i in range(preprocessor.transform_shape_[1])]


def prepare_data(
    path: str = "data/heart.csv",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dataset:
    """Load, split (stratified) and preprocess the data.

    The split is fully deterministic given ``seed`` so that training and
    evaluation operate on exactly the same test set.
    """
    df = load_raw(path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    # First carve off the test set, then split the remainder into train/val.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=seed
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_t = preprocessor.transform(X_val).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)

    return Dataset(
        X_train=X_train_t,
        y_train=y_train.astype(np.float32),
        X_val=X_val_t,
        y_val=y_val.astype(np.float32),
        X_test=X_test_t,
        y_test=y_test.astype(np.float32),
        preprocessor=preprocessor,
        feature_names=_expanded_feature_names(preprocessor),
    )


if __name__ == "__main__":
    data = prepare_data()
    print(f"train: {data.X_train.shape}  val: {data.X_val.shape}  test: {data.X_test.shape}")
    print(f"input dimension after preprocessing: {data.input_dim}")
    print(f"class balance (train): {np.bincount(data.y_train.astype(int))}")
