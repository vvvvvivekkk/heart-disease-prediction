"""Tests for the data loading and preprocessing pipeline."""
import numpy as np

from src.data import (
    BINARY,
    CATEGORICAL,
    CONTINUOUS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_raw,
    prepare_data,
)


def test_load_raw_has_expected_columns():
    df = load_raw("data/heart.csv")
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns


def test_load_raw_drops_duplicates_and_has_no_nulls():
    df = load_raw("data/heart.csv")
    assert df.duplicated().sum() == 0
    assert df.isnull().sum().sum() == 0


def test_target_is_binary():
    df = load_raw("data/heart.csv")
    assert set(df[TARGET_COLUMN].unique()) <= {0, 1}


def test_column_groups_cover_all_features():
    assert set(CONTINUOUS + CATEGORICAL + BINARY) == set(FEATURE_COLUMNS)


def test_prepare_data_shapes_and_disjoint_sizes():
    data = prepare_data(seed=42)
    n = data.X_train.shape[0] + data.X_val.shape[0] + data.X_test.shape[0]
    assert n == 302  # 303 rows minus one duplicate
    # feature width is consistent across splits
    assert data.X_train.shape[1] == data.X_val.shape[1] == data.X_test.shape[1]
    assert data.input_dim == data.X_train.shape[1]
    assert len(data.feature_names) == data.input_dim


def test_prepare_data_is_deterministic():
    a = prepare_data(seed=42)
    b = prepare_data(seed=42)
    assert np.array_equal(a.X_test, b.X_test)
    assert np.array_equal(a.y_test, b.y_test)


def test_continuous_features_are_standardised():
    # After StandardScaler the first len(CONTINUOUS) columns of the train matrix
    # should be ~zero mean / unit variance.
    data = prepare_data(seed=42)
    cont = data.X_train[:, : len(CONTINUOUS)]
    assert np.allclose(cont.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(cont.std(axis=0), 1, atol=1e-1)


def test_no_nan_after_preprocessing():
    data = prepare_data(seed=42)
    for arr in (data.X_train, data.X_val, data.X_test):
        assert not np.isnan(arr).any()
