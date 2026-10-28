"""Data collection, cleaning and preprocessing for the Breast Cancer Wisconsin dataset."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


def load_dataset() -> pd.DataFrame:
    """Phase 1 – Load the dataset as a DataFrame with 30 feature columns and a 'label' column."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.rename(columns={"target": "label"})
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Summarise missing values and duplicate rows."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {config.CLASS_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2 – Drop missing values and duplicate records (none exist in this dataset, but the
    step keeps the pipeline safe if a different copy of the data is used)."""
    return df.dropna().drop_duplicates().reset_index(drop=True)


def split(df: pd.DataFrame):
    """Phase 3a – Separate features/target and make stratified train / validation / test sets.

    Test set       : 20 % of all patients – touched only once, for the final evaluation.
    Validation set : 20 % of the remaining patients – used by early stopping during training.
    Stratification keeps the benign/malignant ratio identical in every subset.
    """
    X = df.drop(columns="label")
    y = df["label"].astype(int)
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=config.VALIDATION_SIZE, stratify=y_trval, random_state=config.SEED)
    return X_train, X_val, X_test, y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()


def scale(X_train, *others):
    """Phase 3b – Standardise every feature to mean 0 and standard deviation 1.

    The scaler is fitted on the training data only, so no information from the
    validation or test sets leaks into training.
    """
    scaler = StandardScaler().fit(X_train)
    return (scaler, scaler.transform(X_train), *[scaler.transform(o) for o in others])
