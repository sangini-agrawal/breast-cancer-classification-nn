"""Phase 7 – Predictive system.

Classify a tumour as benign or malignant from its 30 measurements.

Examples (run from the project root after `python -m src.train`):
    python -m src.predict                         # runs the built-in demo sample
    python -m src.predict --values "11.76,21.6,74.72, ... ,0.06563"
    python -m src.predict --csv new_patients.csv  # CSV with the 30 feature columns
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from . import config

# Measurements of one benign sample from the dataset (used for the demo)
DEMO_SAMPLE = (11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
               0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769,
               12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563)


def load_artifacts():
    if not config.MODEL_PATH.exists() or not config.SCALER_PATH.exists():
        raise SystemExit("Model not found. Train it first with:  python -m src.train")
    return keras.models.load_model(config.MODEL_PATH), joblib.load(config.SCALER_PATH)


def predict(values, model=None, scaler=None) -> list[dict]:
    """values: array-like of shape (n_samples, 30) in the dataset's original units."""
    if model is None or scaler is None:
        model, scaler = load_artifacts()
    X = np.asarray(values, dtype=float).reshape(-1, 30)
    if hasattr(scaler, "feature_names_in_"):
        X = pd.DataFrame(X, columns=scaler.feature_names_in_)  # avoids feature-name warning
    probs = model.predict(scaler.transform(X), verbose=0)
    out = []
    for p in probs:
        label = int(np.argmax(p))
        out.append({"diagnosis": config.CLASS_NAMES[label],
                    "p_malignant": float(p[config.MALIGNANT]), "p_benign": float(p[config.BENIGN]),
                    "confidence": float(p[label])})
    return out


def main():
    ap = argparse.ArgumentParser(description="Breast tumour classifier (neural network)")
    ap.add_argument("--values", help="30 comma-separated measurements for one patient")
    ap.add_argument("--csv", help="CSV file with one patient per row and the 30 feature columns")
    args = ap.parse_args()

    if args.csv:
        rows = pd.read_csv(args.csv).to_numpy()
    elif args.values:
        rows = [[float(v) for v in args.values.split(",")]]
    else:
        print("No input given – running the demo sample.")
        rows = [DEMO_SAMPLE]

    for i, r in enumerate(predict(rows), 1):
        print(f"Patient {i}: the tumour is predicted {r['diagnosis'].upper()} "
              f"(confidence {r['confidence']:.1%}; P(malignant) = {r['p_malignant']:.3f})")
    print("\nNote: this is an educational model and not a medical diagnosis.")


if __name__ == "__main__":
    main()
