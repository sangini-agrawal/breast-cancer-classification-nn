"""Evaluation helpers (Phase 6): accuracy, precision, recall, F1, specificity, ROC-AUC, confusion matrix.

Malignant (label 0) is treated as the *positive* class, because missing a cancer
(false negative) is the most costly mistake in a diagnostic setting.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import config


def compute_metrics(y_true, y_pred, p_malignant) -> dict:
    # Rows/cols ordered [Malignant, Benign] so the top-left cell is true positives
    cm = confusion_matrix(y_true, y_pred, labels=[config.MALIGNANT, config.BENIGN])
    tp, fn, fp, tn = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malignant": float(precision_score(y_true, y_pred, pos_label=config.MALIGNANT)),
        "recall_malignant": float(recall_score(y_true, y_pred, pos_label=config.MALIGNANT)),
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=config.MALIGNANT)),
        "specificity": float(tn / (tn + fp)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score((np.asarray(y_true) == config.MALIGNANT).astype(int), p_malignant)),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def text_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=[config.MALIGNANT, config.BENIGN],
                                  target_names=config.CLASS_NAMES, digits=4)


def roc_points(y_true, p_malignant):
    return roc_curve((np.asarray(y_true) == config.MALIGNANT).astype(int), p_malignant)


def baseline_models():
    """Classical ML models used to benchmark the neural network."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.SEED),
        "K-Nearest Neighbours (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (RBF)": SVC(random_state=config.SEED),
        "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, random_state=config.SEED),
    }


def malignant_score(clf, X):
    """Score that grows with the likelihood of malignancy (used for ROC-AUC)."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, config.MALIGNANT]
    # SVC without probability calibration: decision_function > 0 means class 1 (benign)
    return -clf.decision_function(X)
