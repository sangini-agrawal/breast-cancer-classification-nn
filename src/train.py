"""End-to-end training pipeline.

Run from the project root:
    python -m src.train            # train, evaluate, benchmark, save model + figures
    python -m src.train --no-cv    # skip the 5-fold cross-validation check (faster)
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import joblib
import numpy as np
import tensorflow as tf
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from . import config, data, evaluate, visualize
from .model import build_model

KEY_FEATURES = ["mean radius", "mean texture", "mean perimeter",
                "mean area", "mean concavity", "mean concave points"]


def set_seed(seed: int = config.SEED) -> None:
    keras.utils.set_random_seed(seed)          # seeds Python, NumPy and TensorFlow
    tf.config.experimental.enable_op_determinism()


def train_network(X_train, y_train, X_val, y_val, verbose=2):
    model = build_model(X_train.shape[1])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                               restore_best_weights=True, verbose=1 if verbose else 0)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                        callbacks=[early_stop], verbose=verbose)
    return model, history


def cross_validate(df) -> dict:
    """Stratified k-fold CV on the full dataset for the network and every baseline.
    Gives a more reliable estimate than a single 114-patient test set."""
    X = df.drop(columns="label").to_numpy(); y = df["label"].to_numpy()
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    scores = {"Neural Network (ours)": []} | {n: [] for n in evaluate.baseline_models()}
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        set_seed(config.SEED + fold)
        tr_i, va_i = train_test_split(tr, test_size=config.VALIDATION_SIZE, stratify=y[tr],
                                      random_state=config.SEED)
        scaler = StandardScaler().fit(X[tr_i])
        model, _ = train_network(scaler.transform(X[tr_i]), y[tr_i],
                                 scaler.transform(X[va_i]), y[va_i], verbose=0)
        scores["Neural Network (ours)"].append(model.evaluate(scaler.transform(X[te]), y[te], verbose=0)[1])
        for name, clf in evaluate.baseline_models().items():
            pipe = make_pipeline(StandardScaler(), clone(clf)).fit(X[tr], y[tr])
            scores[name].append(float(pipe.score(X[te], y[te])))
        print(f"  fold {fold}/{config.CV_FOLDS} done – NN accuracy {scores['Neural Network (ours)'][-1]:.4f}")
    return {n: {"fold_accuracy": [float(v) for v in s], "mean": float(np.mean(s)), "std": float(np.std(s))}
            for n, s in scores.items()}


def main(run_cv: bool = True) -> dict:
    set_seed()
    for d in (config.MODELS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Phase 1–2: collect and clean
    df = data.load_dataset()
    quality = data.quality_report(df)
    print("Data quality:", quality)
    df = data.clean(df)

    visualize.class_distribution(df["label"].to_numpy())
    visualize.feature_distributions(df, KEY_FEATURES)
    visualize.correlation_heatmap(df, [c for c in df.columns if c.startswith("mean ")])

    # Phase 3: preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = data.split(df)
    scaler, X_train_std, X_val_std, X_test_std = data.scale(X_train, X_val, X_test)
    print(f"Train: {X_train_std.shape}  Validation: {X_val_std.shape}  Test: {X_test_std.shape}")

    # Phase 4–5: build and train
    model, history = train_network(X_train_std, y_train, X_val_std, y_val)
    model.summary()
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmin(hist["val_loss"]) + 1)
    visualize.training_history(hist, best_epoch=best_epoch)

    # Phase 6: evaluate on the untouched test set
    test_loss, test_acc = model.evaluate(X_test_std, y_test, verbose=0)
    probs = model.predict(X_test_std, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    p_malignant = probs[:, config.MALIGNANT]
    nn_metrics = evaluate.compute_metrics(y_test, y_pred, p_malignant)
    nn_metrics["test_loss"] = float(test_loss)
    report = evaluate.text_report(y_test, y_pred)
    print(f"\nTest accuracy: {test_acc:.4f}   Test loss: {test_loss:.4f}\n")
    print(report)

    visualize.confusion_matrix_plot(np.array(nn_metrics["confusion_matrix"]))
    fpr, tpr, _ = evaluate.roc_points(y_test, p_malignant)
    visualize.roc_curve_plot(fpr, tpr, nn_metrics["roc_auc"])
    visualize.probability_histogram(p_malignant, y_test)

    # Benchmark against classical ML (trained on train + validation, same test set)
    X_trval_std = np.vstack([X_train_std, X_val_std]); y_trval = np.concatenate([y_train, y_val])
    comparison = {"Neural Network (ours)": nn_metrics}
    for name, clf in evaluate.baseline_models().items():
        clf.fit(X_trval_std, y_trval)
        comparison[name] = evaluate.compute_metrics(y_test, clf.predict(X_test_std),
                                                    evaluate.malignant_score(clf, X_test_std))
    visualize.model_comparison(comparison)
    print("Model comparison (test set):")
    for name, m in comparison.items():
        print(f"  {name:<30} acc={m['accuracy']:.4f}  recall(M)={m['recall_malignant']:.4f}  "
              f"F1(M)={m['f1_malignant']:.4f}  AUC={m['roc_auc']:.4f}")

    cv = None
    if run_cv:
        print(f"\n{config.CV_FOLDS}-fold cross-validation (accuracy):")
        cv = cross_validate(df)
        for name, r in cv.items():
            print(f"  {name:<30} {r['mean']:.4f} ± {r['std']:.4f}")

    h1, h2 = config.HIDDEN_UNITS
    visualize.architecture_diagram([
        ("Input", f"{X_train_std.shape[1]} standardised\nfeatures", X_train_std.shape[1]),
        ("Hidden 1", f"Dense · {h1} neurons\nReLU", h1),
        ("Dropout", f"rate {config.DROPOUT_RATE:.0%}\n(training only)", h1),
        ("Hidden 2", f"Dense · {h2} neurons\nReLU", h2),
        ("Output", "Dense · 2 neurons\nSoftmax", 2),
    ])

    # Save artefacts
    model.save(config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    summary = {
        "dataset": quality,
        "split": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
        "hyperparameters": {"hidden_units": list(config.HIDDEN_UNITS), "dropout": config.DROPOUT_RATE,
                            "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE,
                            "max_epochs": config.EPOCHS, "patience": config.PATIENCE, "seed": config.SEED},
        "epochs_trained": len(hist["loss"]), "best_epoch": best_epoch,
        "trainable_params": int(model.count_params()),
        "neural_network": nn_metrics, "comparison": comparison, "cross_validation": cv, "history": hist,
    }
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    config.REPORT_PATH.write_text(report)
    print(f"\nSaved model   -> {config.MODEL_PATH}\nSaved scaler  -> {config.SCALER_PATH}"
          f"\nSaved metrics -> {config.METRICS_PATH}\nFigures       -> {config.FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="skip k-fold cross-validation")
    main(run_cv=not ap.parse_args().no_cv)
