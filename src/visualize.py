"""All charts used in the notebook, report and presentation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import config

# Colour roles (colour-blind-safe pair for the two classes)
BENIGN_C, MALIGNANT_C, THIRD_C = "#2a78d6", "#eb6834", "#1baf7a"
CLASS_COLORS = {config.MALIGNANT: MALIGNANT_C, config.BENIGN: BENIGN_C}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#c53a39"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f0f5fc", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.labelsize": 11, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "font.family": "DejaVu Sans",
})


def _save(fig, name: str, out_dir: Path | None):
    out_dir = Path(out_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def class_distribution(y, out_dir=None):
    counts = [int((y == config.MALIGNANT).sum()), int((y == config.BENIGN).sum())]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(config.CLASS_NAMES, counts, color=[MALIGNANT_C, BENIGN_C], width=0.55)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 6, f"{c}  ({c / sum(counts):.1%})",
                ha="center", va="bottom", color=TEXT, fontsize=11)
    ax.set_ylabel("Number of patients")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Class distribution of the 569 tumour samples")
    return _save(fig, "01_class_distribution", out_dir)


def feature_distributions(df, features, out_dir=None):
    n = len(features); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.4 * rows))
    for ax, feat in zip(axes.ravel(), features):
        lo, hi = df[feat].min(), df[feat].max()
        bins = np.linspace(lo, hi, 30)
        for lbl in (config.BENIGN, config.MALIGNANT):
            ax.hist(df.loc[df.label == lbl, feat], bins=bins, alpha=0.55, color=CLASS_COLORS[lbl],
                    label=config.CLASS_NAMES[lbl], edgecolor=SURFACE, linewidth=0.6)
        ax.set_title(feat.title(), fontsize=11.5)
        ax.set_ylabel("Count")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=11, bbox_to_anchor=(1, 1.02))
    fig.suptitle("How key tumour measurements differ between benign and malignant cases",
                 x=0.01, ha="left", fontweight="bold", fontsize=13.5, y=1.02)
    fig.tight_layout()
    return _save(fig, "02_feature_distributions", out_dir)


def correlation_heatmap(df, features, out_dir=None):
    corr = df[features + ["label"]].corr()
    labels = [f.replace("mean ", "") for f in features] + ["label"]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("Pearson correlation")
    ax.set_title("Correlation between the 10 'mean' features and the label")
    return _save(fig, "03_correlation_heatmap", out_dir)


def training_history(history: dict, out_dir=None, best_epoch: int | None = None):
    ep = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in [(axes[0], "accuracy", "Accuracy per epoch"), (axes[1], "loss", "Loss per epoch")]:
        ax.plot(ep, history[key], color=BENIGN_C, label="Training")
        ax.plot(ep, history[f"val_{key}"], color=MALIGNANT_C, label="Validation")
        if best_epoch:
            ax.axvline(best_epoch, color=TEXT_2, linestyle="--", linewidth=1)
            ax.text(best_epoch, ax.get_ylim()[1], " best epoch", color=TEXT_2, va="top", fontsize=9)
        ax.set_xlabel("Epoch"); ax.set_ylabel(key.capitalize()); ax.set_title(title)
        ax.legend(loc="lower right" if key == "accuracy" else "upper right")
    fig.tight_layout()
    return _save(fig, "04_training_history", out_dir)


def confusion_matrix_plot(cm, out_dir=None, title="Confusion matrix – test set (114 patients)", name="05_confusion_matrix"):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm, cmap=SEQUENTIAL)
    ax.grid(False)
    ax.set_xticks([0, 1], [f"Predicted\n{c}" for c in config.CLASS_NAMES])
    ax.set_yticks([0, 1], [f"Actual\n{c}" for c in config.CLASS_NAMES])
    tags = [["True positive", "False negative"], ["False positive", "True negative"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.5
            ax.text(j, i - 0.08, str(cm[i, j]), ha="center", va="center", fontsize=22, fontweight="bold",
                    color="white" if dark else TEXT)
            ax.text(j, i + 0.22, tags[i][j], ha="center", va="center", fontsize=9,
                    color="white" if dark else TEXT_2)
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    return _save(fig, name, out_dir)


def roc_curve_plot(fpr, tpr, auc, out_dir=None):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_2, linewidth=1, label="Random guess (AUC = 0.50)")
    ax.plot(fpr, tpr, color=BENIGN_C, label=f"Neural network (AUC = {auc:.3f})")
    ax.set_xlabel("False positive rate (1 – specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_title("ROC curve – detecting malignant tumours")
    ax.legend(loc="lower right")
    return _save(fig, "06_roc_curve", out_dir)


def probability_histogram(p_malignant, y_true, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    for lbl in (config.BENIGN, config.MALIGNANT):
        ax.hist(p_malignant[y_true == lbl], bins=bins, color=CLASS_COLORS[lbl], alpha=0.7,
                label=f"Actually {config.CLASS_NAMES[lbl].lower()}", edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(0.5, color=TEXT_2, linestyle="--", linewidth=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.55, "decision\nthreshold 0.5", color=TEXT_2, fontsize=9, va="center")
    ax.set_xlabel("Predicted probability of malignancy"); ax.set_ylabel("Number of test patients")
    ax.set_title("Model confidence on the test set")
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))
    return _save(fig, "07_probability_histogram", out_dir)


def model_comparison(results: dict, out_dir=None):
    """Dot plot of accuracy, F1 and malignant recall for every model (axis does not start at 0,
    so dots – not bars – are used)."""
    names = list(results.keys())[::-1]
    metrics = [("accuracy", "Accuracy", BENIGN_C, "o"), ("f1_malignant", "F1-score (malignant)", MALIGNANT_C, "s"),
               ("recall_malignant", "Recall / sensitivity (malignant)", THIRD_C, "D")]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(names) + 1.6))
    for k, (key, label, color, marker) in enumerate(metrics):
        vals = [results[n][key] for n in names]
        ax.scatter(vals, np.arange(len(names)) + (k - 1) * 0.18, color=color, marker=marker, s=60,
                   label=label, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(names)), names)
    lo = min(results[n][m[0]] for n in names for m in metrics)
    ax.set_xlim(np.floor((lo - 0.01) * 50) / 50, 1.002)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Neural network vs classical machine-learning baselines (test set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9.5)
    return _save(fig, "08_model_comparison", out_dir)


def architecture_diagram(layers, out_dir=None):
    """layers: list of (title, subtitle, n_units) tuples."""
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off"); ax.set_xlim(0, len(layers)); ax.set_ylim(0, 1)
    fills = ["#cde2fb", "#9ec5f4", "#f0efec", "#9ec5f4", "#fbd5c4"]
    for i, (title, sub, units) in enumerate(layers):
        x = i + 0.08
        ax.add_patch(plt.Rectangle((x, 0.2), 0.84, 0.6, facecolor=fills[i % len(fills)], edgecolor="#256abf",
                                   linewidth=1.2, joinstyle="round"))
        ax.text(x + 0.42, 0.62, title, ha="center", va="center", fontweight="bold", fontsize=11.5)
        ax.text(x + 0.42, 0.43, sub, ha="center", va="center", fontsize=9.5, color=TEXT_2, linespacing=1.4)
        if i < len(layers) - 1:
            ax.annotate("", xy=(i + 1.08, 0.5), xytext=(i + 0.92, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT_2, lw=1.4))
    ax.set_title("Neural network architecture", loc="left")
    return _save(fig, "09_architecture", out_dir)
