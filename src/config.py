"""Central configuration: paths and hyper-parameters used across the project."""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "breast_cancer_nn.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42

# ── Data split ───────────────────────────────────────────────────────────
TEST_SIZE = 0.20          # 20 % of patients held out for final testing
VALIDATION_SIZE = 0.20    # 20 % of the training set held out to monitor training (early stopping)
CV_FOLDS = 5              # folds for the cross-validation check

# ── Model / training hyper-parameters ────────────────────────────────────
HIDDEN_UNITS = (16, 8)    # neurons in hidden layers 1 and 2
DROPOUT_RATE = 0.20       # fraction of neurons dropped after hidden layer 1
LEARNING_RATE = 0.001     # Adam optimiser step size
EPOCHS = 100              # upper limit; early stopping usually ends sooner
BATCH_SIZE = 32
PATIENCE = 10             # epochs without val_loss improvement before stopping

# ── Labels ───────────────────────────────────────────────────────────────
# In the scikit-learn copy of the dataset: 0 = malignant, 1 = benign
CLASS_NAMES = ["Malignant", "Benign"]
MALIGNANT, BENIGN = 0, 1
