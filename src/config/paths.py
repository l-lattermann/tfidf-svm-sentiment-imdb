"""Centralized path definitions for the project."""

from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ─── Main Directories ────────────────────────────────────────────────────────────
SRC_DIR     = ROOT / "src"
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "models"
CONFIG_DIR  = SRC_DIR / "config"
TEST_DIR    = ROOT / "tests"
LOG_DIR     = ROOT / "logs"
SVM_DIR     = SRC_DIR / "svm"

# ─── Data Directories ────────────────────────────────────────────────────────────
TRAIN_DATA_DIR         = DATA_DIR / "aclImdb" / "train"
TEST_DATA_DIR          = DATA_DIR / "aclImdb" / "test"
CLEANED_DATA_TXT_DIR   = DATA_DIR / "cleaned_txt"
CLEANED_TRAIN_DIR      = CLEANED_DATA_TXT_DIR / "train"
CLEANED_TEST_DIR       = CLEANED_DATA_TXT_DIR / "test"
ENCODED_DATA_DIR       = DATA_DIR / "IMBD_tfidf"

# ─── Config Files ────────────────────────────────────────────────────────────────
TRAINING_PARAMS = CONFIG_DIR / "training_params.yaml"

# ─── Ensure Directories Exist ─────────────────────────────────────────────────────
for directory in [
    SRC_DIR, DATA_DIR, CONFIG_DIR, TRAIN_DATA_DIR, TEST_DATA_DIR, 
    SVM_DIR, LOG_DIR, CLEANED_TEST_DIR, CLEANED_TRAIN_DIR, 
    ENCODED_DATA_DIR, MODEL_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ─── Public Exports ──────────────────────────────────────────────────────────────
__all__ = [
    "ROOT",
    "SRC_DIR",
    "DATA_DIR",
    "MODEL_DIR",
    "CONFIG_DIR",
    "TRAIN_DATA_DIR",
    "TEST_DATA_DIR",
    "TRAINING_PARAMS",
    "SVM_DIR",
    "LOG_DIR",
    "CLEANED_DATA_TXT_DIR",
    "CLEANED_TRAIN_DIR",
    "CLEANED_TEST_DIR",
    "ENCODED_DATA_DIR",
]