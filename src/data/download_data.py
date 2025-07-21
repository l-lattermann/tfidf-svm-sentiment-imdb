"""Script to download and prepare the IMDb dataset for sentiment analysis."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path
import shutil

# ─── Third-Party Imports ─────────────────────────────────────────────────────────
from datasets import load_dataset, concatenate_datasets

# ─── Project Root Setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ─── Project Imports ─────────────────────────────────────────────────────────────
from src.config.paths import DATA_DIR

# ─── Logging Setup ───────────────────────────────────────────────────────────────
from src.config import logging_config
logger = logging_config.configure_logging()


def download_imdb_dataset(path: Path = DATA_DIR):
    """
    This function downloads the IMDb dataset, splits it into training and testing sets,
    and saves the data in a structured format suitable for sentiment analysis tasks.
    Args:
        path (Path): The directory where the dataset will be saved. Defaults to DATA_DIR.
    Returns:
        None
    """
    
    logger.info("Starting IMDb dataset download and preparation...")
    # Define output directory
    OUTPUT_DIR = DATA_DIR / "aclImdb"
    (OUTPUT_DIR / "train" / "pos").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "neg").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / "pos").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / "neg").mkdir(parents=True, exist_ok=True)
    logger.info("Output directory structure created at %s", OUTPUT_DIR)


    # Load IMDb dataset from HuggingFace
    dataset = load_dataset("imdb")
    if dataset:
        logger.info("IMDb dataset loaded successfully.")
    else:
        logger.error("Failed to load IMDb dataset.")
        sys.exit(1)

    # Merge train and test splits
    dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    # Create a new 80/20 split for training and testing
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    logger.info("Split ratio set to 80/20 for training and testing.")
    logger.info("Number of training samples: %d", len(dataset["train"]))
    logger.info("Number of testing samples: %d", len(dataset["test"]))

    # Save function
    def save_split(split_name):
        split = dataset[split_name]
        for i, example in enumerate(split):
            label = "pos" if example["label"] == 1 else "neg"
            text = example["text"]
            out_dir = OUTPUT_DIR / split_name / label
            file_path = out_dir / f"{i}_{example['label']}.txt"
            file_path.write_text(text, encoding="utf-8")

    # Save train and test splits
    save_split("train")
    save_split("test")
    logger.info("Saved train and test splits to %s", OUTPUT_DIR)

    # Zip the dataset
    shutil.make_archive(OUTPUT_DIR / "aclImdb", 'zip', OUTPUT_DIR)
    logger.info("Zipped the dataset to %s.zip", OUTPUT_DIR / "aclImdb")