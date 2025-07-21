"""Script to load data, encode features, and apply preprocessing pipeline."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
from pathlib import Path
import sys

# ─── Project Module Imports ──────────────────────────────────────────────────────
from src.svm import encoder
from src.config import logging_config


# ─── Logging Setup ───────────────────────────────────────────────────────────────
logger = logging_config.configure_logging()




def encoding_pipeline(train_set, test_set):
    """
    Encodes the training and test datasets using the TF-IDF vectorizer.
    Args:
        train_set: The training dataset.
        test_set: The test dataset.
    Returns:
        encoded_dataset: The dataset containing the encoded features.
    """
    # Start logging
    logger.info("Starting encoding pipeline...")

    # Encode the datasets using the TF-IDF vectorizer
    logger.info("Encoding datasets using TF-IDF vectorizer...")
    encoded_dataset = encoder.tfidf_vectorizer(train_set, test_set)
    logger.info("Encoding completed.")

    return encoded_dataset
