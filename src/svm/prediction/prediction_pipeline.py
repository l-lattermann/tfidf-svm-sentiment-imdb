"""Prediction pipeline for SVM model."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
import random
from pathlib import Path
from sklearn.metrics import classification_report

# ─── Project Module Imports ──────────────────────────────────────────────────────
from src.data import data_loader
from src.config import logging_config
from src.config.paths import TEST_DATA_DIR, MODEL_DIR

# ─── Configure Logging ───────────────────────────────────────────────────────────
logger = logging_config.configure_logging()


def a ():
    # Load the Data
    x_test = []
    y_test = []
    path_neg = Path(TEST_DATA_DIR) / "neg"
    path_pos = Path(TEST_DATA_DIR) / "pos"
    samples = 500
    logger.info("Loading %d samples from the test set for prediction", samples*2)


    neg_files = list(path_neg.glob("*.txt"))
    neg_files = random.sample(neg_files, samples)
    pos_files = list(path_pos.glob("*.txt"))
    pos_files = random.sample(pos_files, samples)
    logger.info("Loading negative reviews from %s", path_neg)
    logger.info("Loading positive reviews from %s", path_pos)

    for i in range(samples):
        with open(neg_files[i], "r") as f:
            neg_review = f.read()
        with open(pos_files[i], "r") as f:
            pos_review = f.read()
        x_test.append(neg_review)
        y_test.append(0)  # Negative review
        x_test.append(pos_review)
        y_test.append(1)  # Positive review
                    

    # Find the Encoder
    path = Path(MODEL_DIR)
    matching_files = list(path.glob("*vectorizer*"))
    logger.info("Loading vectorizer from %s", path)
    logger.info("Found %d vectorizer files", len(matching_files))

    # Load the Encoder
    vectorizer = data_loader.load_encoder(matching_files[0])
    logger.info("Using vectorizer file: %s", matching_files[0])

    # Find the Model
    matching_files = list(path.glob("*svm*"))
    logger.info("Loading model from %s", path)
    logger.info("Found %d model files", len(matching_files))

    # Load the Model
    model = data_loader.load_svm_model(matching_files[0])
    logger.info("Using model file: %s", matching_files[0])

    # Encode the Data
    logger.info("Encoding the data")
    encoded_data = vectorizer.transform(x_test)
    logger.info("Encoded data shape: %s", encoded_data.shape)
    logger.info("Encoded data type: %s", type(encoded_data))

    # Make Predictions
    logger.info("Making predictions")
    y_pred = model.predict(encoded_data)

    # Generate a classification report
    target_names = ["neg", "pos"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    logger.info("Classification Report:\n%s", report)
    return report