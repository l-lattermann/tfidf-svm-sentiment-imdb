
"""Module to load, encode, and save datasets using TfidfDataset."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
from pathlib import Path
from src.config import logging_config

# ─── Third-Party Imports ─────────────────────────────────────────────────────────
from sklearn.datasets import load_files
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Project Imports ──────────────────────────────────────────────────────────────
from src.data.data_classes import TfidfDataset
from src.config.paths import CLEANED_DATA_TXT_DIR, ENCODED_DATA_DIR

# ─── Set up logging ───────────────────────────────────────────────────────────────

logger = logging_config.configure_logging()


def load_texts_from_folder(path, test_mode: bool = False, sample_count: int=20) -> Bunch:
    """
    Load text files from a specified folder and return them as a dataset.
    Args:
        folder_path (str or Path): The path to the folder containing text files.
        test_mode (bool): If True, limit the dataset to the first 20 samples for testing purposes.
        sample_count (int): Number of samples to limit the dataset to when in test mode.
    Returns:
        sklearn.utils.Bunch: A dataset containing the text files and their labels.
        train_data.data       # List of review texts (str)
        train_data.target     # List of labels (0=neg, 1=pos)
        train_data.target_names  # ['neg', 'pos']
    """
    # Start logging
    logger.info(f"Loading text files from {path} with test_mode={test_mode} and sample_count={sample_count}")

    # Load the text files from the specified folder
    data = load_files(path, encoding="utf-8", decode_error="ignore")

    # If testing is enabled, limit the dataset to the first 20 samples
    if test_mode:
        data.data = data.data[:sample_count]
        data.target = data.target[:sample_count]

    # Log the number of samples loaded
    logger.info(f"Loaded {len(data.data)} samples from {path}")

    return data

def save_dataset_as_txt(dataset: Bunch, split: str, path: str = CLEANED_DATA_TXT_DIR):
    """
    Save the dataset to the specified directory, organizing files into 'pos' and 'neg' subdirectories.
    Args:
        dataset (Bunch): The dataset to save, containing 'data' and 'target'.
        DATA_DIR (str or Path): The directory where the dataset will be saved.
        split (str): The split of the dataset ('train' or 'test').
    Returns:
        None
    """
    # Start logging
    logger.info(f"Saving dataset to {path} for split '{split}'")

    # Set the save directory
    SAVE_DIR = path

    # Create a mapping for labels
    label_map = {0: "neg", 1: "pos"}

    # Create subdirectories for each label
    for label_dir in label_map.values():
        (SAVE_DIR / split / label_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {SAVE_DIR / split / label_dir}")

    # Save each text file in the corresponding label directory
    for i, (text, label) in enumerate(zip(dataset.data, dataset.target)):
        label_dir = label_map[label]
        file_path = SAVE_DIR / split / label_dir / f"{i}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.debug(f"Saved file: {file_path}")
    logger.info(f"Saved {len(dataset.data)} files to {SAVE_DIR / split}")

def save_encoded_dataset_as_sparse_matrix(dataset: TfidfDataset, path: str = ENCODED_DATA_DIR):
    """
    Save the dataset as a sparse matrix in the specified directory.
    Args:
        dataset (Bunch): The dataset to save, containing 'data' and 'target'.
        DATA_DIR (str or Path): The directory where the dataset will be saved.
        split (str): The split of the dataset ('train' or 'test').
    Returns:
        None
    """
    # Start logging
    logger.info(f"Saving encoded dataset to %s", path)

    # Create the save directory if it doesn't exist
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


    # Save the sparse matrix and target labels
    dump({
        "X_train": dataset.X_train,
        "y_train": dataset.y_train,
        "X_test": dataset.X_test,
        "y_test": dataset.y_test,
        "vectorizer": dataset.vectorizer
    }, path)
    logger.info("Saved encoded dataset to %s", path)

def load_encoded_dataset_joblib(path: str = ENCODED_DATA_DIR / "complete_dataset.joblib"):
    """
    Load an encoded dataset from a joblib file and return it as a TfidfDataset object.
    Args:
        file_path (str): The path to the joblib file containing the encoded dataset.
    Returns:
        TfidfDataset: An object containing the encoded dataset with attributes:
            - X_train: Sparse matrix of training data features
            - y_train: Labels for the training data
            - X_test: Sparse matrix of test data features
            - y_test: Labels for the test data
            - vectorizer: The TF-IDF vectorizer used for encoding
    """
    # Start logging
    logger.info(f"Loading encoded dataset from {path}")

    # Load the encoded dataset from a joblib file.
    loaded_data = load(path)
    
    # Transform the loaded data into a TfidfDataset object.
    dataset = TfidfDataset(
        name=Path(path).stem,  # Use the file name without extension as the dataset name
        X_train=loaded_data["X_train"],
        y_train=loaded_data["y_train"],
        X_test=loaded_data["X_test"],
        y_test=loaded_data["y_test"],
        vectorizer=loaded_data["vectorizer"]
    )
    logger.info(f"Loaded encoded dataset with {dataset.X_train.shape[0]} training samples and {dataset.X_test.shape[0]} test samples")
    
    return dataset

def save_svm_model(model: GridSearchCV, path: str = ENCODED_DATA_DIR / "svm_model.joblib"):
    """
    Save the trained SVM model to a joblib file.
    Args:
        model (GridSearchCV): The trained SVM model to save.
        path (str): The path where the model will be saved.
    Returns:
        None
    """
    # Start logging
    logger.info(f"Saving SVM model to {path}")

    # Save the model using joblib
    dump(model, path)
    logger.info(f"SVM model saved to {path}")

def load_svm_model(path: str = ENCODED_DATA_DIR / "svm_model.joblib") -> GridSearchCV:
    """
    Load a trained SVM model from a joblib file.
    Args:
        path (str): The path to the joblib file containing the trained SVM model.
    Returns:
        GridSearchCV: The loaded SVM model.
    """
    # Start logging
    logger.info(f"Loading SVM model from {path}")

    # Load the model using joblib
    model = load(path)
    logger.info("SVM model loaded successfully")
    
    return model

def sanitize(val: str) -> str:
    """
    Sanitize a string by removing spaces and special characters.
    Args:
        val (str): The value to sanitize.
    Returns:
        str: The sanitized value.
"""
    return str(val).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")

def param_dict_to_filename(params: dict) -> str:
    return "-".join(f"{k}={sanitize(v)}" for k, v in params.items())

def save_encoder(path: str, vectorizer):
    """
    Save the TF-IDF vectorizer to a joblib file.
    Args:
        path (str): The path where the vectorizer will be saved.
        vectorizer: The TF-IDF vectorizer to save.
    Returns:
        None
    """
    # Ensure the path is a Path object
    if isinstance(path, str):
        path = Path(path)

    # Create the parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start logging
    logger.info(f"Saving encoder to {path}")

    # Save the vectorizer using joblib
    dump(vectorizer, path)
    logger.info(f"Encoder saved to {path}")

def load_encoder(path: str) -> TfidfVectorizer:
    """
    Load the TF-IDF vectorizer from a joblib file.
    Args:
        path (str): The path to the joblib file containing the vectorizer.
    Returns:
        TfidfVectorizer: The loaded TF-IDF vectorizer.
    """
    # Start logging
    logger.info(f"Loading encoder from {path}")

    # Load the vectorizer using joblib
    vectorizer = load(path)
    logger.info("Encoder loaded successfully")

    return vectorizer