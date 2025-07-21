"""TF-IDF Dataset Container.

Encapsulates the training and test data, their labels, and the vectorizer used
for transforming text data into TF-IDF features.
"""

# ─── Standard Library Imports ────────────────────────────────────────────────────
from dataclasses import dataclass

# ─── Third-Party Imports ─────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class TfidfDataset:
    """
    A dataclass to hold the TF-IDF encoded dataset along with the split information.
    This class encapsulates the training and test data, their labels, and the vectorizer used
    for encoding the text data into TF-IDF features.

    Attributes:
    - dataset: Bunch object containing the dataset.
    - split: A string indicating the split of the dataset (e.g., 'train', 'test').
    - X_train: Sparse matrix of TF-IDF features for the training set.
    - y_train: Numpy array of labels for the training set.
    - X_test: Sparse matrix of TF-IDF features for the test set.
    - y_test: Numpy array of labels for the test set.
    - vectorizer: TfidfVectorizer object used to transform the text data.
    """
    name: str
    X_train: csr_matrix
    y_train: np.ndarray
    X_test: csr_matrix
    y_test: np.ndarray
    vectorizer: TfidfVectorizer