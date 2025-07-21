"""Module to load SVM-related data structures and TF-IDF datasets."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
from pathlib import Path
import sys

# ─── Third-Party Imports ─────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.testing import assert_array_equal
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# ─── Path Setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ─── Project Imports ─────────────────────────────────────────────────────────────
from src.data import data_loader
from src.data.tfidf_dataset import TfidfDataset

def test_load_texts_from_folder(tmp_path):
    """
    Test loading text files from a folder into a dataset.
    This test creates a temporary directory with 'neg' and 'pos' subdirectories,
    each containing a text file. It then calls the `load_texts_from_folder` function
    and checks if the returned dataset is structured correctly.
    The dataset should contain the text files and their corresponding labels.
    The expected structure is:
    - tmp_path/
        - neg/
            - neg1.txt
        - pos/
            - pos1.txt
    The dataset should have:
    - data: List of review texts (str)
    - target: List of labels (0=neg, 1=pos)
    - target_names: ['neg', 'pos'] 
    """
    # Arrange
    neg_dir = tmp_path / "neg"
    pos_dir = tmp_path / "pos"
    neg_dir.mkdir()
    pos_dir.mkdir()

    (neg_dir / "neg1.txt").write_text("Bad movie.", encoding="utf-8")
    (pos_dir / "pos1.txt").write_text("Great movie!", encoding="utf-8")

    # Act
    dataset = data_loader.load_texts_from_folder(tmp_path)

    # Assert
    assert isinstance(dataset, Bunch)
    assert len(dataset.data) == 2
    assert sorted(dataset.target_names) == ["neg", "pos"]
    assert set(dataset.target) == {0, 1}
    assert any("Bad movie." in d for d in dataset.data)
    assert any("Great movie!" in d for d in dataset.data)

def test_save_dataset_as_txt(tmp_path):
    """
    Test saving a dataset to text files in a specified directory.
    This test creates a temporary directory and a sample dataset,
    then calls the `save_dataset_as_txt` function to save the dataset.
    It checks if the files are created in the correct structure and contain the expected text.
    """
    # Arrange
    dataset = Bunch(data=["Bad movie.", "Great movie!"], target=[0, 1], target_names=["neg", "pos"])
    split = "train"

    # Act
    data_loader.save_dataset_as_txt(dataset, split, tmp_path)

    # Assert
    neg_file = tmp_path / split / "neg" / "0.txt"
    pos_file = tmp_path / split / "pos" / "1.txt"
    
    assert neg_file.exists()
    assert pos_file.exists()
    
    assert neg_file.read_text(encoding="utf-8") == "Bad movie."
    assert pos_file.read_text(encoding="utf-8") == "Great movie!"

def test_save_and_load_sparse_matrix(tmp_path):
    """
    Test saving and loading a sparse matrix dataset.
    This test creates a sample TfidfDataset, saves it using `save_encoded_dataset_as_sparse_matrix`,
    and then loads it back using `load_encoded_dataset_joblib`.
    It checks if the loaded data matches the original dataset.
    """
    # Arrange
    X_train = csr_matrix([[1, 0], [0, 1]])
    y_train = np.array([0, 1])
    X_test = csr_matrix([[0, 1], [1, 0]])
    y_test = np.array([1, 0])
    vectorizer = TfidfVectorizer()

    # Create a TfidfDataset instance
    dataset = TfidfDataset(
        name="complete_dataset",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        vectorizer=vectorizer
    )

    # Act
    data_loader.save_encoded_dataset_as_sparse_matrix(dataset, tmp_path)

    # Load the dataset back
    loaded_data = data_loader.load_encoded_dataset_joblib(tmp_path / "complete_dataset.joblib")

    # Assert
    assert isinstance(loaded_data, TfidfDataset)
    assert loaded_data.name == "complete_dataset"
    assert issparse(loaded_data.X_train)
    assert issparse(X_train)
    assert_array_equal(loaded_data.X_train.toarray(), X_train.toarray())
    assert_array_equal(loaded_data.y_train, y_train)
    assert issparse(loaded_data.X_test)
    assert issparse(X_test)
    assert_array_equal(loaded_data.X_test.toarray(), X_test.toarray())
    assert_array_equal(loaded_data.y_test, y_test)

def test_save_load_svm_model(tmp_path):
    """
    Test saving and loading a fitted GridSearchCV SVM model with joblib.
    """

    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Fit GridSearchCV
    svm = LinearSVC()
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(svm, param_grid, cv=3)
    grid.fit(X, y)

    # Save model
    model_path = tmp_path / "svm_model.joblib"
    data_loader.save_svm_model(grid, model_path)

    # Load model
    loaded_model = data_loader.load_svm_model(model_path)

    # Assertions
    assert isinstance(loaded_model, GridSearchCV)
    assert loaded_model.best_params_ == grid.best_params_
    assert loaded_model.best_score_ == grid.best_score_
    assert loaded_model.param_grid == grid.param_grid