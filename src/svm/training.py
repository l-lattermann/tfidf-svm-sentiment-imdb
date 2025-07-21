"""SVM Training Script.

Loads encoded data, trains a LinearSVC classifier with optional hyperparameter tuning,
and logs performance metrics.
"""

# ─── Standard Library Imports ────────────────────────────────────────────────────
from pathlib import Path
import sys
import logging
import yaml

# ─── Path Setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ─── Project Module Imports ──────────────────────────────────────────────────────
from src.config.paths import TRAINING_PARAMS
from src.data.tfidf_dataset import TfidfDataset
from src.config import logging_config


# ─── Third-Party Imports ─────────────────────────────────────────────────────────
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# ─── Logging Setup ───────────────────────────────────────────────────────────────
logger = logging_config.configure_logging()

# ─── Load Vectorizer Configuration ───────────────────────────────────────────────
with open(TRAINING_PARAMS, "r") as f:
    training_params = yaml.load(f, Loader=yaml.FullLoader)




def train_svm_model(data: TfidfDataset):
    """
    Train a LinearSVC model using the provided data.
    Args:
        data (TfidfDataset): The dataset containing training and test data.
    Returns:
        grid (GridSearchCV): A fitted GridSearchCV object. 
            - `grid.best_estimator_` gives the best LinearSVC model.
            - `grid.best_params_` provides the best hyperparameter combination.
            - `grid.best_score_` contains the best cross-validated score.
            - `grid.predict(...)` can be used to make predictions with the best model. 
    """
    logger.info("Starting SVM model training...")
    
    # Define SVM model
    svm = LinearSVC()
    logger.info("Initialized LinearSVC model.")

    # Get grid parameters from training params
    grid_params = training_params.get("grid_search_params", {})

    # Define parameter grid (no tfidf params here)
    param_grid = {
        "C": grid_params["C"],
        "tol": grid_params["tol"],
        "max_iter": grid_params["max_iter"],
        "class_weight": grid_params["class_weight"],}
    logger.info("Using parameter grid for Grid Search: %s", param_grid)

    # Grid Search
    tqdm.write("Starting Grid Search...")
    grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(data.X_train, data.y_train)
    tqdm.write("Grid Search completed.")

    # Write the best parameters to the log file
    logger.info("Best parameters found: %s", grid.best_params_)

    # Evaluate the best model
    y_pred = grid.predict(data.X_test)

    # Create a classification report and write it to the log file
    target_names = ["neg", "pos"]
    report = classification_report(data.y_test, y_pred, target_names=target_names)
    logger.info("Classification Report:\n%s", report)

    return grid



