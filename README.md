# IMDb Sentiment Analysis with SVM

This repository contains a complete pipeline to perform sentiment analysis on the IMDb movie reviews dataset using a Support Vector Machine (SVM). The pipeline includes data downloading, preprocessing, TF-IDF vectorization, optional fine-tuning, and model training.

### Best results achieved
---
#### Classification Report:
```
              precision    recall  f1-score   support

         neg       0.91      0.88      0.90      4968
         pos       0.89      0.91      0.90      5032

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
```

- **Precision**: Correctly predicted positives out of all predicted positives.
- **Recall**: Correctly predicted positives out of all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.  
- **Support**: Number of true samples for each class in the test set.
- **Accuracy**: Overall proportion of correct predictions.
- **Macro Avg**: Average of metrics across classes (treats all classes equally).
- **Weighted Avg**: Average weighted by class support (accounts for imbalance).
---

## Project Structure

```
.
├── logs/                                 # Runtime logs and metadata
│   ├── training.log                      # Log output from training pipeline
│   └── prediction.log                    # Log output from prediction pipeline
├── models/                               # Saved ML models and vectorizers
│   ├── vectorizer.joblib                 # Optimal Vectorizer
│   └── model.joblib                      # Optimal Model

├── data/                                 # Raw and processed data storage
│   ├── aclImdb/                          # Original IMDb dataset
│   │   ├── aclImdb.zip                   # Zipped dataset file
│   │   ├── test/                         # Raw test data
│   │   └── train/                        # Raw train data
│   └── cleaned_txt/                      # Cleaned text data
│       ├── test/                         # Cleaned test data
│       └── train/                        # Cleaned train data

├── src/                                  # Source code
│   ├── config/                           # Configuration and logging
│   │   ├── logging_config.py             # Logging setup
│   │   ├── paths.py                      # Centralized file paths
│   │   └── training_params.yaml          # Model training parameters

│   ├── data/                             # Data ingestion and structuring
│   │   ├── data_classes.py               # Data class definitions
│   │   ├── data_loader.py                # Data loading functions
│   │   └── download_data.py              # Dataset download script

│   ├── preprocessing/                    # Text preprocessing logic
│   │   ├── clean_text.py                 # Text cleaning steps
│   │   ├── filters.py                    # Stopword and regex filters
│   │   ├── lemmatization.py              # Lemmatization functions
│   │   └── preprocessing_pipeline.py     # Combined preprocessing flow

│   └── svm/                              # SVM-specific components
│       └── training/
│           ├── gridsearch_trainer.py     # SVM training with hyperparameter search
│           └── vectorizer.py             # TF-IDF vectorizer setup

├── tests/                                # Unit tests
│   ├── conftest.py                       # Shared fixtures
│   ├── config/                           # Config-related tests
│   │   └── test_config_yaml.py           # YAML config test
│   ├── data/                             # Data module tests
│   │   └── test_data_loader.py           # Tests for data loader
│   ├── preprocessing/                    # Preprocessing tests
│   │   ├── test_cleaning.py              # Cleaning test cases
│   │   ├── test_filters.py               # Filter logic tests
│   │   ├── test_lemmatization.py         # Lemmatization tests
│   │   └── test_preprocessing_pipeline.py# Pipeline integration tests
│   ├── logs/                             # Log output testing
│   │   ├── test.log                      # Log output from pytest
│   │   └── coverage.json                 # Coverage metadata
├── pytest.ini                            # Pytest configuration

├── training_pipeline.py                  # Main training entry point
├── prediction_pipeline.py                # Main inference script
├── requirements.txt                      # Python dependencies
└── README.md                             # Project description and instructions
```

## Installation

Clone the repository and install dependencies:

### Mac/Linux:
```bash
git clone https://github.com/l-lattermann/tfidf-svm-sentiment-imdb.git
cd tfidf-svm-sentiment-imdb

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Windows:
```cmd
git clone https://github.com/l-lattermann/tfidf-svm-sentiment-imdb.git
cd tfidf-svm-sentiment-imdb

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Optional Arguments

- `--test`: Run the pipeline in test mode with 20 samples.
- `--skip-prep`: Skip data downloading and preprocessing; uses already preprocessed data.
- `--skip-fine-tune-encoder`: Skip the fine-tuning step and use the default values for training and encoding

Example:

```bash

python training_pipeline.py --test

```

### Basic Usage

Run the complete pipeline:

```bash
# Testrun, downloading, preprocessing the Data (60secons on Mac M2)
python training_pipeline.py --test

# Test deploy the Model for inference
python prediction_pipeline.py
```

## Pipeline Steps

1. **Data Downloading**: Downloads the IMDb dataset.
2. **Preprocessing**: Cleans and preprocesses text data.
3. **TF-IDF Encoding**: Converts text data into numerical format using TF-IDF with the parameters specified in the yaml file.
4. **SVM Training**: Trains the SVM model using GridSearchCV for hyperparameter tuning.
5. **Model Saving**: Saves the trained model and encoder.

## Configuration

Adjust hyperparameters and pipeline configurations in:

- `src/config/training_params.yaml`

## Logging

Logs are stored in the `logs/` directory, documenting pipeline execution and performance metrics.

## Results

Trained SVM models are saved under `models/` with descriptive filenames indicating the hyperparameters used.

---

## Dataset Reference

[Stanford IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

---

**Note:** Ensure necessary permissions are granted for reading/writing files and directories.