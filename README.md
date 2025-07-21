# IMDb Sentiment Analysis with SVM

This repository contains a complete pipeline to perform sentiment analysis on the IMDb movie reviews dataset using a Support Vector Machine (SVM). The pipeline includes data downloading, preprocessing, TF-IDF vectorization, optional fine-tuning, and model training.

### Best results achieved
---
#### Classification Report:
```
              precision    recall  f1-score   support

         neg       0.91      0.89      0.90      4968
         pos       0.89      0.92      0.90      5032

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
├── data                      # Raw and cleaned data storage
├── models                    # Trained SVM models
├── logs                      # Log files
├── src
│   ├── config                # Configuration files and paths
│   ├── data                  # Data downloading and loading utilities
│   ├── preprocessing         # Text preprocessing modules
│   └── svm                   # TF-IDF encoding and SVM training
├── main.py                   # Main pipeline script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
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

### Basic Usage

Run the complete pipeline:

```bash
python main.py
```

### Optional Arguments

- `--test`: Run the pipeline in test mode with 20 samples.
- `--skip-prep`: Skip data downloading and preprocessing; uses already preprocessed data.
- `--skip-fine-tune-encoder`: Skip the fine-tuning step and use the default values for training and encoding

Example:

```bash
python main.py --test
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