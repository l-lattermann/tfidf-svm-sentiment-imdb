"""Main routine orchestrating the download, cleaning, preprocessing, and training for the SVM model on the IMDb dataset."""

# ─── Standard Library Imports ────────────────────────────────────────────────────
import argparse
import yaml
import itertools

# ─── Project Imports ─────────────────────────────────────────────────────────────
import src.data as dat
from src.data import data_loader
import src.preprocessing as prep
import src.svm.training as training
from src.config import logging_config

# ─── Path Imports ────────────────────────────────────────────────────────────────
from src.config.paths import DATA_DIR, TRAIN_DATA_DIR, TEST_DATA_DIR, CLEANED_DATA_TXT_DIR ,CLEANED_TRAIN_DIR, CLEANED_TEST_DIR, ENCODED_DATA_DIR, TRAINING_PARAMS, MODEL_DIR

# ─── Argument Parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run the main pipeline.")
parser.add_argument('--test', action='store_true', help="Run in test mode.")
parser.add_argument('--skip-prep', action='store_true', help="Skip preprocessing step.")
parser.add_argument('--skip-fine-tune-encoder', action='store_true', help="Skip encoder fine-tuning and use default TF-IDF parameters.")
args = parser.parse_args()

# ─── Logging Setup ───────────────────────────────────────────────────────────────
logger = logging_config.configure_logging()
if args.test:
    logger.info("Running in TEST mode (20 samples).")
else:
    logger.info("Running in PRODUCTION mode.")


def main():

    # ─── Test Mode Handling ───────────────────────────────────────────────────────
    if args.test:
        # Set test flag and sample size
        test_flag = True
        samples = 20
        logger.info("Test flag is set. Skipping full dataset download. Running only 20 samples.")
    else:
        # Set test flag and sample size
        test_flag = False
        samples = 0 # 0 means no limit, use full dataset
        logger.info("Running full dataset preparation...")


    # ─── Download Data set ─────────────────────────────────────────────────────────
    if args.skip_prep:
        logger.info("Skipping dataset download.")
    else:
        # Download the IMDb dataset
        dat.download_data.download_imdb_dataset(path=DATA_DIR)
    

    # ─── Load the Datasets ───────────────────────────────────────────────────────────
    """If skip preprocessing is set, load the datasets already cleaned."""
    if args.skip_prep:
        logger.info("Skipping dataset loading. Using existing datasets.")
    
        # Load the datasets from the cleaned directory
        train_set = dat.data_loader.load_texts_from_folder(CLEANED_TRAIN_DIR, test_mode=test_flag, sample_count=samples)
        test_set = dat.data_loader.load_texts_from_folder(CLEANED_TEST_DIR, test_mode=test_flag, sample_count=samples)
        logger.info("Loaded cleaned datasets from %s and %s", CLEANED_TRAIN_DIR, CLEANED_TEST_DIR)
    else:
        logger.info("Loading datasets from %s and %s", TRAIN_DATA_DIR, TEST_DATA_DIR)
        # Load the datasets from the original directories
        train_set = dat.data_loader.load_texts_from_folder(TRAIN_DATA_DIR, test_mode=test_flag, sample_count=samples)
        test_set = dat.data_loader.load_texts_from_folder(TEST_DATA_DIR, test_mode=test_flag, sample_count=samples)


    # ─── Preprocessing the Datasets ──────────────────────────────────────────────────
    if args.skip_prep:  
        logger.info("Skipping preprocessing step.")
    else:
        # Preprocess the datasets
        train_set.data = prep.preprocessing_pipeline.preprocessing_pipeline(train_set.data, name="train_set")
        test_set.data = prep.preprocessing_pipeline.preprocessing_pipeline(test_set.data, name="test_set")

        # Save the preprocessed datasets to the DATA_DIR
        path = CLEANED_DATA_TXT_DIR 
        dat.data_loader.save_dataset_as_txt(train_set, split="train", path=path)
        dat.data_loader.save_dataset_as_txt(test_set, split="test", path=path)


    # ─── Encode the Datasets ────────────────────────────────────────────────────────
    """If fine-tuning the encoder, encode multiple versions of the dataset with different max_features values."""
    # Load the training parameters from the YAML file
    with open(TRAINING_PARAMS, "r") as f:
        training_params = yaml.load(f, Loader=yaml.FullLoader)
    vec_param_grid = training_params.get("vectorizer_param_grid", {})
    logger.info(f"Using TF-IDF parameters: %s", vec_param_grid)


    if args.skip_fine_tune_encoder:
        logger.info("Skipping fine-tuning of the encoder. Using default parameters.")

        # Use the first parameter in the list as the default
        param_dict = {name: value[0] for name, value in vec_param_grid.items()}
        logger.info("Encoded dataset with parameters: %s", param_dict)

        # Encode the dataset using the TF-IDF vectorizer
        data_set = training.encoder.tfidf_vectorizer(train_set, test_set, **param_dict)

        # Create a file path from the encoding parameters
        name = dat.data_loader.param_dict_to_filename(param_dict)
        path = ENCODED_DATA_DIR / f"encoded__{name}.joblib"

        # Save the encoded datasets as sparse matrices
        dat.data_loader.save_encoded_dataset_as_sparse_matrix(data_set, path=path)
        logger.info("Encoded dataset saved to %s", path)

        # Store the encoded dataset in the dictionary
        encoded_datasets[name] = (data_set, param_dict)

    else:
        logger.info("Fine-tuning encoder. Generating multiple encoded datasets with different parameters.")

        # Create a list of all mutations of the parameter grid
        keys, values = zip(*vec_param_grid.items())
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        logger.info("Generated %d combinations of parameters for fine-tuning.", len(combinations))

        # Create a dictionary to store encoded datasets for different max_features
        encoded_datasets = {}
        for param_dict in combinations:
            logger.info("Encoding dataset with parameters: %s", param_dict)
            
            # Encode the datasets using the TF-IDF vectorizer
            data_set = training.encoder.tfidf_vectorizer(train_set, test_set, **param_dict)
            
            # Create a file path from the encoding parameters
            name = dat.data_loader.param_dict_to_filename(param_dict)
            path = ENCODED_DATA_DIR / f"encoded__{name}.joblib"

            # Save the encoded datasets as sparse matrices
            dat.data_loader.save_encoded_dataset_as_sparse_matrix(data_set, path=path)
            logger.info("Encoded dataset saved to %s", path)

            # Store the encoded dataset in the dictionary
            encoded_datasets[name] = (data_set, param_dict)


    # ─── Train the SVM Model ───────────────────────────────────────────────────────
    """If fine-tuning the encoder, train multiple SVM models on all the differently encoded data sets."""

    if args.skip_fine_tune_encoder:
        logger.info("Skipping fine-tuning of the encoder. Training SVM model with default parameters.")

        # Train the SVM model using the encoded datasets
        model = training.training_pipeline.train_svm_model(data_set)

        # Create a file path from the parameters
        name_final_model = encoded_datasets.keys()[0] 
        path = MODEL_DIR / f"svm__{name_final_model}.joblib"

        # Save the trained SVM model
        dat.data_loader.save_svm_model(model, path=path)
        logger.info("SVM model trained and saved to %s", path)
        
    else:
        logger.info("Fine-tuning encoder. Saving only the best model.")

        # Create a directory to save the SVM models
        models = {}

        # Iterate over the encoded datasets and train SVM models
        for name, set_and_params in encoded_datasets.items():

            # Get set and params from tuple 
            data_set = set_and_params[0]
            param_dict = set_and_params[1]
            logger.info(f"Training SVM model with params: %s", param_dict)

            # Train the SVM model using the encoded datasets
            model = training.training_pipeline.train_svm_model(data_set)

            # Store the model in the dictionary
            models[name] = (model, param_dict)

        # Store the best models based on their scores
        best_models = []
        for name, model_and_params in models.items():
            # Get the model and the training parameters
            model = model_and_params[0]
            param_dict = model_and_params[1]

            # Get the model score and the estimator itself
            score = model.best_score_
            model = model.best_estimator_

            # Store the models with their score as key
            best_models.append((model, param_dict, score, name))

        # Get the best model based on the highest score
        best_models.sort(key=lambda x: x[2], reverse=True)
        best_model, best_params, best_score, best_name = best_models[0]

        # Create a file path from the parameters
        name_final_model = best_name
        path = MODEL_DIR / f"svm__{name_final_model}.joblib"

        # Save the best SVM model
        dat.data_loader.save_svm_model(best_model, path=path)
        logger.info("Best SVM model saved with parameters: %s", best_params)
        logger.info("Best SVM model saved with score: %f", best_score)


    # ─── Select the best encoder based on the evaluation ───────────────────────────────────────────────────────
    logger.info("Selected the best encoder based on the evaluation.")
    name_final_encoder = name_final_model 
    logger.info("Selected encoder: %s", name_final_encoder)

    # Get the dataset
    best_dataset = encoded_datasets[name_final_encoder][0]
    
    # Get the best encoder from the encoded datasets
    best_encoder = best_dataset.vectorizer


    # Save the best encoder
    path_encoder = MODEL_DIR / f"vectorizer__{name_final_encoder}.joblib"
    data_loader.save_encoder(path_encoder, best_encoder)
    logger.info("Best encoder saved to %s", path_encoder)


if __name__ == "__main__":
    main()