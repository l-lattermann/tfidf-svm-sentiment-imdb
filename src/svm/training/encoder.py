"""Script to load data, encode features, and apply preprocessing pipeline."""

# ─── Third-Party Imports ─────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Project Imports ─────────────────────────────────────────────────────────────
from src.data.data_classes import TfidfDataset
from src.config import logging_config


# ─── Logging Setup ───────────────────────────────────────────────────────────────
logger = logging_config.configure_logging()



# Encoder
def tfidf_vectorizer(train_data, test_data,
                     max_features=10000,
                     ngram_range=(1, 2),
                     stop_words="english",
                     lowercase=True,
                     use_idf=True,
                     smooth_idf=True,
                     sublinear_tf=False,
                     name="tfidf_vectorizer"):
    """
    Create a TF-IDF vectorizer with specific parameters.
    
    Args:
        train_data (sklearn.utils.Bunch): The training dataset containing 'data' and 'target'.
        test_data (sklearn.utils.Bunch): The test dataset containing 'data' and 'target'.
    Returns:
        X_train (sparse matrix): The TF-IDF transformed training data.
        y_train (array): The labels for the training data.
        X_test (sparse matrix): The TF-IDF transformed test data.
        y_test (array): The labels for the test data.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    logger.debug(f"Encoding data with {name} using max_features={max_features}")

    # Set the parameters for the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=lowercase,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )
    logger.debug(f"Vectorizer initialized with parameters: %s", vectorizer.get_params())

    # Fit the vectorizer on the training data and transform both training and test data
    X_train = vectorizer.fit_transform(train_data.data)
    X_test = vectorizer.transform(test_data.data)
    y_train = train_data.target
    y_test = test_data.target

    logger.debug(f"Training data transformed into TF-IDF matrix with shape: {X_train.shape}")
    logger.debug(f"Test data transformed into TF-IDF matrix with shape: {X_test.shape}")
    logger.debug(f"Training labels: {y_train[:5]}")
    logger.debug(f"Test labels: {y_test[:5]}")

    # Create a TfidfDataset object to hold the encoded data
    encoded_dataset = TfidfDataset(
        name=name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        vectorizer=vectorizer
    )
    logger.debug(f"Encoded dataset created with name: {encoded_dataset.name}")

    return encoded_dataset


    



