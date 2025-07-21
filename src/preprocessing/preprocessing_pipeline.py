"""Preprocessing Pipeline Module.

Combines various preprocessing steps such as cleaning, filtering, and lemmatization.
"""

# ─── Third Party Imports ─────────────────────────────────────────────────────────
from sklearn.utils import Bunch
from tqdm import tqdm

# ─── Local Application Imports ───────────────────────────────────────────────────
from . import clean_text
from . import filters
from . import lemmatization

# ─── Standard Imports──────────────────────────────────────────────────────────────────────────
import logging

# ─── Set up logging ───────────────────────────────────────────────────────────────────────────
from src.config import logging_config
logger = logging_config.configure_logging()





# Ochestration of the preprocessing pipeline
def preprocessing_pipeline(text: str | Bunch, name: str = "default") -> str | list[str]:
    """
    Preprocess the input text by cleaning, filtering, and lemmatizing it.
    Args:
        text (str or list[str]): Input text or list of texts to be preprocessed.
        name (str): Name of the dataset or text source for logging purposes.
    Returns:
        str or list[str]: Preprocessed text or list of preprocessed texts.
    """
    logger.debug(f"Starting preprocessing pipeline for {name}")

    def _pipeline(text: str):
        # Step 1: Clean the text
        cleaned_text = clean_text.regex_cleaning_pipeline(text)

        # Step 2: Filter the text
        filtered_text = filters.filtering_pipeline(cleaned_text)

        # Step 3: Lemmatize the text
        lemmatized_text = lemmatization.lemmatize_text(filtered_text)

        return lemmatized_text

    # Check if the input is a Bunch object or a list of strings
    if isinstance(text, list):
        logger.debug(f"Preprocessing a list of {len(text)} texts for {name}")

        elements =len(text)
        # Apply the preprocessing pipeline to each text in the list
        lemmatized_text = [
                _pipeline(t) for t in tqdm(text, desc=f"Preprocessing {name}", unit="text", total=elements, dynamic_ncols=True)
            ]
    # If the input is a single string, apply the pipeline directly
    else:
        logger.debug(f"Preprocessing a single text for {name}")
        lemmatized_text = _pipeline(text)
    
    logger.debug(f"Completed preprocessing pipeline for {name}")
    return lemmatized_text
            


