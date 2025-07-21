"""Lemmatization Module. This module provides functionality to lemmatize text using spaCy."""

# ─── Third-Party Imports ─────────────────────────────────────────────────────────────────────
import spacy

# ─── Project Imports ───────────────────────────────────────────────────────────────────────
from src.config import logging_config

# ─── Set up logging ───────────────────────────────────────────────────────────────────────────
logger = logging_config.configure_logging()




# Check if the english model already exists
if spacy.util.is_package("en_core_web_sm"):
    logger.info("Using existing spaCy model 'en_core_web_sm'.")
else:
    logger.info("Downloading spaCy model 'en_core_web_sm'.")   
    spacy.cli.download("en_core_web_sm")

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    """
    Lemmatize the input text using spaCy.
    
    Args:
        text (str): Input text to be lemmatized.
    
    Returns:
        str: Lemmatized text.
    """
    logger.debug(f"Lemmatizing text: {text}")

    # Process the text with spaCy
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    logger.debug(f"Lemmatized text: {lemmatized_text}")
    return lemmatized_text
