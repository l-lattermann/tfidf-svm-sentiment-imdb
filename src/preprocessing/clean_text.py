"""
    src/preprocessing/clean_text.py
    This module provides functions for cleaning and preprocessing text data.
    It includes functions to lower case text, strip whitespace, remove non-alphanumeric characters,
    replace wrong quotation marks, remove URLs, and more.
    The functions can be used individually or as part of a pipeline for text preprocessing.

"""

# ─── Standard Library Imports ───────────────────────────────────────────────────────────────────────
import re
import logging

# ─── Project Imports ───────────────────────────────────────────────────────────────────────
from src.config import logging_config

# ─── Set up logging ─────────────────────────────────────────────────────────────────────────────────
logger = logging_config.configure_logging()



def lower_case(text):
    """
    Convert text to lower case.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Lowercased text.
    """
    logger.debug(f"Converting text to lower case: {text}")

    text = text.lower()

    logger.debug(f"Converted text: {text}")
    return text

def strip_whitespace(text):
    """
    Strip leading and trailing whitespace from text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Text with leading and trailing whitespace removed.
    """
    logger.debug(f"Stripping whitespace from text: {text}")

    text = re.sub(r"\s+", " ", text).strip()

    logger.debug(f"Stripped text: {text}")
    return text

def remove_non_alphanumeric_some(text):
    """
    Remove non-alphanumeric characters from text, but keep [!?'\".,#\\-:/%$]
    Args:
        text (str): Input text.

    Returns:
        str: Text with non-alphanumeric characters removed, except for specified punctuation.
    """
    logger.debug(f"Removing non-alphanumeric characters from text: {text}")

    text = re.sub(r"[^a-zA-Z0-9\s!?'\".,#\-:/%$]", "", text)

    logger.debug(f"Text after removing non-alphanumeric characters: {text}")
    return text

def replace_wrong_quotation_marks(text):
    """
    Replace wrong quotation marks in text with standard double quotes.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Text with wrong quotation marks replaced.
    """
    logger.debug(f"Replacing wrong quotation marks in text: {text}")

    text = re.sub(r"[`´]", "'", text)

    logger.debug(f"Text after replacing quotation marks: {text}")
    return text

def remove_non_alphanumeric_all(text):
    """
    Remove non-alphanumeric characters from text, but keep [!?.,'.]

    Args:
        text (str): Input text.

    Returns:
        str: Text with non-alphanumeric characters removed, except for specified punctuation.
    """
    logger.debug(f"Removing non-alphanumeric characters from text: {text}")

    text = re.sub(r"[^a-zA-Z0-9\s.,'!?]", "", text)

    logger.debug(f"Text after removing non-alphanumeric characters: {text}")
    return text   

def remove_multiple_punctuation(text):
    """
    Remove multiple consecutive punctuation characters from text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Text with multiple consecutive punctuation characters removed.
    """
    logger.debug(f"Removing multiple punctuation from text: {text}")

    text = re.sub(r"([!?.,'\"-])\1+", r"\1", text)

    logger.debug(f"Text after removing multiple punctuation: {text}")
    return text

def remove_numeric_and_punctuation(text):
    """
    Remove all alphanumeric characters and punctuation from text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Text with all alphanumeric characters and punctuation removed.
    """
    logger.debug(f"Removing numeric and punctuation from text: {text}")
    
    text = re.sub(r"[^A-Za-z\s]", "", text)

    logger.debug(f"Text after removing numeric and punctuation: {text}")
    return text

def remove_urls(text):
    """
    Remove URLs from text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Text with URLs removed
    """
    logger.debug(f"Removing URLs from text: {text}")

    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"www\.[^\s]+", "", text)

    logger.debug(f"Text after removing URLs: {text}")
    return text

def remove_html_tags(text):
    """
    Remove HTML tags from text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Text with HTML tags removed.
    """
    logger.debug(f"Removing HTML tags from text: {text}")

    text = re.sub(r"<[^>]+>", "", text)

    logger.debug(f"Text after removing HTML tags: {text}")
    return text




# ====================================== PIPELINE =======================================
def regex_cleaning_pipeline(text, rem_all_nonalphabetic=True):
    """
    Clean the input text by applying a series of preprocessing steps.
    
    Args:
        text (str): The input text to clean.
        rem_all_nonalphabetic (bool): If True, removes all non-alphabetic characters and punctuation.
    
    Returns:
        str: The cleaned text.
    """
    logger.debug(f"Starting regex cleaning pipeline for text: {text}")


    # Lowercase the text
    text = lower_case(text)

    # Remove HTML tags
    text = remove_html_tags(text)
    
    # Replace wrong quotation marks
    text = replace_wrong_quotation_marks(text)
    
    # Remove URLs
    text = remove_urls(text)
    
    # Remove non-alphanumeric characters (all)
    if not rem_all_nonalphabetic:
        text = remove_non_alphanumeric_all(text)
        text = remove_multiple_punctuation(text)
    
    # Remove numeric and punctuation if specified
    elif rem_all_nonalphabetic:
        text = remove_numeric_and_punctuation(text)

    # Strip whitespace
    text = strip_whitespace(text)

    logger.debug(f"Final cleaned text: {text}")
    return text