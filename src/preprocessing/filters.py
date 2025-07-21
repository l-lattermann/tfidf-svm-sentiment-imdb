"""Filters for text preprocessing."""

# ─── Imports ─────────────────────────────────────────────────────────────────────
import re

# ─── Project Imports ───────────────────────────────────────────────────────────────────────
from src.config import logging_config

# ─── Set up logging ─────────────────────────────────────────────────────────
logger = logging_config.configure_logging()

def expand_contractions(text):
    """
    Expand contractions in the text.
    
    Args:
        text (str): input text.
    
    Returns:
        str: Text with contractions expanded.
    """
    logger.debug(f"Expanding contractions in text: {text}")

    # Define a dictionary of contractions and their expansions
    contractions = {
        "aren't": "are not",        "arent": "are not",
        "isn't": "is not",          "isnt": "is not",
        "we're": "we are",          "were": "we are",
        "can't": "cannot",          "cant": "cannot",
        "it's": "it is",            "its": "it is",
        "we've": "we have",         "weve": "we have",
        "couldn't": "could not",    "couldnt": "could not",
        "we'll": "we will",         "well": "we will",
        "didn't": "did not",        "didnt": "did not",
        "it'll": "it will",         "itll": "it will",
        "we'd": "we would",         "wed": "we would",
        "don't": "do not",          "dont": "do not",
        "mustn't": "must not",      "mustnt": "must not",
        "doesn't": "does not",      "doesnt": "does not",
        "she's": "she is",          "shes": "she is",
        "weren't": "were not",      "werent": "were not",
        "hadn't": "had not",        "hadnt": "had not",
        "she'll": "she will",       "shell": "she will",
        "what's": "what is",        "whats": "what is",
        "haven't": "have not",      "havent": "have not",
        "where's": "where is",      "wheres": "where is",
        "he's": "he is",            "hes": "he is",
        "she'd": "she would",       "shed": "she would",
        "who's": "who is",          "whos": "who is",
        "who'll": "who will",       "wholl": "who will",
        "he'll": "he will",         "hell": "he will",
        "shouldn't": "should not", "shouldnt": "should not",
        "won't": "will not",        "wont": "will not",
        "he'd": "he would",         "hed": "he would",
        "that's": "that is",        "thats": "that is",
        "wouldn't": "would not",    "wouldnt": "would not",
        "there's": "there is",      "theres": "there is",
        "you're": "you are",        "youre": "you are",
        "here's": "here is",        "heres": "here is",
        "they're": "they are",      "theyre": "they are",
        "they've": "they have",     "theyve": "they have",
        "i'm": "i am",              "im": "i am",
        "they'll": "they will",     "theyll": "they will",
        "you'll": "you will",       "youll": "you will",
        "i've": "i have",           "ive": "i have",
        "they'd": "they would",     "theyd": "they would",
        "you'd": "you would",       "youd": "you would",
        "i'll": "i will",           "ill": "i will",
        "i'd": "i would",           "id": "i would",
        "wasn't": "was not",        "wasnt": "was not",
        "let's": "let us",          "lets": "let us",
    }
    
    
    for contraction, expansion in contractions.items():
        text = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text, flags=re.IGNORECASE)
    
    logger.debug(f"Text after expanding contractions: {text}")
    return text

def slang_handling(text):
    """
    Handle common slang terms in the text.
    
    Args:
        text (str): input text.
    
    Returns:
        str: Text with slang terms replaced.
    """
    logger.debug(f"Handling slang terms in text: {text}")

    # Define a dictionary of slang terms and their replacements
    slang = {
        'wtf': 'what the fuck',
        'fyi': 'for your information',
        'sux': 'is bad',
        'cam': 'camera',
        'coz': 'because',
        'brb': 'be right back',
        'sayin': 'saying',
        'imdb': 'internet movie database',
        'atm': 'at the moment',
        'wth': 'what the fuck',
        'cryin': 'crying',
        'lmao': 'funny',
        'omfg': 'oh god',
        'omg': 'oh god',
        'ily': 'i love you',
        'dvd': 'digital versatile disc',
        'talkin': 'talking',
        'dunno': 'do not know',
        'os': 'operating system',
        'ptsd': 'post traumatic stress disorder',
        'doin': 'doing',
        'bg': 'background',
        'prolly': 'probably',
        'af': 'very much',
        'ewww': 'disgusting',
        'aint': 'is not',
        'rofl': 'funny',
        'shoulda': 'should have',
        'nvm': 'never mind',
        'yall': 'you all',
        'thx': 'thank you',
        'btw': 'by the way',
        'cuz': 'because',
        'rip': 'rest in peace',
        'romcom': 'romantic comedy',
        'lol': 'funny',
        'vfx': 'visual effects',
        'lemme': 'let me',
        'comin': 'coming',
        'imho': 'in my opinion',
        'bluray': 'blu-ray disc',
        'livin': 'living',
        'gg': 'good',
        'gotta': 'got to',
        'kinda': 'kind of',
        'tho': 'though',
        'yellin': 'yelling',
        'gimme': 'give me',
        'dc': 'detective comics',
        'bgm': 'background music',
        'bc': 'because',
        'ff': 'fast forward',
        'woulda': 'would have',
        'gettin': 'getting',
        'lookin': 'looking',
        'cgi': 'computer generated imagery',
        'walkin': 'walking',
        'hd': 'high definition',
        'mpaa': 'motion picture association of america',
        'fx': 'effects',
        'nah': 'no',
        'gonna': 'going to',
        'sfx': 'sound effects',
        'hires': 'high resolution',
        'naw': 'no',
        'rn': 'right now',
        'idk': 'i do not know',
        'ty': 'thank you',
        'imo': 'in my opinion',
        'betcha': 'bet you',
        'musta': 'must have',
        'wanna': 'want to',
        'nope': 'no',
        'ffs': 'out of frustration',
        'coulda': 'could have',
        'watcha': 'what are you',
        'givin': 'giving',
        'ost': 'original soundtrack',
        'mc': 'main character',
        'soo': 'so',
        'makin': 'making',
        'irl': 'in real life',
        'lotta': 'lot of',
        'tbh': 'to be honest',
        'goin': 'going',
        'ya': 'you',
        'fr': 'for real',
        'outta': 'out of',
        'plz': 'please',
        'pls': 'please',
        'u': 'you'
    }

    # Replace slang terms with their full forms
    for slang_term, replacement in slang.items():
        text = re.sub(r'\b' + re.escape(slang_term) + r'\b', replacement, text, flags=re.IGNORECASE)

    logger.debug(f"Text after handling slang terms: {text}")    
    return text

# ====================================== PIPELINE =======================================
def filtering_pipeline(text):
    """
    Apply a series of text preprocessing steps to clean the input text.
    
    Args:
        text (str): input text.
    
    Returns:
        str: Cleaned text.
    """
    logger.debug(f"Starting filtering pipeline for text: {text}")

    # Expand contractions
    text = expand_contractions(text)
    
    # Handle slang terms
    text = slang_handling(text)

    logger.debug(f"Final filtered text: {text}")
    return text