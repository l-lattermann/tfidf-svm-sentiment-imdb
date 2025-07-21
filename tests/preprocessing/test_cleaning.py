"""Tests for the text cleaning functions in the preprocessing module."""

# ─── Module Imports ──────────────────────────────────────────────────────────────
from src.preprocessing import clean_text

# ─── Standard Library Imports ────────────────────────────────────────────────────
import pytest


def test_lower_case():
    """ Check if the lower_case function converts text to lower case. """

    # Test case for lower_case function
    text = "Hello World!"
    expected = "hello world!"
    result = clean_text.lower_case(text)
    assert result == expected, f"Expected '{expected}', but got '{result}"

def test_strip_whitespace():
    """ Check if the strip_whitespace function removes leading and trailing whitespace. """

    # Test case for strip_whitespace function
    text = "   Hello World!   "
    expected = "Hello World!"
    result = clean_text.strip_whitespace(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_replace_wrong_quotation_marks():
    """ Check if the replace_wrong_quotation_marks function replaces wrong quotation marks. """

    # Test case for replace_wrong_quotation_marks function
    text = "This is a `test` with wrong ´quotation´ marks."
    expected = "This is a 'test' with wrong 'quotation' marks."
    result = clean_text.replace_wrong_quotation_marks(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_remove_non_alphanumeric_some():
    """ Check if the remove_non_alphanumeric function removes non-alphanumeric characters. """

    # Test case for remove_non_alphanumeric function
    text = "Keep all this characters: !?'\".,#\\-:/%$ But remove this: @&*()_+=<>`~"
    expected = "Keep all this characters: !?'\".,#-:/%$ But remove this: "
    result = clean_text.remove_non_alphanumeric_some(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_remove_non_alphanumeric_all():
    """ Check if the remove_non_alphanumeric function removes non-alphanumeric characters. """

    # Test case for remove_non_alphanumeric function
    text = "Keep all this characters: !?.,' But remove this: @&*()_+=<§$%&/(/)(&=)/()/>`~"
    expected = "Keep all this characters !?.,' But remove this "
    result = clean_text.remove_non_alphanumeric_all(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_remove_numeric_and_punctuation():
    """ Check if the remove_numeric_and_punctuation function removes numeric and punctuation characters. """

    # Test case for remove_numeric_and_punctuation function
    text = "Remove all alphanumeric characters and punctuation: 1234567890!@#$%^&*()'_+"
    expected = "Remove all alphanumeric characters and punctuation "
    result = clean_text.remove_numeric_and_punctuation(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_remove_multiple_punctuation():
    """ Check if the remove_multiple_punctuation function removes multiple punctuation characters. """

    # Test case for remove_multiple_punctuation function
    text = "Hello!!! How are you??? I'm fine... really!!!"
    expected = "Hello! How are you? I'm fine. really!"
    result = clean_text.remove_multiple_punctuation(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_remove_urls():
    """ Check if the remove_urls function removes URLs from text. """

    # Test case for remove_urls function
    text = (
        "Visit our website at https://example.com/watch?v=Mc9JAra8WZU&list=PLMWaZv9C0otal1FO "
        "for more information or just type www.example.com."
    )
    expected = "Visit our website at  for more information or just type "
    result = clean_text.remove_urls(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_remove_html_tags():
    """ Check if the remove_html_tags function removes HTML tags from text. """

    # Test case for remove_html_tags function
    text = "<p>This is a <strong>test</strong> with <a href='https://example.com'>links</a>.</p>"
    expected = "This is a test with links."
    result = clean_text.remove_html_tags(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

# Test the pipeline function
@pytest.mark.parametrize("input_text, rem_all_nonalphabetic, expected_output", [
    # Lowercasing, whitespace, some non-alphanum
    ("  Hello!!!  ", False, "hello!"),
    ("  Hello!!!  ", True, "hello"),

    # Wrong quotation marks
    ("This is a `test` with wrong ´quotation´ marks.", False, "this is a 'test' with wrong 'quotation' marks."),
    ("This is a `test` with wrong ´quotation´ marks.", True, "this is a test with wrong quotation marks"),

    # Remove Urls
    ("Visit our website at https://example.com/watch?v=Mc9JAra8WZU&list=PLMWaZv9C0otal1FO for more information, or just type www.example.com.", 
     False, 
     "visit our website at for more information, or just type"),
    ("Visit our website at https://example.com/watch?v=Mc9JAra8WZU&list=PLMWaZv9C0otal1FO for more information or just type www.example.com.", 
     True, 
     "visit our website at for more information or just type"),
   
])
def test_regex_cleaning(input_text, rem_all_nonalphabetic, expected_output):
    cleaned = clean_text.regex_cleaning_pipeline(input_text, rem_all_nonalphabetic=rem_all_nonalphabetic)
    assert cleaned == expected_output, f"Expected '{expected_output}', but got '{cleaned}'"