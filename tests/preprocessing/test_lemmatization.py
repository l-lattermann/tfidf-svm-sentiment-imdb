from src.preprocessing import lemmatization

# Test Lemmatization
def test_lemmatization():
    """ Check if the lemmatization function lemmatizes words correctly. """
    
    text = [
        "running",
        "ran",
        "children",
        "mice",
        "went",
        "I went to school, but it was closed, so I kept searching and was successful."
    ]

    expected = [
        "run",
        "run",
        "child",
        "mouse",
        "go",
        "I go to school , but it be closed , so I keep search and be successful ."
    ]

    for text_item, expected_item in zip(text, expected):
        result = lemmatization.lemmatize_text(text_item)
        assert result == expected_item, f"Expected '{expected_item}', but got '{result}'"