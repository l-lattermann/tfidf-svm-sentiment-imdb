from src.preprocessing import filters
import pytest

# Test the individual functions first
def test_expand_contractions():
    """ Check if the contraction handling function expands contractions correctly. """
    
    text = [
        "i'm going to the store.",
        "he's not here.",
        "they've already left.",
        "she'll be back, don't worry.",
    ]

    expected = [
        "i am going to the store.",
        "he is not here.",
        "they have already left.",
        "she will be back, do not worry.",
    ]

    for text_item, expected_item in zip(text, expected):
        result = filters.expand_contractions(text_item)
        assert result == expected_item, f"Expected '{expected_item}', but got '{result}'"
    
def test_slang_handling():
    """ Check if the slang handling function replaces slang terms correctly. """
    
    text = [
        "wtf is going on?",
        "fyi, the meeting is at 3pm.",
        "lmao, that was hilarious!",
        "omfg, did you see that?",
        "wtf, this sux!",
    ]   

    expected = [
        "what the fuck is going on?",
        "for your information, the meeting is at 3pm.",
        "funny, that was hilarious!",
        "oh god, did you see that?",
        "what the fuck, this is bad!"
    ]


    for text_item, expected_item in zip(text, expected):
        result = filters.slang_handling(text_item)
        assert result == expected_item, f"Expected '{expected_item}', but got '{result}'"


# Test the pipeline
@pytest.mark.parametrize("input_text, expected_output", [
    ("i'm gonna go now, fyi.", "i am going to go now, for your information."),
    ("he's not here, lmao.", "he is not here, funny."),
    ("they've left. omfg!", "they have left. oh god!"),
    ("she'll be back, don't worry. sux, right?", "she will be back, do not worry. is bad, right?"),
    ("wtf, shouldn't he be here?", "what the fuck, should not he be here?"),
])
def test_filtering_pipeline(input_text, expected_output):
    result = filters.filtering_pipeline(input_text)
    assert expected_output == result, f"Expected '{expected_output}', but got '{result}'"

