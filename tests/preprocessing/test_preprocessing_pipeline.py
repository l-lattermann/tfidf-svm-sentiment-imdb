from src.preprocessing import preprocessing_pipeline

def test_preprocessing_pipeline():
    test_text = """
        OMG! I can't believe this "Movie"!!! It had 10 explosions in the first 5 minutes... 🤯
        Check it out at https://example.com/movie?id=123 – you'll love it! 
        Also, what's with the 'acting'? Lol. That guy was like: “I’m gonna save the world!” – and then *boom*.
        Total nonsense, but kinda fun tbh...
        """
    
    expected_output = (
        "oh god I can not believe this movie it have explosion in the first minute "
        "check it out at you will love it "
        "also what be with the act funny that guy be like I be go to save the world and then boom "
        "total nonsense but kind of fun to be honest"
    )

    assert expected_output == preprocessing_pipeline.preprocessing_pipeline(test_text)