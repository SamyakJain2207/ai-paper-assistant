from app.services.nlp.preprocessing import (
    normalize_text,
    preprocess_text,
    remove_stopwords,
    tokenize,
)


def test_normalize_text():
    text = "  Attention   Mechanisms ARE Powerful!  "

    assert normalize_text(text) == "attention mechanisms are powerful!"


def test_tokenize():
    text = "Attention mechanisms are powerful."

    assert tokenize(text) == [
        "Attention",
        "mechanisms",
        "are",
        "powerful",
    ]


def test_preprocess_text():
    text = "  Attention   Mechanisms ARE Powerful!  "

    assert preprocess_text(text) == [
        "attention",
        "mechanisms",
        "are",
        "powerful",
    ]

def test_remove_stopwords():
    tokens = ["the", "transformer", "is", "a", "neural", "network"]

    assert remove_stopwords(tokens) == [
        "transformer",
        "neural",
        "network",
    ]