from app.services.nlp.text_cleaning import (
    clean_text,
    lemmatize,
    process_text,
    remove_stopwords,
)


def test_remove_stopwords():
    doc = process_text("The transformer is a neural network.")

    assert remove_stopwords(doc) == [
        "transformer",
        "neural",
        "network",
        ".",
    ]


def test_lemmatize():
    doc = process_text("The models are running studies.")

    assert lemmatize(doc) == [
        "the",
        "model",
        "be",
        "run",
        "study",
        ".",
    ]


def test_clean_text():
    doc = process_text("The models are running studies.")

    assert clean_text(doc) == [
        "model",
        "run",
        "study",
    ]