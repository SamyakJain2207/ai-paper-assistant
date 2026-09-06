import math
import pytest
from app.services.nlp.tfidf import (
    TFIDFModel,
    compare_with_sklearn,
    compute_idf,
    compute_tf,
    compute_tfidf,
)


def test_compute_tf():
    tokens = ["transformer", "model", "transformer", "attention"]
    tf = compute_tf(tokens)

    assert tf["transformer"] == 2 / 4
    assert tf["model"] == 1 / 4
    assert tf["attention"] == 1 / 4
    assert compute_tf([]) == {}


def test_compute_idf_smooth():
    corpus = [
        ["transformer", "attention", "model"],
        ["transformer", "bert", "language"],
        ["transformer", "gpt", "model"],
    ]
    idf = compute_idf(corpus, smooth=True)

    # "transformer" appears in all 3 docs: ln((3 + 1) / (3 + 1)) + 1 = ln(1) + 1 = 1.0
    assert math.isclose(idf["transformer"], 1.0, rel_tol=1e-5)

    # "attention" appears in 1 doc: ln((3 + 1) / (1 + 1)) + 1 = ln(2) + 1 ≈ 1.693147
    expected_rare = math.log(4 / 2) + 1.0
    assert math.isclose(idf["attention"], expected_rare, rel_tol=1e-5)

    # Rarer terms must have strictly higher IDF than common terms
    assert idf["attention"] > idf["transformer"]
    assert idf["bert"] > idf["model"]


def test_compute_tfidf():
    idf = {"transformer": 1.0, "attention": 2.0}
    tokens = ["transformer", "transformer", "attention", "model"]
    # TF: transformer=0.5, attention=0.25, model=0.25
    tfidf = compute_tfidf(tokens, idf)

    assert math.isclose(tfidf["transformer"], 0.5 * 1.0)
    assert math.isclose(tfidf["attention"], 0.25 * 2.0)


def test_tfidf_model():
    corpus = [
        ["deep", "neural", "network"],
        ["convolutional", "neural", "network"],
        ["recurrent", "neural", "network"],
        ["transformer", "attention", "mechanism"],
    ]
    model = TFIDFModel(smooth=True)
    model.fit(corpus)

    assert "neural" in model.vocab
    assert "transformer" in model.vocab
    assert model.num_docs == 4

    scores = model.transform(["neural", "transformer", "transformer"])
    # transformer is rarer than neural across the corpus, so it should have a higher score per token
    assert scores["transformer"] > scores["neural"]


def test_compare_with_sklearn_runs():
    corpus = [
        "attention is all you need",
        "transformers are powerful models",
        "bert is a bidirectional transformer model",
    ]
    res = compare_with_sklearn(corpus)
    # If sklearn is installed, ensure both produced dictionaries
    if "error" not in res:
        assert len(res["scratch"]) == len(corpus)
        assert len(res["sklearn"]) == len(corpus)
