import numpy as np
import pytest
from app.services.embeddings.word2vec import Word2VecPipeline

SAMPLE_CORPUS = [
    ["attention", "mechanism", "connects", "encoder", "and", "decoder"],
    ["the", "transformer", "model", "uses", "self", "attention"],
    ["recurrent", "neural", "networks", "process", "sequential", "tokens"],
    ["convolutional", "neural", "networks", "extract", "spatial", "features"],
    ["encoder", "maps", "input", "sequence", "to", "continuous", "representations"],
    ["decoder", "generates", "output", "sequence", "one", "token", "at", "time"],
    ["multi", "head", "attention", "allows", "joint", "attending", "to", "information"],
    ["attention", "is", "all", "you", "need", "for", "machine", "translation"],
    ["bert", "is", "designed", "to", "pretrain", "deep", "bidirectional", "representations"],
    ["bert", "uses", "transformer", "encoder", "architecture"],
]


@pytest.fixture
def trained_cbow_pipeline():
    pipeline = Word2VecPipeline(
        vector_size=32,
        window=3,
        min_count=1,
        sg=0,  # CBOW
        epochs=40,
        seed=42,
    )
    pipeline.train(SAMPLE_CORPUS)
    return pipeline


@pytest.fixture
def trained_skipgram_pipeline():
    pipeline = Word2VecPipeline(
        vector_size=32,
        window=3,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=40,
        seed=42,
    )
    pipeline.train(SAMPLE_CORPUS)
    return pipeline


def test_pipeline_initialization():
    cbow = Word2VecPipeline(sg=0)
    skipgram = Word2VecPipeline(sg=1)
    assert cbow.architecture_name == "CBOW"
    assert skipgram.architecture_name == "Skip-gram"
    assert cbow.vocab_size == 0
    assert not cbow.has_word("attention")


def test_training_and_vocabulary(trained_cbow_pipeline):
    assert trained_cbow_pipeline.vocab_size > 15
    assert trained_cbow_pipeline.has_word("attention")
    assert trained_cbow_pipeline.has_word("transformer")
    assert not trained_cbow_pipeline.has_word("nonexistentword123")


def test_get_vector_shape_and_values(trained_cbow_pipeline):
    vec = trained_cbow_pipeline.get_vector("attention")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (32,)
    assert not np.all(vec == 0)


def test_oov_raises_keyerror(trained_cbow_pipeline):
    with pytest.raises(KeyError, match="Out-Of-Vocabulary"):
        trained_cbow_pipeline.get_vector("quantization")


def test_word_similarity(trained_cbow_pipeline):
    sim = trained_cbow_pipeline.similarity("encoder", "decoder")
    assert -1.0 <= sim <= 1.0


def test_most_similar(trained_cbow_pipeline):
    results = trained_cbow_pipeline.most_similar(positive=["transformer"], topn=3)
    assert len(results) == 3
    assert all(isinstance(word, str) and isinstance(score, float) for word, score in results)


def test_skipgram_training(trained_skipgram_pipeline):
    assert trained_skipgram_pipeline.vocab_size > 15
    vec = trained_skipgram_pipeline.get_vector("bert")
    assert vec.shape == (32,)


def test_sentence_vector_pooling(trained_cbow_pipeline):
    sent_vec = trained_cbow_pipeline.sentence_vector("Transformer uses attention mechanism")
    assert sent_vec.shape == (32,)
    assert not np.all(sent_vec == 0)

    # Empty / entirely OOV sentence returns zeros
    oov_vec = trained_cbow_pipeline.sentence_vector("qubit quantum superposition")
    assert oov_vec.shape == (32,)
    assert np.all(oov_vec == 0)


def test_limitation_word_order_invariance(trained_cbow_pipeline):
    """
    Empirical demonstration of Bag-of-Vectors failure:
    Reversing word order yields an IDENTICAL sentence vector.
    """
    v1 = trained_cbow_pipeline.sentence_vector("encoder connects to decoder")
    v2 = trained_cbow_pipeline.sentence_vector("decoder connects to encoder")

    # Word2Vec mean pooling is order-invariant: cos_sim is exactly 1.0!
    sim = trained_cbow_pipeline.sentence_similarity(
        "encoder connects to decoder",
        "decoder connects to encoder",
    )
    assert pytest.approx(sim, rel=1e-4) == 1.0


def test_model_save_and_load(trained_cbow_pipeline, tmp_path):
    save_file = tmp_path / "test_w2v.model"
    trained_cbow_pipeline.save(save_file)
    assert save_file.exists()

    loaded = Word2VecPipeline.load(save_file)
    assert loaded.vocab_size == trained_cbow_pipeline.vocab_size
    assert loaded.architecture_name == "CBOW"

    vec_original = trained_cbow_pipeline.get_vector("attention")
    vec_loaded = loaded.get_vector("attention")
    np.testing.assert_allclose(vec_original, vec_loaded, rtol=1e-5)
