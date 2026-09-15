import numpy as np
import pytest

from app.models.block import BlockType, ContentBlock
from app.models.document import Document
from app.models.metadata import PaperMetadata
from app.models.section import Section
from app.services.embeddings.sentence_transformer import SentenceEmbeddingService


@pytest.fixture(scope="module")
def service():
    # Share a single model instance across tests in this module to minimize load time
    return SentenceEmbeddingService(model_name="all-MiniLM-L6-v2")


def test_service_dimension(service):
    assert service.dimension == 384


def test_embed_single_text(service):
    text = "The Transformer architecture relies entirely on an attention mechanism."
    vec = service.embed_text(text)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Verify L2 normalization: norm should be ~1.0
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_batch(service):
    texts = [
        "Self-attention connects all positions of a sequence.",
        "BERT pre-trains deep bidirectional representations.",
        "Recurrent neural networks compute sequentially.",
    ]
    vectors = service.embed_batch(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 384)
    for v in vectors:
        assert pytest.approx(np.linalg.norm(v), rel=1e-3) == 1.0


def test_resolves_word_order_invariance(service):
    """
    Unlike Word2Vec (which produces 1.00000000 and mathematically identical vectors),
    Sentence Transformers use Positional Encodings and Self-Attention,
    producing distinct coordinates for inverted syntax.
    """
    s1 = "model uses attention instead of recurrent layers"
    s2 = "model uses recurrent layers instead of attention"

    v1 = service.embed_text(s1)
    v2 = service.embed_text(s2)
    sim = service.compute_similarity(v1, v2)

    # Word2Vec produced exact 1.00000000 and v1 == v2.
    # SBERT distinguishes the tokens via positional encodings.
    assert sim < 1.0
    assert not np.allclose(v1, v2, atol=1e-3)


def test_resolves_negation_blindness(service):
    """
    Sentence Transformers attend to the negation token 'not',
    meaningfully lowering semantic similarity between opposite claims.
    """
    s_pos = "this transformer architecture is effective"
    s_neg = "this transformer architecture is not effective"

    v_pos = service.embed_text(s_pos)
    v_neg = service.embed_text(s_neg)
    sim = service.compute_similarity(v_pos, v_neg)

    # Word2Vec had ~0.98 similarity; SBERT is noticeably lower (< 0.85)
    assert sim < 0.85


def test_semantic_proximity(service):
    """
    Phrasings with different words but identical meaning should have high similarity.
    """
    s1 = "How do transformers handle relationships between distant words?"
    s2 = "Self-attention mechanisms capture long-range token dependencies."
    s_unrelated = "The weather forecast predicts heavy rain tomorrow."

    v1 = service.embed_text(s1)
    v2 = service.embed_text(s2)
    v_unrelated = service.embed_text(s_unrelated)

    sim_related = service.compute_similarity(v1, v2)
    sim_unrelated = service.compute_similarity(v1, v_unrelated)

    assert sim_related > 0.25
    assert sim_unrelated < 0.15
    assert sim_related > sim_unrelated * 2


def test_embed_document_blocks_and_search(service):
    mock_doc = Document(
        doc_id="test_doc_001",
        metadata=PaperMetadata(title="Attention Is All You Need"),
        sections=[
            Section(section_id="sec_1", title="Introduction", level=1),
            Section(section_id="sec_2", title="Attention Mechanism", level=1),
        ],
        content_blocks=[
            ContentBlock(
                block_id="b1",
                type=BlockType.PARAGRAPH,
                text="The dominant sequence transduction models are based on complex recurrent neural networks.",
                page_number=1,
                section_id="sec_1",
            ),
            ContentBlock(
                block_id="b2",
                type=BlockType.PARAGRAPH,
                text="An attention function can be described as mapping a query and a set of key-value pairs to an output.",
                page_number=2,
                section_id="sec_2",
            ),
        ],
    )

    embedded = service.embed_document_blocks(mock_doc)

    assert len(embedded) == 2
    assert embedded[0].block_id == "b1"
    assert embedded[0].section_title == "Introduction"
    assert embedded[1].section_title == "Attention Mechanism"
    assert embedded[0].vector_dim == 384

    # Perform semantic search query
    results = service.search_blocks(
        query="How are queries keys and values mapped to output?",
        blocks=embedded,
        top_k=1,
    )

    assert len(results) == 1
    top_block, score = results[0]
    # Block 2 (the attention mechanism paragraph) must be retrieved
    assert top_block.block_id == "b2"
    assert score > 0.50
