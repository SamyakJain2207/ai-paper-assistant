from app.models.block import BlockType, ContentBlock
from app.models.document import Document
from app.models.metadata import PaperMetadata
from app.models.section import Section
from app.services.nlp.keywords import (
    extract_document_keywords,
    extract_section_keywords,
)
from app.services.nlp.sentence_scoring import (
    score_sentences,
    split_sentences,
)
from app.services.nlp.summarizer import generate_extractive_summary
from app.services.nlp.tfidf import TFIDFModel


def create_mock_document() -> Document:
    sections = [
        Section(section_id="sec_1", title="Introduction", level=1),
        Section(section_id="sec_2", title="Model Architecture", level=1),
        Section(section_id="sec_3", title="Conclusion", level=1),
    ]
    blocks = [
        ContentBlock(
            block_id="b1",
            type=BlockType.PARAGRAPH,
            text="Recurrent neural networks have long been the dominant sequence model. Attention mechanisms allow modeling dependencies regardless of distance.",
            page_number=1,
            section_id="sec_1",
        ),
        ContentBlock(
            block_id="b2",
            type=BlockType.PARAGRAPH,
            text="The Transformer relies entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs. Multi-head attention allows the model to jointly attend to information.",
            page_number=2,
            section_id="sec_2",
        ),
        ContentBlock(
            block_id="b3",
            type=BlockType.PARAGRAPH,
            text="We presented the Transformer, the first sequence transduction model based entirely on attention. We plan to extend the Transformer to problems involving input modalities other than text.",
            page_number=3,
            section_id="sec_3",
        ),
    ]
    return Document(
        doc_id="mock_doc_123",
        metadata=PaperMetadata(title="Attention Is All You Need"),
        sections=sections,
        content_blocks=blocks,
    )


def test_split_sentences():
    text = "The Transformer is fast. It uses self-attention. It achieves state-of-the-art results!"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The Transformer is fast."


def test_score_sentences():
    corpus = [
        ["transformer", "self", "attention"],
        ["recurrent", "network", "sequence"],
    ]
    model = TFIDFModel(smooth=True).fit(corpus)
    sents = [
        "The Transformer uses self-attention mechanism.",
        "A weather report discusses heavy rain.",
    ]
    scored = score_sentences(sents, model)
    assert len(scored) == 2
    # The first sentence with domain words should score significantly higher than unrelated weather words
    assert scored[0].score > scored[1].score


def test_extract_section_keywords():
    doc = create_mock_document()
    sec_keywords = extract_section_keywords(doc, top_k=3)

    assert "Introduction" in sec_keywords
    assert "Model Architecture" in sec_keywords
    # Verify keywords are returned as list of tuples (word, score)
    intro_words = [kw[0] for kw in sec_keywords["Introduction"]]
    assert len(intro_words) > 0


def test_extract_document_keywords():
    doc = create_mock_document()
    doc_keywords = extract_document_keywords(doc, top_k=5)

    assert len(doc_keywords) > 0
    words = [kw[0] for kw in doc_keywords]
    # "attention" or "transformer" should be among the top document keywords
    assert any(term in words for term in ["attention", "transformer", "model"])


def test_generate_extractive_summary():
    doc = create_mock_document()
    summary = generate_extractive_summary(doc, num_sentences=2)

    assert isinstance(summary, str)
    assert len(summary) > 0
    # Selected sentences should include key concepts
    assert "Transformer" in summary or "attention" in summary.lower()
