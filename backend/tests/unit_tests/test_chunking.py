import pytest
from app.models.document import BoundingBox, ContentBlock, Document, PaperMetadata, Section
from app.services.retrieval.chunking import ChunkingConfig, SemanticChunker, estimate_token_count


def make_dummy_block(block_id: str, text: str, page: int = 1, sec_title: str = "Introduction") -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        text=text,
        page_number=page,
        section_title=sec_title,
        bounding_box=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=page),
    )


def make_dummy_doc(blocks: list[ContentBlock], title: str = "Attention Is All You Need") -> Document:
    meta = PaperMetadata(
        title=title,
        authors=["Vaswani et al."],
        year=2017,
        arxiv_id="1706.03762",
        total_pages=max(b.page_number for b in blocks) if blocks else 1,
    )
    return Document(
        metadata=meta,
        content_blocks=blocks,
        sections=[Section(id="s1", title="Introduction", level=1, page_number=1)],
    )


def test_token_estimator():
    assert estimate_token_count("") == 0
    assert estimate_token_count("hello world") == 2
    # 10 words -> ~13 tokens
    words_10 = "one two three four five six seven eight nine ten"
    assert estimate_token_count(words_10) == 13


def test_token_budget_compliance():
    # Create 8 sentences, each around 20 words (~26 tokens) -> total ~208 tokens
    sentences = [
        f"Sentence number {i} provides critical context regarding neural machine translation performance."
        for i in range(1, 9)
    ]
    block = make_dummy_block("b1", " ".join(sentences), page=1)
    doc = make_dummy_doc([block])

    config = ChunkingConfig(max_tokens=80, overlap_tokens=0, prepend_headers=False)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= config.max_tokens


def test_sentence_boundary_integrity():
    text = (
        "First sentence defines self attention mechanisms. "
        "Second sentence explains why recurrent layers are replaced. "
        "Third sentence demonstrates the computational speedup achieved on TPUs."
    )
    block = make_dummy_block("b1", text, page=1)
    doc = make_dummy_doc([block])

    config = ChunkingConfig(max_tokens=40, overlap_tokens=0)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    for c in chunks:
        # Every chunk must end with valid sentence punctuation
        assert c.content_text.strip().endswith((".", "!", "?"))


def test_sliding_window_overlap():
    s1 = "Sentence one discusses foundational transformer architectures."
    s2 = "Sentence two explains the multi head attention projections."
    s3 = "Sentence three evaluates scaled dot product scoring functions."
    s4 = "Sentence four summarizes the empirical results on WMT translation."

    text = f"{s1} {s2} {s3} {s4}"
    block = make_dummy_block("b1", text, page=1)
    doc = make_dummy_doc([block])

    # Tight token budget (~25 tokens per sentence, max_tokens=40, overlap_tokens=20)
    config = ChunkingConfig(max_tokens=40, overlap_tokens=20, prepend_headers=False)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    # Chunk 1 and Chunk 2 should share overlapping text
    overlap_found = any(
        sentence in chunks[0].content_text and sentence in chunks[1].content_text
        for sentence in [s1, s2, s3, s4]
    )
    assert overlap_found, "Consecutive chunks should share overlapping sentence context"


def test_contextual_header_injection():
    text = "The Transformer is the first transduction model relying entirely on self-attention."
    block = make_dummy_block("b1", text, page=2, sec_title="3.1 Model Architecture")
    doc = make_dummy_doc([block], title="Attention Is All You Need")

    config = ChunkingConfig(prepend_headers=True)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    chunk = chunks[0]

    # embed_text must include document title, section title, and page
    assert "[Document: Attention Is All You Need" in chunk.embed_text
    assert "Section: 3.1 Model Architecture" in chunk.embed_text
    assert "Page 2" in chunk.embed_text

    # content_text should remain clean and natural
    assert not chunk.content_text.startswith("[Document:")
    assert chunk.content_text == text


def test_section_isolation():
    b1 = make_dummy_block("b1", "Abstract text explaining main contributions.", page=1, sec_title="Abstract")
    b2 = make_dummy_block("b2", "Introduction text providing background context.", page=1, sec_title="Introduction")
    doc = make_dummy_doc([b1, b2])

    # Even with huge max_tokens, blocks in different sections must NEVER merge
    config = ChunkingConfig(max_tokens=1000)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].section_title == "Abstract"
    assert chunks[1].section_title == "Introduction"
    assert "Abstract text" in chunks[0].content_text
    assert "Introduction text" in chunks[1].content_text


def test_lineage_and_page_tracking():
    b1 = make_dummy_block("b1", "Text on page 3 continuing a section.", page=3, sec_title="Methods")
    b2 = make_dummy_block("b2", "Text on page 4 completing the discussion.", page=4, sec_title="Methods")
    doc = make_dummy_doc([b1, b2])

    config = ChunkingConfig(max_tokens=500)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_block_ids == ["b1", "b2"]
    assert chunk.page_numbers == [3, 4]
    assert chunk.primary_page == 3
    assert "Pages 3-4" in chunk.embed_text


def test_trailing_small_chunk_merged():
    # Long block that splits into 1 main chunk and a tiny 5-token remainder
    main_sentences = " ".join([f"Sentence number {i} discusses deep transformer architecture details." for i in range(1, 6)])
    tiny_remainder = "Final short sentence."
    block = make_dummy_block("b1", f"{main_sentences} {tiny_remainder}", page=1)
    doc = make_dummy_doc([block])

    # Config with max_tokens=60, min_tokens=30. The tiny remainder should merge into the previous chunk.
    config = ChunkingConfig(max_tokens=65, min_tokens=30, overlap_tokens=0)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert "Final short sentence." in chunks[0].content_text


def test_parent_section_text_and_reference_detection():
    b1 = make_dummy_block("b1", "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.", page=10, sec_title="References")
    b2 = make_dummy_block("b2", "[2] Devlin et al. BERT. NAACL 2019.", page=10, sec_title="References")
    doc = make_dummy_doc([b1, b2])

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.is_reference is True
    assert c.chunk_type == "reference"
    assert c.parent_section_text is not None
    assert "Vaswani" in c.parent_section_text and "Devlin" in c.parent_section_text

