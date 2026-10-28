import pytest
from app.models.block import BlockType, ContentBlock
from app.models.document import Document
from app.models.metadata import PaperMetadata
from app.models.section import Section
from app.services.retrieval.chunking import (
    ChunkingConfig,
    SemanticChunker,
    estimate_token_count,
)


def make_dummy_block(
    block_id: str,
    text: str,
    page: int = 1,
    sec_id: str = "sec_intro",
) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        type=BlockType.PARAGRAPH,
        text=text,
        page_number=page,
        section_id=sec_id,
        bbox=[0.0, 0.0, 100.0, 100.0],
    )


def make_dummy_doc(
    blocks: list[ContentBlock],
    title: str = "Attention Is All You Need",
    sections: list[Section] | None = None,
) -> Document:
    meta = PaperMetadata(
        title=title,
        authors=["Vaswani et al."],
        arxiv_id="1706.03762",
        total_pages=max(b.page_number for b in blocks) if blocks else 1,
    )
    if sections is None:
        seen_ids: set[str] = set()
        sec_list: list[Section] = []
        for b in blocks:
            if b.section_id and b.section_id not in seen_ids:
                seen_ids.add(b.section_id)
                sec_list.append(Section(section_id=b.section_id, title="Introduction", level=1))
        if not sec_list:
            sec_list = [Section(section_id="sec_intro", title="Introduction", level=1)]
    else:
        sec_list = sections

    return Document(
        doc_id="doc_dummy",
        metadata=meta,
        content_blocks=blocks,
        sections=sec_list,
    )


def test_token_estimator():
    assert estimate_token_count("") == 0
    assert estimate_token_count("hello world") == 2
    words_10 = "one two three four five six seven eight nine ten"
    assert estimate_token_count(words_10) == 13


def test_token_budget_compliance():
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
        assert c.content_text.strip().endswith((".", "!", "?"))


def test_sliding_window_overlap():
    s1 = "Sentence one discusses foundational transformer architectures in deep learning models."
    s2 = "Sentence two explains the multi head attention projections and dimensional mappings."
    s3 = "Sentence three evaluates scaled dot product scoring functions with softmax activations."
    s4 = "Sentence four summarizes the empirical results on WMT translation benchmarks."

    text = f"{s1} {s2} {s3} {s4}"
    block = make_dummy_block("b1", text, page=1)
    doc = make_dummy_doc([block])

    # Each sentence is ~13 tokens; with max_tokens=35, chunk 1 takes s1+s2 (26 tokens).
    # With overlap_tokens=18, s2 (13 tokens) is retained in chunk 2.
    config = ChunkingConfig(max_tokens=35, overlap_tokens=18, prepend_headers=False)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    overlap_found = any(
        sentence in chunks[0].content_text and sentence in chunks[1].content_text
        for sentence in [s1, s2, s3, s4]
    )
    assert overlap_found, "Consecutive chunks should share overlapping sentence context"


def test_contextual_header_injection():
    text = "The Transformer is the first transduction model relying entirely on self-attention."
    block = make_dummy_block("b1", text, page=2, sec_id="sec_arch")
    sections = [Section(section_id="sec_arch", title="3.1 Model Architecture", level=2)]
    doc = make_dummy_doc([block], title="Attention Is All You Need", sections=sections)

    config = ChunkingConfig(prepend_headers=True)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    chunk = chunks[0]

    assert "[Document: Attention Is All You Need" in chunk.embed_text
    assert "Section: 3.1 Model Architecture" in chunk.embed_text
    assert "Page 2" in chunk.embed_text

    assert not chunk.content_text.startswith("[Document:")
    assert chunk.content_text == text


def test_section_isolation():
    b1 = make_dummy_block("b1", "Abstract text explaining main contributions.", page=1, sec_id="sec_abs")
    b2 = make_dummy_block("b2", "Introduction text providing background context.", page=1, sec_id="sec_intro")
    sections = [
        Section(section_id="sec_abs", title="Abstract", level=1),
        Section(section_id="sec_intro", title="Introduction", level=1),
    ]
    doc = make_dummy_doc([b1, b2], sections=sections)

    config = ChunkingConfig(max_tokens=1000)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].section_title == "Abstract"
    assert chunks[1].section_title == "Introduction"
    assert "Abstract text" in chunks[0].content_text
    assert "Introduction text" in chunks[1].content_text


def test_lineage_and_page_tracking():
    b1 = make_dummy_block("b1", "Text on page 3 continuing a section.", page=3, sec_id="sec_methods")
    b2 = make_dummy_block("b2", "Text on page 4 completing the discussion.", page=4, sec_id="sec_methods")
    sections = [Section(section_id="sec_methods", title="Methods", level=1)]
    doc = make_dummy_doc([b1, b2], sections=sections)

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
    # 4 sentences (~10 tokens each) fill chunk 1 when max_tokens=45.
    main_sentences = " ".join([
        f"Sentence number {i} discusses deep transformer architecture details."
        for i in range(1, 5)
    ])
    tiny_remainder = "Final short sentence ends here."
    block = make_dummy_block("b1", f"{main_sentences} {tiny_remainder}", page=1, sec_id="sec_arch")
    sections = [Section(section_id="sec_arch", title="Architecture", level=1)]
    doc = make_dummy_doc([block], sections=sections)

    # With max_tokens=45, the first 4 sentences take ~40 tokens.
    # The remainder (~6 tokens) is < min_tokens=30, so it gets merged into chunk 1.
    config = ChunkingConfig(max_tokens=45, min_tokens=30, overlap_tokens=0)
    chunker = SemanticChunker(config=config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert "Final short sentence ends here." in chunks[0].content_text


def test_parent_section_text_and_reference_detection():
    b1 = make_dummy_block(
        "b1", "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.", page=10, sec_id="sec_ref"
    )
    b2 = make_dummy_block(
        "b2", "[2] Devlin et al. BERT. NAACL 2019.", page=10, sec_id="sec_ref"
    )
    sections = [Section(section_id="sec_ref", title="References", level=1)]
    doc = make_dummy_doc([b1, b2], sections=sections)

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.is_reference is True
    assert c.chunk_type == "reference"
    assert c.parent_section_text is not None
    assert "Vaswani" in c.parent_section_text and "Devlin" in c.parent_section_text
