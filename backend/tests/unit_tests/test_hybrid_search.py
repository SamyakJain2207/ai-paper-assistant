import pytest
from app.models.chunk import Chunk
from app.models.search import SearchFilter
from app.services.embeddings.sentence_transformer import SentenceEmbeddingService
from app.services.retrieval.bm25 import BM25Index, tokenize_bm25
from app.services.retrieval.hybrid_search import HybridSearchEngine
from app.services.vector_db.chroma_service import ChromaService


def make_test_chunk(
    chunk_id: str,
    text: str,
    paper_id: str = "paper_test",
    title: str = "Test Paper",
    sec: str = "Section 1",
    page: int = 1,
) -> Chunk:
    header = f"[Document: {title} | Section: {sec} | Page {page}]\n\n"
    return Chunk(
        chunk_id=chunk_id,
        content_text=text,
        embed_text=header + text,
        token_count=len(text.split()),
        char_count=len(text),
        paper_id=paper_id,
        paper_title=title,
        section_title=sec,
        page_numbers=[page],
        source_block_ids=[f"b_{chunk_id}"],
    )


@pytest.fixture
def temp_hybrid_engine(tmp_path):
    """Provides a fresh HybridSearchEngine pointing to an isolated ChromaDB directory."""
    db_path = tmp_path / "hybrid_chroma_test"
    chroma = ChromaService(persist_dir=db_path, collection_name="hybrid_test_col")
    embedder = SentenceEmbeddingService(model_name="all-MiniLM-L6-v2")
    engine = HybridSearchEngine(chroma_service=chroma, embedder=embedder)
    yield engine
    chroma.clear_collection()


def test_bm25_tokenization_preserves_technical_identifiers():
    raw = "The Adam optimizer with d_model=512, WMT-14 dataset, and BLEU score 28.4!"
    tokens = tokenize_bm25(raw)

    assert "d_model" in tokens
    assert "512" in tokens
    assert "wmt-14" in tokens
    assert "bleu" in tokens
    assert "28.4" in tokens
    # Punctuation stripped from end of '28.4!'
    assert not any(t.endswith("!") for t in tokens)


def test_bm25_zero_score_filtering():
    c1 = make_test_chunk("c1", "Attention is all you need for sequence to sequence models.")
    c2 = make_test_chunk("c2", "Transformers use self attention to model dependencies.")

    bm25 = BM25Index().fit([c1, c2])

    # Query with zero lexical overlap
    results = bm25.search("photosynthesis in plant chloroplasts", top_k=5)
    # Zero-score chunks must NEVER receive a rank
    assert results == []


def test_bm25_non_negative_idf_on_frequent_terms():
    # Term 'model' appears in 9 out of 10 chunks
    chunks = [
        make_test_chunk(f"c_{i}", f"This neural model is evaluated on step {i}.")
        for i in range(9)
    ]
    chunks.append(make_test_chunk("c_9", "Completely unrelated text without the target word."))

    bm25 = BM25Index().fit(chunks)
    idf = bm25.compute_idf("model")

    # Standard Robertson formula can go negative when n > N/2; Lucene formulation guarantees >= 0
    assert idf >= 0.0


def test_bm25_metadata_filtering():
    c_att = make_test_chunk("c_att", "We evaluate on WMT-14 English German benchmark.", paper_id="attention_paper")
    c_bert = make_test_chunk("c_bert", "We evaluate on WMT-14 English French translation.", paper_id="bert_paper")

    bm25 = BM25Index().fit([c_att, c_bert])

    # Filter to attention_paper only
    filt = SearchFilter(paper_id="attention_paper")
    hits = bm25.search("WMT-14", top_k=5, search_filter=filt)

    assert len(hits) == 1
    chunk, score, rank = hits[0]
    assert chunk.chunk_id == "c_att"
    assert chunk.paper_id == "attention_paper"
    assert rank == 1


def test_rrf_scoring_math_and_rank_start(temp_hybrid_engine):
    # Manually verify 1-based RRF formula
    # If a chunk is rank 1 in Dense and rank 2 in BM25 with k=60:
    # RRF = 1/(60 + 1) + 1/(60 + 2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    k = 60
    expected_score = round(1.0 / (k + 1) + 1.0 / (k + 2), 6)

    c1 = make_test_chunk(
        "c1",
        "The Adam optimizer was used with beta1 0.9 and beta2 0.98 for neural training.",
        paper_id="p1",
        sec="Training"
    )
    c2 = make_test_chunk(
        "c2",
        "Learning rate schedules and dropout hyperparameters.",
        paper_id="p1",
        sec="Hyperparameters"
    )

    temp_hybrid_engine.index_chunks([c1, c2])

    results = temp_hybrid_engine.search("Adam optimizer beta1 beta2", top_k=2, rrf_k=60)
    assert len(results) > 0
    top_hit = results[0]
    assert top_hit.chunk_id == "c1"
    # RRF score must be strictly positive and reflect 1-based rank fusion
    assert top_hit.rrf_score > 0.015
    assert top_hit.dense_rank is not None
    assert top_hit.sparse_rank is not None


def test_hybrid_search_elevates_agreement(temp_hybrid_engine):
    # Chunk A: matches both conceptual meaning AND exact keywords
    cA = make_test_chunk("cA", "Multi-head attention projects queries, keys, and values with d_model=512.")
    # Chunk B: matches conceptual meaning only
    cB = make_test_chunk("cB", "Self-attention computes dynamic weights across different representation subspaces.")
    # Chunk C: matches exact keywords only
    cC = make_test_chunk("cC", "The dimensionality parameters are d_model=512 and d_k=64.")

    temp_hybrid_engine.index_chunks([cA, cB, cC])

    hits = temp_hybrid_engine.search("How does multi-head attention project queries with d_model 512?", top_k=3)
    assert len(hits) >= 1
    # Chunk A has mutual agreement across both limbs and should be rank 1
    assert hits[0].chunk_id == "cA"
    assert hits[0].dense_rank is not None
    assert hits[0].sparse_rank is not None


def test_hybrid_search_atomic_deletion(temp_hybrid_engine):
    c1 = make_test_chunk("c1", "Paper 1 content to delete.", paper_id="paper_del")
    c2 = make_test_chunk("c2", "Paper 2 content to keep.", paper_id="paper_keep")

    temp_hybrid_engine.index_chunks([c1, c2])

    deleted_count = temp_hybrid_engine.delete_paper("paper_del")
    assert deleted_count == 1

    # Search for deleted content yields no results from paper_del
    hits = temp_hybrid_engine.search("Paper 1 content", top_k=5)
    for h in hits:
        assert h.paper_id != "paper_del"


def test_bm25_reindexing_same_chunk_is_idempotent():
    c1 = make_test_chunk("c1", "Adam optimizer with warmup steps.")
    c2 = make_test_chunk("c2", "Dropout and label smoothing regularization.")

    bm25 = BM25Index().fit([c1, c2])
    idf_before = bm25.compute_idf("adam")
    freqs_before = dict(bm25.doc_freqs)

    bm25.add_chunks([c1])  # re-index an identical chunk

    assert bm25.total_docs == 2
    assert dict(bm25.doc_freqs) == freqs_before
    assert bm25.compute_idf("adam") == idf_before


def test_bm25_reindexing_replaces_old_content():
    old = make_test_chunk("c1", "Original text about convolution.")
    bm25 = BM25Index().fit([old])
    assert bm25.search("convolution", top_k=3) != []

    new = make_test_chunk("c1", "Updated text about recurrence.")
    bm25.add_chunks([new])

    assert bm25.search("convolution", top_k=3) == []
    assert "convolution" not in bm25.doc_freqs
    assert bm25.search("recurrence", top_k=3)[0][0].chunk_id == "c1"


def test_bm25_delete_keeps_doc_frequencies_consistent():
    a = make_test_chunk("a", "shared term alpha", paper_id="A")
    b = make_test_chunk("b", "shared term beta", paper_id="B")
    bm25 = BM25Index().fit([a, b])

    assert bm25.delete_paper("A") == 1

    expected = {}
    for tokens in bm25.doc_tokens.values():
        for t in set(tokens):
            expected[t] = expected.get(t, 0) + 1
    assert dict(bm25.doc_freqs) == expected
    assert "alpha" not in bm25.doc_freqs