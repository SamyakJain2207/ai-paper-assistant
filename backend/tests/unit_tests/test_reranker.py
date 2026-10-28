import pytest
from app.models.chunk import Chunk
from app.models.search import HybridSearchResult
from app.services.embeddings.sentence_transformer import SentenceEmbeddingService
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.hybrid_search import HybridSearchEngine
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.vector_db.chroma_service import ChromaService


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the passage (no model download)."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls += 1
        out = []
        for query, passage in pairs:
            q_words = set(query.lower().split())
            p_words = set(passage.lower().split())
            out.append(float(len(q_words & p_words)))
        return out


def make_result(chunk_id: str, text: str, rrf: float) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        content_text=text,
        embed_text=f"[Document: T]\n\n{text}",
        rrf_score=rrf,
        paper_id="p",
        paper_title="T",
    )


def make_chunk(chunk_id: str, text: str, paper_id: str = "p") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content_text=text,
        embed_text=f"[Document: T | Section: S | Page 1]\n\n{text}",
        token_count=len(text.split()),
        char_count=len(text),
        paper_id=paper_id,
        paper_title="T",
        section_title="S",
        page_numbers=[1],
    )


def test_rerank_orders_by_cross_score_not_rrf():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    results = [
        make_result("a", "unrelated words only", rrf=0.033),   # best RRF, worst relevance
        make_result("b", "adam optimizer beta1 beta2 values", rrf=0.020),
    ]
    out = reranker.rerank("adam optimizer beta1", results)

    assert [r.chunk_id for r in out] == ["b", "a"]
    assert out[0].cross_score == 3.0
    assert out[0].final_score == out[0].cross_score
    assert out[1].cross_score == 0.0


def test_rerank_does_not_mutate_inputs_and_respects_top_k():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    results = [make_result(f"c{i}", "word " * i, rrf=0.01) for i in range(1, 5)]
    out = reranker.rerank("word", results, top_k=2)

    assert len(out) == 2
    assert all(r.cross_score is None for r in results)  # originals untouched


def test_rerank_ties_are_deterministic_via_rrf_then_chunk_id():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    results = [
        make_result("z", "same", rrf=0.010),
        make_result("a", "same", rrf=0.010),
        make_result("m", "same", rrf=0.020),
    ]
    out = reranker.rerank("same", results)
    assert [r.chunk_id for r in out] == ["m", "a", "z"]


def test_rerank_edge_cases():
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)
    results = [make_result("a", "text", rrf=0.01)]

    assert reranker.rerank("q", []) == []
    assert reranker.rerank("q", results, top_k=0) == []
    # blank query: keep the RRF order and never call the model
    assert reranker.rerank("   ", results)[0].chunk_id == "a"
    assert model.calls == 0
    with pytest.raises(ValueError):
        CrossEncoderReranker(batch_size=0)


@pytest.fixture
def reranking_engine(tmp_path):
    chroma = ChromaService(persist_dir=tmp_path / "rerank_db", collection_name="rerank_test_col")
    embedder = SentenceEmbeddingService(model_name="all-MiniLM-L6-v2")
    engine = HybridSearchEngine(
        chroma_service=chroma,
        embedder=embedder,
        bm25_index=BM25Index(),
        reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
    )
    yield engine
    chroma.clear_collection()


def test_engine_reranks_and_rerank_false_restores_rrf_order(reranking_engine):
    engine = reranking_engine
    engine.index_chunks([
        make_chunk("c1", "Adam optimizer beta1 0.9 beta2 0.98 epsilon"),
        make_chunk("c2", "dropout rate and label smoothing regularization"),
        make_chunk("c3", "the optimizer used warmup steps"),
    ])

    reranked = engine.search("adam optimizer beta1", top_k=3)
    assert reranked[0].cross_score is not None
    assert reranked[0].final_score == reranked[0].cross_score
    assert [r.final_score for r in reranked] == sorted(
        [r.final_score for r in reranked], reverse=True
    )

    plain = engine.search("adam optimizer beta1", top_k=3, rerank=False)
    assert all(r.cross_score is None for r in plain)
    assert all(r.final_score == r.rrf_score for r in plain)


def test_engine_without_reranker_is_unchanged(tmp_path):
    chroma = ChromaService(persist_dir=tmp_path / "plain_db", collection_name="plain_test_col")
    engine = HybridSearchEngine(chroma, SentenceEmbeddingService("all-MiniLM-L6-v2"))
    engine.index_chunks([make_chunk("c1", "multi head attention projects queries")])
    hits = engine.search("attention queries", top_k=1)  # rerank=True but no reranker configured
    assert hits and hits[0].cross_score is None