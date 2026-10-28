from typing import Any

from sentence_transformers import CrossEncoder

from app.models.search import HybridSearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Second-stage reranker powered by a Sentence Transformers CrossEncoder.

    A bi-encoder (dense search) embeds the query and each passage separately, while
    a cross-encoder reads the (query, passage) pair together. That is far more
    accurate but too slow to run over a whole corpus, so it is applied only to the
    small candidate pool produced by hybrid search (BM25 + dense via RRF).

    Design decisions:
    - Final ordering uses the cross-encoder score ALONE. Cross-encoder scores,
      cosine similarities, BM25 scores and RRF scores live on different scales, so
      blending them (e.g. rrf + w * cross) would reintroduce the scale-mixing
      problem that RRF was chosen to avoid. RRF only decides WHO enters the pool.
    - The passage scored is ``content_text`` (clean prose). The header-enriched
      ``embed_text`` is reserved for the embedding and BM25 stages.
    - Input results are never mutated; copies carrying ``cross_score`` and
      ``final_score`` are returned.
    - The model is lazy-loaded on first use, mirroring SentenceEmbeddingService.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        # An already-constructed model (anything exposing ``predict``) can be injected,
        # e.g. for unit tests or to share one model instance across services.
        self._model: Any | None = model

    @property
    def model(self) -> Any:
        """Lazy-load the underlying CrossEncoder neural model."""
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                device=self._device,
                max_length=self.max_length,
            )
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score each passage against the query. Higher means more relevant."""
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int | None = None,
    ) -> list[HybridSearchResult]:
        """
        Reorder hybrid search results by cross-encoder relevance.

        Args:
            query: The original user query.
            results: Candidate pool from HybridSearchEngine.
            top_k: Keep only the best ``top_k`` after reranking (None keeps all).

        Returns:
            New HybridSearchResult objects with ``cross_score`` set and
            ``final_score`` equal to ``cross_score``, sorted best-first. Ties are
            broken by the original RRF score, then chunk_id, for determinism.
        """
        if not results or (top_k is not None and top_k <= 0):
            return []
        if not query or not query.strip():
            return results[:top_k] if top_k is not None else list(results)

        cross_scores = self.score(query, [r.content_text for r in results])

        reranked = [
            r.model_copy(update={"cross_score": cs, "final_score": cs})
            for r, cs in zip(results, cross_scores)
        ]
        reranked.sort(key=lambda r: (-r.cross_score, -r.rrf_score, r.chunk_id))
        return reranked[:top_k] if top_k is not None else reranked