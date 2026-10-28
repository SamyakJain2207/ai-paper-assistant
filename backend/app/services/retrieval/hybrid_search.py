from app.models.chunk import Chunk
from app.models.embedding import EmbeddedBlock
from app.models.search import HybridSearchResult, SearchFilter, SearchResult
from app.services.embeddings.sentence_transformer import SentenceEmbeddingService
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.vector_db.chroma_service import ChromaService

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_POOL = 25
DEFAULT_RERANK_POOL = 20


class HybridSearchEngine:
    """
    Hybrid Search Engine fusing Dense Vector Search (ChromaDB) and Sparse Lexical
    Search (BM25Okapi) using Reciprocal Rank Fusion (RRF).

    Key Features:
    - Unified ingestion and deletion across both dense and sparse indices.
    - True 1-based Reciprocal Rank Fusion (RRF) with configurable smoothing (k=60) and weights.
    - Strict metadata filtering applied to BOTH retrievers before ranking.
    - Zero-score filtration on the lexical side (inside BM25Index) so irrelevant
      hits never enter RRF.
    - Scale-invariant fusion: never mixes unbounded BM25 scores with cosine similarities.
    - Deterministic ordering: ties in RRF score are broken by best rank, then chunk_id.
    - Optional second stage: a CrossEncoderReranker re-orders the top fused
      candidates by (query, passage) relevance. RRF decides who enters the pool;
      the cross-encoder score alone decides the final order.

    Design note: BM25Index is the single source of truth for full Chunk objects
    (exposed here as ``chunks_by_id``). ChromaDB persists across sessions while the
    BM25 index is in-memory, so after a restart dense hits fall back to the metadata
    stored in ChromaDB until the chunks are re-indexed.
    """

    def __init__(
        self,
        chroma_service: ChromaService,
        embedder: SentenceEmbeddingService,
        bm25_index: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.chroma_service = chroma_service
        self.embedder = embedder
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker

    @property
    def chunks_by_id(self) -> dict[str, Chunk]:
        """Full Chunk objects currently held by the lexical index (read-only view)."""
        return self.bm25_index.chunks_by_id

    # ------------------------------------------------------------------
    # Ingestion & deletion
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """
        Unified ingestion path:
        1. Embeds contextual embed_text with SentenceEmbeddingService.
        2. Upserts into the ChromaDB persistent collection (grouped by paper).
        3. Adds chunks into BM25Index for lexical retrieval.

        The fallible steps (embedding, ChromaDB upsert) run first, so a failure
        there leaves the in-memory BM25 index untouched.
        """
        if not chunks:
            return 0

        # 1. Compute dense vectors from the contextual embed_text
        vectors = self.embedder.embed_batch([c.embed_text for c in chunks])

        # 2. Build embedded blocks, grouped by paper
        papers_map: dict[str, list[EmbeddedBlock]] = {}
        paper_titles: dict[str, str] = {}
        for chunk, vector in zip(chunks, vectors):
            block = EmbeddedBlock(
                block_id=chunk.chunk_id,
                text=chunk.content_text,
                page_number=chunk.primary_page,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                embedding=vector.tolist(),
                metadata={
                    **chunk.metadata,
                    "paper_title": chunk.paper_title,
                    "paper_id": chunk.paper_id,
                    "page_numbers": chunk.page_numbers,
                    "is_reference": chunk.is_reference,
                    "embed_text": chunk.embed_text,
                    "parent_section_text": chunk.parent_section_text or "",
                },
            )
            papers_map.setdefault(chunk.paper_id, []).append(block)
            paper_titles.setdefault(chunk.paper_id, chunk.paper_title)

        # 3. Upsert into ChromaDB (idempotent by block_id)
        for paper_id, blocks in papers_map.items():
            self.chroma_service.index_embedded_blocks(
                blocks=blocks,
                paper_id=paper_id,
                paper_title=paper_titles[paper_id],
            )

        # 4. Update the lexical index
        self.bm25_index.add_chunks(chunks)

        return len(chunks)

    def delete_paper(self, paper_id: str) -> int:
        """
        Delete a paper from both the dense and sparse indices so no zombie
        chunks remain in either retriever.
        """
        chroma_deleted = self.chroma_service.delete_paper(paper_id)
        bm25_deleted = self.bm25_index.delete_paper(paper_id)
        return max(chroma_deleted, bm25_deleted)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = DEFAULT_RRF_K,
        search_filter: SearchFilter | None = None,
        filter_paper_id: str | None = None,
        rerank: bool = True,
        rerank_pool: int = DEFAULT_RERANK_POOL,
    ) -> list[HybridSearchResult]:
        """
        Execute Hybrid Search combining ChromaDB dense ANN search and BM25 lexical search,
        optionally followed by cross-encoder reranking.

        Uses 1-based Reciprocal Rank Fusion (RRF):
            Score(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))

        Args:
            query_text: Natural-language or keyword query.
            top_k: Number of fused hits to return.
            candidate_pool: Hits requested from EACH retriever before fusion
                (raised to top_k if smaller).
            dense_weight / sparse_weight: Per-retriever RRF weights.
            rrf_k: RRF smoothing constant (industry default 60).
            search_filter: Structured filter (paper, section, page range) applied
                to BOTH retrievers.
            filter_paper_id: Shortcut to scope both retrievers to one paper.
                Takes precedence over ``search_filter.paper_id``.
            rerank: Apply the cross-encoder when one was configured. Ignored
                (plain RRF ordering) if the engine has no reranker.
            rerank_pool: Number of top fused candidates sent to the cross-encoder
                (raised to top_k if smaller).
        """
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("dense_weight and sparse_weight must be non-negative")
        if top_k <= 0 or not query_text or not query_text.strip():
            return []

        pool = max(candidate_pool, top_k)
        effective_filter = self._resolve_filter(search_filter, filter_paper_id)
        effective_paper_id = effective_filter.paper_id if effective_filter else None

        # --- Stage 1: Dense Retrieval (ChromaDB) ---
        dense_hits = self.chroma_service.query_by_text(
            query_text=query_text,
            embedder=self.embedder,
            top_k=pool,
            search_filter=effective_filter,
        )
        # chunk_id -> (1-based rank, SearchResult)
        dense_ranked: dict[str, tuple[int, SearchResult]] = {
            hit.block_id: (rank, hit) for rank, hit in enumerate(dense_hits, start=1)
        }

        # --- Stage 2: Sparse Lexical Retrieval (BM25Okapi) ---
        sparse_hits = self.bm25_index.search(
            query=query_text,
            top_k=pool,
            search_filter=effective_filter,
            filter_paper_id=effective_paper_id,
            filter_stopwords=True,
        )
        # chunk_id -> (1-based rank, bm25 score, Chunk)
        sparse_ranked: dict[str, tuple[int, float, Chunk]] = {
            chunk.chunk_id: (rank, score, chunk) for chunk, score, rank in sparse_hits
        }

        # --- Stage 3: Reciprocal Rank Fusion (RRF) ---
        candidate_ids = set(dense_ranked) | set(sparse_ranked)
        if not candidate_ids:
            return []

        fused: list[tuple[str, float]] = []
        for cid in candidate_ids:
            rrf_score = 0.0
            if cid in dense_ranked:
                rrf_score += dense_weight / (rrf_k + dense_ranked[cid][0])
            if cid in sparse_ranked:
                rrf_score += sparse_weight / (rrf_k + sparse_ranked[cid][0])
            fused.append((cid, rrf_score))

        # Highest RRF first; ties broken by best (lowest) rank in either retriever,
        # then chunk_id, so results are deterministic across runs.
        no_rank = pool + 1
        fused.sort(
            key=lambda item: (
                -item[1],
                min(
                    dense_ranked[item[0]][0] if item[0] in dense_ranked else no_rank,
                    sparse_ranked[item[0]][0] if item[0] in sparse_ranked else no_rank,
                ),
                item[0],
            )
        )

        # --- Stage 4: Build output models for the fused top candidates ---
        use_reranker = rerank and self.reranker is not None
        fetch_k = max(top_k, rerank_pool) if use_reranker else top_k

        results: list[HybridSearchResult] = []
        for cid, rrf_score in fused[:fetch_k]:
            dense_entry = dense_ranked.get(cid)
            sparse_entry = sparse_ranked.get(cid)
            results.append(
                self._build_result(
                    chunk_id=cid,
                    rrf_score=rrf_score,
                    dense_entry=dense_entry,
                    sparse_entry=sparse_entry,
                )
            )

        # --- Stage 5: Cross-encoder reranking (optional) ---
        if use_reranker:
            return self.reranker.rerank(query_text, results, top_k=top_k)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_filter(
        search_filter: SearchFilter | None,
        filter_paper_id: str | None,
    ) -> SearchFilter | None:
        """
        Merge ``filter_paper_id`` into a single SearchFilter so that both
        retrievers receive exactly the same constraints.
        """
        if filter_paper_id is None:
            return search_filter
        if search_filter is None:
            return SearchFilter(paper_id=filter_paper_id)
        return search_filter.model_copy(update={"paper_id": filter_paper_id})

    def _build_result(
        self,
        chunk_id: str,
        rrf_score: float,
        dense_entry: tuple[int, SearchResult] | None,
        sparse_entry: tuple[int, float, Chunk] | None,
    ) -> HybridSearchResult:
        """
        Assemble a HybridSearchResult. Prefers the full Chunk object (from the
        lexical index) and falls back to metadata stored in ChromaDB when the
        chunk is only known to the dense side.
        """
        dense_rank = dense_entry[0] if dense_entry else None
        dense_score = dense_entry[1].score if dense_entry else None
        sparse_rank = sparse_entry[0] if sparse_entry else None
        sparse_score = sparse_entry[1] if sparse_entry else None

        chunk = sparse_entry[2] if sparse_entry else self.chunks_by_id.get(chunk_id)

        if chunk is not None:
            return HybridSearchResult(
                chunk_id=chunk_id,
                content_text=chunk.content_text,
                embed_text=chunk.embed_text,
                rrf_score=round(rrf_score, 6),
                final_score=round(rrf_score, 6),
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                dense_score=dense_score,
                sparse_score=sparse_score,
                paper_id=chunk.paper_id,
                paper_title=chunk.paper_title,
                section_title=chunk.section_title,
                page_numbers=chunk.page_numbers,
                parent_section_text=chunk.parent_section_text,
                is_reference=chunk.is_reference,
                metadata=chunk.metadata,
            )

        # Dense-only fallback (chunk not present in the in-memory index)
        if dense_entry is None:
            raise LookupError(f"Chunk '{chunk_id}' is unknown to both retrievers")
        hit = dense_entry[1]
        return HybridSearchResult(
            chunk_id=chunk_id,
            content_text=hit.text,
            embed_text=str(hit.metadata.get("embed_text", hit.text)),
            rrf_score=round(rrf_score, 6),
            final_score=round(rrf_score, 6),
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            dense_score=dense_score,
            sparse_score=sparse_score,
            paper_id=hit.paper_id,
            paper_title=str(hit.metadata.get("paper_title", hit.paper_id)),
            section_title=hit.section_title,
            page_numbers=[hit.page_number],
            parent_section_text=hit.metadata.get("parent_section_text") or None,
            is_reference=bool(hit.metadata.get("is_reference", False)),
            metadata=hit.metadata,
        )