from typing import Any

from app.models.chunk import Chunk
from app.models.embedding import EmbeddedBlock
from app.models.search import HybridSearchResult, SearchFilter
from app.services.embeddings.sentence_transformer import SentenceEmbeddingService
from app.services.retrieval.bm25 import BM25Index
from app.services.vector_db.chroma_service import ChromaService


class HybridSearchEngine:
    """
    Production Hybrid Search Engine fusing Dense Vector Search (ChromaDB)
    and Sparse Lexical Search (BM25Okapi) using Reciprocal Rank Fusion (RRF).

    Key Features:
    - Unified atomic ingestion and deletion across both dense and sparse indices.
    - True 1-based Reciprocal Rank Fusion (RRF) with configurable smoothing (k=60) and weights.
    - Strict metadata filtering applied to BOTH retrievers before ranking.
    - Zero-score filtration on the lexical side to prevent irrelevant hits from entering RRF.
    - Scale-invariant fusion: never mixes unbounded BM25 scores with cosine similarities.
    """

    def __init__(
        self,
        chroma_service: ChromaService,
        embedder: SentenceEmbeddingService,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.chroma_service = chroma_service
        self.embedder = embedder
        self.bm25_index = bm25_index or BM25Index()
        # In-memory registry of indexed chunks by chunk_id for full object retrieval
        self.chunks_by_id: dict[str, Chunk] = dict(self.bm25_index.chunks_by_id)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """
        Unified ingestion path:
        1. Embeds contextual embed_text with SentenceEmbeddingService.
        2. Upserts into ChromaDB persistent collection.
        3. Adds chunks into BM25Index for lexical retrieval.
        """
        if not chunks:
            return 0

        # Update in-memory registry
        for c in chunks:
            self.chunks_by_id[c.chunk_id] = c

        # 1. Update BM25 index
        self.bm25_index.add_chunks(chunks)

        # 2. Compute embeddings for ChromaDB
        embed_texts = [c.embed_text for c in chunks]
        vectors = self.embedder.embed_batch(embed_texts)

        embedded_blocks: list[EmbeddedBlock] = []
        for i, c in enumerate(chunks):
            block = EmbeddedBlock(
                block_id=c.chunk_id,
                text=c.content_text,
                page_number=c.primary_page,
                section_id=c.section_id,
                section_title=c.section_title,
                embedding=vectors[i].tolist(),
                metadata={
                    **c.metadata,
                    "paper_title": c.paper_title,
                    "paper_id": c.paper_id,
                    "page_numbers": c.page_numbers,
                    "is_reference": c.is_reference,
                    "embed_text": c.embed_text,
                    "parent_section_text": c.parent_section_text or "",
                },
            )
            embedded_blocks.append(block)

        # Group by paper_id and upsert into ChromaDB
        papers_map: dict[str, list[EmbeddedBlock]] = {}
        for eb in embedded_blocks:
            pid = eb.metadata.get("paper_id", "unknown")
            papers_map.setdefault(pid, []).append(eb)

        for pid, p_blocks in papers_map.items():
            first_title = p_blocks[0].metadata.get("paper_title")
            self.chroma_service.index_embedded_blocks(
                blocks=p_blocks,
                paper_id=pid,
                paper_title=first_title,
            )

        return len(chunks)

    def delete_paper(self, paper_id: str) -> int:
        """
        Atomic deletion across both dense and sparse indices.
        Permanently eliminates zombie chunks.
        """
        # 1. Delete from ChromaDB
        chroma_deleted = self.chroma_service.delete_paper(paper_id)

        # 2. Delete from BM25 index
        bm25_deleted = self.bm25_index.delete_paper(paper_id)

        # 3. Clean registry
        to_del = [cid for cid, c in self.chunks_by_id.items() if c.paper_id == paper_id]
        for cid in to_del:
            del self.chunks_by_id[cid]

        return max(chroma_deleted, bm25_deleted)

    

class HybridSearchEngine:
@@
-    def __init__(
-        self,
-        chroma_service: ChromaService,
-        embedder: SentenceEmbeddingService,
-        bm25_index: BM25Index | None = None,
-    ) -> None:
-        self.chroma_service = chroma_service
-        self.embedder = embedder
-        self.bm25_index = bm25_index or BM25Index()
-        # In-memory registry of indexed chunks by chunk_id for full object retrieval
-        self.chunks_by_id: dict[str, Chunk] = dict(self.bm25_index.chunks_by_id)
+    def __init__(
+        self,
+        chroma_service: ChromaService,
+        embedder: SentenceEmbeddingService,
+        bm25_index: BM25Index | None = None,
+    ) -> None:
+        self.chroma_service = chroma_service
+        self.embedder = embedder
+        self.bm25_index = bm25_index or BM25Index()
+        # In‑memory registry of indexed chunks by chunk_id for full object retrieval
+        self.chunks_by_id: dict[str, Chunk] = dict(self.bm25_index.chunks_by_id)
+
+        # ------------------------------------------------------------
+        # Load configuration for hybrid search (BM25 params, RRF, cross‑encoder)
+        # ------------------------------------------------------------
+        default_cfg = {
+            "bm25": {"k1": 1.5, "b": 0.75},
+            "rrf": {"k": 60, "dense_weight": 1.0, "sparse_weight": 1.0},
+            "cross_encoder": {
+                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
+                "weight": 0.0,  # set >0 to blend cross‑encoder score
+                "candidate_pool": 25,
+            },
+        }
+        cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "hybrid_search.json")
+        if os.path.exists(cfg_path):
+            try:
+                with open(cfg_path, "r", encoding="utf-8") as f:
+                    user_cfg = json.load(f)
+                # shallow merge (user values override defaults)
+                for top_key, sub_cfg in user_cfg.items():
+                    if top_key in default_cfg and isinstance(sub_cfg, dict):
+                        default_cfg[top_key].update(sub_cfg)
+                    else:
+                        default_cfg[top_key] = sub_cfg
+            except Exception as e:
+                # If config is malformed we fall back to defaults; log for debugging
+                print(f"[HybridSearchEngine] Failed to load config {cfg_path}: {e}")
+        self.cfg = default_cfg
+
+        # Apply BM25 hyper‑parameters if they differ from defaults
+        bm25_cfg = self.cfg.get("bm25", {})
+        if "k1" in bm25_cfg or "b" in bm25_cfg:
+            self.bm25_index.k1 = bm25_cfg.get("k1", self.bm25_index.k1)
+            self.bm25_index.b = bm25_cfg.get("b", self.bm25_index.b)
+
+        # Instantiate cross‑encoder service based on config
+        cross_cfg = self.cfg.get("cross_encoder", {})
+        self.cross_encoder = CrossEncoderService(cross_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
+        self.cross_weight = cross_cfg.get("weight", 0.0)
+        self.cross_candidate_pool = cross_cfg.get("candidate_pool", 25)
@@
-    def search(
-        self,
-        query_text: str,
-        top_k: int = 5,
-        candidate_pool: int = 25,
-        dense_weight: float = 1.0,
-        sparse_weight: float = 1.0,
-        rrf_k: int = 60,
-        search_filter: SearchFilter | None = None,
-        filter_paper_id: str | None = None,
-    ) -> list[HybridSearchResult]:
-        """
-        Execute Hybrid Search combining ChromaDB Dense ANN search and BM25 Lexical search.
-        Uses 1-based Reciprocal Rank Fusion (RRF):
-            Score(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
-        """
+    def search(
+        self,
+        query_text: str,
+        top_k: int = 5,
+        candidate_pool: int | None = None,
+        dense_weight: float | None = None,
+        sparse_weight: float | None = None,
+        rrf_k: int | None = None,
+        search_filter: SearchFilter | None = None,
+        filter_paper_id: str | None = None,
+    ) -> list[HybridSearchResult]:
+        """Execute hybrid search with optional cross‑encoder re‑ranking.
+
+        Parameters can be overridden per call; if ``None`` the values from the
+        configuration file are used.
+        """
+        # Resolve configuration defaults when arguments are omitted
+        candidate_pool = candidate_pool if candidate_pool is not None else self.cfg["cross_encoder"].get("candidate_pool", 25)
+        dense_weight = dense_weight if dense_weight is not None else self.cfg["rrf"].get("dense_weight", 1.0)
+        sparse_weight = sparse_weight if sparse_weight is not None else self.cfg["rrf"].get("sparse_weight", 1.0)
+        rrf_k = rrf_k if rrf_k is not None else self.cfg["rrf"].get("k", 60)
*** End Patch ***        effective_paper_id = filter_paper_id or (search_filter.paper_id if search_filter else None)

        # --- Stage 1: Dense Retrieval (ChromaDB) ---
        dense_hits = self.chroma_service.query_by_text(
            query_text=query_text,
            embedder=self.embedder,
            top_k=candidate_pool,
            search_filter=search_filter,
            where={"paper_id": effective_paper_id} if effective_paper_id and not search_filter else None,
        )

        # Map dense hits by chunk_id: (rank_1_based, score, SearchResult)
        dense_ranked: dict[str, tuple[int, float, Any]] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            dense_ranked[hit.block_id] = (rank, hit.score, hit)

        # --- Stage 2: Sparse Lexical Retrieval (BM25Okapi) ---
        sparse_hits = self.bm25_index.search(
            query=query_text,
            top_k=candidate_pool,
            search_filter=search_filter,
            filter_paper_id=effective_paper_id,
            filter_stopwords=True,
        )

        # Map sparse hits by chunk_id: (rank_1_based, score, Chunk)
        sparse_ranked: dict[str, tuple[int, float, Chunk]] = {}
        for chunk, score, rank in sparse_hits:
            sparse_ranked[chunk.chunk_id] = (rank, score, chunk)

        # --- Stage 3: Reciprocal Rank Fusion (RRF) ---
        all_candidate_ids = set(dense_ranked.keys()) | set(sparse_ranked.keys())
        if not all_candidate_ids:
            return []

        fusion_candidates: list[dict[str, Any]] = []

        for cid in all_candidate_ids:
            rrf_score = 0.0
            dense_rank = None
            dense_score = None
            sparse_rank = None
            sparse_score = None

            # Add Dense contribution (1-based rank)
            if cid in dense_ranked:
                d_rank, d_score, _ = dense_ranked[cid]
                dense_rank = d_rank
                dense_score = d_score
                rrf_score += dense_weight / (rrf_k + d_rank)

            # Add Sparse contribution (1-based rank)
            if cid in sparse_ranked:
                s_rank, s_score, _ = sparse_ranked[cid]
                sparse_rank = s_rank
                sparse_score = s_score
                rrf_score += sparse_weight / (rrf_k + s_rank)

            # Resolve chunk metadata
            chunk_obj = self.chunks_by_id.get(cid)
            if chunk_obj is not None:
                content_text = chunk_obj.content_text
                embed_text = chunk_obj.embed_text
                paper_id = chunk_obj.paper_id
                paper_title = chunk_obj.paper_title
                section_title = chunk_obj.section_title
                page_numbers = chunk_obj.page_numbers
                parent_section = chunk_obj.parent_section_text
                is_ref = chunk_obj.is_reference
                meta = chunk_obj.metadata
            elif cid in dense_ranked:
                _, _, d_hit = dense_ranked[cid]
                content_text = d_hit.text
                embed_text = d_hit.metadata.get("embed_text", d_hit.text)
                paper_id = d_hit.paper_id
                paper_title = d_hit.metadata.get("paper_title", paper_id)
                section_title = d_hit.section_title
                page_numbers = [d_hit.page_number]
                parent_section = d_hit.metadata.get("parent_section_text")
                is_ref = bool(d_hit.metadata.get("is_reference", False))
                meta = d_hit.metadata
            else:
                continue

            fusion_candidates.append({
                "chunk_id": cid,
                "content_text": content_text,
                "embed_text": embed_text,
                "rrf_score": round(rrf_score, 6),
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section_title": section_title,
                "page_numbers": page_numbers,
                "parent_section_text": parent_section,
                "is_reference": is_ref,
                "metadata": meta,
            })

        # --- Stage 4: Cross‑Encoder Re‑ranking (optional) ---
        if self.cross_weight > 0.0:
            # Prepare texts for the cross‑encoder: combine query with passage
            cross_texts = []
            candidate_ids = []
            for cand in fusion_candidates:
                # The cross‑encoder expects a single string "[query] [SEP] [passage]"
                # Use a space separator; tokenizer will handle max_length internally.
                cross_texts.append(f"{query_text} [SEP] {cand['content_text']}")
                candidate_ids.append(cand['chunk_id'])
            # Run in batches (default batch size 16) to respect CPU memory
            cross_scores = self.cross_encoder.score_batch(cross_texts)
            # Assign scores back to the corresponding candidates
            for cid, c_score in zip(candidate_ids, cross_scores):
                for cand in fusion_candidates:
                    if cand['chunk_id'] == cid:
                        cand['cross_score'] = float(c_score)
                        # Combine with RRF using configured weight
                        cand['final_score'] = cand['rrf_score'] + self.cross_weight * cand['cross_score']
                        break
        else:
            for cand in fusion_candidates:
                cand['cross_score'] = None
                cand['final_score'] = cand['rrf_score']

        fusion_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Build output models
        results: list[HybridSearchResult] = []
        for cand in fusion_candidates[:top_k]:
            results.append(HybridSearchResult(**cand))

        return results
