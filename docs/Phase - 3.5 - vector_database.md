# Phase 3.5 — Embedded Vector Database (ChromaDB Integration)

---

## 1. Architectural Scope & Context

In the master 5-module system architecture:
- **Module 1**: Document Processing (PDF $\to$ `Document` objects with layout & sections)
- **Module 2**: Classical NLP Pipeline (Preprocessing, tokenization, TF-IDF, keywords)
- **Module 3**: Embedding Service (Word2Vec, Sentence Transformers, pure vector representations)
- **Module 4**: Retrieval Engine (Hybrid search, re-ranking, chunking)
- **Module 5**: Knowledge Graph & Specialized Agents

Phase 3.5 finalizes **Module 3 (Embedding Service)** by introducing **persistent storage, HNSW indexing, and metadata-aware retrieval**—completely independent of any LLM.

---

## 2. Why ChromaDB?

| Requirement | Evaluation | ChromaDB Capability |
| :--- | :--- | :--- |
| **Persistence** | Vectors must survive app restarts | Embedded SQLite storage at `data/chroma_db/` |
| **Index Speed** | Sub-millisecond ANN search | Native Hierarchical Navigable Small World (HNSW) graph |
| **Zero Server Ops**| No external Docker containers on Windows | Pure in-process Python client (`chromadb.PersistentClient`) |
| **Provenance** | Link vectors to exact PDF pages/sections | First-class metadata storage bundled with chunk text |
| **Metric Space** | Cosine distance for unit embeddings | Configured via `metadata={"hnsw:space": "cosine"}` |

---

## 3. Implementation Details

### A. Data Schemas ([`backend/app/models/search.py`](../backend/app/models/search.py))
* **`SearchResult`**: Encapsulates retrieved blocks with text, page number, section heading, paper ID, and calibrated cosine similarity:
  $$\text{Score} = 1.0 - \text{Distance}_{\text{cosine}} \in [-1.0, 1.0]$$
* **`SearchFilter`**: Provides structured, type-safe filtering (`paper_id`, `section_title`, `min_page`, `max_page`) and automatically serializes into ChromaDB's native `$and`, `$eq`, `$gte`, `$lte` boolean query language via `.to_chroma_where()`.

### B. Core Service ([`backend/app/services/vector_db/chroma_service.py`](../backend/app/services/vector_db/chroma_service.py))
* **Metadata Sanitization (`sanitize_chroma_metadata`)**: Automatically flattens nested lists (e.g., `authors: ["Vaswani", "Shazeer"]` $\to$ `"Vaswani, Shazeer"`) and strips `None` values to comply with ChromaDB's primitive-type constraints.
* **Idempotent Upsert (`index_embedded_blocks`)**: Updates existing blocks if re-indexed without creating duplicate vectors.
* **Paper Lifecycle Management (`delete_paper`)**: Executes atomic query-based deletes (`collection.delete(where={"paper_id": paper_id})`) to permanently prevent "zombie chunks" when papers are removed or updated.
* **Library Audit (`list_indexed_papers`)**: Aggregates distinct papers, total block counts, and unique page coverage across the knowledge base.

---

## 4. Empirical Verification & Test Suite

### Automated Unit Tests ([`backend/tests/unit_tests/test_chroma_service.py`](../backend/tests/unit_tests/test_chroma_service.py))
7 passing automated tests verify:
1. `test_initialization_and_empty_state`: Cold-start resilience.
2. `test_index_and_retrieve_blocks`: Accurate nearest-neighbor cosine retrieval.
3. `test_metadata_filtering_paper_id`: Strict document scoping with zero data leakage.
4. `test_structured_search_filter`: Compound range queries on page numbers and sections.
5. `test_upsert_idempotency`: Overwriting existing IDs without vector duplication.
6. `test_delete_paper_lifecycle`: Complete eradication of target paper vectors.
7. `test_persistence_across_restarts`: Verifies SQLite/HNSW disk persistence across client destruction and re-instantiation.

### Full Research Paper Benchmark ([`backend/tests/paper_evaluations/evaluate_vector_db.py`](../backend/tests/paper_evaluations/evaluate_vector_db.py))
* Ingests full *Attention Is All You Need* (149 blocks) and *BERT* (275 blocks) into `data/chroma_db/`.
* Evaluates global retrieval vs. scoped retrieval on ambiguous hardware and pre-training queries.
* Demonstrates query latency under $3\text{ ms}$ on local CPU.
