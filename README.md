# AI Paper Assistant

An intelligent research paper assistant that parses academic PDFs, extracts structured content, applies classical NLP intelligence, and builds towards modern dense semantic search and LLM-powered Retrieval-Augmented Generation (RAG).

---

## Project Architecture & Roadmap

See [`docs/MASTER_ARCHITECTURE_PLAN.md`](docs/MASTER_ARCHITECTURE_PLAN.md) for the comprehensive technical blueprint covering the 5 modules, the interactive paper navigator, and the specialized multi-agent layer.

```text
Module 1: Document Processing [COMPLETED]
   │  └── PDF ➔ PyMuPDF ➔ Document Object (Title, Sections, Blocks, Metadata)
   ▼
Module 2: Classical NLP Pipeline [COMPLETED]
   │  └── Cleaning ➔ Sentence Split ➔ Tokenize ➔ TF-IDF ➔ Keywords
   ▼
Module 3: Embedding Service (Zero LLM Dependency) [COMPLETED]
   │  └── Word2Vec Baseline ➔ all-MiniLM-L6-v2 ➔ ChromaDB Persistence
   ▼
Module 4: Production Retrieval Engine [UP NEXT]
   │  └── Production Chunking ➔ Hybrid Search (Dense + BM25) ➔ Cross-Encoder Reranker
   ▼
Module 5: Knowledge Graph & Interactive Paper Navigator [PLANNED]
   │  └── Relational Entity Triples ➔ Non-Linear Semantic Paper Structure
   ▼
Module 6: Specialized Multi-Agent Layer [PLANNED]
      └── Explainer, Implementation, Experiment, Code, and Comparison Agents
```

---

## Phase Documentation

- **Master Plan**: See [`docs/MASTER_ARCHITECTURE_PLAN.md`](docs/MASTER_ARCHITECTURE_PLAN.md) for the end-to-end engineering roadmap.
- **Phase 0 — Document Data Modeling & Architecture**: See [`docs/Phase - 0 - document_structure.md`](docs/Phase%20-%200%20-%20document_structure.md) for the decoupled data architecture (`PaperMetadata`, `Section`, `ContentBlock`, `Document`) and why raw PDF text fails.
- **Phase 1 — Parser & Section Hierarchy**: See [`docs/Phase - 1 - parser.md`](docs/Phase%20-%201%20-%20parser.md) for PDF block extraction, section hierarchy detection, and arXiv/Crossref metadata enrichment.
- **Phase 2 — Classical NLP & Text Intelligence**: See [`docs/Phase - 2 - classical_nlp.md`](docs/Phase%20-%202%20-%20classical_nlp.md) for the from-scratch TF-IDF implementation, keyword extraction, challenges encountered (math formulas, affiliation noise, section TF-IDF paradox), and why modern NLP transitioned to dense embeddings.
- **Phase 3.1 — Word Embeddings (Word2Vec)**: See [`docs/Phase - 3.1 - word2vec.md`](docs/Phase%20-%203.1%20-%20word2vec.md) for the CBOW & Skip-gram models trained on research papers, vector arithmetic, and empirical demonstrations of static embedding limitations (polysemy, order invariance, negation blindness).
- **Phase 3.2 — Sentence Vector Pooling Baselines**: See [`docs/Phase - 3.2 - sentence_pooling_baselines.md`](docs/Phase%20-%203.2%20-%20sentence_pooling_baselines.md) for TF-IDF weighted Word2Vec sentence pooling, mathematical formulations, and interactive exploratory notebooks.
- **Phase 3.3 — Dense Semantic Embeddings (Sentence Transformers)**: See [`docs/Phase - 3.3 - sentence_transformers.md`](docs/Phase%20-%203.3%20-%20sentence_transformers.md) for the Siamese bi-encoder architecture (`all-MiniLM-L6-v2`), `EmbeddedBlock` schema, and head-to-head empirical benchmark across full research papers.
- **Phase 3.5 — Embedded Vector Database (ChromaDB)**: See [`docs/Phase - 3.5 - vector_database.md`](docs/Phase%20-%203.5%20-%20vector_database.md) for persistent storage at `data/chroma_db/`, HNSW indexing with cosine distance metric, idempotent upsert pipeline, metadata-scoped queries, and paper lifecycle management.
- **Phase 4.1 — Production Semantic Chunking**: See [`docs/Phase - 4.1 - production_chunking.md`](docs/Phase%20-%204.1%20-%20production_chunking.md) for section-aware sliding-window chunking (300 max tokens, 50 overlap), sentence-boundary preservation, and contextual header injection (`embed_text` vs `content_text`).

---

## Repository Structure

```text
ai-paper-assistant/
├── backend/
│   ├── app/
│   │   ├── models/            # Pydantic data models (Document, Section, ContentBlock, Metadata)
│   │   └── services/
│   │       ├── parser/        # Phase 1: PDF parser and section detector
│   │       ├── nlp/           # Phase 2: Preprocessing, TF-IDF, Keywords, Summarizer
│   │       └── embeddings/    # Phase 3: Word2Vec & Sentence Transformers
│   └── tests/
│       ├── unit_tests/        # Automated pytest suites (parsers, NLP, embeddings)
│       └── paper_evaluations/ # End-to-end evaluation scripts on real research papers
├── data/
│   ├── papers/                # Raw research paper PDFs (Attention, BERT)
│   └── processed/             # Processed artifacts & trained models
├── docs/                      # Technical design & phase retrospective notes
├── notebooks/                 # Exploratory research notebooks & interactive demonstrations
└── pyproject.toml             # Project dependencies and pytest configuration
```

---