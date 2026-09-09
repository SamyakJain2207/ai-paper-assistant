# AI Paper Assistant

An intelligent research paper assistant that parses academic PDFs, extracts structured content, applies classical NLP intelligence, and builds towards modern dense semantic search and LLM-powered Retrieval-Augmented Generation (RAG).

---

## Project Roadmap

```text
Phase 1: PDF to Structured Paper
   │
   ▼
Phase 2: Classical NLP & Text Intelligence (TF-IDF, Keywords, Extractive Summary)
   │
   ▼
Phase 3: Dense Representations & Semantic Search (Embeddings, Vector DB)
   │
   ▼
Phase 4: AI Research Assistant (RAG & Abstractive LLM)
```

---

## Phase Documentation

- **Phase 0 — Document Data Modeling & Architecture**: See [`docs/Phase - 0 - document_structure.md`](docs/Phase%20-%200%20-%20document_structure.md) for the decoupled data architecture (`PaperMetadata`, `Section`, `ContentBlock`, `Document`) and why raw PDF text fails.
- **Phase 1 — Parser & Section Hierarchy**: See [`docs/Phase - 1 - parser.md`](docs/Phase%20-%201%20-%20parser.md) for PDF block extraction, section hierarchy detection, and arXiv/Crossref metadata enrichment.
- **Phase 2 — Classical NLP & Text Intelligence**: See [`docs/Phase - 2 - classical_nlp.md`](docs/Phase%20-%202%20-%20classical_nlp.md) for the from-scratch TF-IDF implementation, keyword extraction, challenges encountered (math formulas, affiliation noise, section TF-IDF paradox), and why modern NLP transitioned to dense embeddings.
- **Phase 3.1 — Word Embeddings (Word2Vec)**: See [`docs/Phase - 3.1 - word2vec.md`](docs/Phase%20-%203.1%20-%20word2vec.md) for the CBOW & Skip-gram models trained on research papers, vector arithmetic, and empirical demonstrations of static embedding limitations (polysemy, order invariance, negation blindness).

---

## Repository Structure

```text
ai-paper-assistant/
├── backend/
│   ├── app/
│   │   ├── models/            # Pydantic data models (Document, Section, ContentBlock, Metadata)
│   │   └── services/
│   │       ├── parser/        # Phase 1: PDF parser and section detector
│   │       └── nlp/           # Phase 2: Preprocessing, TF-IDF, Keywords, Summarizer
│   └── tests/                 # Unit tests and system evaluation scripts
├── data/
│   ├── papers/                # Raw research paper PDFs (Attention, BERT)
│   └── processed/             # Processed artifacts
├── docs/                      # Technical design & phase retrospective notes
└── pyproject.toml             # Project dependencies and pytest configuration
```

---