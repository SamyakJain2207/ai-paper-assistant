# AI Paper Assistant — Master Architecture & Engineering Blueprint

---

## 1. System Vision & Core Philosophy

The **AI Paper Assistant** is an intelligent academic research assistant designed to transform unstructured research paper PDFs into structured, queryable knowledge representations. It mirrors production-grade enterprise Retrieval-Augmented Generation (RAG) and Agentic systems, adhering to strict modular decoupling:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ MASTER END-TO-END PIPELINE                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Module 1: Document Processing                                               │
│   PDF Binary ➔ Clean Layout Structure (Title, Sections, Chunks, Metadata)   │
│                                │                                            │
│                                ▼                                            │
│ Module 2: NLP Pipeline                                                      │
│   Linguistic & Statistical Analysis (Tokens, POS, NER, TF-IDF, Keywords)    │
│                                │                                            │
│                                ▼                                            │
│ Module 3: Embedding Service (Pure Representations — Zero LLM Dependency)   │
│   Continuous Semantic Vectors (Word2Vec, all-MiniLM-L6-v2, ChromaDB HNSW)   │
│                                │                                            │
│                                ▼                                            │
│ Module 4: Production Retrieval Engine                                       │
│   Hybrid Search (Dense + BM25) ➔ Cross-Encoder Reranking ➔ Best Context     │
│                                │                                            │
│                                ▼                                            │
│ Module 5: Knowledge Graph & Interactive Paper Navigator                     │
│   Concept Graph (Nodes/Edges) ➔ Non-Linear Structural Paper Navigation      │
│                                │                                            │
│                                ▼                                            │
│ Specialized Agent Layer                                                     │
│   Multi-Agent System (Explainer, Implementation, Experiment, Comparison)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Specifications

### Module 1 — Document Processing
* **Input**: Raw academic PDF file.
* **Output**: Structured `Document` object:
  ```json
  {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer", ...],
    "sections": [
      {
        "title": "3.2 Multi-Head Attention",
        "level": 2,
        "blocks": [...]
      }
    ],
    "references": [...],
    "figures": [...],
    "tables": [...],
    "equations": [...],
    "text": "..."
  }
  ```
* **Technologies**: PyMuPDF (`fitz`), `pdfplumber`, GROBID (future enhancement).
* **Status**: **Completed (Phase 0 & Phase 1)**.

---

### Module 2 — NLP Pipeline
* **Purpose**: Syntactic, linguistic, and statistical intelligence across extracted text.
* **Pipeline Sequence**:
  $$\text{Cleaning} \longrightarrow \text{Sentence Segmentation} \longrightarrow \text{Tokenization} \longrightarrow \text{Lemmatization} \longrightarrow \text{POS} \longrightarrow \text{NER} \longrightarrow \text{Dependency Parsing} \longrightarrow \text{Keyphrase Extraction}$$
* **Output**: Structured semantic metadata and linguistic features.
* **Status**: **Core statistical NLP completed (Phase 2)**; enriched linguistic annotations integrated as needed.

---

### Module 3 — Embedding Service
* **Core Rule**: *This module does not know anything about LLMs. Only embeddings and metric spaces.*
* **Responsibilities**:
  1. Sentence & passage dense representations.
  2. Chunk embeddings.
  3. Metric similarity calculations (Cosine similarity).
  4. Vector database persistence (HNSW indexing, metadata payloads).
* **Models & Technology**:
  * Word2Vec (CBOW & Skip-gram baseline).
  * HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`, 384-D unit vectors).
  * ChromaDB embedded persistent vector database (`data/chroma_db/`).
* **Status**: **Completed (Phase 3.1, 3.2, 3.3, 3.5)**.

---

### Module 4 — Retrieval Engine (CURRENT ROADMAP)
* **Purpose**: Bridge raw vectors into a production-grade, high-precision retrieval pipeline for RAG.
* **Responsibilities**:
  1. **Production Chunking**: Section-aware hierarchical splitting with token overlap and contextual header prepending.
  2. **Metadata Filtering**: Scoping queries by document ID, section title, or page range.
  3. **Hybrid Search**: Fusing Dense Semantic Search (ChromaDB) + Sparse Lexical Search (BM25) via Reciprocal Rank Fusion (RRF):
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + r_m(d)}$$
  4. **Cross-Encoder Re-ranking**: Two-stage retrieval using `ms-marco-MiniLM-L-6-v2` to eliminate semantic false positives.
  5. **Best Context Assembly**: Token budgeting, provenance tracking, and chronological ordering ready for LLM consumption.
* **Status**: **Next up for implementation**.

---

### Module 5 — Knowledge Graph & Interactive Paper Navigator
* **Purpose**: Enable non-linear conceptual exploration that transcends standard text chunk retrieval.

#### 1. Conceptual Knowledge Graph
Instead of only storing isolated chunks, store relational concept entities:
```text
[Transformer] ──(uses)──► [Self-Attention] ──(introduced in)──► [Attention Is All You Need]
                               │
                               ▼ (improved by)
                            [RoPE] ──(used in)──► [Llama]
```
Enables relational queries: *"Show me all papers and techniques that build upon Scaled Dot-Product Attention."*

#### 2. Interactive Paper Navigator
Transforms the linear PDF viewer into an active semantic research tool:
```text
Introduction ──► Problem ──► Existing Methods ──► Proposed Method ──► Training ──► Experiments ──► Limitations ──► Future Work
```
Clicking each functional node reveals:
- **Explanation**: Plain-language synthesis of the section's core idea.
- **Summary**: Key takeaways and mathematical definitions.
- **Prerequisites**: Background papers or concepts needed to understand this section.

---

### Module 6 — Specialized Multi-Agent Layer
Instead of a generic single-prompt chatbot, implement specialized autonomous agents equipped with dedicated system prompts and toolsets:

1. **Paper Explainer Agent**:
   - Focus: High-level conceptual analogies, intuition building, and academic walkthroughs.
2. **Implementation Agent**:
   - Focus: Turning theoretical equations into step-by-step engineering implementation roadmaps:
     $$\text{Tokenization} \longrightarrow \text{Embedding Layer} \longrightarrow \text{Multi-Head Attention} \longrightarrow \text{Residual Blocks} \longrightarrow \text{Training Loop}$$
3. **Experiment Agent**:
   - Focus: Hardware setup, dataset benchmarks (BLEU, GLUE), ablation study analysis, and hyperparameter schedules.
4. **Code Agent**:
   - Focus: Generating clean, idiomatic PyTorch/JAX code reproducing components from the paper.
5. **Comparison Agent**:
   - Focus: Head-to-head structural, mathematical, and empirical comparisons across multiple papers (e.g., *BERT vs. RoBERTa* or *Transformer vs. Mamba*).

---

## 3. Progress Tracker

| Module / Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 0** | Document Schemas (`ContentBlock`, `Section`, `Document`) | ✅ Completed |
| **Phase 1** | PyMuPDF Layout Parser & Section Hierarchy Detection | ✅ Completed |
| **Phase 2** | Classical NLP (TF-IDF, Tokenization, Extractive Summarizer) | ✅ Completed |
| **Phase 3.1** | Word2Vec CBOW & Skip-gram Baseline Models | ✅ Completed |
| **Phase 3.2** | TF-IDF Weighted Sentence Pooling Baselines | ✅ Completed |
| **Phase 3.3** | Dense Sentence Transformers (`all-MiniLM-L6-v2`, 384-D) | ✅ Completed |
| **Phase 3.5** | ChromaDB Persistent Vector Database & Metadata Filtering | ✅ Completed |
| **Module 4** | Production Retrieval Engine (Hybrid Search + Cross-Encoder Reranker) | ⏳ **Up Next** |
| **Module 5** | Knowledge Graph & Interactive Paper Navigator | 📅 Planned |
| **Module 6** | Specialized Multi-Agent Layer (Explainer, Implementation, etc.) | 📅 Planned |
