# Phase 4.2 — Hybrid Search Engine (BM25 + Dense Vectors via RRF)

---

## 1. Architectural Scope & Context

In Module 4 (Retrieval Engine), relying exclusively on dense vector search or sparse keyword search introduces critical retrieval failure modes. Phase 4.2 builds a **Hybrid Search Engine** that executes Dense Vector Search (ChromaDB) and Sparse Lexical Search (BM25Okapi) in parallel and unifies their outputs via **Reciprocal Rank Fusion (RRF)**.

```text
                             User Academic Query
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   [Dense Vector Search]                               [Sparse Lexical Search]
    (ChromaDB HNSW / 384-D)                              (BM25Okapi Inverted)
            │                                                   │
   Top-M Candidate Pool                                Top-M Candidate Pool
   (e.g., Top 25 Dense)                                (e.g., Top 25 Sparse)
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                        Reciprocal Rank Fusion (RRF)
                  Score(d) = Σ w / (k + r)  [1-based rank]
                                      │
                                      ▼
                               Top-K Fused Hits
                          (Elevates Mutual Agreement)
```

---

## 2. Why Neither Retriever Can Stand Alone

| Retrieval Challenge | Example Query | Dense Vector (ChromaDB) | Sparse Lexical (BM25) | Hybrid Search (RRF) |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Acronyms / Numbers** | *"WMT 2014 BLEU score 28.4"* | ❌ Drifts to general translation/evaluation paragraphs | ✅ Matches exact tokens immediately (Rank #1) | ✅ Elevated to Rank #1 |
| **Conceptual Paraphrase** | *"How words relate without recurrence"* | ✅ Matches Multi-Head Self-Attention (Rank #1) | ❌ Zero term overlap (Score = 0.0) | ✅ Elevated to Rank #1 |
| **Variable / Code Tokens** | *"hyperparameters d_model and d_k"* | ⚠️ Partial semantic match | ✅ Matches exact identifier syntax | ✅ Mutual Agreement Boost |

---

## 3. Mathematical Formulation: Reciprocal Rank Fusion (RRF)

### Why Raw Score Addition Fails
- **Dense Cosine Similarity**: Strictly bounded in $[-1.0, 1.0]$ (typically $0.3\text{--}0.8$ for relevant hits).
- **BM25 Relevance Score**: Unbounded positive log-odds (typically $2.0\text{--}30.0+$).
- Adding raw scores directly ($S_{\text{raw}} = \text{Score}_{\text{dense}} + \text{Score}_{\text{bm25}}$) is mathematically invalid: the BM25 score would completely dominate the vector score, drowning out dense semantic retrieval.

### The RRF Solution
RRF is a **scale-invariant, rank-based aggregation algorithm** (Cormack et al., SIGIR 2009):
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + r_m(d)}$$

Where:
- $r_m(d) \in \{1, 2, 3, \dots, M\}$ is the **1-based rank** of document $d$ in retriever $m$.
- $k$ is a smoothing parameter (industry standard default $k = 60$) that prevents top ranks from overwhelmingly eclipsing slightly lower ranks.
- $w_m$ is an optional retriever weight (default $w_{\text{dense}} = 1.0, w_{\text{sparse}} = 1.0$).

### Properties of RRF:
1. **Agreement Reward**: If a chunk ranks #1 in Dense and #1 in BM25:
   $$\text{RRF} = \frac{1}{60 + 1} + \frac{1}{60 + 1} = \frac{1}{61} + \frac{1}{61} \approx 0.032786$$
   If a chunk ranks #1 in only one retriever and is absent in the other, its score is $\frac{1}{61} \approx 0.016393$. Mutual agreement receives an enormous relative boost.
2. **Tolerance to Zero Overlap**: If a query is purely conceptual and has zero BM25 lexical match, the chunk still advances through its dense score without penalty.
3. **1-Based Rank Strictness**:
   - Ranks are strictly 1-based ($r \in [1, M]$). Using 0-based ranks ($r \in [0, M-1]$) creates artificial weighting skew between rank #0 and rank #1 and risks division-by-zero if $k=0$.

---

## 4. Architectural & Engineering Decisions Audit

During the design and implementation of Phase 4.2, we evaluated several technical tradeoffs across the dense, sparse, and fusion layers:

```text
┌───────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Hybrid Search Decision Area                   │ Concrete Architectural Implementation                  │
├───────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. Vector Distance Metric                     │ Explicit HNSW Cosine space + similarity calibration    │
│ 2. BM25 Architecture & Scaling                │ In-memory native Python index (<1ms) vs. external daemon│
│ 3. Lexical Tokenization for Papers            │ Custom regex preserving snake_case, hyphens & decimals │
│ 4. Small-Corpus IDF Formulation               │ Non-negative Robertson/Lucene IDF: ln(1 + ...)         │
│ 5. BM25 Zero-Score Filtration                 │ Strict score > 0.0 cutoff before candidate ranking     │
│ 6. Text Indexing Consistency                  │ Both limbs index enriched `embed_text` with headers    │
│ 7. Multi-Store Lifecycle Synchronization      │ Atomic ingestion and deletion via shared chunk IDs     │
│ 8. Metadata Filtering Strategy                │ Pre-ranking filtering applied to both retrievers       │
└───────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

### Decision 1: ChromaDB Distance Metric & Similarity Calibration
- **Issue**: ChromaDB defaults to $L_2$ squared distance (`squared_l2`). Bi-encoders like `all-MiniLM-L6-v2` operate on unit hyperspheres where angular distance (cosine similarity) is the true semantic measure. Furthermore, Chroma returns distances (lower is better), which inverts ranking if treated as similarity.
- **Decision**:
  1. Explicitly configured `metadata={"hnsw:space": "cosine"}` at collection creation in [`ChromaService`](../backend/app/services/vector_db/chroma_service.py).
  2. Calibrated distance to similarity: $\text{sim\_score} = 1.0 - \text{distance}$ so higher is always better.

### Decision 2: In-Memory Native BM25 vs. External Clusters (Elasticsearch / OpenSearch)
- **Issue**: Standard enterprise guides often default to recommending external search clusters (Elasticsearch, OpenSearch, Solr).
- **Decision**: We deliberately implemented a custom in-memory [`BM25Index`](../backend/app/services/retrieval/bm25.py) using Python standard library primitives (`math`, `re`, `collections.Counter`).
- **Rationale**:
  - For a desktop academic paper assistant managing 1 to 100 papers (~500 to 25,000 chunks), our custom in-memory index executes queries in **$0.15\text{--}0.8\text{ms}$** and uses under **$10\text{MB}$** of RAM.
  - Adding an external Lucene cluster introduces JVM overhead, Docker daemon dependencies, network serialization, and multi-gigabyte memory footprints for zero user-perceptible benefit.

### Decision 3: Technical Tokenization for Academic Research Text
- **Issue**: Naive tokenizers (`text.split()` or basic word tokenizers) discard hyphens and underscores, splitting `d_model` into `d` and `model`, or retaining punctuation like `"28.4,"` and `"WMT-14."`.
- **Decision**: Implemented [`tokenize_bm25`](../backend/app/services/retrieval/bm25.py) with regex:
  ```python
  TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*")
  ```
  Preserves `snake_case` variables, hyphenated benchmarks/models, numerical decimals, and alphanumeric codes while stripping surrounding punctuation and normalizing casing.

### Decision 4: Non-Negative Robertson/Lucene IDF on Small Corpora
- **Issue**: Under standard Robertson BM25:
  $$\text{IDF}(q) = \ln \left( \frac{N - n(q) + 0.5}{n(q) + 0.5} \right)$$
  When a term appears in more than half the corpus chunks ($n > N/2$), the numerator is smaller than the denominator, making the log argument $< 1$ and the resulting IDF **negative**. In a single-paper context, queries targeting the paper's core theme (e.g. *"attention"*) receive negative scores and get actively demoted!
- **Decision**: Implemented the Lucene-calibrated non-negative Robertson IDF:
  $$\text{IDF}(q) = \ln \left( 1.0 + \frac{N - n(q) + 0.5}{n(q) + 0.5} \right)$$
  This guarantees $\text{IDF} \ge 0.0$ for all terms regardless of corpus size or frequency.

### Decision 5: Zero-Score Lexical Filtration
- **Issue**: Dense vector search always produces a similarity score for every document in the database. In contrast, BM25 assigns `0.0` to any document lacking lexical overlap. If zero-score documents are retained, arbitrary non-matching documents get ranked #15–#25 and receive artificial RRF fusion points.
- **Decision**: Strictly enforce `if chunk_score > 0.0:` prior to sorting and ranking in [`BM25Index.search`](../backend/app/services/retrieval/bm25.py). Only genuine lexical matches receive a rank and enter RRF.

### Decision 6: Index Text Representation Symmetry
- **Issue**: If ChromaDB indexes contextual text with headers (`embed_text`), but BM25 indexes raw passage text (`content_text`), the two retrievers evaluate different document realities. A query searching for a section name (e.g. *"multi-head attention"*) would be seen by the vector index but invisible to the lexical index.
- **Decision**: Both `ChromaService` and `BM25Index` index the exact same enriched [`embed_text`](../backend/app/services/retrieval/hybrid_search.py) containing hierarchical section anchors.

### Decision 7: Multi-Store Lifecycle Synchronization & Atomic Deletions
- **Issue**: Having separate storage systems risks desynchronization: deleting a paper from ChromaDB while leaving chunks in BM25 leads to "zombie chunks" returning in searches.
- **Decision**: [`HybridSearchEngine`](../backend/app/services/retrieval/hybrid_search.py) acts as the unified facade:
  - Shared `chunk_id`s across both indices.
  - `index_chunks(chunks)` embeds and indexes across both stores atomically.
  - `delete_paper(paper_id)` purges vectors from Chroma and inverted entries from BM25 synchronously.

### Decision 8: Pre-Ranking Metadata Filtering
- **Issue**: Performing post-retrieval filtering (e.g., retrieving Top-25 globally and then filtering for `paper_id == "attention"`) can return zero results if the global top hits belong to another paper in the corpus.
- **Decision**: Pre-filtering is enforced on both retrievers:
  - ChromaDB evaluates metadata predicates using its internal HNSW index (`where={"paper_id": ...}`).
  - BM25 applies `SearchFilter` constraints (paper ID, section title, page range) to prune non-matching chunks *before* scoring and ranking.

---

## 5. Scope Discipline: Intentionally Excluded & Deferred Alternatives

We consciously rejected or deferred the following features during Phase 4.2 to maintain architectural simplicity, local execution speed, and modular isolation:

1. **Distributed Search Clusters (Elasticsearch / OpenSearch)**:
   - *Status*: **Rejected**.
   - *Rationale*: Massive resource overhead (JVM, Docker, memory) with zero performance advantage over native in-memory BM25 for desktop corpora ($\le 100$ papers).
2. **Model Asymmetry Prefixes (`query:` / `passage:`)**:
   - *Status*: **Excluded for current model**.
   - *Rationale*: Our bi-encoder is `all-MiniLM-L6-v2`, which is symmetric and does not expect prompt prefixes. Prefixing is reserved as a modular configuration if asymmetric models (e.g. E5, BGE) are adopted in future phases.
3. **LLM-Based Query Rewriting & Sub-Query Expansion**:
   - *Status*: **Deferred to Module 6 (Specialized Agent Layer)**.
   - *Rationale*: The Retrieval Engine (Module 4) must remain deterministic, ultra-fast ($<50\text{ms}$), and independent of external LLM API latency, costs, and network dependencies. Conversational query decomposition and re-framing belong strictly in the Agent Layer.

---

## 6. Implementation Components Summary

| File | Component | Primary Responsibility |
| :--- | :--- | :--- |
| [`backend/app/models/search.py`](../backend/app/models/search.py) | `HybridSearchResult` | Schema containing content, metadata, dual scores, dual ranks, and fused RRF score |
| [`backend/app/services/retrieval/bm25.py`](../backend/app/services/retrieval/bm25.py) | `BM25Index` | In-memory BM25Okapi, non-negative IDF, technical regex tokenizer, zero-score filter |
| [`backend/app/services/retrieval/hybrid_search.py`](../backend/app/services/retrieval/hybrid_search.py) | `HybridSearchEngine` | Synchronized ingestion/deletion, dual candidate pooling ($M=25$), 1-based RRF fusion |
| [`backend/tests/unit_tests/test_hybrid_search.py`](../backend/tests/unit_tests/test_hybrid_search.py) | Unit Tests | 7 automated unit tests verifying tokenization, RRF math, zero-score filtering, filters, and deletes |
| [`backend/tests/paper_evaluations/evaluate_hybrid_search.py`](../backend/tests/paper_evaluations/evaluate_hybrid_search.py) | Benchmark | Evaluation on real papers (*Attention Is All You Need* & *LoRA*) comparing Dense vs. BM25 vs. Hybrid |

---

## 7. Verification & Tests

### Automated Unit Tests
```powershell
uv run python -m pytest backend/tests/unit_tests/test_hybrid_search.py -v
```
Verifies:
1. `test_bm25_tokenization_preserves_identifiers`: Validates `d_model`, `WMT-14`, `28.4`, and trailing punctuation stripping.
2. `test_bm25_zero_score_filtering`: Verifies non-overlapping queries return zero results without polluting candidate ranks.
3. `test_bm25_non_negative_idf`: Verifies terms in $>50\%$ of chunks retain non-negative IDF scores.
4. `test_hybrid_search_metadata_filtering`: Confirms `SearchFilter(paper_id=...)` isolates targets across both dense and sparse paths.
5. `test_rrf_scoring_accuracy`: Verifies exact 1-based formula: $w_1 / (60 + r_1) + w_2 / (60 + r_2)$.
6. `test_hybrid_search_mutual_agreement_boost`: Confirms chunks ranking high in both retrievers outrank single-retriever hits.
7. `test_hybrid_search_atomic_deletion`: Verifies `delete_paper` removes all traces from both ChromaDB and BM25.

---

## 8. Next Step: Phase 4.3 (Two-Stage Cross-Encoder Re-ranking)

With Hybrid Search retrieving a high-recall candidate pool ($Top\text{-}25$), Phase 4.3 introduces a **Cross-Encoder Re-ranker** (`ms-marco-MiniLM-L-6-v2`) to perform full query-passage cross-attention, eliminating false-positive semantic matches and delivering the verified **Best Context** (top 3–5 chunks) to the downstream LLM.
