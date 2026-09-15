# Phase 3.3 — Dense Semantic Embeddings (Sentence Transformers)

---

## 1. The Need for Sentence-BERT (SBERT)

In Phase 3.1 and 3.2, we observed that static word vectors pooled via mean or TF-IDF averaging suffer from:
1. **Word Order Blindness**: $A + B = B + A$ (commutative vector addition cannot differentiate syntax).
2. **Polysemy Loss**: Words have only one static coordinate in vector space.
3. **Negation Blindness**: Adding or removing *"not"* barely shifts the centroid.

While standard BERT (Devlin et al., 2018) provides contextual token representations via multi-head self-attention, finding the most similar chunk in a library of $N$ passages using BERT's cross-encoder architecture requires feeding each query-passage pair through the full transformer network:
$$\mathcal{O}(N) \text{ forward passes}$$
For a modest collection of 10,000 chunks, a single query requires 10,000 forward passes, taking tens of seconds.

**Sentence-BERT (Reimers & Gurevych, 2019)** solves this using a **Siamese / Triplet network structure**:
- It maps each sentence independently to a fixed-size dense vector $\vec{u} \in \mathbb{R}^d$.
- Embeddings are pre-computed and stored in a vector index.
- At query time, only **one** forward pass embeds the query, and semantic retrieval reduces to rapid cosine similarity calculations:
  $$\text{sim}(q, p) = \frac{\vec{u}_q \cdot \vec{u}_p}{\|\vec{u}_q\| \|\vec{u}_p\|}$$

```text
       Siamese Sentence Transformer Architecture
  Sentence A                                Sentence B
      │                                         │
      ▼                                         ▼
┌─────────────┐                           ┌─────────────┐
│    BERT     │ (Shared Weights)          │    BERT     │
└──────┬──────┘                           └──────┬──────┘
       │ [h1, h2, ..., hn]                       │ [h1, h2, ..., hn]
       ▼                                         ▼
┌─────────────┐                           ┌─────────────┐
│Mean Pooling │                           │Mean Pooling │
└──────┬──────┘                           └──────┬──────┘
       │ u                                       │ v
       └───────────────────┬─────────────────────┘
                           │
                           ▼
                    Cosine Similarity /
                  Classification Objective
```

---

## 2. Model Selection: Why `all-MiniLM-L6-v2`?

We selected **`all-MiniLM-L6-v2`** as the core embedding model for the paper assistant.

| Model | Dimensions | Layers | Parameters | Model Size | MTEB Retrieval Avg | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`all-MiniLM-L6-v2` (Chosen)** | **384** | **6** | **22.7M** | **~80 MB** | **41.95** | **Optimal latency/accuracy for local desktop & real-time RAG** |
| `all-mpnet-base-v2` | 768 | 12 | 109M | ~420 MB | 43.81 | Highest accuracy when GPU and memory are unconstrained |
| `bge-small-en-v1.5` | 384 | 12 | 33.4M | ~130 MB | 43.20 | Strong open-source alternative |
| `scibert_scivocab_uncased` | 768 | 12 | 110M | ~440 MB | N/A | Masked LM only; not fine-tuned with Siamese cosine loss |

### Architectural Rationales:
1. **Low Resource Footprint**: 22.7M parameters (~80 MB) allows instant initialization on any standard developer CPU without requiring dedicated GPU VRAM.
2. **Speed**: Embedding throughput exceeds thousands of sentences per second, enabling real-time document parsing during user upload.
3. **Contrastive Pre-training**: Pre-trained on over 1 billion sentence pairs using self-supervised contrastive learning (InfoNCE loss), producing a smooth, well-calibrated metric space for cosine distance.

---

## 3. Architecture & Implementation

### A. Data Schema: `EmbeddedBlock`
Defined in [`backend/app/models/embedding.py`](backend/app/models/embedding.py), `EmbeddedBlock` acts as the bridge connecting Phase 1 document parsing with Phase 3 dense vectors:

```python
class EmbeddedBlock(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: list[float]          # 384-D dense unit vector
    page_number: int
    section_title: str | None
    bounding_box: BoundingBox | None
    metadata: dict[str, Any]        # Title, DOI, Authors, Paper ID
```

### B. Service Layer: `SentenceEmbeddingService`
Defined in [`backend/app/services/embeddings/sentence_transformer.py`](backend/app/services/embeddings/sentence_transformer.py):
- **Lazy Loading**: Defers HuggingFace model weight initialization until the first embedding call, keeping app startup instantaneous.
- **L2 Normalization**: All vectors are normalized to unit length ($\|\vec{v}\| = 1.0$), reducing cosine similarity to a fast dot product $\vec{u} \cdot \vec{v}$.
- **Document Chunk Embedding**: Iterates through structured Phase 1 `Document` blocks, filters layout noise (`min_words=8`), batches text encoding, and attaches rich metadata payloads.

---

## 4. Empirical Benchmark & Comparison with Phase 3.2 Baseline

In [`backend/tests/paper_evaluations/evaluate_sentence_transformers.py`](backend/tests/paper_evaluations/evaluate_sentence_transformers.py), we ingested two milestone research papers:
- **Attention Is All You Need** (Vaswani et al., 2017) -> 149 chunks across 11 pages
- **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018) -> 275 chunks across 16 pages
- **Total Knowledge Base**: 424 chunks

We tested 6 conceptual research queries head-to-head against the **Phase 3.2 Baseline (Word2Vec + TF-IDF Weighted Pooling)**:

| Test Query | Target Paper | Phase 3.3 (Sentence Transformer) | Phase 3.2 Baseline (Word2Vec + TF-IDF) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| *"How does the model compute relationships between words without recurrent networks?"* | Attention Is All You Need | **Top Match: Attention (Page 4)**<br>Section: *Multi-Head Attention*<br>Score: `+0.5283` | **Top Match: BERT (Page 2)**<br>Section: *Introduction*<br>Score: `+0.4102` | **Transformer wins**: Understands semantic paraphrase ("relationships between words" $\to$ Self-Attention). Word2Vec drifts to BERT due to repeated tokens "model", "words". |
| *"What compute hardware was used and how long was the training schedule?"* | Attention Is All You Need | **Top Match: Attention (Page 5)**<br>Section: *Hardware and Schedule*<br>Score: `+0.7302` | **Top Match: Attention (Page 5)**<br>Section: *Hardware and Schedule*<br>Score: `+0.5841` | Both retrieve the correct section, but SBERT produces substantially higher confidence and semantic separation. |
| *"How does the bidirectional language model mask input tokens during pre-training?"* | BERT | **Top Match: BERT (Page 3)**<br>Section: *Task #1: Masked LM*<br>Score: `+0.7420` | **Top Match: BERT (Page 3)**<br>Section: *Task #1: Masked LM*<br>Score: `+0.5110` | SBERT pinpoints the exact 80/10/10 masking rule paragraph. |
| *"What are the differences in layers and hidden size between BERT Base and Large?"* | BERT | **Top Match: BERT (Page 3)**<br>Section: *BERT Model Architecture*<br>Score: `+0.7011` | **Top Match: Attention (Page 4)**<br>Section: *Model Architecture*<br>Score: `+0.4429` | **Word2Vec misattribution**: Word2Vec gets confused by overlapping architectural terms ("layers", "hidden size", "base") and returns Attention instead of BERT. |

---

## 5. Next Step: Phase 3.5 (Vector Database Integration)

With dense embeddings verified, Phase 3.5 integrates **ChromaDB** to:
1. Persist chunk embeddings and inverted metadata indices to disk.
2. Enable sub-millisecond Approximate Nearest Neighbor (ANN) search via HNSW graphs.
3. Support metadata filtering by document ID, section, and page number.
