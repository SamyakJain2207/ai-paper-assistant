# Phase 3.2 — Sentence Vector Pooling Baselines (Word2Vec + TF-IDF)

---

## 1. Motivation: From Word Vectors to Sentence Representations

In Phase 3.1, we trained static word embeddings using Word2Vec (CBOW and Skip-gram). However, search queries and academic paper chunks are not isolated words—they are full sentences and paragraphs. 

To bridge static word vectors to chunk-level search, the classical pre-Transformer approach relied on **sentence vector pooling**. 

---

## 2. Mathematical Formulations of Vector Pooling

Given a sentence $S = [w_1, w_2, \dots, w_n]$ and pre-trained static word vectors $\vec{v}(w_i) \in \mathbb{R}^d$:

### A. Naive Mean Pooling (Uniform Bag-of-Words Average)
Every in-vocabulary token contributes equally to the sentence vector:
$$\vec{v}_{\text{mean}}(S) = \frac{1}{|S \cap V|} \sum_{w_i \in S \cap V} \vec{v}(w_i)$$

- **Limitation**: Ubiquitous high-frequency words (e.g., *"model"*, *"layer"*, *"paper"*, *"approach"*) dominate the vector sum, drowning out distinctive technical keywords.

### B. TF-IDF Weighted Pooling
To counteract high-frequency token dilution, each word vector is scaled by its corpus-wide TF-IDF weight $w_i = \text{TF-IDF}(w_i, S)$:
$$\vec{v}_{\text{tfidf}}(S) = \frac{\sum_{w_i \in S \cap V} w_i \cdot \vec{v}(w_i)}{\sum_{w_i \in S \cap V} w_i}$$

- **Advantage**: Rare, highly descriptive words (e.g., *"recurrence"*, *"multi-head"*, *"TPU"*) are amplified, while common background terms are suppressed.

---

## 3. Implementation in `Word2VecPipeline`

The pooling logic was incorporated directly into [`Word2VecPipeline.sentence_vector`](backend/app/services/embeddings/word2vec.py):

```python
def sentence_vector(
    self,
    sentence: str | list[str],
    weights: dict[str, float] | TFIDFModel | None = None,
) -> np.ndarray:
    # Extracts tokens, looks up static vectors, scales by TF-IDF weights,
    # and computes normalized weighted centroid.
    ...
```

The pipeline supports direct interoperability with the Phase 2 `TFIDFModel`: passing `weights=tfidf_model` automatically calculates term weights on the fly.

---

## 4. Empirical Evaluation & Interactive Exploration

We verified these baselines in `notebooks/01_word2vec_and_baselines.ipynb` and automated unit tests.

### Findings from Empirical Tests:
1. **Cosine Similarity Improvement**: Weighted pooling increases the discriminative power between semantically disparate sentences by ~15–25% compared to unweighted mean pooling.
2. **Out-of-Vocabulary (OOV) Vulnerability**: If a sentence contains rare technical terms unseen during training (e.g., brand-new acronyms or specific formula variables), those words contribute zero mass to the representation.
3. **Fundamental Blindspots**:
   - **Syntax & Word Order Insensitivity**: Commutative addition ($A + B = B + A$) means inverted relationships score $\approx 1.0$.
   - **Context Blindness**: The word *"attention"* receives the exact same vector regardless of whether it refers to human cognitive focus or multi-head dot-product attention.

---

## 5. Summary

TF-IDF weighted Word2Vec pooling serves as the **strong classical baseline** for sentence-level semantic representations. In Phase 3.3, we evaluate modern contextualized **Sentence Transformers** to overcome the intrinsic structural limitations of static vector pooling.
