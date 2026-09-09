# Phase 3.1 — Word2Vec: Mechanics, Exploration & The Ceiling of Static Word Embeddings

---

## 1. The Bridge from Phase 2 (TF-IDF) to Phase 3 (Dense Embeddings)

In Phase 2, we extracted keywords and ranked sentences using **TF-IDF**. While mathematically elegant, TF-IDF represents text in a **sparse, orthogonal vocabulary space**:

- The vocabulary size $|V|$ is large ($5,000$ to $50,000+$ words).
- Every word is an independent basis vector:
  $$\vec{w}_{\text{attention}} \cdot \vec{w}_{\text{recurrence}} = 0$$
- If a user searches for *"memory constraints in deep models"*, but the paper writes *"computational bottleneck in recurrent architectures"*, TF-IDF registers **zero overlap** because the exact string tokens differ.

To solve this, NLP moved to **Dense Continuous Representations** (Word Embeddings):
Mapping every word $w \in V$ to a dense, low-dimensional continuous vector $\vec{v} \in \mathbb{R}^d$ (typically $d \in [50, 300]$).

### The Distributional Hypothesis
The mathematical foundation of all embedding algorithms is J.R. Firth's famous 1957 linguistic insight:
> *"You shall know a word by the company it keeps."*

Words that appear in similar contextual windows across a corpus share similar semantic, syntactic, and functional meanings.

---

## 2. Word2Vec Mechanics: CBOW vs. Skip-gram

Introduced by Mikolov et al. at Google in 2013, **Word2Vec** is a shallow two-layer neural network designed to learn word vectors by solving a fake auxiliary prediction task.

```text
               CBOW                                 Skip-gram
   (Continuous Bag-of-Words)

    Context Words:                         Target Word:
    [w(t-2), w(t-1), w(t+1), w(t+2)]                  [w(t)]
                 │                                      │
                 ▼                                      ▼
           ┌───────────┐                          ┌───────────┐
           │ Projection│                          │ Projection│
           │  (Average)│                          │  (Lookup) │
           └─────┬─────┘                          └─────┬─────┘
                 │                                      │
                 ▼                                      ▼
           Predicted Word:                        Predicted Context:
               [w(t)]                      [w(t-2), w(t-1), w(t+1), w(t+2)]
```

### 1. Continuous Bag-of-Words (CBOW, `sg=0`)
- **Task**: Predict the target center word $w_t$ given the surrounding context words:
  $$P(w_t \mid w_{t-c}, \dots, w_{t+c})$$
- **Mechanism**: Projects all context words into the hidden layer and averages their vectors.
- **Strengths**:
  - Trains much faster (one update per context window).
  - Better statistical smoothing for frequent words.
- **Weakness**: Averages away fine-grained nuances of rare words.

### 2. Continuous Skip-gram (`sg=1`)
- **Task**: Predict the surrounding context words given the target center word:
  $$\sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t)$$
- **Strengths**:
  - Does not average the context.
  - Treats each (target, context) pair independently, making it far superior at learning rich representations for **rare technical terms** and specialized vocabulary.
- **Weakness**: Computationally slower than CBOW.

### Optimization: Negative Sampling (SGNS)
Computing the exact softmax over a vocabulary of $50,000$ words requires calculating the partition function $\sum_{w' \in V} \exp(\vec{v}_{w'}^\top \vec{u})$ on every step, which is prohibitively expensive.
Word2Vec solves this using **Negative Sampling**: turning the multi-class prediction into binary logistic regression, distinguishing the true context word from $k$ randomly drawn "noise" (negative) words:

$$\mathcal{L}_{\text{SGNS}} = \log \sigma(\vec{v}_{w_O}^\top \vec{u}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-\vec{v}_{w_i}^\top \vec{u}_{w_I}) \right]$$

---

## 3. Geometric Properties: Vector Arithmetic & Cosine Similarity

Because word co-occurrence statistics are mapped into continuous linear space, semantic relationships manifest as **geometric offset vectors**:

$$\vec{v}(\text{"king"}) - \vec{v}(\text{"man"}) + \vec{v}(\text{"woman"}) \approx \vec{v}(\text{"queen"})$$

In scientific literature, linear direction vectors often encode functional roles:
$$\vec{v}(\text{"decoder"}) - \vec{v}(\text{"output"}) + \vec{v}(\text{"input"}) \approx \vec{v}(\text{"encoder"})$$

### Cosine Similarity Metric
To measure how close two words or representations are, we compute the cosine of the angle between their unit vectors:

$$\text{CosineSimilarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

- $+1.0$: Exactly identical direction (maximum semantic alignment)
- $0.0$: Orthogonal (completely unrelated)
- $-1.0$: Diametrically opposite direction

---

## 4. The 5 Fundamental Limitations of Static Word Embeddings

While Word2Vec was a historic paradigm shift, it possesses severe structural flaws that prevent it from being used for modern document search and research paper understanding:

### Limitation 1: Polysemy & Single-Vector Constraint (Context Blindness)
- In Word2Vec, each unique string token has exactly **one static vector** in the embedding matrix $W \in \mathbb{R}^{|V| \times d}$.
- Polysemous words (words with multiple meanings) are forced to occupy a single point:
  - *"head"* in *"multi-head attention"* (neural network component) vs. *"he shook his head"* (human body part).
  - *"bank"* in *"river bank"* vs. *"investment bank"*.
  - *"transformer"* in electrical grid hardware vs. *"transformer"* sequence architecture.
- The resulting vector is a muddy weighted compromise of all senses, heavily biased toward whatever sense was most frequent in the training corpus.
- **Crucial flaw**: Word2Vec cannot dynamically modify a word's representation based on the sentence it appears in.

### Limitation 2: The Out-Of-Vocabulary (OOV) Wall
- Word2Vec operates on a strictly closed vocabulary $V$ learned during training.
- If a user queries a term not seen during training (e.g., new models like *"RoBERTa"*, domain chemical names, or simple typos like *"attension"*), Word2Vec **cannot represent it at all**.
- It either throws a `KeyError` or returns a meaningless generic `<UNK>` vector.

### Limitation 3: Subword & Morphological Blindness
- Word2Vec treats words as atomic, indivisible tokens.
- It shares zero parameters across morphological variants:
  `"train"`, `"training"`, `"pretrained"`, and `"retrainable"` are four completely independent entries.
- If `"pretrain"` is in the vocabulary but `"pretraining"` appeared only once and was filtered out by `min_count`, the model has zero knowledge that they share a root!
- *(FastText later partially mitigated this with character n-grams, and Modern Transformers fully resolved it with Byte-Pair Encoding / WordPiece subword tokenization).*

### Limitation 4: The Bag-of-Vectors Failure (Sentence & Passage Representation)
Word2Vec embeds **words**, but research paper search requires comparing **sentences, paragraphs, and chunks**.
The standard heuristic to create a sentence vector from word vectors is **Average Pooling**:

$$\vec{v}_{\text{sentence}} = \frac{1}{N} \sum_{i=1}^N \vec{v}(w_i)$$

This creates two catastrophic failures:
1. **Word Order Invariance**:
   - $\vec{v}(\text{"model uses attention instead of recurrent layers"})$
   - $\vec{v}(\text{"model uses recurrent layers instead of attention"})$
   Because addition is commutative ($\vec{a} + \vec{b} = \vec{b} + \vec{a}$), both sentences produce **the exact same vector** ($\text{cos\_sim} = 1.000000$), despite expressing diametrically opposite technical realities!
2. **Negation Blindness**:
   - $\vec{v}(\text{"this architecture is effective"})$
   - $\vec{v}(\text{"this architecture is not effective"})$
   Adding the vector for *"not"* simply shifts the composite vector by a fraction of a percent. The cosine similarity remains $\approx 0.95+$, failing to reflect the negated truth value.
3. **Semantic Dilution / Hubness**:
   - Averaging 40 words in a technical paragraph pulls the vector towards the centroid of the vector space, washing out specific claims into generic background noise.

### Limitation 5: Data Hunger & Small-Corpus Instability
- Word2Vec requires millions to billions of tokens to learn reliable geometric relationships.
- When trained on small domain corpora (like a few research papers), rare words co-occur with only a few other words by chance, creating noisy, distorted vector coordinates.

---

## 5. Empirical Verification & Experiments

We implemented and verified Word2Vec across four dedicated scripts in `backend/`:
- [`backend/app/services/embeddings/word2vec.py`](backend/app/services/embeddings/word2vec.py): Core `Word2VecPipeline` (CBOW & Skip-gram).
- [`backend/tests/test_word2vec.py`](backend/tests/test_word2vec.py): Automated unit test suite verifying training, OOV exceptions, vector arithmetic, and save/load roundtrips.
- [`backend/tests/evaluate_word2vec.py`](backend/tests/evaluate_word2vec.py): End-to-end evaluation pipeline trained on *Attention Is All You Need* and *BERT*.
- [`backend/tests/demo_context_and_order.py`](backend/tests/demo_context_and_order.py): Empirical demonstration of polysemy, word order inversion, and negation blindness.

### Experimental Findings on Academic Papers

| Test Dimension | Observed Result | Takeaway |
| :--- | :--- | :--- |
| **Corpus Extraction** | 708 clean prose sentences, ~18,000 tokens | Clean sentence splitting filters out formulas and layout noise. |
| **Vocabulary Size** | 1,489 unique words ($\ge 2$ occurrences) | Technical domain vocabulary learned in dense 100-D space. |
| **Semantic Similarity** | `encoder` $\leftrightarrow$ `decoder`: $+0.61$ | Related functional concepts cluster closely in vector space. |
| **Polysemy ("bank")** | $\vec{v}(\text{"bank"})_{\text{river}} \equiv \vec{v}(\text{"bank"})_{\text{finance}}$ | Static lookup table cannot adapt to sentence context. |
| **Order Inversion** | *"A sent to B"* vs *"B sent to A"*: $\text{sim} = \mathbf{1.000000}$ | Commutative vector addition completely destroys sequence order. |
| **Negation Blindness** | *"effective"* vs *"not effective"*: $\text{sim} = \mathbf{0.94+}$ | Polarity inversion is impossible with naive vector pooling. |

---

## 6. The Evolution: From Word2Vec to Sentence Transformers

The timeline of semantic NLP highlights how each architecture addressed the previous limitation:

```text
2013: Word2Vec (Mikolov et al.)
      ├── Static lookup table
      └── Fixed vector per word (Polysemy & OOV issues)
        │
2014: GloVe (Pennington et al.)
      └── Matrix factorization on global co-occurrence counts
        │
2016: FastText (Bojanowski et al.)
      └── Subword character n-grams (fixes OOV & morphology)
        │
2018: ELMo (Peters et al.)
      └── Bi-directional LSTM yielding contextualized word representations
        │
2018: BERT (Devlin et al.)
      └── Multi-head self-attention: token representations conditioned on entire bidirectional context
        │
2019: Sentence-BERT / SBERT (Reimers & Gurevych)
      └── Siamese & Triplet networks producing fixed-dimension, dense sentence embeddings optimized for cosine similarity
```

In **Step 3.2 & 3.3**, we move from static word averages to **Sentence Transformers**, enabling genuine dense semantic retrieval of research paper chunks!

