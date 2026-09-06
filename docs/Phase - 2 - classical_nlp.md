# Phase 2 — Classical NLP & Text Intelligence: Development, Challenges & Lessons

---

## 1. Phase 0 Foundation: Designing the Structured Document Representation

### Why Raw Text Extraction Fails
A PDF is not a word-processor document; it is a visual coordinate layout format designed for printing. It contains no native concept of "paragraphs", "sections", "subsections", or "footnotes". 

If raw text is dumped directly into an NLP pipeline:
- Section headings get blended into paragraph text.
- Header and footer watermarks (*e.g., "arXiv:1706.03762v7"*) interrupt sentences.
- Math equations and table values fragment into isolated, meaningless words.
- Multi-column reading orders get interleaved.

To build meaningful text intelligence, we first had to impose a structured semantic data model over the raw PDF stream.

### The Document Data Architecture (`backend/app/models/`)

We designed four decoupled, strongly-typed models:

1. **`PaperMetadata`**:
   - Stores bibliographic identity: `title`, `authors`, `publication_date`, `doi`, and `total_pages`.
   - Uses a local-first heuristic fallback on Page 1 bold spans and enriches metadata via arXiv and Crossref APIs with a $\ge 0.70$ Jaccard title similarity threshold.
2. **`Section`**:
   - Stores hierarchical section trees: `section_id`, `title`, `level` (1 for main sections, 2 for subsections, 3 for sub-subsections), and `parent_id`.
   - Enables the system to understand paper organization (e.g. *Section 3.2 is a child of Section 3*).
3. **`ContentBlock` & `BlockType`**:
   - Classifies every discrete text block into `BlockType.HEADING` or `BlockType.PARAGRAPH`.
   - Tags each block with `page_number`, `bbox` coordinates, and `section_id`.
4. **`Document`**:
   - The master object binding metadata, sections, and content blocks into a cohesive, traversable paper representation.

---

## 2. Phase 2: Classical NLP Vision & Initial Architecture

### The Core Objective
> **Extract useful textual intelligence from the structured document without an LLM.**

Our pipeline was architected in six progressive stages:

```text
               PHASE 1
                  │
                  ▼
        Structured Research Paper
                  │
                  ▼
        ┌─────────────────────┐
        │  Text Preprocessing │  (De-hyphenation, citation & URL removal)
        └──────────┬──────────┘
                   │
         Clean Lemmatized Text   (spaCy linguistic pipeline)
                   │
                   ▼
             ┌──────────┐
             │  TF-IDF  │         (From-scratch mathematical model)
             └────┬─────┘
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
    Keywords   Sentence   Term/document
               importance representation
                  │
                  ▼
          Extractive Summary
```

### Initial Implementation: Mathematical Foundations
We implemented TF-IDF from scratch to understand its mechanics before relying on external libraries:

- **Term Frequency (TF)**: Normalized by document length to prevent long paragraphs from dominating:
  $$\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}$$

- **Inverse Document Frequency (IDF)**: Used smooth IDF to prevent division by zero and negative values:
  $$\text{IDF}(t, D) = \ln\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1$$

- **TF-IDF**:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
  *(Out-of-vocabulary terms receive a score of $0.0$)*.

We validated our implementation against `scikit-learn`'s `TfidfVectorizer` to guarantee mathematical parity.

---

## 3. Real-World Difficulties Discovered During Testing

When we executed the pipeline on real research papers (*Attention Is All You Need* and *BERT*), the initial implementation revealed several critical edge-case failures:

### Challenge 1: The "Title Section" and Author Affiliation Trap
* **Symptom**: Section 1 keywords for *Attention Is All You Need* were extracted as:
  `google (0.420), brain (0.231), research (0.194), aidan@cs.toronto.edu (0.087)`
* **Root Cause**: The section detector classified the title heading on Page 1 as the first section. The block immediately following contained author emails and institutional affiliations (*Google Brain, Google Research, University of Toronto*). Because that block was tagged as a body paragraph, TF-IDF treated author affiliations as the defining keywords of that section!
* **Resolution**:
  - Filtered out author blocks, email addresses (`@`), and institutional affiliation patterns on Page 1.
  - Excluded sections whose title matches the paper title from being treated as body sections.
  - Filtered metadata terms (`arxiv`, `preprint`, `doi`, `http`) from keyword extraction.

### Challenge 2: The Math Equation & Formula Fragmentation Trap
* **Symptom**: The first extractive summary test produced a single, garbled formula string:
  `MultiHead( Q, K, V ) i , KW K i , z i ∈ R d , such as a hidden lrate = d − 0 . 5 [ 30 ].`
* **Root Cause**:
  1. Standalone mathematical equations were extracted by the PDF parser as `BlockType.PARAGRAPH`.
  2. Math expressions fragmented into isolated symbols and variables ($Q, K, V, W, z, d$).
  3. Because these variables appeared almost exclusively in that single block, their IDF was abnormally high.
  4. With standard length normalization ($|S|^{0.8}$), 3-word formula fragments achieved higher scores than genuine 25-word English sentences!
  5. The diversity filter selected one formula fragment from each section, producing a string of stitched math pieces.
* **Resolution (`is_valid_prose_sentence`)**:
  - **Verb requirement**: A real English sentence must contain at least one finite or auxiliary verb (`token.pos_ in ("VERB", "AUX")`). Formula lines contain no verbs and are rejected.
  - **Alphabetic ratio**: Enforced that $\ge 65\%$ of characters must be letters, rejecting symbol-heavy formulas ($\in, \le, =, \sum$).
  - **Formula symbol ban**: Explicitly excluded mathematical notation characters (`∈`, `≤`, `≥`, `∑`, `d_model`).
  - **Smoothing**: Adjusted length normalization to $\frac{\sum \text{TF-IDF}}{\text{token\_count} + 2}$ to prevent short fragments from outscoring complete sentences.

### Challenge 3: Acknowledgements Leaking into the Conclusion
* **Symptom**: The conclusion summary included:
  `Making generation less sequential is another research goals of ours. Acknowledgements We are grateful to Nal Kalchbrenner...`
* **Root Cause**: In the PDF layout, "Acknowledgements" followed Section 6 without a detected heading, so PyMuPDF lumped it into the conclusion section's text.
* **Resolution**: Added `is_acknowledgement_sentence` to detect and filter out acknowledgement markers (*"acknowledgements"*, *"grateful to"*, *"fruitful comments"*, *"supported by"*).

### Challenge 4: The Core Paradox of Section-Level TF-IDF for Summarization
* **Symptom**: Summaries picked isolated implementation facts (*"Each layer has two sub-layers. Training took 3."*) rather than explaining what the paper was actually about.
* **Root Cause**:
  - The central concepts of a paper (*"model"*, *"attention"*, *"transformer"*, *"sequence"*) appear in **almost every section**.
  - Because they appear everywhere, their local Inverse Document Frequency ($\text{IDF}$) drops!
  - Meanwhile, narrow implementation parameters (*"sub-layer"*, *"projections"*, *"training took"*) appear in only one subsection, giving them artificially high local IDF scores!
  - Consequently, naive section-level TF-IDF actively biases **against** the core thesis of the paper and **favors** isolated implementation minutiae.
* **Resolution**:
  - Transitioned from section-level TF-IDF to **Paper-Wide Keyword Salience**: scoring sentences by the density of the paper's top 25 global domain keywords.
  - Incorporated **Edmundson's Cue Method**: prioritizing sentences with indicative markers (*"we propose"*, *"we present"*, *"in this work"*, *"outperforms"*, *"state-of-the-art"*).
  - Restricted candidate sentence pools for whole-paper summaries to synthesis sections: **Abstract**, **Introduction**, and **Conclusion**.

---

## 4. The Glass Ceiling of Classical NLP: Why We Must Move to Phase 3

After applying all these principled refinements, our extractive summary produced:

> *"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs... Recurrent models typically factor computation along the symbol positions of the input and output sequences. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism... In this work, we presented the Transformer, the first sequence transduction model based entirely on attention..."*

### The Honest Evaluation
For an algorithm running purely on word frequencies with zero neural weights, this is a remarkable technical feat: out of 400+ candidate sentences, it identified the RNN bottleneck, the Transformer proposal, the self-attention mechanism, and the SOTA translation benchmark.

**However, as a user-facing product, it hits a hard ceiling:**

1. **Extractive vs. Abstractive Summarization**:
   - Classical NLP can only **copy-paste verbatim sentences** (extractive).
   - It cannot rephrase, condense, synthesize, or explain ideas in fresh prose (abstractive).
   - What users expect when they read a "summary" is an abstractive synthesis.
2. **Redundancy without Complex Heuristics**:
   - Notice that Sentence 4 (*"In this work we propose the Transformer..."*) and Sentence 5 (*"In this work, we presented the Transformer..."*) repeat the same core point. 
   - Classical methods require heuristics like Maximal Marginal Relevance (MMR) to manually prune word overlap.
3. **Disjointed Narrative Flow**:
   - Verbatim sentences plucked from different sections lack connective tissue (*"However,"*, *"To solve this,"*, *"In contrast"*). The output reads like disconnected bullet points forced into a paragraph.
4. **Keyword Blindness (No Semantic Understanding)**:
   - TF-IDF treats words as independent atomic symbols. It has no awareness that:
     - `"attention mechanism"`, `"self-attention"`, and `"soft alignment"` are conceptually related.
     - `"RNN"`, `"LSTM"`, and `"recurrent network"` refer to the same paradigm.
   - If a user searches for *"memory constraints in language models"*, TF-IDF will fail if the paper uses the phrase *"computational bottleneck in recurrent architectures"*.

---

## 5. What We Keep & The Bridge to Phase 3

We are not discarding Phase 2. In modern enterprise search and retrieval systems, the gold standard is **Hybrid Search**:

$$\text{Search Score} = \alpha \cdot \text{Dense Vector Score (Embeddings)} + (1 - \alpha) \cdot \text{Sparse Vector Score (TF-IDF / BM25)}$$

- **The Phase 2 Assets We Carry Forward**:
  - The structured PDF extraction pipeline.
  - The robust text cleaning, de-hyphenation, and tokenization.
  - The TF-IDF model and keyword extractor, which will serve as the **sparse retrieval half** of our hybrid search engine.
- **The Phase 3 Frontier**:
  - Overcoming keyword blindness through **Dense Vector Embeddings** (Word2Vec $\rightarrow$ Sentence Transformers).
  - Storing semantic embeddings in a **Vector Database**.
  - Enabling true **Semantic Search** based on meaning rather than exact word matching.
  - Setting the stage for **Phase 4: RAG & Abstractive LLM Assistant**.
