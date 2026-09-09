import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.embeddings.word2vec import Word2VecPipeline
from app.services.parser.pdf_parser import parse_pdf


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format rows into a clean CLI table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)


def run_word2vec_evaluation():
    print("=" * 80)
    print("PHASE 3.1 — WORD2VEC: TRAINING, VECTOR ARITHMETIC & LIMITATIONS")
    print("=" * 80)

    # 1. Locate and parse papers
    papers_dir = (
        Path("data/papers")
        if Path("data/papers").exists()
        else Path("../data/papers")
    )
    pdf_files = sorted(list(papers_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"[!] No PDF files found in {papers_dir.resolve()}.")
        return

    print(f"\n[1] Parsing {len(pdf_files)} Research Papers:")
    docs = []
    for pdf_path in pdf_files:
        doc = parse_pdf(pdf_path)
        docs.append(doc)
        print(f"    • {pdf_path.name:<32} (Blocks: {len(doc.content_blocks)}, Sections: {len(doc.sections)})")

    # 2. Extract and tokenize sentences
    print("\n[2] Extracting & Tokenizing Clean Prose Sentences...")
    sentences = Word2VecPipeline.extract_sentences_from_documents(docs)
    total_tokens = sum(len(s) for s in sentences)
    print(f"    • Total Clean Sentences Extracted: {len(sentences):,}")
    print(f"    • Total Tokens in Corpus:          {total_tokens:,}")
    print(f"    • Example Sentence:                {' '.join(sentences[0])}")

    # 3. Train CBOW Model
    print("\n[3] Training Word2Vec CBOW (Continuous Bag-of-Words, sg=0)...")
    cbow_pipeline = Word2VecPipeline(
        vector_size=100,
        window=5,
        min_count=2,
        sg=0,  # CBOW
        epochs=100,
        seed=42,
    )
    cbow_pipeline.train(sentences)
    print(f"    ✓ CBOW Vocabulary Size: {cbow_pipeline.vocab_size} unique words")

    # 4. Train Skip-gram Model
    print("\n[4] Training Word2Vec Skip-gram (sg=1)...")
    skipgram_pipeline = Word2VecPipeline(
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,  # Skip-gram
        epochs=100,
        seed=42,
    )
    skipgram_pipeline.train(sentences)
    print(f"    ✓ Skip-gram Vocabulary Size: {skipgram_pipeline.vocab_size} unique words")

    # Save models to data/processed
    processed_dir = Path("data/processed") if Path("data/processed").exists() else Path("../data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    cbow_pipeline.save(processed_dir / "word2vec_cbow.model")
    skipgram_pipeline.save(processed_dir / "word2vec_skipgram.model")
    print(f"    ✓ Models saved to {processed_dir.resolve()}")

    # 5. Semantic Word Similarities (CBOW vs Skip-gram comparison)
    print("\n" + "=" * 80)
    print("[5] SEMANTIC SIMILARITY COMPARISON: CBOW vs SKIP-GRAM")
    print("=" * 80)
    word_pairs = [
        ("encoder", "decoder"),
        ("attention", "recurrent"),
        ("bert", "transformer"),
        ("layer", "layers"),
        ("training", "gpu"),
        ("model", "attention"),
        ("random", "transformer"),
    ]

    sim_rows = []
    for w1, w2 in word_pairs:
        if cbow_pipeline.has_word(w1) and cbow_pipeline.has_word(w2):
            sim_cbow = cbow_pipeline.similarity(w1, w2)
            sim_sg = skipgram_pipeline.similarity(w1, w2)
            sim_rows.append([f"{w1} <-> {w2}", f"{sim_cbow:+.4f}", f"{sim_sg:+.4f}"])
        else:
            missing = [w for w in (w1, w2) if not cbow_pipeline.has_word(w)]
            sim_rows.append([f"{w1} <-> {w2}", f"OOV ({', '.join(missing)})", "N/A"])

    print(format_table(["Word Pair", "CBOW Similarity", "Skip-gram Similarity"], sim_rows))

    # 6. Nearest Neighbors (Exploration)
    print("\n" + "=" * 80)
    print("[6] TOP NEAREST NEIGHBORS (SEMANTIC CLUSTERING)")
    print("=" * 80)
    probe_words = ["attention", "transformer", "recurrent", "encoder", "layer"]
    for word in probe_words:
        if cbow_pipeline.has_word(word):
            cbow_nn = cbow_pipeline.most_similar(positive=[word], topn=3)
            sg_nn = skipgram_pipeline.most_similar(positive=[word], topn=3)
            cbow_str = ", ".join(f"{w} ({s:.3f})" for w, s in cbow_nn)
            sg_str = ", ".join(f"{w} ({s:.3f})" for w, s in sg_nn)
            print(f"  Target: '{word}'")
            print(f"    • CBOW:      {cbow_str}")
            print(f"    • Skip-gram: {sg_str}")

    # 7. Vector Arithmetic: Analogical Reasoning
    print("\n" + "=" * 80)
    print("[7] VECTOR ARITHMETIC / ANALOGY EXPERIMENTS")
    print("=" * 80)
    print("  Testing: vec(decoder) - vec(output) + vec(input) ≈ ?")
    try:
        analogy = cbow_pipeline.most_similar(
            positive=["decoder", "input"],
            negative=["output"],
            topn=4,
        )
        print("  Results:")
        for rank, (w, s) in enumerate(analogy, start=1):
            print(f"    {rank}. {w:<16} (score: {s:.4f})")
    except KeyError as e:
        print(f"  Analogy skipped due to vocabulary: {e}")

    # 8. EMPIRICAL DEMONSTRATION OF WORD2VEC LIMITATIONS
    print("\n" + "=" * 80)
    print("[8] EMPIRICAL DEMONSTRATION OF STATIC EMBEDDING LIMITATIONS")
    print("=" * 80)

    # Limitation 1: Polysemy / Single Vector per Word
    print("\n--- Limitation 1: Polysemy & Context Blindness ---")
    print("A word has exactly ONE static vector, regardless of how it is used.")
    target_poly = "head" if cbow_pipeline.has_word("head") else "state"
    if cbow_pipeline.has_word(target_poly):
        vec = cbow_pipeline.get_vector(target_poly)
        print(f"  • Word: '{target_poly}'")
        print(f"    Context A: 'multi-head attention mechanism' (neural network structure)")
        print(f"    Context B: 'he tilted his head in thought'  (human anatomy)")
        print(f"    -> Word2Vec assigns the EXACT SAME vector (shape: {vec.shape}, norm: {np.linalg.norm(vec):.2f})")
        print(f"    -> It cannot adapt to the sentence meaning!")

    # Limitation 2: Out-Of-Vocabulary (OOV)
    print("\n--- Limitation 2: Out-Of-Vocabulary (OOV) Blindness ---")
    oov_candidates = ["roberta", "deberta", "attension", "unparallelized"]
    for oov_word in oov_candidates:
        try:
            cbow_pipeline.get_vector(oov_word)
        except KeyError as e:
            print(f"  • Looking up '{oov_word}': Caught expected error -> {e}")

    # Limitation 3: Bag-of-Vectors Failure (Word Order Invariance)
    print("\n--- Limitation 3: Bag-of-Vectors Failure: Word Order Invariance ---")
    s1 = "model uses attention instead of recurrent layers"
    s2 = "model uses recurrent layers instead of attention"
    sim_order = cbow_pipeline.sentence_similarity(s1, s2)
    print(f"  Sentence 1: '{s1}'")
    print(f"  Sentence 2: '{s2}'")
    print(f"  -> Semantic meaning: OPPOSITE / INVERTED")
    print(f"  -> Word2Vec Pooled Cosine Similarity: {sim_order:.6f}")
    print(f"  -> Conclusion: Word2Vec sentence pooling is completely blind to word order!")

    # Limitation 4: Negation Blindness
    print("\n--- Limitation 4: Negation Blindness ---")
    s3 = "this transformer architecture is effective"
    s4 = "this transformer architecture is not effective"
    sim_neg = cbow_pipeline.sentence_similarity(s3, s4)
    print(f"  Sentence 1: '{s3}'")
    print(f"  Sentence 2: '{s4}'")
    print(f"  -> Semantic meaning: OPPOSITE (Affirmation vs Negation)")
    print(f"  -> Word2Vec Pooled Cosine Similarity: {sim_neg:.6f}")
    print(f"  -> Conclusion: Adding 'not' merely dilutes the vector slightly; it cannot flip polarity.")

    # Limitation 5: Corpus Size & Data Hunger
    print("\n--- Limitation 5: Data Hunger on Domain Corpora ---")
    print(f"  • Trained on 2 papers (~{len(sentences)} sentences, ~{total_tokens:,} tokens).")
    print("  • High-frequency words ('attention', 'layer') have decent neighbors.")
    print("  • Low-frequency domain words have noisy or unhelpful vectors.")
    print("  • Word2Vec needs hundreds of millions of tokens to learn robust multi-dimensional geometry.")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE — READY TO TRANSITION TO SENTENCE TRANSFORMERS (3.2 / 3.3)")
    print("=" * 80)


if __name__ == "__main__":
    run_word2vec_evaluation()
