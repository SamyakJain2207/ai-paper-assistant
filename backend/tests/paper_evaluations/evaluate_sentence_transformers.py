import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.embeddings.sentence_transformer import SentenceEmbeddingService
from app.services.embeddings.word2vec import Word2VecPipeline
from app.services.nlp.preprocessing import clean_paper_text, tokenize
from app.services.nlp.tfidf import TFIDFModel
from app.services.parser.pdf_parser import parse_pdf


def sparse_tfidf_sim(d1: dict[str, float], d2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse TF-IDF dictionaries."""
    common = set(d1.keys()) & set(d2.keys())
    if not common:
        return 0.0
    dot = sum(d1[k] * d2[k] for k in common)
    n1 = np.sqrt(sum(v ** 2 for v in d1.values()))
    n2 = np.sqrt(sum(v ** 2 for v in d2.values()))
    return float(dot / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0


def run_full_paper_evaluation():
    print("=" * 85)
    print("PHASE 3.3 — FULL RESEARCH PAPER EMBEDDING & SEMANTIC RETRIEVAL TEST")
    print("=" * 85)

    # 1. Load Sentence Transformer
    print("\n[1] Initializing Sentence Transformer (all-MiniLM-L6-v2)...")
    embedder = SentenceEmbeddingService(model_name="all-MiniLM-L6-v2")
    print(f"    ✓ Model loaded:     {embedder.model_name}")
    print(f"    ✓ Embedding Dim:    {embedder.dimension} continuous features")

    # 2. Parse and Embed Full Papers
    print("\n[2] Ingesting & Embedding Full Research Papers...")
    papers_dir = (
        Path("data/papers")
        if Path("data/papers").exists()
        else (
            Path("../data/papers")
            if Path("../data/papers").exists()
            else Path("../../data/papers")
        )
    )
    pdf_files = sorted(list(papers_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"[!] No PDFs found in {papers_dir}")
        return

    all_blocks = []
    paper_sentences_for_tfidf = []

    for pdf in pdf_files:
        doc = parse_pdf(pdf)
        blocks = embedder.embed_document_blocks(doc, min_words=8)
        all_blocks.extend(blocks)
        for b in blocks:
            tokens = [t.lower() for t in tokenize(clean_paper_text(b.text))]
            if tokens:
                paper_sentences_for_tfidf.append(tokens)

        print(f"    • {pdf.name:<32} -> {len(blocks):3d} embedded chunks across {doc.metadata.total_pages} pages")

    print(f"    ✓ Total Knowledge Base Chunks: {len(all_blocks):,}")

    # 3. Train Word2Vec & Fit TF-IDF on the same blocks (Phase 3.2 Baseline)
    print("\n[3] Training Phase 3.2 Baseline (Word2Vec + TF-IDF Weights)...")
    tfidf_model = TFIDFModel(smooth=True).fit(paper_sentences_for_tfidf)
    print(f"    ✓ TF-IDF Vocabulary: {len(tfidf_model.vocab):,} terms")

    w2v_pipeline = Word2VecPipeline(vector_size=100, window=5, min_count=1, epochs=30, workers=1)
    w2v_pipeline.train(paper_sentences_for_tfidf)
    print(f"    ✓ Word2Vec Vocabulary: {w2v_pipeline.vocab_size:,} terms (100-D static vectors)")

    # Pre-pool Word2Vec vectors for all chunks
    w2v_block_vectors = np.array([
        w2v_pipeline.sentence_vector(b.text, weights=tfidf_model)
        for b in all_blocks
    ])
    norms = np.linalg.norm(w2v_block_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    w2v_block_vectors_norm = w2v_block_vectors / norms

    # 4. The Semantic Acid Test: 6 Conceptual Queries
    print("\n" + "=" * 85)
    print("[4] HEAD-TO-HEAD RETRIEVAL: SENTENCE TRANSFORMER vs. WORD2VEC + TF-IDF")
    print("=" * 85)

    test_queries = [
        # Queries for "Attention Is All You Need"
        (
            "Attention Is All You Need",
            "Self-Attention Mechanism",
            "How does the model compute relationships between words without recurrent networks?",
        ),
        (
            "Attention Is All You Need",
            "Hardware & Schedule",
            "What compute hardware was used and how long was the training schedule?",
        ),
        (
            "Attention Is All You Need",
            "Positional Encodings",
            "How does the model know the relative order or position of tokens in a sequence?",
        ),
        # Queries for "BERT"
        (
            "BERT",
            "Masked LM Pre-training",
            "How does the bidirectional language model mask input tokens during pre-training?",
        ),
        (
            "BERT",
            "Next Sentence Prediction",
            "How does the model determine if two sentences logically follow one another?",
        ),
        (
            "BERT",
            "Model Architecture / Size",
            "What are the differences in layers and hidden size between BERT Base and Large?",
        ),
    ]

    for target_paper, concept, query in test_queries:
        print("\n" + "-" * 85)
        print(f"TARGET:  [{target_paper}] — {concept}")
        print(f"QUERY:   \"{query}\"")
        print("-" * 85)

        # --- A. Sentence Transformer (Phase 3.3) ---
        sbert_results = embedder.search_blocks(query, all_blocks, top_k=1)
        sbert_block, sbert_score = sbert_results[0]
        sbert_paper = sbert_block.metadata.get("title", "Unknown")
        sbert_match = target_paper.lower().split()[0] in sbert_paper.lower()

        print(f"  [Phase 3.3 — Sentence Transformer (all-MiniLM-L6-v2)]")
        print(f"    • Paper:        {sbert_paper} (Page {sbert_block.page_number})")
        print(f"    • Section:      {sbert_block.section_title or 'Main Body'}")
        print(f"    • Cosine Sim:   {sbert_score:+.4f}")
        print(f"    • Evaluation:   {'✓ CORRECT PAPER' if sbert_match else '✗ RETRIEVAL MISS'}")
        print(f"    • Snippet:      \"{sbert_block.text[:160].replace(chr(10), ' ')}...\"")

        # --- B. Word2Vec + TF-IDF Weighted Pooling (Phase 3.2 Baseline) ---
        q_w2v = w2v_pipeline.sentence_vector(query, weights=tfidf_model)
        q_norm = np.linalg.norm(q_w2v)
        if q_norm > 0:
            q_w2v_norm = q_w2v / q_norm
            w2v_sims = np.dot(w2v_block_vectors_norm, q_w2v_norm)
            top_w2v_idx = int(np.argmax(w2v_sims))
            w2v_score = float(w2v_sims[top_w2v_idx])
            w2v_block = all_blocks[top_w2v_idx]
        else:
            w2v_block = all_blocks[0]
            w2v_score = 0.0

        w2v_paper = w2v_block.metadata.get("title", "Unknown")
        w2v_match = target_paper.lower().split()[0] in w2v_paper.lower()

        print(f"\n  [Phase 3.2 — Baseline (Word2Vec + TF-IDF Pooling)]")
        print(f"    • Paper:        {w2v_paper} (Page {w2v_block.page_number})")
        print(f"    • Section:      {w2v_block.section_title or 'Main Body'}")
        print(f"    • Cosine Sim:   {w2v_score:+.4f}")
        print(f"    • Evaluation:   {'✓ CORRECT PAPER' if w2v_match else '✗ RETRIEVAL MISS'}")
        print(f"    • Snippet:      \"{w2v_block.text[:160].replace(chr(10), ' ')}...\"")

    print("\n" + "=" * 85)
    print("FULL PAPER HEAD-TO-HEAD EVALUATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_full_paper_evaluation()
