import sys
import time
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.search import SearchFilter
from app.services.embeddings.sentence_transformer import SentenceEmbeddingService
from app.services.parser.pdf_parser import parse_pdf
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.chunking import ChunkingConfig, SemanticChunker
from app.services.retrieval.hybrid_search import HybridSearchEngine
from app.services.vector_db.chroma_service import ChromaService


def run_hybrid_evaluation():
    print("=" * 90)
    print("PHASE 4.2 — HYBRID SEARCH BENCHMARK: DENSE (CHROMADB) vs. SPARSE (BM25) vs. HYBRID (RRF)")
    print("=" * 90)

    # 1. Initialize Services
    print("\n[1] Initializing ChromaDB, Sentence Transformer, and BM25 Index...")
    db_path = Path("data/chroma_db")
    chroma = ChromaService(persist_dir=db_path, collection_name="hybrid_evaluation_papers")
    embedder = SentenceEmbeddingService(model_name="all-MiniLM-L6-v2")
    bm25 = BM25Index()
    engine = HybridSearchEngine(chroma_service=chroma, embedder=embedder, bm25_index=bm25)

    # 2. Ingest and Chunk Academic Papers
    print("\n[2] Ingesting and Chunking Full Research Papers...")
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

    chunker = SemanticChunker(ChunkingConfig(max_tokens=300, overlap_tokens=50, prepend_headers=True))
    all_chunks = []

    for pdf in pdf_files:
        doc = parse_pdf(pdf)
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"    • {pdf.name:<32} -> {len(chunks):3d} production chunks across {doc.metadata.total_pages} pages")

    print(f"\n[3] Synchronizing Chunks to ChromaDB & BM25 Index ({len(all_chunks):,} total chunks)...")
    t0 = time.perf_counter()
    engine.index_chunks(all_chunks)
    sync_time = time.perf_counter() - t0
    print(f"    ✓ Ingestion, embedding, and indexing complete in {sync_time:.2f}s")
    print(f"    ✓ BM25 Indexed Chunks: {bm25.total_docs:,}")
    print(f"    ✓ ChromaDB Stored Chunks: {chroma.total_blocks:,}")

    # 4. Head-to-Head Benchmark Queries
    print("\n" + "=" * 90)
    print("[4] EVALUATION BENCHMARK: 4 DIVERSE RESEARCH QUERIES")
    print("=" * 90)

    test_queries = [
        # Query 1: Exact metric & acronym (Lexical strength)
        (
            "Exact Acronym & Metric (BM25 Strength)",
            "WMT 2014 English-to-German translation BLEU score 28.4",
            None,
        ),
        # Query 2: Conceptual paraphrase without exact keywords (Dense strength)
        (
            "Conceptual Paraphrase (Dense Strength)",
            "How does the model compute pairwise token relationships without recurrent network steps?",
            None,
        ),
        # Query 3: Technical parameters & hyperparameter naming (Mutual agreement)
        (
            "Hyperparameters & Architecture (Mutual Agreement)",
            "What is the hidden size, number of layers, and attention heads in BERT-Base?",
            None,
        ),
        # Query 4: Metadata scoped query (Disambiguation)
        (
            "Metadata Scoped Disambiguation (filter='attention_is_all_you_need')",
            "What compute hardware was used and how long was the training schedule?",
            SearchFilter(paper_id="1706_03762"),
        ),
    ]

    for label, query, s_filter in test_queries:
        print("\n" + "-" * 90)
        print(f"TEST:   [{label}]")
        print(f"QUERY:  \"{query}\"")
        if s_filter:
            print(f"FILTER: paper_id='{s_filter.paper_id}'")
        print("-" * 90)

        # 1. Dense Only
        t_dense = time.perf_counter()
        dense_hits = chroma.query_by_text(query, embedder=embedder, top_k=1, search_filter=s_filter)
        dense_lat = (time.perf_counter() - t_dense) * 1000

        # 2. BM25 Only
        t_bm25 = time.perf_counter()
        bm25_hits = bm25.search(query, top_k=1, search_filter=s_filter)
        bm25_lat = (time.perf_counter() - t_bm25) * 1000

        # 3. Hybrid Search (RRF)
        t_hyb = time.perf_counter()
        hybrid_hits = engine.search(query, top_k=1, search_filter=s_filter)
        hyb_lat = (time.perf_counter() - t_hyb) * 1000

        # Output Dense Top Hit
        print(f"\n  [1. Dense Only (ChromaDB) — {dense_lat:.1f}ms]")
        if dense_hits:
            h = dense_hits[0]
            print(f"      • Paper:     {h.metadata.get('paper_title')} (Page {h.page_number})")
            print(f"      • Section:   {h.section_title}")
            print(f"      • Cosine:    {h.score:+.4f}")
            print(f"      • Snippet:   \"{h.text[:130].replace(chr(10), ' ')}...\"")
        else:
            print("      • No results found.")

        # Output BM25 Top Hit
        print(f"\n  [2. Sparse Only (BM25Okapi) — {bm25_lat:.1f}ms]")
        if bm25_hits:
            c, score, rank = bm25_hits[0]
            print(f"      • Paper:     {c.paper_title} (Page {c.primary_page})")
            print(f"      • Section:   {c.section_title}")
            print(f"      • BM25:      {score:.4f}")
            print(f"      • Snippet:   \"{c.content_text[:130].replace(chr(10), ' ')}...\"")
        else:
            print("      • No results (score <= 0.0 filtered).")

        # Output Hybrid Top Hit
        print(f"\n  [3. Hybrid Search (Reciprocal Rank Fusion) — {hyb_lat:.1f}ms]")
        if hybrid_hits:
            hyb = hybrid_hits[0]
            print(f"      • Paper:     {hyb.paper_title} (Page {hyb.page_numbers[0]})")
            print(f"      • Section:   {hyb.section_title}")
            print(f"      • RRF Score: {hyb.rrf_score:.6f} (Dense Rank: #{hyb.dense_rank}, BM25 Rank: #{hyb.sparse_rank})")
            print(f"      • Snippet:   \"{hyb.content_text[:130].replace(chr(10), ' ')}...\"")
        else:
            print("      • No results found.")

    print("\n" + "=" * 90)
    print("PHASE 4.2 HYBRID SEARCH EVALUATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    run_hybrid_evaluation()
