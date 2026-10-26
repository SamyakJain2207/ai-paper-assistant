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
from app.services.vector_db.chroma_service import ChromaService


def run_vector_db_evaluation():
    print("=" * 85)
    print("PHASE 3.5 — CHROMADB VECTOR DATABASE PERSISTENCE & FILTERED RETRIEVAL BENCHMARK")
    print("=" * 85)

    # 1. Initialize ChromaDB Service & Embedding Model
    print("\n[1] Initializing ChromaDB Persistent Storage & Sentence Transformer...")
    db_path = Path("data/chroma_db")
    chroma = ChromaService(persist_dir=db_path, collection_name="academic_papers")
    embedder = SentenceEmbeddingService(model_name="all-MiniLM-L6-v2")

    print(f"    ✓ Persistent Database Path: {db_path.resolve()}")
    print(f"    ✓ Active Collection:       {chroma.collection_name}")
    print(f"    ✓ Metric Space:            Cosine Distance (HNSW)")
    print(f"    ✓ Initial Blocks Stored:   {chroma.total_blocks:,}")

    # 2. Ingest Full Research Papers (Idempotent Upsert)
    print("\n[2] Ingesting Full Research Papers into ChromaDB...")
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

    ingest_start = time.perf_counter()
    for pdf in pdf_files:
        paper_id = pdf.stem.lower().replace("-", "_")
        doc = parse_pdf(pdf)

        # Embed and upsert blocks into ChromaDB
        blocks = embedder.embed_document_blocks(doc, min_words=8)
        upserted_count = chroma.index_embedded_blocks(
            blocks=blocks,
            paper_id=paper_id,
            paper_title=doc.metadata.title or pdf.stem,
        )
        print(
            f"    • {pdf.name:<32} -> {upserted_count:3d} blocks indexed across {doc.metadata.total_pages} pages (ID: '{paper_id}')"
        )

    ingest_elapsed = time.perf_counter() - ingest_start
    print(f"    ✓ Total Vector DB Blocks:  {chroma.total_blocks:,} (Ingestion & indexing took {ingest_elapsed:.2f}s)")

    # 3. Collection Metadata Audit
    print("\n[3] Auditing Indexed Library Summary...")
    papers_summary = chroma.list_indexed_papers()
    for p in papers_summary:
        print(f"    • Paper ID: [{p['paper_id']}]")
        print(f"      Title:        \"{p['title']}\"")
        print(f"      Block Count:  {p['total_blocks']} chunks across {p['total_pages']} pages")

    # 4. Global Semantic Search Benchmark
    print("\n" + "=" * 85)
    print("[4] GLOBAL SEMANTIC RETRIEVAL (Searching across ALL papers)")
    print("=" * 85)

    global_queries = [
        (
            "Attention Is All You Need",
            "How does the model compute scaled dot-product attention for queries, keys, and values?",
        ),
        (
            "BERT",
            "How are input tokens masked and corrupted during bidirectional pre-training?",
        ),
    ]

    for expected_paper, query in global_queries:
        print("\n" + "-" * 85)
        print(f"QUERY:    \"{query}\"")
        print(f"EXPECTED: [{expected_paper}]")
        print("-" * 85)

        t0 = time.perf_counter()
        hits = chroma.query_by_text(query_text=query, embedder=embedder, top_k=2)
        latency_ms = (time.perf_counter() - t0) * 1000

        if hits:
            top = hits[0]
            matched = expected_paper.lower().split()[0] in top.metadata.get("paper_title", "").lower()
            print(f"  Top Match (Latency: {latency_ms:.2f}ms):")
            print(f"    • Paper:        {top.metadata.get('paper_title', 'Unknown')}")
            print(f"    • Section:      {top.section_title or 'Main Body'} (Page {top.page_number})")
            print(f"    • Cosine Sim:   {top.score:+.4f} (Distance: {top.distance:.4f})")
            print(f"    • Evaluation:   {'✓ CORRECT PAPER RETRIEVED' if matched else '✗ RETRIEVAL DRIFT'}")
            print(f"    • Snippet:      \"{top.text[:180].replace(chr(10), ' ')}...\"")

    # 5. Metadata Scoped Search Benchmark (Disambiguation Test)
    print("\n" + "=" * 85)
    print("[5] METADATA-FILTERED SEARCH (Scoped Query Disambiguation)")
    print("=" * 85)
    print("Scenario: The user asks an identical query: 'What compute hardware was used for training?'")
    print("We demonstrate how metadata filtering scopes the answer to the requested paper.\n")

    ambiguous_query = "What compute hardware was used and how long was the training schedule?"

    # Scope 1: Filter to Attention paper
    print("  --- Scope A: filter={'paper_id': 'attention_is_all_you_need'} ---")
    filt_att = SearchFilter(paper_id="attention_is_all_you_need")
    hits_att = chroma.query_by_text(ambiguous_query, embedder=embedder, top_k=1, search_filter=filt_att)
    if hits_att:
        h = hits_att[0]
        print(f"    • Paper:      {h.metadata.get('paper_title')}")
        print(f"    • Section:    {h.section_title} (Page {h.page_number})")
        print(f"    • Cosine Sim: {h.score:+.4f}")
        print(f"    • Snippet:    \"{h.text[:180].replace(chr(10), ' ')}...\"")

    # Scope 2: Filter to BERT paper
    print("\n  --- Scope B: filter={'paper_id': 'bert'} ---")
    filt_bert = SearchFilter(paper_id="bert")
    hits_bert = chroma.query_by_text(ambiguous_query, embedder=embedder, top_k=1, search_filter=filt_bert)
    if hits_bert:
        h = hits_bert[0]
        print(f"    • Paper:      {h.metadata.get('paper_title')}")
        print(f"    • Section:    {h.section_title} (Page {h.page_number})")
        print(f"    • Cosine Sim: {h.score:+.4f}")
        print(f"    • Snippet:    \"{h.text[:180].replace(chr(10), ' ')}...\"")

    # 6. Persistence Verification (Re-load from disk)
    print("\n" + "=" * 85)
    print("[6] VERIFYING DISK PERSISTENCE")
    print("=" * 85)
    del chroma  # Simulate complete application shutdown

    reloaded_chroma = ChromaService(persist_dir=db_path, collection_name="academic_papers")
    print(f"    ✓ Re-opened ChromaDB from {db_path}")
    print(f"    ✓ Total Blocks Restored from Disk: {reloaded_chroma.total_blocks:,}")
    assert reloaded_chroma.total_blocks > 0, "Error: database did not persist blocks to disk!"
    print("    ✓ Verified: Vectors and HNSW indices persist across application restarts.")

    print("\n" + "=" * 85)
    print("PHASE 3.5 VECTOR DATABASE EVALUATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_vector_db_evaluation()
