import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.parser.pdf_parser import parse_pdf
from app.services.retrieval.chunking import ChunkingConfig, SemanticChunker


def run_chunking_evaluation():
    print("=" * 85)
    print("PHASE 4.1 — PRODUCTION CHUNKING EVALUATION & COMPARISON BENCHMARK")
    print("=" * 85)

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

    config = ChunkingConfig(max_tokens=300, overlap_tokens=50, prepend_headers=True)
    chunker = SemanticChunker(config=config)

    print(f"\n[1] Active Chunking Configuration:")
    print(f"    • Target Max Tokens:    {config.max_tokens} tokens (~230 words)")
    print(f"    • Sliding Window Overlap: {config.overlap_tokens} tokens (~38 words)")
    print(f"    • Header Prepending:    {config.prepend_headers} (Injects [Document | Section | Page])")

    print("\n[2] Processing Full Academic Papers...")

    for pdf in pdf_files:
        print("\n" + "-" * 85)
        print(f"DOCUMENT: {pdf.name}")
        print("-" * 85)

        doc = parse_pdf(pdf)
        raw_blocks = doc.content_blocks
        production_chunks = chunker.chunk_document(doc)

        token_counts = [c.token_count for c in production_chunks]
        char_counts = [c.char_count for c in production_chunks]

        print(f"  • Raw Layout Blocks:      {len(raw_blocks):,} blocks")
        print(f"  • Production Chunks:      {len(production_chunks):,} chunks")
        print(f"  • Avg Token Count/Chunk:  {np.mean(token_counts):.1f} ± {np.std(token_counts):.1f} tokens")
        print(f"  • Min / Max Tokens:       {np.min(token_counts)} / {np.max(token_counts)} tokens")
        print(f"  • Multi-Page Chunks:      {sum(1 for c in production_chunks if len(c.page_numbers) > 1)} chunks")

        # Show Sample Chunk
        sample_chunk = production_chunks[min(5, len(production_chunks) - 1)]
        print(f"\n  [Sample Chunk #{sample_chunk.chunk_id}]")
        print(f"    • Section:      {sample_chunk.section_title}")
        print(f"    • Pages:        {sample_chunk.page_numbers}")
        print(f"    • Token Count:  {sample_chunk.token_count} tokens ({sample_chunk.char_count} chars)")
        print(f"    • Source Blocks:{sample_chunk.source_block_ids}")
        print(f"    • Embed Header: \"{sample_chunk.embed_text.splitlines()[0]}\"")
        print(f"    • Content Text: \"{sample_chunk.content_text[:180].replace(chr(10), ' ')}...\"")

    print("\n" + "=" * 85)
    print("PHASE 4.1 PRODUCTION CHUNKING EVALUATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_chunking_evaluation()
