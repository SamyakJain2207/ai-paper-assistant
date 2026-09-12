import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def run_paper_test(paper_name: str = "Attention-is-all-you-need.pdf"):
    print("=" * 75, flush=True)
    print(f"PHASE 2 SYSTEM TEST: {paper_name}", flush=True)
    print("=" * 75, flush=True)

    papers_dir = (
        Path("data/papers")
        if Path("data/papers").exists()
        else (
            Path("../data/papers")
            if Path("../data/papers").exists()
            else Path("../../data/papers")
        )
    )
    pdf_path = papers_dir / paper_name

    if not pdf_path.exists():
        available = [p.name for p in papers_dir.glob("*.pdf")]
        print(f"Error: {pdf_path} not found.", flush=True)
        print(f"Available papers in {papers_dir}: {available}", flush=True)
        return

    # 1. Parse PDF Document
    print("\n[1/5] Parsing PDF into structured Document...", flush=True)
    from app.models.block import BlockType
    from app.services.parser.pdf_parser import parse_pdf

    doc = parse_pdf(pdf_path)

    paras = [b for b in doc.content_blocks if b.type == BlockType.PARAGRAPH]
    headings = [b for b in doc.content_blocks if b.type == BlockType.HEADING]

    print(f"  Title:          {doc.metadata.title}", flush=True)
    print(f"  Authors:        {', '.join(doc.metadata.authors[:4])}{'...' if len(doc.metadata.authors) > 4 else ''}", flush=True)
    print(f"  Total Pages:    {doc.metadata.total_pages}", flush=True)
    print(f"  Total Sections: {len(doc.sections)} detected", flush=True)
    print(f"  Content Blocks: {len(doc.content_blocks)} ({len(paras)} paragraphs, {len(headings)} headings)", flush=True)

    from app.services.nlp import (
        extract_document_keywords,
        extract_section_keywords,
        generate_extractive_summary,
        generate_section_summary,
    )

    # 2. Document Keywords
    print("\n[2/5] Extracting Top Document Keywords (TF-IDF)...", flush=True)
    doc_keywords = extract_document_keywords(doc, top_k=10)
    print("-" * 75, flush=True)
    for rank, (word, score) in enumerate(doc_keywords, start=1):
        print(f"  {rank:2d}. {word:<18} (score: {score:.4f})", flush=True)

    # 3. Section Keywords
    print("\n[3/5] Extracting Section-Level Keywords...", flush=True)
    print("-" * 75, flush=True)
    sec_keywords = extract_section_keywords(doc, top_k=5)
    displayed = 0
    for title, kws in sec_keywords.items():
        if not kws or any(ex in title.lower() for ex in ["reference", "table", "figure"]):
            continue
        kw_str = ", ".join(f"{w} ({s:.3f})" for w, s in kws[:4])
        print(f"  * {title[:38]:<40} -> {kw_str}", flush=True)
        displayed += 1
        if displayed >= 8:
            break

    # 4. Extractive Summary
    print("\n[4/5] Generating Extractive Summary (Top 5 sentences)...", flush=True)
    print("-" * 75, flush=True)
    summary = generate_extractive_summary(doc, num_sentences=5)
    print(summary, flush=True)

    # 5. Conclusion Summary
    print("\n[5/5] Extracting Conclusion Summary...", flush=True)
    print("-" * 75, flush=True)
    conclusion_summary = generate_section_summary(
        doc, section_title_query="conclusion", num_sentences=2
    )
    if conclusion_summary:
        print(conclusion_summary, flush=True)
    else:
        print("(No designated Conclusion section heading found)", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("TEST COMPLETE", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Attention-is-all-you-need.pdf"
    run_paper_test(target)
