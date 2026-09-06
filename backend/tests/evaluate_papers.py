import sys
from pathlib import Path

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.nlp import (
    extract_document_keywords,
    extract_section_keywords,
    generate_extractive_summary,
    generate_section_summary,
)
from app.services.parser.pdf_parser import parse_pdf


def run_evaluation():
    papers_dir = (
        Path("data/papers")
        if Path("data/papers").exists()
        else Path("../data/papers")
    )

    pdf_files = sorted(list(papers_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {papers_dir.resolve()}")
        return

    print("=" * 80)
    print("PHASE 2 — CLASSICAL NLP / TEXT INTELLIGENCE EVALUATION")
    print("=" * 80)

    for pdf_path in pdf_files:
        print(f"\n\n{'#' * 80}")
        print(f"PROCESSING PAPER: {pdf_path.name}")
        print(f"{'#' * 80}")

        doc = parse_pdf(pdf_path)

        print("\n--- 1. DOCUMENT METADATA ---")
        print(f"Title:            {doc.metadata.title}")
        print(f"Authors:          {', '.join(doc.metadata.authors[:4])}{'...' if len(doc.metadata.authors) > 4 else ''}")
        print(f"Total Sections:   {len(doc.sections)}")
        print(f"Content Blocks:   {len(doc.content_blocks)}")

        print("\n--- 2. TOP DOCUMENT KEYWORDS (TF-IDF) ---")
        doc_keywords = extract_document_keywords(doc, top_k=10)
        for rank, (word, score) in enumerate(doc_keywords, start=1):
            print(f"  {rank:2d}. {word:<18} (score: {score:.4f})")

        print("\n--- 3. SECTION-LEVEL KEYWORDS (TF-IDF) ---")
        section_keywords = extract_section_keywords(doc, top_k=5)
        # Select representative sections to display cleanly
        displayed_sections = 0
        for title, kws in section_keywords.items():
            # Skip empty or reference sections for the preview
            if not kws or any(ex in title.lower() for ex in ["reference", "table", "figure"]):
                continue
            kw_str = ", ".join(f"{w} ({s:.3f})" for w, s in kws[:4])
            print(f"  [{title[:40]}]: {kw_str}")
            displayed_sections += 1
            if displayed_sections >= 8:
                break

        print("\n--- 4. EXTRACTIVE SUMMARY (TOP 5 SENTENCES) ---")
        summary = generate_extractive_summary(doc, num_sentences=5)
        print(summary)

        print("\n--- 5. CONCLUSION / SUMMARY SECTION EXTRACTION ---")
        conclusion_summary = generate_section_summary(
            doc, section_title_query="conclusion", num_sentences=2
        )
        if conclusion_summary:
            print(conclusion_summary)
        else:
            print("(No designated Conclusion section found; extracted from main text)")


if __name__ == "__main__":
    run_evaluation()
