from pathlib import Path

from app.services.parser.pdf_parser import parse_pdf

PAPERS_DIR = Path("data/papers") if Path("data/papers").exists() else Path("../data/papers")


def run_manual_test():
    for pdf_path in PAPERS_DIR.glob("*.pdf"):
        print(f"\n{'=' * 60}")
        print(f"PAPER: {pdf_path.name}")
        print("=" * 60)

        document = parse_pdf(pdf_path)

        print("\nMETADATA:")
        print(f"Title: {document.metadata.title}")
        print(f"Authors: {document.metadata.authors}")
        print(f"Publication date: {document.metadata.publication_date}")
        print(f"DOI: {document.metadata.doi}")
        print(f"Total pages: {document.metadata.total_pages}")

        print("\nSECTIONS:")
        for section in document.sections:
            indent = "  " * (section.level - 1)
            print(
                f"{indent}- {section.title} "
                f"(level={section.level}, parent={section.parent_id})"
            )

        print(f"\nContent blocks: {len(document.content_blocks)}")


if __name__ == "__main__":
    run_manual_test()
