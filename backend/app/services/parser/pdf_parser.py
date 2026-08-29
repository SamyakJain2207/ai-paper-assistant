import hashlib
from pathlib import Path

import pymupdf
from app.models.document import Document
from app.models.metadata import PaperMetadata
from app.services.parser.section_detector import detect_sections


def generate_doc_id(pdf_path: Path) -> str:
    """Generate a stable ID from the PDF contents."""
    file_hash = hashlib.sha256()

    with pdf_path.open("rb") as file:
        while chunk := file.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def parse_pdf(pdf_path: str | Path) -> Document:
    """Extract text, metadata, and sections from a PDF."""
    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as pdf:
        metadata = pdf.metadata

        paper_metadata = PaperMetadata(
            title=metadata.get("title") or None,
            authors=[
                author.strip()
                for author in (metadata.get("author") or "").split(",")
                if author.strip()
            ],
            total_pages=len(pdf),
        )

        raw_blocks = []

        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("dict")["blocks"]

            for block_index, block in enumerate(blocks):
                if "lines" not in block:
                    continue

                lines = block["lines"]

                text_parts = []

                font_sizes = []

                for line in lines:
                    for span in line["spans"]:
                        text_parts.append(span["text"])
                        font_sizes.append(span["size"])

                text = " ".join(text_parts).strip()

                if not text:
                    continue

                raw_blocks.append(
                    {
                        "block_id": f"page_{page_number}_block_{block_index}",
                        "text": text,
                        "page_number": page_number,
                        "bbox": list(block["bbox"]),
                        "font_size": max(font_sizes),
                        "type": None,
                        "section_id": None,
                    }
                )

        sections, content_blocks = detect_sections(raw_blocks)

        return Document(
            doc_id=generate_doc_id(pdf_path),
            metadata=paper_metadata,
            sections=sections,
            content_blocks=content_blocks,
        )