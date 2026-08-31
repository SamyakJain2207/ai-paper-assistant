import datetime
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
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


def extract_blocks(pdf: pymupdf.Document) -> list[dict]:
    """Extract structured text blocks from a PDF."""
    blocks = []

    for page_number, page in enumerate(pdf, start=1):
        page_blocks = page.get_text("dict")["blocks"]

        for block_index, block in enumerate(page_blocks):
            if "lines" not in block:
                continue

            text_parts = []
            spans = []

            for line in block["lines"]:
                for span in line["spans"]:
                    text_parts.append(span["text"])

                    spans.append(
                        {
                            "text": span["text"],
                            "font_size": span["size"],
                            "font": span["font"],
                            "flags": span["flags"],
                        }
                    )

            text = " ".join(text_parts).strip()

            if not text:
                continue

            blocks.append(
                {
                    "block_id": f"page_{page_number}_block_{block_index}",
                    "text": text,
                    "page_number": page_number,
                    "bbox": list(block["bbox"]),
                    "spans": spans,
                    "section_id": None,
                }
            )

    return blocks


def extract_metadata(
    pdf: pymupdf.Document,
) -> PaperMetadata:
    """Extract metadata from the PDF's embedded metadata."""
    metadata = pdf.metadata

    authors = [
        author.strip()
        for author in (metadata.get("author") or "").split(",")
        if author.strip()
    ]

    return PaperMetadata(
        title=metadata.get("title") or None,
        authors=authors,
        total_pages=len(pdf),
    )


def is_bold_font(font: str, flags: int) -> bool:
    """Helper to check if span font attributes indicate bold weight."""
    font = font.lower()
    return "bold" in font or "medi" in font or "heavy" in font or (flags & 16) != 0


def extract_authors_fallback(raw_blocks: list[dict], title: str) -> list[str]:
    """Fallback author extraction on Page 1: parses the block following the title."""
    for block in raw_blocks:
        if block["page_number"] != 1:
            continue

        text = block["text"].strip()
        # Skip if title or abstract
        if (
            text.lower() == "abstract"
            or title.lower() in text.lower()
            or text.lower() in title.lower()
        ):
            continue

        spans = block.get("spans", [])
        if not spans:
            continue

        bold_spans = [
            s for s in spans if is_bold_font(s.get("font", ""), s.get("flags", 0))
        ]
        if not bold_spans:
            continue

        names = []
        for s in bold_spans:
            t = s["text"].strip()
            # Clean up asterisk/footnote indicators
            t_clean = re.sub(r"[\*†‡§¶#\d]+$", "", t).strip()
            # Must look like a name (capitalized words, no email/domain/blacklist)
            is_valid_name = len(t_clean) >= 3 and not any(
                k in t_clean.lower()
                for k in [
                    "google",
                    "university",
                    "department",
                    "research",
                    "microsoft",
                    "school",
                    "college",
                    "institute",
                    "laboratory",
                ]
            )
            if is_valid_name and all(
                w[0].isupper() or w[0] in "-\u0141"
                for w in t_clean.split()
                if w and w[0].isalpha()
            ):
                names.append(t_clean)

        if names:
            return names

    return []


def parse_date_string(date_str: str) -> datetime.date | None:
    """Robust parser to convert various publication date strings into datetime.date."""
    if not date_str:
        return None

    # Check for ISO formats like 2018-10-11
    iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso_match:
        return datetime.date(
            int(iso_match.group(1)),
            int(iso_match.group(2)),
            int(iso_match.group(3)),
        )

    months = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    date_str_lower = date_str.lower()
    for idx, month in enumerate(months, start=1):
        if month in date_str_lower:
            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", date_str)
            if not year_match:
                continue
            year = int(year_match.group(1))
            day_match = re.search(
                r"\b([1-9]|[12]\d|3[01])\b",
                date_str.replace(year_match.group(0), ""),
            )
            day = int(day_match.group(1)) if day_match else 1
            return datetime.date(year, idx, day)

    # Fallback to year only (default to December 1st of that year)
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", date_str)
    if year_match:
        return datetime.date(int(year_match.group(1)), 12, 1)

    return None


def extract_local_arxiv_info(
    pdf: pymupdf.Document,
) -> tuple[str | None, datetime.date | None]:
    """Scan Page 1 text for arXiv ID or publication date markers locally."""
    arxiv_id = None
    pub_date = None

    page1_text = pdf[0].get_text()

    # Search for arXiv:YYMM.NNNNN
    arxiv_match = re.search(r"arxiv:(\d{4}\.\d{4,5})", page1_text, re.IGNORECASE)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)

    # Watermark date check: e.g. [cs.CL] 24 May 2019
    watermark_match = re.search(
        r"\[cs\.[a-z]+\]\s+(\d{1,2}\s+[a-z]+\s+\d{4})",
        page1_text,
        re.IGNORECASE,
    )
    if watermark_match:
        pub_date = parse_date_string(watermark_match.group(1))
    else:
        # Fallback to general date regex in page 1
        date_match = re.search(
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b|"
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|"
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
            page1_text,
            re.IGNORECASE,
        )
        if date_match:
            pub_date = parse_date_string(date_match.group(0))

    # NeurIPS/NIPS year fallback
    if not pub_date:
        nips_match = re.search(
            r"\b(?:nips|neurips)\s+(\d{4})\b", page1_text, re.IGNORECASE
        )
        if nips_match:
            pub_date = datetime.date(int(nips_match.group(1)), 12, 1)

    return arxiv_id, pub_date


def enrich_via_arxiv_api(arxiv_id: str | None, title: str | None) -> dict | None:
    """Query the official arXiv API using HTTPS by ID or Title to get metadata."""
    if not arxiv_id and not title:
        return None

    # Construct the API URL
    if arxiv_id:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    else:
        # Format title query
        formatted_title = f'ti:"{title}"'
        url = f"https://export.arxiv.org/api/query?search_query={formatted_title}&max_results=1"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            entry = root.find("atom:entry", ns)
            if entry is not None:
                title_res = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                title_res = re.sub(r"\s+", " ", title_res)
                published = entry.find("atom:published", ns).text.strip()

                doi_el = entry.find("arxiv:doi", ns)
                doi = doi_el.text.strip() if doi_el is not None else None

                id_url = entry.find("atom:id", ns).text.strip()
                extracted_id = id_url.split("/abs/")[-1].split("v")[0]

                authors = [
                    a.find("atom:name", ns).text.strip()
                    for a in entry.findall("atom:author", ns)
                ]

                return {
                    "title": title_res,
                    "published": parse_date_string(published),
                    "doi": doi or f"10.48550/arXiv.{extracted_id}",
                    "authors": authors,
                    "arxiv_id": extracted_id,
                }
    except httpx.RequestError:
        # Gracefully handle connection timeouts/exceptions
        return None


def parse_pdf(pdf_path: str | Path) -> Document:
    """Parse a research paper into our structured Document model."""
    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as pdf:
        metadata = extract_metadata(pdf)
        raw_blocks = extract_blocks(pdf)

        # 1. Fallback Title locally
        if (not metadata.title or ".pdf" in metadata.title.lower()) and raw_blocks:
            metadata.title = raw_blocks[0]["text"].strip()

        # 2. Fallback Authors locally if embedded authors metadata is empty
        if not metadata.authors:
            metadata.authors = extract_authors_fallback(
                raw_blocks, metadata.title or ""
            )

        # 3. Local arXiv ID and Date extraction
        arxiv_id, local_date = extract_local_arxiv_info(pdf)
        metadata.publication_date = local_date

        if arxiv_id:
            metadata.doi = f"10.48550/arXiv.{arxiv_id}"

        # 4. Attempt remote metadata enrichment via arXiv API
        enriched = enrich_via_arxiv_api(arxiv_id, metadata.title)
        if enriched:
            if enriched.get("title"):
                metadata.title = enriched["title"]
            if enriched.get("authors"):
                metadata.authors = enriched["authors"]
            if enriched.get("published"):
                metadata.publication_date = enriched["published"]
            if enriched.get("doi"):
                metadata.doi = enriched["doi"]

        sections, content_blocks = detect_sections(raw_blocks)

        return Document(
            doc_id=generate_doc_id(pdf_path),
            metadata=metadata,
            sections=sections,
            content_blocks=content_blocks,
        )
