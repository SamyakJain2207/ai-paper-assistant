import re

from app.models.block import BlockType, ContentBlock
from app.models.section import Section

NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*([A-Z]\b(?:\.\d+)*|\d+(?:\.\d+)*|[IVX]+\b)\.?\s+(.+)"
)


def get_heading_level(text: str) -> int | None:
    """Return the hierarchy level of a numbered heading."""
    match = NUMBERED_HEADING_PATTERN.match(text)

    if not match:
        return None

    return len(match.group(1).split("."))


def looks_like_numbered_heading(text: str) -> bool:
    """Check whether text follows a numbered section-heading pattern."""
    return bool(NUMBERED_HEADING_PATTERN.match(text))


def is_span_bold(span: dict) -> bool:
    """Check if a single span's font styling indicates bold weight."""
    font = span.get("font", "").lower()
    flags = span.get("flags", 0)
    return "bold" in font or "medi" in font or "heavy" in font or (flags & 16) != 0


def get_block_formatting(block: dict) -> tuple[float, bool, float]:
    """Return the maximum font size, whether the block is bold, and the first span size."""
    spans = block.get("spans", [])
    non_empty = [s for s in spans if s.get("text", "").strip()]

    if not non_empty:
        return 0.0, False, 0.0

    font_size = max(span["font_size"] for span in spans)
    first_span_size = non_empty[0]["font_size"]

    # A block is considered bold if all non-empty spans are bold
    bold = all(is_span_bold(s) for s in non_empty)

    return font_size, bold, first_span_size


def looks_like_unnumbered_heading(
    text: str,
    font_size: float,
    is_bold: bool,
) -> bool:
    """Detect likely unnumbered headings using conservative heuristics."""
    if not text or len(text) > 120:
        return False

    if "\n" in text:
        return False

    if looks_like_numbered_heading(text):
        return False

    words = text.split()
    if len(words) > 12:
        return False

    # Unnumbered headings should be bold and not tiny (e.g. captions/table labels)
    if font_size < 9.5:
        return False

    if is_bold:
        return True

    # If not bold, it must be significantly larger than body text.
    return font_size >= 12.0


def is_false_heading(
    text: str,
    font_size: float,
    is_bold: bool,
    first_span_size: float,
) -> bool:
    """
    Reject blocks that are likely footnotes, equations, tables,
    references, URLs, preprint watermarks, or ordinary prose.
    """
    words = text.split()
    if not words:
        return True

    # Headings are short.
    if len(words) > 15:
        return True

    # Check for URLs or email addresses.
    text_lower = text.lower()
    if (
        "http://" in text_lower
        or "https://" in text_lower
        or "www." in text_lower
        or "@" in text_lower
    ):
        return True

    # Preprint watermarks or metadata.
    if "arxiv:" in text_lower or "arxiv preprint" in text_lower or "doi:" in text_lower:
        return True

    # Long numeric-heavy blocks are usually tables/equations.
    digit_count = sum(char.isdigit() for char in text)
    if digit_count > 8:
        return True

    # Common non-heading prefixes.
    if text.startswith(
        (
            "Note ",
            "See ",
            "For example,",
            "Figure ",
            "Table ",
            "Fig. ",
            "Download ",
            "ISBN ",
        )
    ):
        return True

    # Footnote superscript check: if the first span (e.g. footnote number)
    # is significantly smaller than the rest of the text block.
    if first_span_size > 0.0 and font_size > 0.0 and first_span_size < 0.9 * font_size:
        return True

    # Reject bibliography years or page numbers matched as sections (e.g. 2016., 2003., 100.)
    match = NUMBERED_HEADING_PATTERN.match(text)
    if match:
        first_num = match.group(1).split(".")[0]
        if first_num.isdigit():
            val = int(first_num)
            if val >= 50:
                return True

    return False


def should_merge(b1: dict, b2: dict) -> bool:
    """Verify if two consecutive blocks are lines of a single multi-line heading."""
    if b1["page_number"] != b2["page_number"]:
        return False

    font_size_1, is_bold_1, _ = get_block_formatting(b1)
    font_size_2, is_bold_2, _ = get_block_formatting(b2)

    # Both blocks must be bold headings of the same font size
    if not is_bold_1 or not is_bold_2:
        return False

    if abs(font_size_1 - font_size_2) > 0.2:
        return False

    # The continuation line must not start with a section number prefix
    if looks_like_numbered_heading(b2["text"].strip()):
        return False

    # Gap must be small (normally 1.0 to 1.5 lines of space)
    gap = b2["bbox"][1] - b1["bbox"][3]
    if not (-5.0 <= gap <= max(20.0, 1.8 * font_size_1)):
        return False

    # Must overlap horizontally in the same column/content area
    x_overlap = min(b1["bbox"][2], b2["bbox"][2]) > max(b1["bbox"][0], b2["bbox"][0])
    if not x_overlap:
        return False

    combined_text = b1["text"] + " " + b2["text"]
    return len(combined_text) <= 200


def merge_blocks(b1: dict, b2: dict) -> dict:
    """Merge coordinates and texts of two heading blocks."""
    merged = b1.copy()
    merged["text"] = b1["text"].strip() + " " + b2["text"].strip()
    merged["bbox"] = [
        min(b1["bbox"][0], b2["bbox"][0]),
        min(b1["bbox"][1], b2["bbox"][1]),
        max(b1["bbox"][2], b2["bbox"][2]),
        max(b1["bbox"][3], b2["bbox"][3]),
    ]
    merged["spans"] = b1["spans"] + b2["spans"]
    return merged


def split_inline_headers(blocks: list[dict]) -> list[dict]:
    """
    Split blocks starting with a bold inline heading run of at least 2 words
    and followed by regular text into separate heading and paragraph blocks.
    """
    split_blocks: list[dict] = []

    for block in blocks:
        spans = block.get("spans", [])
        if not spans:
            split_blocks.append(block)
            continue

        bold_prefix = []
        has_regular = False

        for span in spans:
            is_bold = is_span_bold(span)
            is_whitespace = not span.get("text", "").strip()

            if is_bold or (is_whitespace and bold_prefix):
                if not has_regular:
                    bold_prefix.append(span)
            else:
                has_regular = True

        if not bold_prefix or not has_regular:
            split_blocks.append(block)
            continue

        bold_text = " ".join(s["text"] for s in bold_prefix).strip()
        words = bold_text.split()

        # Require at least 2 words for the inline heading to avoid splitting
        # on paragraphs that simply start with a single bold word (like "BERT is...").
        # Also limit to 10 words to avoid splitting full bold paragraphs.
        if len(words) < 2 or len(words) > 10:
            split_blocks.append(block)
            continue

        split_index = len(bold_prefix)
        heading_spans = bold_prefix
        paragraph_spans = spans[split_index:]

        # Remove leading whitespace spans from paragraph
        while paragraph_spans and not paragraph_spans[0].get("text", "").strip():
            paragraph_spans.pop(0)

        if not paragraph_spans:
            split_blocks.append(block)
            continue

        heading_text = " ".join(s["text"] for s in heading_spans).strip()
        paragraph_text = " ".join(s["text"] for s in paragraph_spans).strip()

        heading_block = {
            "block_id": f"{block['block_id']}_h",
            "text": heading_text,
            "page_number": block["page_number"],
            "bbox": block["bbox"],
            "spans": heading_spans,
            "section_id": None,
        }

        paragraph_block = {
            "block_id": f"{block['block_id']}_p",
            "text": paragraph_text,
            "page_number": block["page_number"],
            "bbox": block["bbox"],
            "spans": paragraph_spans,
            "section_id": None,
        }

        split_blocks.append(heading_block)
        split_blocks.append(paragraph_block)

    return split_blocks


def detect_sections(
    blocks: list[dict],
) -> tuple[list[Section], list[ContentBlock]]:
    """Identify headings and build a hierarchical section structure."""
    sections: list[Section] = []
    content_blocks: list[ContentBlock] = []
    section_stack: list[Section] = []

    # Run a pre-processing pass to split inline headers first
    split_blocks = split_inline_headers(blocks)

    # Run a pre-processing pass to merge consecutive multi-line headings
    processed_blocks: list[dict] = []
    for block in split_blocks:
        if not block["text"].strip():
            continue
        if processed_blocks and should_merge(processed_blocks[-1], block):
            processed_blocks[-1] = merge_blocks(processed_blocks[-1], block)
        else:
            processed_blocks.append(block)

    seen_first_heading = False
    seen_abstract_or_intro = False

    for block in processed_blocks:
        text = block["text"].strip()

        if not text:
            continue

        is_numbered = looks_like_numbered_heading(text)
        numbered_level = get_heading_level(text)

        font_size, is_bold, first_span_size = get_block_formatting(block)

        is_unnumbered = looks_like_unnumbered_heading(
            text=text,
            font_size=font_size,
            is_bold=is_bold,
        )

        # Reject suspicious numbered blocks.
        if is_numbered and (
            is_false_heading(text, font_size, is_bold, first_span_size)
            or (not is_bold and font_size <= 11.0)
        ):
            is_numbered = False
            numbered_level = None

        if is_unnumbered and is_false_heading(
            text, font_size, is_bold, first_span_size
        ):
            is_unnumbered = False

        if is_numbered:
            level = numbered_level
            block_type = BlockType.HEADING

        elif is_unnumbered:
            level = 2
            block_type = BlockType.HEADING

        else:
            level = None
            block_type = BlockType.PARAGRAPH

        # On page 1, reject heading candidates between the paper title and the Abstract/Introduction.
        if block["page_number"] == 1:
            is_intro_or_abstract = text.lower() == "abstract" or text.startswith(
                ("1 ", "1. ")
            )
            if is_intro_or_abstract:
                seen_abstract_or_intro = True

            if block_type == BlockType.HEADING:
                if not seen_first_heading:
                    seen_first_heading = True
                elif not seen_abstract_or_intro:
                    block_type = BlockType.PARAGRAPH
                    level = None

        if block_type == BlockType.HEADING:
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()

            parent_id = section_stack[-1].section_id if section_stack else None

            section_id = f"sec_{len(sections) + 1}"

            section = Section(
                section_id=section_id,
                title=text,
                level=level,
                parent_id=parent_id,
            )

            sections.append(section)
            section_stack.append(section)

            block["section_id"] = section_id

        else:
            block["section_id"] = (
                section_stack[-1].section_id if section_stack else None
            )

        content_blocks.append(
            ContentBlock(
                block_id=block["block_id"],
                type=block_type,
                text=text,
                page_number=block["page_number"],
                section_id=block["section_id"],
                bbox=block["bbox"],
            )
        )

    return sections, content_blocks
