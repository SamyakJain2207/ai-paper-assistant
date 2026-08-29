import re

from app.models.block import BlockType, ContentBlock
from app.models.section import Section


def is_heading(block: dict) -> bool:
    """Determine whether a PyMuPDF text block is likely a section heading."""
    text = block["text"].strip()

    if not text or len(text) > 150:
        return False

    # Ignore blocks containing multiple lines.
    if "\n" in text:
        return False

    # Headings are usually short and visually prominent.
    if block["font_size"] < 10:
        return False

    if block["font_size"] >= 12:
        return True

    # Common numbered heading patterns: 1, 1.1, 2.3.1, etc.
    if re.match(r"^\d+(\.\d+)*\.?\s+\S+", text):
        return True

    return bool(re.match(r"^\d+(\.\d+)*\.?\s+\S+", text))


def determine_heading_level(text: str) -> int:
    """Determine hierarchy level from numbered heading text."""
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", text)

    if match:
        return len(match.group(1).split("."))

    return 1


def detect_sections(
    blocks: list[dict],
) -> tuple[list[Section], list[ContentBlock]]:
    """Identify headings and build a section hierarchy."""
    sections = []
    content_blocks = []

    section_stack: list[Section] = []

    for block in blocks:
        text = block["text"].strip()

        if not text:
            continue

        if is_heading(block):
            level = determine_heading_level(text)

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

            block["type"] = BlockType.HEADING
            block["section_id"] = section_id

        else:
            current_section = section_stack[-1] if section_stack else None

            block["type"] = BlockType.PARAGRAPH
            block["section_id"] = (
                current_section.section_id if current_section else None
            )

        content_blocks.append(
            ContentBlock(
                block_id=block["block_id"],
                type=block["type"],
                text=text,
                page_number=block["page_number"],
                section_id=block["section_id"],
                bbox=block["bbox"],
            )
        )

    return sections, content_blocks