from enum import Enum

from pydantic import BaseModel


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"


class ContentBlock(BaseModel):
    block_id: str
    type: BlockType
    text: str
    page_number: int
    section_id: str | None = None
    bbox: list[float] | None = None