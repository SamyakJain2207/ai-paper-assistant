from typing import Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Represents a production chunk derived from one or more ContentBlocks within a section.
    Includes contextual embedding text, clean content text, token metrics, and provenance.
    """

    chunk_id: str = Field(
        description="Deterministic unique chunk identifier: {paper_id}_s{sec_idx}_c{chunk_idx}"
    )
    content_text: str = Field(
        description="Clean, raw passage text intended for LLM context injection"
    )
    embed_text: str = Field(
        description="Contextually-anchored text with injected headers used for embedding generation"
    )
    token_count: int = Field(description="Estimated token length of content_text")
    char_count: int = Field(description="Total character count of content_text")
    paper_id: str = Field(description="Unique identifier of the source paper")
    paper_title: str = Field(description="Title of the source research paper")
    section_title: str | None = Field(default=None, description="Section heading hierarchy path")
    section_id: str | None = Field(default=None, description="Section ID if available")
    page_numbers: list[int] = Field(
        default_factory=list, description="All unique PDF page numbers spanned by this chunk"
    )
    source_block_ids: list[str] = Field(
        default_factory=list, description="IDs of original Phase 1 ContentBlocks aggregated in this chunk"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata (authors, year, doi, etc.)"
    )
    parent_section_text: str | None = Field(
        default=None, description="Full text of parent section for parent-child context expansion"
    )
    chunk_type: str = Field(
        default="prose", description="Semantic content type: prose, table, equation, or reference"
    )
    is_reference: bool = Field(
        default=False, description="Whether this chunk belongs to bibliography or references"
    )

    @property
    def primary_page(self) -> int:
        """First page where this chunk appears."""
        return self.page_numbers[0] if self.page_numbers else 1
