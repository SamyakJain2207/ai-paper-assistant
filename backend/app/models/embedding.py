from typing import Any
from pydantic import BaseModel, Field


class EmbeddedBlock(BaseModel):
    """
    Represents a structured paper text block paired with its dense vector embedding.
    Binds Phase 1 layout & hierarchy metadata with Phase 3 semantic representations.
    """

    block_id: str = Field(description="Unique block identifier from Phase 1")
    text: str = Field(description="Cleaned prose text of the content block")
    page_number: int = Field(description="Page number in original PDF")
    section_id: str | None = Field(default=None, description="Section ID if mapped")
    section_title: str | None = Field(default=None, description="Section heading title")
    embedding: list[float] = Field(description="Dense continuous vector (e.g. 384-D for all-MiniLM-L6-v2)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Paper-level metadata (title, doi, etc.)")

    @property
    def vector_dim(self) -> int:
        return len(self.embedding)
