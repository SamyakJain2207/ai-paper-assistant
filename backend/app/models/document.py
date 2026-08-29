from app.models.block import ContentBlock
from app.models.metadata import PaperMetadata
from app.models.section import Section
from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    metadata: PaperMetadata
    sections: list[Section] = Field(default_factory=list)
    content_blocks: list[ContentBlock] = Field(default_factory=list)