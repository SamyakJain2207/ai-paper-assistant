from datetime import date

from pydantic import BaseModel


class PaperMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = []
    publication_date: date | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    total_pages: int | None = None