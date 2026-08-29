from pydantic import BaseModel


class Section(BaseModel):
    section_id: str
    title: str
    level: int
    parent_id: str | None = None