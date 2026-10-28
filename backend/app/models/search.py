from typing import Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """
    Represents a single retrieved block from the vector database.
    Includes both raw distance and calibrated cosine similarity.
    """

    block_id: str = Field(description="Unique identifier for the content block")
    text: str = Field(description="Raw text content of the retrieved chunk")
    score: float = Field(description="Calibrated cosine similarity score [-1.0, 1.0]")
    distance: float = Field(description="Raw vector distance metric from ChromaDB")
    page_number: int = Field(description="Page number in the original PDF")
    section_title: str | None = Field(default=None, description="Section heading if available")
    paper_id: str = Field(description="Identifier for the source research paper")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flattened paper-level and chunk-level metadata attributes",
    )


class SearchFilter(BaseModel):
    """
    Structured filter options for querying the vector database.
    Can be translated into ChromaDB's 'where' clause format.
    """

    paper_id: str | None = Field(default=None, description="Scope query to a specific paper")
    section_title: str | None = Field(default=None, description="Scope query to a specific section")
    min_page: int | None = Field(default=None, description="Minimum page number constraint")
    max_page: int | None = Field(default=None, description="Maximum page number constraint")

    def to_chroma_where(self) -> dict[str, Any] | None:
        """
        Translates structured filter fields into a ChromaDB 'where' dictionary.
        Handles single conditions and composite $and expressions.
        """
        conditions: list[dict[str, Any]] = []

        if self.paper_id is not None:
            conditions.append({"paper_id": {"$eq": self.paper_id}})

        if self.section_title is not None:
            conditions.append({"section_title": {"$eq": self.section_title}})

        if self.min_page is not None and self.max_page is not None:
            if self.min_page == self.max_page:
                conditions.append({"page_number": {"$eq": self.min_page}})
            else:
                conditions.append({"page_number": {"$gte": self.min_page}})
                conditions.append({"page_number": {"$lte": self.max_page}})
        elif self.min_page is not None:
            conditions.append({"page_number": {"$gte": self.min_page}})
        elif self.max_page is not None:
            conditions.append({"page_number": {"$lte": self.max_page}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


class HybridSearchResult(BaseModel):
    """
    Represents a unified retrieval hit resulting from Reciprocal Rank Fusion (RRF)
    of dense vector search (ChromaDB) and sparse lexical search (BM25).
    """

    chunk_id: str = Field(description="Unique deterministic chunk ID")
    content_text: str = Field(description="Clean passage text for LLM consumption")
    embed_text: str = Field(description="Enriched text with prepended document and section headers")
    rrf_score: float = Field(description="Reciprocal Rank Fusion score: sum(w / (k + rank))")
    dense_rank: int | None = Field(default=None, description="1-based rank in dense search list")
    sparse_rank: int | None = Field(default=None, description="1-based rank in BM25 search list")
    dense_score: float | None = Field(default=None, description="Raw cosine similarity [-1.0, 1.0]")
    sparse_score: float | None = Field(default=None, description="Raw BM25Okapi relevance score")
    cross_score: float | None = Field(
        default=None,
        description="Cross-encoder relevance score; set only when the hit was reranked",
    )
    final_score: float | None = Field(
        default=None,
        description="Score that determines final order: cross_score if reranked, else rrf_score",
    )
    paper_id: str = Field(description="Source paper identifier")
    paper_title: str = Field(description="Source paper title")
    section_title: str | None = Field(default=None, description="Section heading")
    page_numbers: list[int] = Field(default_factory=list, description="Pages spanned by chunk")
    parent_section_text: str | None = Field(default=None, description="Full parent section text")
    is_reference: bool = Field(default=False, description="Whether chunk is a bibliography entry")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata payload")