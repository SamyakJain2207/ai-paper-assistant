from pathlib import Path
from typing import Any
import chromadb
from chromadb.api.models.Collection import Collection

from app.models.document import Document
from app.models.embedding import EmbeddedBlock
from app.models.search import SearchFilter, SearchResult
from app.services.embeddings.sentence_transformer import SentenceEmbeddingService


def sanitize_chroma_metadata(raw_metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    Sanitize metadata for ChromaDB compatibility.
    Chroma requires all metadata values to be primitive types (str, int, float, bool).
    Nested lists are joined into comma-separated strings; None values are omitted.
    """
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in raw_metadata.items():
        if value is None:
            continue
        if isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = value
        elif isinstance(value, (list, tuple, set)):
            sanitized[key] = ", ".join(str(item) for item in value)
        else:
            sanitized[key] = str(value)
    return sanitized


class ChromaService:
    """
    Persistent ChromaDB vector database service for academic research papers.
    Handles embedding storage, HNSW indexing (cosine metric), metadata-filtered
    retrieval, and paper lifecycle management (atomic deletes, collection stats).
    """

    DEFAULT_COLLECTION = "paper_blocks"

    def __init__(
        self,
        persist_dir: Path | str = "data/chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize embedded persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def total_blocks(self) -> int:
        """Return total number of indexed blocks across all papers."""
        return self.collection.count()

    def index_embedded_blocks(
        self,
        blocks: list[EmbeddedBlock],
        paper_id: str,
        paper_title: str | None = None,
    ) -> int:
        """
        Upsert embedded blocks into the collection with sanitized metadata.
        Idempotent: updates existing blocks with the same block_id without duplicating.
        """
        if not blocks:
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for b in blocks:
            ids.append(b.block_id)
            embeddings.append(b.embedding)
            documents.append(b.text)

            # Build sanitized metadata dictionary
            meta = sanitize_chroma_metadata(b.metadata)
            meta["paper_id"] = paper_id
            if paper_title:
                meta["paper_title"] = paper_title
            elif "title" in meta:
                meta["paper_title"] = meta["title"]
            else:
                meta["paper_title"] = paper_id

            meta["page_number"] = b.page_number
            if b.section_title:
                meta["section_title"] = b.section_title
            if b.section_id:
                meta["section_id"] = b.section_id

            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def index_document(
        self,
        document: Document,
        embedder: SentenceEmbeddingService,
        min_words: int = 8,
    ) -> list[EmbeddedBlock]:
        """
        End-to-end ingestion helper: takes a Phase 1 parsed Document,
        computes dense vectors via SentenceEmbeddingService, and indexes
        all blocks into ChromaDB.
        """
        paper_id = (
            getattr(document.metadata, "arxiv_id", None)
            or document.metadata.doi
            or ((document.metadata.title or "paper").lower().replace(" ", "_")[:50])
        )
        blocks = embedder.embed_document_blocks(document, min_words=min_words)
        self.index_embedded_blocks(
            blocks=blocks,
            paper_id=paper_id,
            paper_title=document.metadata.title,
        )
        return blocks

    def query_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve top-k nearest neighbor blocks using cosine distance.
        Converts raw cosine distance into calibrated cosine similarity (1.0 - distance).
        Accepts raw 'where' dict or structured 'SearchFilter'.
        """
        if self.total_blocks == 0:
            return []

        # Resolve where condition
        effective_where = where
        if search_filter is not None:
            effective_where = search_filter.to_chroma_where()

        n_results = min(top_k, self.total_blocks)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if effective_where:
            query_kwargs["where"] = effective_where

        results = self.collection.query(**query_kwargs)

        search_results: list[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, block_id in enumerate(ids):
            dist = float(distances[i]) if i < len(distances) else 0.0
            # Convert cosine distance (0.0 to 2.0) to similarity (1.0 to -1.0)
            sim_score = round(1.0 - dist, 4)

            meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
            doc_text = docs[i] if i < len(docs) else ""

            search_results.append(
                SearchResult(
                    block_id=block_id,
                    text=doc_text,
                    score=sim_score,
                    distance=round(dist, 4),
                    page_number=int(meta.get("page_number", 1)),
                    section_title=meta.get("section_title"),
                    paper_id=str(meta.get("paper_id", "unknown")),
                    metadata=meta,
                )
            )

        return search_results

    def query_by_text(
        self,
        query_text: str,
        embedder: SentenceEmbeddingService,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """Convenience method: embeds raw query string and runs query_similar."""
        q_vec = embedder.embed_text(query_text)
        return self.query_similar(
            query_embedding=q_vec,
            top_k=top_k,
            where=where,
            search_filter=search_filter,
        )

    def delete_paper(self, paper_id: str) -> int:
        """
        Atomic paper deletion: purges all vectors, text passages, and metadata
        associated with the given paper_id. Prevents zombie chunks.
        """
        existing = self.collection.get(where={"paper_id": paper_id})
        count = len(existing["ids"]) if existing and "ids" in existing else 0
        if count > 0:
            self.collection.delete(where={"paper_id": paper_id})
        return count

    def list_indexed_papers(self) -> list[dict[str, Any]]:
        """
        Inspect collection and return a structured summary of all indexed papers,
        their titles, total block counts, and unique page coverage.
        """
        records = self.collection.get(include=["metadatas"])
        if not records or not records["metadatas"]:
            return []

        papers: dict[str, dict[str, Any]] = {}
        for meta in records["metadatas"]:
            pid = str(meta.get("paper_id", "unknown"))
            if pid not in papers:
                papers[pid] = {
                    "paper_id": pid,
                    "title": meta.get("paper_title") or meta.get("title") or pid,
                    "total_blocks": 0,
                    "pages": set(),
                }
            papers[pid]["total_blocks"] += 1
            if "page_number" in meta:
                papers[pid]["pages"].add(meta["page_number"])

        return [
            {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "total_blocks": p["total_blocks"],
                "total_pages": len(p["pages"]),
            }
            for p in papers.values()
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic metrics of the vector database."""
        return {
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "total_blocks": self.total_blocks,
            "metric": "cosine",
            "indexed_papers_count": len(self.list_indexed_papers()),
        }

    def clear_collection(self) -> None:
        """Wipe all records in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
