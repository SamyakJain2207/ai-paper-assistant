from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.block import BlockType
from app.models.document import Document
from app.models.embedding import EmbeddedBlock
from app.services.nlp.preprocessing import clean_paper_text

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceEmbeddingService:
    """
    Bi-encoder dense representation service powered by Sentence Transformers.
    Converts sentences, paper passages, and queries into contextual dense vectors
    specifically optimized for cosine similarity and semantic search.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None
        self._device = device

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the underlying SentenceTransformer neural model."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality (e.g. 384 for all-MiniLM-L6-v2)."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    def embed_text(self, text: str) -> np.ndarray:
        """
        Compute dense embedding vector for a single string.
        Returns a 1D NumPy float32 array of shape (dimension,).
        """
        cleaned = clean_paper_text(text)
        if not cleaned:
            # Return zero vector for empty text
            return np.zeros(self.dimension, dtype=np.float32)

        vec = self.model.encode(
            cleaned,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute dense embedding vectors for a batch of strings.
        Returns a 2D NumPy float32 array of shape (N, dimension).
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        cleaned_texts = [clean_paper_text(t) for t in texts]
        # Replace empty strings with a single space to avoid empty encoding issues
        safe_texts = [t if t else " " for t in cleaned_texts]

        vectors = self.model.encode(
            safe_texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_document_blocks(
        self,
        document: Document,
        min_words: int = 5,
        batch_size: int = 32,
    ) -> list[EmbeddedBlock]:
        """
        Transform all content blocks from a parsed Document into EmbeddedBlocks.
        Binds Phase 1 layout, bounding boxes, and section hierarchy with dense embeddings.
        """
        # Build section lookup table: section_id -> section_title
        section_titles: dict[str, str] = {
            sec.section_id: sec.title for sec in document.sections
        }

        # Filter and collect candidate blocks
        candidate_blocks = []
        texts_to_embed = []

        for block in document.content_blocks:
            cleaned = clean_paper_text(block.text)
            words = cleaned.split()
            # Filter out tiny fragments, isolated math symbols, or empty blocks
            if len(words) >= min_words:
                candidate_blocks.append(block)
                texts_to_embed.append(cleaned)

        if not candidate_blocks:
            return []

        # Batch encode all candidate texts
        vectors = self.embed_batch(texts_to_embed, batch_size=batch_size)

        paper_meta = {
            "title": document.metadata.title,
            "doi": document.metadata.doi,
            "total_pages": document.metadata.total_pages,
            "doc_id": document.doc_id,
        }

        embedded_blocks: list[EmbeddedBlock] = []
        for block, vec in zip(candidate_blocks, vectors, strict=True):
            sec_title = section_titles.get(block.section_id) if block.section_id else None

            embedded_blocks.append(
                EmbeddedBlock(
                    block_id=block.block_id,
                    text=clean_paper_text(block.text),
                    page_number=block.page_number,
                    section_id=block.section_id,
                    section_title=sec_title,
                    embedding=vec.tolist(),
                    metadata=paper_meta,
                )
            )

        return embedded_blocks

    @staticmethod
    def compute_similarity(vec1: np.ndarray | list[float], vec2: np.ndarray | list[float]) -> float:
        """
        Compute cosine similarity between two dense vectors.
        Because vectors are L2-normalized upon embedding, dot product == cosine similarity.
        """
        u = np.asarray(vec1, dtype=np.float32)
        v = np.asarray(vec2, dtype=np.float32)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0

        return float(np.dot(u, v) / (norm_u * norm_v))

    def search_blocks(
        self,
        query: str,
        blocks: list[EmbeddedBlock],
        top_k: int = 5,
    ) -> list[tuple[EmbeddedBlock, float]]:
        """
        Perform in-memory dense cosine similarity search over a collection of EmbeddedBlocks.
        Ranks blocks by semantic relevance to the query.
        """
        if not blocks or not query.strip():
            return []

        query_vec = self.embed_text(query)

        # Convert block embeddings to 2D matrix
        block_matrix = np.array([b.embedding for b in blocks], dtype=np.float32)

        # Compute dot product against all normalized block vectors: shape (N,)
        similarities = np.dot(block_matrix, query_vec)

        # Retrieve top_k indices sorted descending
        k = min(top_k, len(blocks))
        top_indices = np.argsort(similarities)[::-1][:k]

        return [(blocks[idx], float(similarities[idx])) for idx in top_indices]
