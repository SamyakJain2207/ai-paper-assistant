import re
from typing import Any
from pydantic import BaseModel, Field

from app.models.chunk import Chunk
from app.models.document import ContentBlock, Document
from app.services.nlp.preprocessing import clean_paper_text
from app.services.nlp.sentence_scoring import split_sentences


class ChunkingConfig(BaseModel):
    """Configuration settings for semantic production chunking."""

    max_tokens: int = Field(default=300, description="Target maximum tokens per chunk")
    overlap_tokens: int = Field(
        default=50, description="Target token overlap between consecutive chunks in a section"
    )
    min_tokens: int = Field(
        default=30, description="Minimum tokens for a chunk; smaller units are merged"
    )
    prepend_headers: bool = Field(
        default=True,
        description="Whether to inject contextual parent header into embed_text",
    )


def estimate_token_count(text: str) -> int:
    """
    Fast and robust token count estimator for English academic text.
    In standard subword tokenization (WordPiece/BPE), 1 word ~ 1.3 tokens.
    """
    words = len(text.strip().split())
    if words == 0:
        return 0
    return max(1, int(words * 1.3))


class SemanticChunker:
    """
    Production-grade section-aware semantic chunker for academic papers.
    - Preserves logical section boundaries (no cross-section bleed).
    - Respects grammatical sentence boundaries (no mid-sentence cuts).
    - Enforces sliding-window token overlap for semantic continuity.
    - Injects contextual parent headers for high-precision vector embeddings.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Transform a structured Document into a list of production Chunks.
        Groups content blocks by section, splits into token-bounded sentence windows,
        and generates dual representations (clean content_text vs contextual embed_text).
        """
        paper_id = (
            document.metadata.arxiv_id
            or document.metadata.doi
            or document.metadata.title.lower().replace(" ", "_")[:50]
        )
        paper_id = re.sub(r"[^a-zA-Z0-9_-]", "_", paper_id)
        paper_title = document.metadata.title or "Untitled Research Paper"

        # 1. Group content blocks by section title to avoid cross-section bleed
        section_groups: list[dict[str, Any]] = []
        current_group: dict[str, Any] | None = None

        for block in document.content_blocks:
            cleaned = clean_paper_text(block.text)
            if not cleaned or len(cleaned.split()) < 3:
                continue

            sec_title = block.section_title or "Main Content"
            sec_id = block.section_id

            if current_group is None or current_group["section_title"] != sec_title:
                if current_group is not None:
                    section_groups.append(current_group)
                current_group = {
                    "section_title": sec_title,
                    "section_id": sec_id,
                    "blocks": [block],
                }
            else:
                current_group["blocks"].append(block)

        if current_group is not None:
            section_groups.append(current_group)

        # 2. Chunk each section independently
        all_chunks: list[Chunk] = []

        for sec_idx, group in enumerate(section_groups, start=1):
            sec_title = group["section_title"]
            sec_id = group["section_id"]
            blocks: list[ContentBlock] = group["blocks"]

            sec_chunks = self._chunk_section_blocks(
                blocks=blocks,
                sec_idx=sec_idx,
                sec_title=sec_title,
                sec_id=sec_id,
                paper_id=paper_id,
                paper_title=paper_title,
                doc_metadata=document.metadata.model_dump(),
            )
            all_chunks.extend(sec_chunks)

        return all_chunks

    def _chunk_section_blocks(
        self,
        blocks: list[ContentBlock],
        sec_idx: int,
        sec_title: str,
        sec_id: str | None,
        paper_id: str,
        paper_title: str,
        doc_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk all sentences within a single section using sliding-window overlap."""
        # 1. Detect section type (e.g., references/bibliography vs prose)
        is_reference = bool(
            re.search(r"\b(references?|bibliography|works cited)\b", sec_title.lower())
        )

        # 2. Build complete parent section text for parent-child retrieval
        parent_section_text = " ".join(
            clean_paper_text(b.text) for b in blocks if clean_paper_text(b.text)
        )

        # 3. Collect all sentences paired with block provenance and page numbers
        sentence_units: list[dict[str, Any]] = []

        for block in blocks:
            cleaned_block = clean_paper_text(block.text)
            sentences = split_sentences(cleaned_block)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                sentence_units.append({
                    "text": s_clean,
                    "tokens": estimate_token_count(s_clean),
                    "page_number": block.page_number,
                    "block_id": block.block_id,
                })

        if not sentence_units:
            return []

        chunks: list[Chunk] = []
        chunk_idx = 1
        unit_idx = 0
        total_units = len(sentence_units)

        while unit_idx < total_units:
            # Check remaining tokens: if remaining units are too small and we already have a chunk,
            # merge them into the previous chunk to avoid creating orphan fragments
            remaining_tokens = sum(u["tokens"] for u in sentence_units[unit_idx:])
            if chunks and remaining_tokens < self.config.min_tokens:
                # Merge into the last chunk
                prev_chunk = chunks[-1]
                remaining_text = " ".join(u["text"] for u in sentence_units[unit_idx:])
                new_content = f"{prev_chunk.content_text} {remaining_text}"
                prev_chunk.content_text = new_content
                prev_chunk.token_count = estimate_token_count(new_content)
                prev_chunk.char_count = len(new_content)
                for u in sentence_units[unit_idx:]:
                    if u["page_number"] not in prev_chunk.page_numbers:
                        prev_chunk.page_numbers.append(u["page_number"])
                    if u["block_id"] not in prev_chunk.source_block_ids:
                        prev_chunk.source_block_ids.append(u["block_id"])
                prev_chunk.page_numbers.sort()
                prev_chunk.source_block_ids.sort()

                # Rebuild embed_text
                if self.config.prepend_headers:
                    pages_str = (
                        f"Page {prev_chunk.page_numbers[0]}"
                        if len(prev_chunk.page_numbers) == 1
                        else f"Pages {prev_chunk.page_numbers[0]}-{prev_chunk.page_numbers[-1]}"
                    )
                    prev_chunk.embed_text = (
                        f"[Document: {paper_title} | Section: {sec_title} | {pages_str}]\n\n{new_content}"
                    )
                else:
                    prev_chunk.embed_text = new_content
                break

            current_sentences: list[str] = []
            current_tokens = 0
            current_pages: set[int] = set()
            current_block_ids: set[str] = set()

            # Greedily accumulate sentences up to max_tokens
            while unit_idx < total_units:
                unit = sentence_units[unit_idx]
                unit_tokens = unit["tokens"]

                # If single sentence exceeds max_tokens on its own, include it solo
                if not current_sentences and unit_tokens >= self.config.max_tokens:
                    current_sentences.append(unit["text"])
                    current_tokens += unit_tokens
                    current_pages.add(unit["page_number"])
                    current_block_ids.add(unit["block_id"])
                    unit_idx += 1
                    break

                if current_tokens + unit_tokens > self.config.max_tokens:
                    break

                current_sentences.append(unit["text"])
                current_tokens += unit_tokens
                current_pages.add(unit["page_number"])
                current_block_ids.add(unit["block_id"])
                unit_idx += 1

            if not current_sentences:
                break

            content_text = " ".join(current_sentences)
            token_count = estimate_token_count(content_text)
            char_count = len(content_text)
            sorted_pages = sorted(list(current_pages))

            # Differentiate general prose from bibliography/references
            chunk_type = "reference" if is_reference else "prose"

            # Build contextual embed header
            if self.config.prepend_headers:
                pages_str = (
                    f"Page {sorted_pages[0]}"
                    if len(sorted_pages) == 1
                    else f"Pages {sorted_pages[0]}-{sorted_pages[-1]}"
                )
                header = (
                    f"[Document: {paper_title} | Section: {sec_title} | {pages_str}]\n\n"
                )
                embed_text = header + content_text
            else:
                embed_text = content_text

            chunk_id = f"{paper_id}_s{sec_idx}_c{chunk_idx}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content_text=content_text,
                    embed_text=embed_text,
                    token_count=token_count,
                    char_count=char_count,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    section_title=sec_title,
                    section_id=sec_id,
                    page_numbers=sorted_pages,
                    source_block_ids=sorted(list(current_block_ids)),
                    metadata=doc_metadata,
                    parent_section_text=parent_section_text,
                    chunk_type=chunk_type,
                    is_reference=is_reference,
                )
            )
            chunk_idx += 1

            # Determine sliding window overlap for the next chunk
            if unit_idx < total_units and self.config.overlap_tokens > 0:
                accumulated_overlap = 0
                overlap_steps = 0
                lookback_idx = unit_idx - 1

                while lookback_idx >= 0:
                    cand_tokens = sentence_units[lookback_idx]["tokens"]
                    if accumulated_overlap + cand_tokens <= self.config.overlap_tokens:
                        accumulated_overlap += cand_tokens
                        overlap_steps += 1
                        lookback_idx -= 1
                    else:
                        break

                if overlap_steps > 0 and overlap_steps < len(current_sentences):
                    unit_idx = unit_idx - overlap_steps

        return chunks
