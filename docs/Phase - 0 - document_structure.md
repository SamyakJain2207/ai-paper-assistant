# Phase 0 — Document Data Modeling & Architecture

---

## 1. Motivation: Why Raw PDF Text Fails

A PDF is not a structured text document; it is a **visual layout specification** designed for rendering ink or pixels on a 2D canvas. 

Internally, a PDF does not store concepts like *"paragraphs"*, *"headings"*, *"sections"*, or *"footnotes"*. Instead, it contains low-level character draw instructions:
- `BT /F1 12.0 Tf 72.0 712.0 Td (Introduction) Tj ET`

If you extract raw text from a PDF without an explicit data model:
1. **Headings get lost in body text**: A section heading like `3.2 Attention` is merged directly with the first sentence of the following paragraph.
2. **Page headers and footers contaminate text**: Running headers (*"arXiv:1706.03762v7 [cs.CL] 6 Dec 2017"*) and page numbers break sentences mid-thought.
3. **Multi-column layouts interleave**: Text from Column 1 and Column 2 on the same page often get mixed into alternating lines.
4. **Metadata is missing or inaccurate**: The PDF's embedded author and title fields are frequently blank, generic (e.g. *"untitled"* or the filename), or truncated.

Before building any parser (Phase 1) or NLP algorithms (Phase 2), we had to define a **structured semantic data model** that represents a research paper as an intelligent, queryable entity.

---

## 2. The Four Core Data Models (`backend/app/models/`)

We implemented four decoupled, strongly-typed Pydantic models:

```text
                           Document
                     ┌────────┴────────┐
                     ▼                 ▼
               PaperMetadata     sections: list[Section]
                                       │
                                       ▼
                          content_blocks: list[ContentBlock]
```

### 1. `PaperMetadata` (`backend/app/models/metadata.py`)
Stores the academic identity of the paper:
```python
class PaperMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = []
    publication_date: date | None = None
    doi: str | None = None
    total_pages: int | None = None
```
- **Design Rationale**: A research paper is defined by its citation identity. Having structured metadata allows the assistant to attribute quotes, generate bibtex citations, filter by author, and verify publications via external APIs (Crossref / arXiv).

### 2. `Section` (`backend/app/models/section.py`)
Represents the structural tree of the paper:
```python
class Section(BaseModel):
    section_id: str
    title: str
    level: int
    parent_id: str | None = None
```
- **Design Rationale**: Research papers are not flat sequences of paragraphs; they are hierarchical trees:
  - `Level 1`: Main sections (`1 Introduction`, `3 Model Architecture`, `6 Conclusion`)
  - `Level 2`: Subsections (`3.1 Encoder and Decoder Stacks`, `3.2 Attention`)
  - `Level 3`: Sub-subsections (`3.2.1 Scaled Dot-Product Attention`)
- By recording `level` and `parent_id`, downstream algorithms can reason about scope (e.g. aggregating all subsections of *Model Architecture* or extracting keywords at any level of granularity).

### 3. `ContentBlock` & `BlockType` (`backend/app/models/block.py`)
The fundamental atomic unit of content extracted from the document:
```python
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
```
- **Design Rationale**:
  - **`BlockType` distinction**: Crucial for downstream intelligence. If a heading is marked as `HEADING`, the summarizer knows never to pick it as a body sentence, and the keyword extractor treats it as structural metadata.
  - **`section_id` foreign key**: Links every paragraph block back to its parent `Section`.
  - **`page_number` & `bbox`**: Preserves physical coordinates `[x0, y0, x1, y1]`. This makes the data model future-proof for UI features like PDF highlighting, bounding-box overlays, and multi-modal vision models.

### 4. `Document` (`backend/app/models/document.py`)
The master container uniting the entire paper:
```python
class Document(BaseModel):
    doc_id: str
    metadata: PaperMetadata
    sections: list[Section] = Field(default_factory=list)
    content_blocks: list[ContentBlock] = Field(default_factory=list)
```
- **Deterministic `doc_id`**: Generated via a SHA-256 hash of the PDF bytes (`generate_doc_id(pdf_path)`). This ensures document identity is content-addressable and stable across restarts and re-indexing.

---

## 3. How Downstream Phases Build on This Foundation

This data structure is the backbone of the entire project lifecycle:

| Phase | How It Interacts with the `Document` Model |
| :--- | :--- |
| **Phase 1 (Parser)** | Populates `sections` and `content_blocks` using font size, bold flags, numbering regexes, and run-in header splitting. Enriches `metadata` via arXiv/Crossref APIs. |
| **Phase 2 (Classical NLP)** | Uses `BlockType.PARAGRAPH` to gather pure text, groups paragraphs by `section_id` to compute section-level TF-IDF and keywords, and extracts candidate sentences for summarization. |
| **Phase 3 (Dense Vectors & Search)** | Chunks text by `ContentBlock` and `Section` boundaries (rather than arbitrary character counts), attaching `section_id` and `title` as metadata in vector databases. |
| **Phase 4 (RAG & LLM Assistant)** | Retrieves exact content blocks with precise citations: *"According to Section 3.2 on Page 4..."*, enabling hallucination-free grounded answers. |
