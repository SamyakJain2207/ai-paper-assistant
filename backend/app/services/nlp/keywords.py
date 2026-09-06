from collections import defaultdict
from app.models.block import BlockType
from app.models.document import Document
from app.services.nlp.preprocessing import clean_paper_text
from app.services.nlp.text_cleaning import clean_text, process_text
from app.services.nlp.tfidf import TFIDFModel

METADATA_TERMS = {
    "arxiv", "preprint", "doi", "http", "https", "et", "al",
    "figure", "table", "fig", "author", "university", "google",
    "department", "research", "equal", "contribution", "email",
}


def extract_keywords_from_scores(
    tfidf_scores: dict[str, float], top_k: int = 10, min_len: int = 3
) -> list[tuple[str, float]]:
    """Extract and sort top K domain keywords from TF-IDF scores."""
    filtered = [
        (term, score)
        for term, score in tfidf_scores.items()
        if len(term) >= min_len
        and not term.isdigit()
        and score > 0
        and term.lower() not in METADATA_TERMS
        and "@" not in term
    ]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    return filtered[:top_k]


def is_author_or_header_block(text: str, page_number: int, paper_title: str | None) -> bool:
    """Identify author affiliation, email, or header noise blocks on page 1."""
    if page_number > 1:
        return False

    text_lower = text.lower()
    if "@" in text or "google.com" in text_lower or "toronto.edu" in text_lower:
        return True

    if paper_title and paper_title.lower() in text_lower:
        return True

    # Check for heavy affiliation keywords on page 1
    affiliation_words = ["google brain", "google research", "university", "department of"]
    return any(w in text_lower for w in affiliation_words)


def get_section_paragraphs(document: Document) -> dict[str, list[str]]:
    """
    Extract body paragraphs grouped by section title.
    Only includes BlockType.PARAGRAPH blocks.
    Excludes title headers and author affiliations.
    """
    section_title_map = {sec.section_id: sec.title for sec in document.sections}
    paper_title = document.metadata.title or ""
    section_paras: dict[str, list[str]] = defaultdict(list)

    for block in document.content_blocks:
        if block.type != BlockType.PARAGRAPH:
            continue

        if is_author_or_header_block(block.text, block.page_number, paper_title):
            continue

        title = section_title_map.get(block.section_id)
        # Skip section if it's identical to the paper title
        if title and paper_title and title.strip().lower() == paper_title.strip().lower():
            continue

        if not title:
            title = "Abstract" if block.page_number == 1 else "General"

        cleaned_text = clean_paper_text(block.text)
        if len(cleaned_text.split()) >= 4:
            section_paras[title].append(cleaned_text)

    return section_paras


def extract_section_keywords(
    document: Document, top_k: int = 5
) -> dict[str, list[tuple[str, float]]]:
    """Extract top keywords for each section using section-level TF-IDF."""
    section_paras = get_section_paragraphs(document)
    if not section_paras:
        return {}

    section_tokens: dict[str, list[str]] = {}
    for title, paras in section_paras.items():
        doc_spacy = process_text(" ".join(paras))
        tokens = clean_text(doc_spacy)
        if tokens:
            section_tokens[title] = tokens

    if not section_tokens:
        return {}

    model = TFIDFModel(smooth=True).fit(list(section_tokens.values()))

    return {
        title: extract_keywords_from_scores(model.transform(tokens), top_k=top_k)
        for title, tokens in section_tokens.items()
    }


def extract_document_keywords(
    document: Document, top_k: int = 10
) -> list[tuple[str, float]]:
    """Extract top keywords across the entire paper."""
    section_paras = get_section_paragraphs(document)
    if not section_paras:
        return []

    section_tokens = []
    all_tokens = []
    for paras in section_paras.values():
        tokens = clean_text(process_text(" ".join(paras)))
        if tokens:
            section_tokens.append(tokens)
            all_tokens.extend(tokens)

    if not section_tokens:
        return []

    model = TFIDFModel(smooth=True).fit(section_tokens)
    doc_scores = model.transform(all_tokens)
    return extract_keywords_from_scores(doc_scores, top_k=top_k)
