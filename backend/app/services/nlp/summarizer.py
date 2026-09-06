from collections import defaultdict
from app.models.document import Document
from app.services.nlp.keywords import (
    extract_document_keywords,
    get_section_paragraphs,
)
from app.services.nlp.sentence_scoring import (
    ScoredSentence,
    score_sentences,
    split_sentences,
)

EXCLUDED_SECTIONS = {
    "reference", "bibliography", "acknowledgement",
    "acknowledgment", "appendix", "appendices", "table", "figure"
}

# Standard summary-rich sections in research papers
SUMMARY_SECTIONS = ["abstract", "introduction", "conclusion"]

ACKNOWLEDGEMENT_MARKERS = ["acknowledgement", "grateful to", "thank", "supported by", "fellowship"]


def is_acknowledgement_sentence(text: str) -> bool:
    """Check if a sentence belongs to inline acknowledgements."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in ACKNOWLEDGEMENT_MARKERS)


def generate_extractive_summary(
    document: Document,
    num_sentences: int = 5,
    target_sections: list[str] | None = None,
    max_per_section: int = 2,
) -> str:
    """
    Generate an informative extractive summary of a research paper.

    Strategy:
    1. Compute top paper-wide domain keywords (TF-IDF salience).
    2. Focus candidate sentences on key synthesis sections (Abstract, Intro, Conclusion)
       or user-specified target sections.
    3. Score sentences using keyword density + indicative cue phrases ("we propose", "outperforms").
    4. Select top sentences enforcing section diversity across sections.
    5. Re-sort chronologically to preserve narrative coherence.
    """
    section_paras = get_section_paragraphs(document)
    if not section_paras:
        return ""

    # 1. Compute paper-wide salient keywords as scoring weights
    doc_keywords = extract_document_keywords(document, top_k=25)
    if not doc_keywords:
        return ""
    keyword_weights = dict(doc_keywords)

    # 2. Determine eligible sections
    # If no target_sections specified, default to summary-rich sections if present
    desired_sections = target_sections
    if not desired_sections:
        has_summary_sections = any(
            any(s in title.lower() for s in SUMMARY_SECTIONS)
            for title in section_paras.keys()
        )
        if has_summary_sections:
            desired_sections = SUMMARY_SECTIONS

    # 3. Collect candidate prose sentences
    candidates: list[tuple[str, str, int]] = []
    order_idx = 0

    for title, paras in section_paras.items():
        title_lower = title.lower()
        if any(ex in title_lower for ex in EXCLUDED_SECTIONS):
            continue

        if desired_sections and not any(ds in title_lower for ds in desired_sections):
            continue

        for para in paras:
            for sent in split_sentences(para):
                if not is_acknowledgement_sentence(sent):
                    candidates.append((sent, title, order_idx))
                order_idx += 1

    # Fallback to all sections if target filter returned nothing
    if not candidates:
        for title, paras in section_paras.items():
            if any(ex in title.lower() for ex in EXCLUDED_SECTIONS):
                continue
            for para in paras:
                for sent in split_sentences(para):
                    if not is_acknowledgement_sentence(sent):
                        candidates.append((sent, title, order_idx))
                    order_idx += 1

    if not candidates:
        return ""

    # 4. Score candidate sentences using paper-wide keyword salience
    scored: list[ScoredSentence] = []
    for text, title, idx in candidates:
        scored.extend(
            score_sentences([text], keyword_weights, section_title=title, start_index=idx)
        )

    if not scored:
        return ""

    # 5. Rank by importance score descending
    scored.sort(key=lambda s: s.score, reverse=True)

    # 6. Select top sentences with section diversity
    selected: list[ScoredSentence] = []
    sec_counts: dict[str, int] = defaultdict(int)

    for s in scored:
        title = s.section_title or "General"
        if max_per_section and sec_counts[title] >= max_per_section:
            continue
        selected.append(s)
        sec_counts[title] += 1
        if len(selected) >= num_sentences:
            break

    # If diversity limit kept us below num_sentences, fill remaining
    if len(selected) < num_sentences:
        selected_ids = {id(s) for s in selected}
        for s in scored:
            if id(s) not in selected_ids:
                selected.append(s)
                if len(selected) >= num_sentences:
                    break

    # 7. Re-order chronologically and return cohesive summary
    selected.sort(key=lambda s: s.order_index)
    return " ".join(s.text for s in selected)


def generate_section_summary(
    document: Document, section_title_query: str = "conclusion", num_sentences: int = 2
) -> str:
    """
    Generate an extractive summary for a specific section (e.g. Conclusion)
    using the exact same keyword-salience and cue-phrase strategy.
    """
    return generate_extractive_summary(
        document,
        num_sentences=num_sentences,
        target_sections=[section_title_query],
        max_per_section=num_sentences,
    )
