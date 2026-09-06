from dataclasses import dataclass
from app.services.nlp.text_cleaning import clean_text, process_text
from app.services.nlp.tfidf import TFIDFModel

MATH_SYMBOLS = {"∈", "≤", "≥", "∑", "∏", "√", "∝", "∼", "±", "×", "÷", "≠", "≡", "d_model"}
INVALID_STARTERS = ("see ", "figure ", "table ", "fig. ", "where ", "for each ", "equation ", "eq. ")
CUE_PHRASES = [
    "we propose", "we present", "we introduce", "in this work",
    "in this paper", "our model", "we show", "we find",
    "outperform", "state-of-the-art", "results show", "rely entirely",
]


@dataclass
class ScoredSentence:
    text: str
    score: float
    order_index: int
    section_title: str | None = None
    token_count: int = 0


def is_valid_prose_sentence(sent_doc) -> bool:
    """
    Verify that a sentence is genuine English prose, filtering out
    isolated math formulas, figure/table references, and fragment lines.
    """
    text = sent_doc.text.strip()
    words = text.split()

    if len(words) < 3 or len(words) > 65:
        return False

    text_lower = text.lower()
    if any(text_lower.startswith(prefix) for prefix in INVALID_STARTERS):
        return False

    # Check letter ratio (must be mostly alphabetic prose)
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count / max(len(text), 1) < 0.65:
        return False

    if any(sym in text for sym in MATH_SYMBOLS):
        return False

    # Must contain at least one verb
    if not any(t.pos_ in ("VERB", "AUX") for t in sent_doc):
        return False

    if not text[0].isupper():
        return False

    return True


def split_sentences(text: str) -> list[str]:
    """Split paragraph into natural language prose sentences using spaCy."""
    doc = process_text(text)
    return [
        " ".join(sent.text.split()).strip()
        for sent in doc.sents
        if is_valid_prose_sentence(sent)
    ]


def score_sentence_salience(
    sentence_text: str,
    keyword_weights: dict[str, float],
) -> tuple[float, int]:
    """
    Score an informative sentence using paper keyword salience and indicative cues.
    """
    doc = process_text(sentence_text)
    tokens = clean_text(doc)
    if not tokens:
        return 0.0, 0

    # Keyword overlap score: how many core paper keywords are discussed
    kw_score = sum(keyword_weights.get(t, 0.0) for t in tokens)
    if kw_score == 0.0:
        return 0.0, len(tokens)

    # Length-balanced density
    base_score = kw_score / (len(tokens) ** 0.5)

    # Cue phrase bonus for core contribution sentences (Edmundson's cue method)
    text_lower = sentence_text.lower()
    multiplier = 1.5 if any(cue in text_lower for cue in CUE_PHRASES) else 1.0

    return base_score * multiplier, len(tokens)


def score_sentences(
    sentences: list[str],
    weights_or_model: dict[str, float] | TFIDFModel,
    section_title: str | None = None,
    start_index: int = 0,
) -> list[ScoredSentence]:
    """Score an ordered list of candidate sentences."""
    if isinstance(weights_or_model, TFIDFModel):
        weights = weights_or_model.idf
    else:
        weights = weights_or_model

    scored = []
    for idx, text in enumerate(sentences, start=start_index):
        score, count = score_sentence_salience(text, weights)
        if score > 0:
            scored.append(
                ScoredSentence(
                    text=text,
                    score=score,
                    order_index=idx,
                    section_title=section_title,
                    token_count=count,
                )
            )
    return scored
