import re
from spacy.lang.en.stop_words import STOP_WORDS


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_paper_text(text: str) -> str:
    """
    Clean raw text extracted from PDF content blocks.
    Removes hyphenations, bracketed citations, URLs, and emails.
    """
    # De-hyphenate words split across line breaks
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Remove bracketed citations [1], [2, 3]
    text = re.sub(r"\[\d+(?:[,\s–-]+\d+)*\]", "", text)
    # Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text into word tokens using word boundaries."""
    return re.findall(r"\b\w+\b", text)


def preprocess_text(text: str) -> list[str]:
    """Normalize and tokenize text."""
    return tokenize(normalize_text(text))


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out common English stopwords from a list of tokens."""
    return [token for token in tokens if token.lower() not in STOP_WORDS]
