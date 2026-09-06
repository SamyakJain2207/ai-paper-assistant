import spacy
from spacy.tokens import Doc

# Disable unused NER component to accelerate model load time
nlp = spacy.load("en_core_web_sm", disable=["ner"])


def process_text(text: str) -> Doc:
    """Run spaCy NLP pipeline on text."""
    return nlp(text)


def remove_stopwords(doc: Doc) -> list[str]:
    """Extract token texts excluding stopwords."""
    return [token.text for token in doc if not token.is_stop]


def lemmatize(doc: Doc) -> list[str]:
    """Extract lemma strings for all tokens."""
    return [token.lemma_ for token in doc]


def clean_text(doc: Doc) -> list[str]:
    """
    Clean text for classical NLP:
    extracts lowercase lemmas, removing stopwords, punctuation, whitespace, and single-char noise.
    """
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.lemma_.strip()) > 1
    ]