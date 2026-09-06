import math
from collections import Counter


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Compute Term Frequency (TF) for a single document.

    TF(t, d) = count(t, d) / total_tokens(d)
    
    Measures how frequently a term appears in a document,
    normalized by document length to prevent bias towards longer texts.
    """
    if not tokens:
        return {}

    total_tokens = len(tokens)
    counts = Counter(tokens)

    return {term: count / total_tokens for term, count in counts.items()}


def compute_idf(
    corpus: list[list[str]], smooth: bool = True
) -> dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) across a collection of documents.

    Smooth IDF:
        IDF(t, D) = ln((N + 1) / (DF(t) + 1)) + 1
    Standard IDF:
        IDF(t, D) = ln(N / DF(t))

    Where:
        N = total number of documents in corpus
        DF(t) = number of documents containing term t
    """
    num_docs = len(corpus)
    if num_docs == 0:
        return {}

    # Count in how many documents each unique term appears
    doc_freq: dict[str, int] = Counter()
    for doc_tokens in corpus:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    idf: dict[str, float] = {}
    for term, df in doc_freq.items():
        if smooth:
            idf[term] = math.log((num_docs + 1) / (df + 1)) + 1.0
        else:
            idf[term] = math.log(num_docs / df) if df > 0 else 0.0

    return idf


def compute_tfidf(
    tokens: list[str], idf: dict[str, float]
) -> dict[str, float]:
    """
    Compute TF-IDF scores for tokens in a document using precomputed IDF weights.

    TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
    """
    tf = compute_tf(tokens)
    # Out-of-vocabulary terms have zero weight
    default_idf = 0.0

    return {
        term: tf_val * idf.get(term, default_idf)
        for term, tf_val in tf.items()
    }


class TFIDFModel:
    """
    A lightweight, from-scratch TF-IDF model designed for classical NLP
    on research papers.
    """

    def __init__(self, smooth: bool = True):
        self.smooth = smooth
        self.doc_freq: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.num_docs: int = 0
        self.vocab: set[str] = set()

    def fit(self, corpus: list[list[str]]) -> "TFIDFModel":
        """
        Fit the model on a corpus of tokenized documents (e.g. paper sections).
        """
        self.num_docs = len(corpus)
        self.doc_freq = Counter()

        for doc_tokens in corpus:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                self.doc_freq[term] += 1

        self.vocab = set(self.doc_freq.keys())
        self.idf = compute_idf(corpus, smooth=self.smooth)
        return self

    def transform(self, tokens: list[str]) -> dict[str, float]:
        """
        Transform a tokenized document into a dictionary of {term: tfidf_score}.
        """
        return compute_tfidf(tokens, self.idf)

    def fit_transform(self, corpus: list[list[str]]) -> list[dict[str, float]]:
        """
        Fit on the corpus and return TF-IDF representations for all documents.
        """
        self.fit(corpus)
        return [self.transform(doc) for doc in corpus]


def compare_with_sklearn(
    corpus_texts: list[str],
) -> dict:
    """
    Helper function to compare our scratch TF-IDF results against
    scikit-learn's TfidfVectorizer.
    Returns both results for side-by-side comparison.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {"error": "scikit-learn is not installed in the environment."}

    # 1. Run scikit-learn
    vectorizer = TfidfVectorizer(smooth_idf=True, norm=None)
    X = vectorizer.fit_transform(corpus_texts)
    feature_names = vectorizer.get_feature_names_out()

    sklearn_results = []
    for row in X.toarray():
        doc_scores = {
            feature_names[i]: float(score)
            for i, score in enumerate(row)
            if score > 0
        }
        sklearn_results.append(doc_scores)

    # 2. Run our model
    tokenized_corpus = [
        text.lower().split() for text in corpus_texts
    ]
    model = TFIDFModel(smooth=True)
    scratch_results = model.fit_transform(tokenized_corpus)

    return {
        "scratch": scratch_results,
        "sklearn": sklearn_results,
    }
