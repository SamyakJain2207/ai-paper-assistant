from pathlib import Path
from typing import Self
import numpy as np
from gensim.models import Word2Vec

from app.models.document import Document
from app.services.nlp.preprocessing import clean_paper_text, tokenize
from app.services.nlp.sentence_scoring import split_sentences


class Word2VecPipeline:
    """
    Word2Vec embedding pipeline for research papers.
    Supports Continuous Bag-of-Words (CBOW, sg=0) and Skip-gram (sg=1),
    vector arithmetic, similarity queries, and document sentence pooling.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        sg: int = 0,  # 0 = CBOW, 1 = Skip-gram
        epochs: int = 50,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.seed = seed
        self.workers = workers
        self.model: Word2Vec | None = None

    @property
    def architecture_name(self) -> str:
        return "Skip-gram" if self.sg == 1 else "CBOW"

    @property
    def vocab_size(self) -> int:
        if self.model is None:
            return 0
        return len(self.model.wv)

    @property
    def vocabulary(self) -> list[str]:
        if self.model is None:
            return []
        return list(self.model.wv.index_to_key)

    @staticmethod
    def extract_sentences_from_documents(documents: list[Document]) -> list[list[str]]:
        """
        Extract clean, tokenized sentences from structured Document objects.
        Uses Phase 1 blocks and Phase 2 cleaning/sentence splitting.
        """
        tokenized_sentences: list[list[str]] = []

        for doc in documents:
            for block in doc.content_blocks:
                # Clean block text (strip citations, URLs, hyphenations)
                cleaned_block = clean_paper_text(block.text)
                if not cleaned_block:
                    continue

                # Split into valid grammatical sentences
                sentences = split_sentences(cleaned_block)
                for sentence in sentences:
                    # Tokenize into lowercase tokens
                    tokens = [t.lower() for t in tokenize(sentence) if len(t) > 1]
                    if len(tokens) >= 3:
                        tokenized_sentences.append(tokens)

        return tokenized_sentences

    @staticmethod
    def extract_sentences_from_text(raw_text: str) -> list[list[str]]:
        """Extract clean, tokenized sentences from raw string text."""
        cleaned = clean_paper_text(raw_text)
        sentences = split_sentences(cleaned)
        tokenized_sentences: list[list[str]] = []
        for sentence in sentences:
            tokens = [t.lower() for t in tokenize(sentence) if len(t) > 1]
            if len(tokens) >= 3:
                tokenized_sentences.append(tokens)
        return tokenized_sentences

    def train(self, sentences: list[list[str]]) -> Word2Vec:
        """
        Train Word2Vec model on pre-tokenized sentences.
        """
        if not sentences:
            raise ValueError("Cannot train Word2Vec on an empty sentence list.")

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            seed=self.seed,
            workers=self.workers,
            compute_loss=True,
        )
        return self.model

    def has_word(self, word: str) -> bool:
        """Check if a word exists in the vocabulary."""
        if self.model is None:
            return False
        return word.lower() in self.model.wv

    def get_vector(self, word: str) -> np.ndarray:
        """
        Retrieve static vector embedding for a word.
        Raises KeyError if word is Out-of-Vocabulary (OOV).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        word_clean = word.lower()
        if word_clean not in self.model.wv:
            raise KeyError(
                f"Word '{word}' is Out-Of-Vocabulary (OOV). "
                f"Vocabulary size: {self.vocab_size} terms."
            )
        return self.model.wv[word_clean]

    def similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words in the vocabulary."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        w1 = word1.lower()
        w2 = word2.lower()
        if w1 not in self.model.wv:
            raise KeyError(f"Word '{word1}' is Out-Of-Vocabulary.")
        if w2 not in self.model.wv:
            raise KeyError(f"Word '{word2}' is Out-Of-Vocabulary.")

        return float(self.model.wv.similarity(w1, w2))

    def most_similar(
        self,
        positive: list[str] | None = None,
        negative: list[str] | None = None,
        topn: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find top-n nearest neighbors or execute vector arithmetic:
        v_result = sum(positive) - sum(negative)
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        pos = [w.lower() for w in (positive or [])]
        neg = [w.lower() for w in (negative or [])]

        # Verify all query tokens exist in vocabulary
        for w in pos + neg:
            if w not in self.model.wv:
                raise KeyError(f"Word '{w}' is Out-Of-Vocabulary.")

        results = self.model.wv.most_similar(
            positive=pos,
            negative=neg,
            topn=topn,
        )
        return [(word, float(score)) for word, score in results]

    def sentence_vector(
        self,
        sentence: str | list[str],
        weights: dict[str, float] | object | None = None,
    ) -> np.ndarray:
        """
        Demonstrate naive bag-of-words sentence vector pooling.
        Averages word vectors with optional TF-IDF weights or a TFIDFModel instance.
        Returns a zero vector if no words from the sentence are in vocabulary.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        if isinstance(sentence, str):
            tokens = [t.lower() for t in tokenize(clean_paper_text(sentence))]
        else:
            tokens = [t.lower() for t in sentence]

        if weights is not None:
            if hasattr(weights, "transform"):
                # Accepts a Phase 2 TFIDFModel instance directly
                token_weight_map: dict[str, float] = weights.transform(tokens)
            elif isinstance(weights, dict):
                token_weight_map = weights
            else:
                token_weight_map = None
        else:
            token_weight_map = None

        vectors: list[np.ndarray] = []
        token_weights: list[float] = []

        for t in tokens:
            if t in self.model.wv:
                v = self.model.wv[t]
                w = token_weight_map.get(t, 0.0) if token_weight_map is not None else 1.0
                if w > 0.0:
                    vectors.append(v * w)
                    token_weights.append(w)

        if not vectors or sum(token_weights) == 0:
            return np.zeros(self.vector_size, dtype=np.float32)

        pooled = np.sum(vectors, axis=0) / sum(token_weights)
        return pooled.astype(np.float32)

    def sentence_similarity(
        self,
        sent1: str | list[str],
        sent2: str | list[str],
        weights: dict[str, float] | object | None = None,
    ) -> float:
        """
        Compute cosine similarity between two sentences using naive vector pooling.
        Accepts optional weights (dict or Phase 2 TFIDFModel).
        """
        v1 = self.sentence_vector(sent1, weights=weights)
        v2 = self.sentence_vector(sent2, weights=weights)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def save(self, path: str | Path) -> None:
        """Persist Word2Vec model to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save untrained model.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load persisted Word2Vec model from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        loaded_gensim = Word2Vec.load(str(p))
        instance = cls(
            vector_size=loaded_gensim.vector_size,
            window=loaded_gensim.window,
            min_count=loaded_gensim.min_count,
            sg=loaded_gensim.sg,
            epochs=loaded_gensim.epochs,
        )
        instance.model = loaded_gensim
        return instance
