import math
import re
from collections import Counter
from typing import Any

from app.models.chunk import Chunk
from app.models.search import SearchFilter

# Common conversational filler and stopwords for query refinement
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "s", "same", "she", "should", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves", "tell", "explain", "describe",
    "paper", "model", "show", "find"
}

# Regex preserving alphanumeric words, snake_case (d_model), hyphens (WMT-14, ERR-4021), and decimals (28.4)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*")


def tokenize_bm25(text: str) -> list[str]:
    """
    Robust tokenizer for technical research text.
    Preserves snake_case, hyphenated variables, decimals, and alphanumeric codes.
    Lowercases and strips trailing punctuation.
    """
    if not text:
        return []
    matches = TOKEN_PATTERN.findall(text.lower())
    return [m for m in matches if len(m) > 1 or m.isalnum()]


class BM25Index:
    """
    In-memory BM25Okapi lexical retrieval index for academic research paper Chunks.
    - Uses non-negative Robertson/Lucene IDF formulation (prevents negative IDF on small corpora).
    - Preserves technical identifiers (snake_case, formulas, acronyms, decimals).
    - Filters zero-score chunks so non-matching documents never enter RRF.
    - Synchronized with metadata filtering (e.g., paper_id, page ranges).
    - Indexes embed_text (consistent with dense embeddings).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks_by_id: dict[str, Chunk] = {}
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_lens: dict[str, int] = {}
        self.doc_freqs: dict[str, int] = Counter()
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.idf_cache: dict[str, float] = {}

    def fit(self, chunks: list[Chunk]) -> "BM25Index":
        """Build the index from a list of Chunks."""
        self.chunks_by_id.clear()
        self.doc_tokens.clear()
        self.doc_lens.clear()
        self.doc_freqs.clear()
        self.idf_cache.clear()

        for c in chunks:
            self._add_single_chunk(c)

        self._recompute_corpus_stats()
        return self

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally add chunks to the index."""
        for c in chunks:
            self._add_single_chunk(c)
        self._recompute_corpus_stats()

    def delete_paper(self, paper_id: str) -> int:
        """Remove all chunks associated with a specific paper_id."""
        to_delete = [
            cid for cid, c in self.chunks_by_id.items() if c.paper_id == paper_id
        ]
        for cid in to_delete:
            del self.chunks_by_id[cid]
            del self.doc_tokens[cid]
            del self.doc_lens[cid]

        # Recompute doc frequencies from scratch
        self.doc_freqs.clear()
        for tokens in self.doc_tokens.values():
            for t in set(tokens):
                self.doc_freqs[t] += 1

        self._recompute_corpus_stats()
        return len(to_delete)

    def _add_single_chunk(self, chunk: Chunk) -> None:
        """Index enriched embed_text of a single chunk."""
        self.chunks_by_id[chunk.chunk_id] = chunk
        tokens = tokenize_bm25(chunk.embed_text)
        self.doc_tokens[chunk.chunk_id] = tokens
        self.doc_lens[chunk.chunk_id] = len(tokens)
        for t in set(tokens):
            self.doc_freqs[t] += 1

    def _recompute_corpus_stats(self) -> None:
        self.total_docs = len(self.chunks_by_id)
        if self.total_docs > 0:
            self.avg_doc_len = sum(self.doc_lens.values()) / self.total_docs
        else:
            self.avg_doc_len = 0.0
        self.idf_cache.clear()

    def compute_idf(self, term: str) -> float:
        """
        Compute non-negative Lucene/Robertson IDF:
        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        Guarantees IDF >= 0 even when term appears in >50% of documents.
        """
        if term in self.idf_cache:
            return self.idf_cache[term]

        n_q = self.doc_freqs.get(term, 0)
        if n_q == 0 or self.total_docs == 0:
            idf = 0.0
        else:
            numerator = self.total_docs - n_q + 0.5
            denominator = n_q + 0.5
            idf = math.log(1.0 + (numerator / denominator))

        self.idf_cache[term] = idf
        return idf

    def search(
        self,
        query: str,
        top_k: int = 25,
        search_filter: SearchFilter | None = None,
        filter_paper_id: str | None = None,
        filter_stopwords: bool = True,
    ) -> list[tuple[Chunk, float, int]]:
        """
        Execute BM25 search across indexed chunks.
        Returns list of (Chunk, bm25_score, 1-based rank).
        Filters score > 0.0 so non-overlapping chunks never receive a rank.
        Applies metadata filters BEFORE ranking.
        """
        if self.total_docs == 0:
            return []

        # Tokenize query
        raw_tokens = tokenize_bm25(query)
        if filter_stopwords and len(raw_tokens) > 2:
            query_tokens = [t for t in raw_tokens if t not in STOPWORDS]
            if not query_tokens:
                query_tokens = raw_tokens
        else:
            query_tokens = raw_tokens

        if not query_tokens:
            return []

        # Resolve filter constraints
        target_paper_id = filter_paper_id or (search_filter.paper_id if search_filter else None)
        min_page = search_filter.min_page if search_filter else None
        max_page = search_filter.max_page if search_filter else None
        target_sec = search_filter.section_title if search_filter else None

        scores: list[tuple[Chunk, float]] = []

        for cid, chunk in self.chunks_by_id.items():
            # Apply metadata filters
            if target_paper_id and chunk.paper_id != target_paper_id:
                continue
            if min_page is not None and not any(p >= min_page for p in chunk.page_numbers):
                continue
            if max_page is not None and not any(p <= max_page for p in chunk.page_numbers):
                continue
            if target_sec and chunk.section_title != target_sec:
                continue

            doc_tokens = self.doc_tokens[cid]
            doc_len = self.doc_lens[cid]
            token_counts = Counter(doc_tokens)

            chunk_score = 0.0
            for qt in query_tokens:
                if qt not in token_counts:
                    continue
                tf = token_counts[qt]
                idf = self.compute_idf(qt)
                # BM25 term frequency saturation formula
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / denom)
                chunk_score += term_score

            # Critical Rule: Only retain chunks with genuine lexical overlap (score > 0)
            if chunk_score > 0.0:
                scores.append((chunk, round(chunk_score, 4)))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Truncate to top_k and assign 1-based ranks
        results: list[tuple[Chunk, float, int]] = []
        for rank, (chunk, score) in enumerate(scores[:top_k], start=1):
            results.append((chunk, score, rank))

        return results
