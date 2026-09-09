import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.embeddings.word2vec import Word2VecPipeline


def run_context_and_order_demo():
    print("=" * 80)
    print("DEMO: WORD2VEC ON POLYSEMY (CONTEXT), WORD ORDER & NEGATION")
    print("=" * 80)

    # We train a small focused corpus so that the vocabulary has all necessary words
    sentences = [
        # Context A: Financial bank
        ["the", "customer", "went", "to", "the", "investment", "bank", "to", "deposit", "money"],
        ["the", "bank", "approved", "the", "loan", "with", "interest"],
        # Context B: River bank
        ["the", "fisherman", "sat", "by", "the", "river", "bank", "watching", "the", "flowing", "water"],
        ["trees", "grow", "along", "the", "river", "bank", "near", "the", "water"],
        # Context C: Head (Anatomy vs Transformer)
        ["the", "patient", "suffered", "an", "injury", "to", "the", "head"],
        ["he", "turned", "his", "head", "to", "look", "at", "the", "screen"],
        ["multi", "head", "attention", "computes", "projections", "in", "parallel"],
        ["each", "attention", "head", "focuses", "on", "different", "positions"],
        # Sentences for order and negation testing
        ["server", "alpha", "sent", "all", "the", "data", "to", "server", "beta"],
        ["server", "beta", "sent", "all", "the", "data", "to", "server", "alpha"],
        ["the", "proposed", "model", "is", "fast", "and", "effective"],
        ["the", "proposed", "model", "is", "not", "fast", "and", "not", "effective"],
    ]

    pipeline = Word2VecPipeline(
        vector_size=50,
        window=4,
        min_count=1,  # Keep all demo words
        sg=1,         # Skip-gram
        epochs=100,
        seed=42,
    )
    pipeline.train(sentences)

    # =========================================================================
    # EXPERIMENT 1: POLYSEMY (SAME WORD, DIFFERENT SENSES)
    # =========================================================================
    print("\n" + "-" * 80)
    print("EXPERIMENT 1: POLYSEMY — SAME WORD, TWO COMPLETELY DIFFERENT MEANINGS")
    print("-" * 80)

    s_river = "the fisherman sat by the river bank near the water"
    s_finance = "the customer went to the investment bank to deposit money"

    print(f"Sentence A (Nature):  '{s_river}'")
    print(f"Sentence B (Finance): '{s_finance}'")
    print("\nKey Shared Word: 'bank'")

    vec_bank_in_A = pipeline.get_vector("bank")
    vec_bank_in_B = pipeline.get_vector("bank")

    # Are they different vectors?
    are_vectors_identical = np.array_equal(vec_bank_in_A, vec_bank_in_B)
    print(f"  • Is vec('bank') in Sentence A different from vec('bank') in Sentence B? -> {not are_vectors_identical}")
    print(f"  • Norm of vec('bank'): {np.linalg.norm(vec_bank_in_A):.4f}")
    print(f"  • First 5 dimensions of vec('bank'): {np.round(vec_bank_in_A[:5], 4)}")

    # What is the pooled sentence similarity?
    sim_bank = pipeline.sentence_similarity(s_river, s_finance)
    print(f"\n  -> Pooled Cosine Similarity between Sentence A and Sentence B: {sim_bank:.4f}")
    print("  -> EXPLANATION: Word2Vec only has ONE entry in its dictionary for 'bank'.")
    print("     It cannot know whether 'bank' means a financial institution or the side of a river.")
    print("     The vector for 'bank' is pulled toward both 'river' and 'investment' simultaneously,")
    print("     giving it a compromised, blurry representation that falsely bridges both sentences!")

    # =========================================================================
    # EXPERIMENT 2: WORD ORDER INVERSION (WHO SENT WHAT TO WHOM?)
    # =========================================================================
    print("\n" + "-" * 80)
    print("EXPERIMENT 2: WORD ORDER INVERSION — COMMUTATIVE PROPERTY FAILURE")
    print("-" * 80)

    s_order_1 = "server alpha sent all the data to server beta"
    s_order_2 = "server beta sent all the data to server alpha"

    print(f"Sentence 1: '{s_order_1}'")
    print(f"Sentence 2: '{s_order_2}'")

    v1 = pipeline.sentence_vector(s_order_1)
    v2 = pipeline.sentence_vector(s_order_2)
    diff = np.linalg.norm(v1 - v2)
    sim_order = pipeline.sentence_similarity(s_order_1, s_order_2)

    print(f"  • Vector Euclidean Distance ||v1 - v2||: {diff:.8f}")
    print(f"  • Cosine Similarity:                     {sim_order:.6f}")
    print("  -> EXPLANATION: In Word2Vec, a sentence vector is computed by adding word vectors:")
    print("     v_sentence = v('server') + v('alpha') + v('sent') + ... + v('beta')")
    print("     Because vector addition is commutative (a + b = b + a),")
    print("     switching subject and object yields THE EXACT SAME mathematical vector!")

    # =========================================================================
    # EXPERIMENT 3: NEGATION BLINDNESS (OPPOSITE MEANING, HIGH SIMILARITY)
    # =========================================================================
    print("\n" + "-" * 80)
    print("EXPERIMENT 3: NEGATION BLINDNESS — POLAR OPPOSITES STILL MATCH")
    print("-" * 80)

    s_pos = "the proposed model is fast and effective"
    s_neg = "the proposed model is not fast and not effective"

    print(f"Sentence Affirmative: '{s_pos}'")
    print(f"Sentence Negative:    '{s_neg}'")

    sim_neg = pipeline.sentence_similarity(s_pos, s_neg)
    print(f"  • Cosine Similarity: {sim_neg:.4f}")
    print("  -> EXPLANATION: 6 out of 7 words in both sentences are identical.")
    print("     Adding the vector for 'not' simply acts as a small perturbation.")
    print("     Word2Vec has no logical reasoning engine to flip the truth polarity of a statement.")

    print("\n" + "=" * 80)
    print("SUMMARY: WHY SENTENCE TRANSFORMERS ARE NEEDED (STEP 3.2 & 3.3)")
    print("=" * 80)
    print("• Word2Vec solved WORD-LEVEL semantic similarity (e.g. 'cat' ≈ 'kitten').")
    print("• But Word2Vec CANNOT do SENTENCE-LEVEL semantic understanding:")
    print("    1. No context-dependent vectors (Polysemy failure).")
    print("    2. No syntax or sequence awareness (Word order blindness).")
    print("    3. No compositional semantics (Negation blindness).")
    print("• Transformers (BERT / SBERT) solved this using Self-Attention & Positional Encodings!")
    print("=" * 80)


if __name__ == "__main__":
    run_context_and_order_demo()
