"""Capability 1 - course-pair overlap.
Baseline (Week 2): TF-IDF cosine of the whole course text.
Core (Week 3): chunk-level embeddings with MaxSim - every unit of A is matched to its closest unit of B and vice versa;
matched units are the common topics, unmatched ones are unique to each course."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from curriculum_engine.data import chunks, course_text


class Tfidf:
    """Lexical baseline, fitted on training courses only."""

    def __init__(self, train_courses):
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
        self.vec.fit([course_text(c) for c in train_courses])

    def matrix(self, courses):
        return self.vec.transform([course_text(c) for c in courses])

    def cosine(self, a, b):
        A, B = self.matrix([a, b])
        return float(A.multiply(B).sum())


def chunk_vectors(emb, c):
    return emb([t for _, t in chunks(c)])


def maxsim(A, B, tau):
    """Bidirectional MaxSim of two chunk-embedding matrices (rows L2-normalised).
    soft = mean best-match similarity (symmetrised); coverage_a = share of A's chunks with a match >= tau in B; overlap = F-measure."""
    if not len(A) or not len(B):
        return {"S": np.zeros((len(A), len(B))), "best_a": np.zeros(len(A)), "best_b": np.zeros(len(B)),
                "cov_a": 0.0, "cov_b": 0.0, "overlap": 0.0, "soft": 0.0}
    S = A @ B.T
    ab, ba = S.max(1), S.max(0)
    ca, cb = float((ab >= tau).mean()), float((ba >= tau).mean())
    f = 2 * ca * cb / (ca + cb) if ca + cb else 0.0
    return {"S": S, "best_a": ab, "best_b": ba, "cov_a": ca, "cov_b": cb, "overlap": f, "soft": float((ab.mean() + ba.mean()) / 2)}


def topics(a, b, m, tau, k=12):
    """Common topics (matched unit pairs) and the units unique to each course."""
    la, lb = [t for t, _ in chunks(a)], [t for t, _ in chunks(b)]
    common = []
    for i in np.argsort(-m["best_a"]):
        j = int(m["S"][i].argmax())
        if m["S"][i, j] >= tau:
            common.append({"topic_a": la[i], "topic_b": lb[j], "similarity": round(float(m["S"][i, j]), 3)})
    return {"common_topics": common[:k],
            "unique_to_a": [la[i] for i in range(len(la)) if m["best_a"][i] < tau][:k],
            "unique_to_b": [lb[j] for j in range(len(lb)) if m["best_b"][j] < tau][:k]}
