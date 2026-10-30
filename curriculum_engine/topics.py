"""Capability 7 - topic model: clusters of unit embeddings named by class-based TF-IDF (BERTopic-style, without the extra
dependency). Fitted on training courses; every unit is then assigned to its nearest topic centroid."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from curriculum_engine import SEED
from curriculum_engine.data import chunks


class TopicModel:
    def __init__(self, emb, k=40):
        self.emb, self.k = emb, k

    def fit(self, courses):
        texts = [t for c in courses for _, t in chunks(c)]
        titles = [l for c in courses for l, _ in chunks(c)]
        E = self.emb(texts)
        self.k = min(self.k, len(texts))                  # tiny datasets (tests, a single uploaded syllabus)
        km = KMeans(self.k, n_init=4, random_state=SEED).fit(E)
        lab = km.labels_
        C = km.cluster_centers_
        self.centroids = C / np.linalg.norm(C, axis=1, keepdims=True)
        small = len(texts) < 50                           # tiny inputs: keep every term
        cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1 if small else 3, max_df=1.0 if small else 0.5,
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
        X = cv.fit_transform(texts)
        terms = np.array(cv.get_feature_names_out())
        tf = np.vstack([np.asarray(X[lab == k].sum(0)).ravel() for k in range(self.k)])
        tf = tf / np.maximum(tf.sum(1, keepdims=True), 1)
        ctf = tf * np.log(1 + X.sum() / np.maximum(np.asarray(X.sum(0)).ravel(), 1) / self.k)
        self.terms = [list(terms[np.argsort(-ctf[k])[:5]]) for k in range(self.k)]
        self.names = [", ".join(t[:3]) for t in self.terms]
        sims = (E @ self.centroids.T)[np.arange(len(lab)), lab]
        self.info = [{"topic": k, "name": self.names[k], "terms": self.terms[k], "units": int((lab == k).sum()),
                      "coherence": round(float(sims[lab == k].mean()), 3),
                      "examples": [titles[i] for i in np.flatnonzero(lab == k)[:8]]} for k in range(self.k)]
        return self

    def assign(self, texts):
        if not texts:
            return []
        return [self.names[i] for i in (self.emb(texts) @ self.centroids.T).argmax(1)]
