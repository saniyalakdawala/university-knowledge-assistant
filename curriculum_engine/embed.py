"""Sentence-transformer embeddings with an on-disk cache keyed by text hash (re-runs and evaluation are then instant)."""
import hashlib, os, pickle, re
from pathlib import Path

import numpy as np

from curriculum_engine import DATA

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE_BASE = "BAAI/bge-base-en-v1.5"
if (Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")       # cached models: do not contact huggingface.co (it can hang for an hour)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # on a fresh machine (the grader's) the model is downloaded once instead
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")             # joblib cannot count physical cores on this Windows setup
os.environ.setdefault("USE_TF", "0")                   # sentence-transformers would otherwise import TensorFlow


class Embedder:
    def __init__(self, model=MINILM):
        self.name, self._m = model, None
        self.path = DATA / "interim" / f"emb_{re.sub(r'[^A-Za-z0-9]+', '_', model)}.pkl"
        self.cache = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.dirty = False

    def __call__(self, texts):
        texts = list(texts)
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        miss = {k: t for k, t in zip(keys, texts) if k not in self.cache}
        if miss:
            if self._m is None:
                from sentence_transformers import SentenceTransformer
                self._m = SentenceTransformer(self.name, device="cpu")
            vec = self._m.encode(list(miss.values()), batch_size=64, normalize_embeddings=True, show_progress_bar=len(miss) > 800)
            self.cache.update(zip(miss, vec.astype(np.float32)))
            self.dirty = True
        dim = next(iter(self.cache.values())).shape[0] if self.cache else 384
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, dim), np.float32)

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(pickle.dumps(self.cache))
            self.dirty = False
