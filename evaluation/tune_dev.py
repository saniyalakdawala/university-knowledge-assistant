"""Choose retrieval settings on the DEV split only (no answer generation). Writes evaluation/results/tuning_dev.json.
Test questions are never loaded here."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation" / "results"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("LLM_DISK_CACHE", str(OUT / "llm_cache.json"))
os.environ.setdefault("GROQ_MAX_RETRIES", "8")
os.environ.setdefault("GROQ_TIMEOUT", "60")

import numpy as np

from backend.rag_service import UniversityRAGService
from evaluation import metrics as M
from evaluation.run_benchmark import load_questions

qs, facts = load_questions("dev")
qs = [q for q in qs if q["facts"]]


def score(svc, rows):
    res = []
    for q in qs:
        ranked = svc.retrieve(q["question"])["ranked"][:10]
        res.append(M.retrieval_scores([facts[f] for f in q["facts"]], [rows[c] for c in ranked]))
    return {k: round(float(np.mean([r[k] for r in res])), 4) for k in ("recall@3", "recall@5", "mrr@10", "ndcg@10", "precision@5")}


log = []
for words in (100, 130, 160):
    svc = UniversityRAGService(max_chunk_words=words, query_expansion=False)
    rows = {i: {"text": c.text, "page": int(c.page), "end_page": int(c.end_page)} for i, c in enumerate(svc.chunks.itertuples())}
    grid = [(20, .5), (20, .7), (20, .9), (30, .7), (30, 1.0)] if words == 130 else [(20, .7)]
    for cands, weight in grid:
        svc.rerank_candidates, svc.rerank_weight = cands, weight
        t0 = time.time()
        m = score(svc, rows)
        log.append({"expansion": False, "max_chunk_words": words, "rerank_candidates": cands, "rerank_weight": weight,
                    **m, "seconds_per_q": round((time.time() - t0) / len(qs), 2)})
        print(log[-1], flush=True)
        (OUT / "tuning_dev.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
best = max(log, key=lambda r: (r["recall@5"], r["mrr@10"]))

# the chosen setting with Groq query expansion, and with the expansion alone switched off the reranker
svc = UniversityRAGService(max_chunk_words=best["max_chunk_words"], query_expansion=True,
                           rerank_candidates=best["rerank_candidates"], rerank_weight=best["rerank_weight"])
rows = {i: {"text": c.text, "page": int(c.page), "end_page": int(c.end_page)} for i, c in enumerate(svc.chunks.itertuples())}
t0 = time.time()
m = score(svc, rows)
log.append({**{k: best[k] for k in ("max_chunk_words", "rerank_candidates", "rerank_weight")}, "expansion": True, **m,
            "seconds_per_q": round((time.time() - t0) / len(qs), 2)})
print(log[-1], flush=True)
(OUT / "tuning_dev.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
