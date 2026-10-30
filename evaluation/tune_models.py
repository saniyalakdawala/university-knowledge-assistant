"""Are other embedding / reranking models needed? DEV split only. Writes evaluation/results/tuning_models.json."""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation" / "results"
os.environ.setdefault("LLM_DISK_CACHE", str(OUT / "llm_cache.json"))
os.environ.setdefault("GROQ_MAX_RETRIES", "8")
os.environ.setdefault("GROQ_TIMEOUT", "60")

import numpy as np

from backend.rag_service import UniversityRAGService
from evaluation import metrics as M
from evaluation.run_benchmark import load_questions

qs, facts = load_questions("dev")
qs = [q for q in qs if q["facts"]]
MINILM, BGE = "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"
L6, L12 = "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"
log = {}
for emb, ce in [(MINILM, L6), (BGE, L6), (MINILM, L12), (BGE, L12)]:
    svc = UniversityRAGService(emb_model_name=emb, ce_model_name=ce, rerank_candidates=30)
    rows = {i: {"text": c.text, "page": int(c.page), "end_page": int(c.end_page)} for i, c in enumerate(svc.chunks.itertuples())}
    for weight, rerank_with, expand in [(1.0, "question", False), (0.8, "question", False), (0.8, "question", True), (0.8, "both", True), (0.6, "both", True)]:
        svc.rerank_weight, svc.rerank_with, svc.query_expansion = weight, rerank_with, expand
        by, t0 = defaultdict(list), time.time()
        for q in qs:
            ranked = svc.retrieve(q["question"])["ranked"][:10]
            s = M.retrieval_scores([facts[f] for f in q["facts"]], [rows[c] for c in ranked])
            by[q["category"]].append(s)
            by["all"].append(s)
        name = f"{emb.split('/')[-1]}|{ce.split('-')[-2]}-{ce.split('-')[-1]}|w{weight}|{rerank_with}|exp={expand}"
        log[name] = {c: {k: round(float(np.mean([r[k] for r in v])), 3) for k in ("recall@3", "recall@5", "recall@10", "mrr@10")} for c, v in by.items()}
        print(name, "ALL", log[name]["all"], "PARA", log[name]["paraphrased"]["recall@5"], "INF", log[name]["informal"]["recall@5"],
              "DIR", log[name]["direct"]["mrr@10"], f"{(time.time() - t0) / len(qs):.2f}s", flush=True)
        (OUT / "tuning_models.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
