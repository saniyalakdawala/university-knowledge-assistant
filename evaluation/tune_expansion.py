"""How should Groq rewrites enter retrieval? DEV split only. Writes evaluation/results/tuning_expansion.json."""
import json
import os
import sys
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
svc = UniversityRAGService(rerank_candidates=30, rerank_weight=1.0)
rows = {i: {"text": c.text, "page": int(c.page), "end_page": int(c.end_page)} for i, c in enumerate(svc.chunks.itertuples())}

CONFIGS = {
    "no_expansion": dict(query_expansion=False, variant_weight=1.0, rerank_with="question"),
    "rewrites_rrf_1.0_ce_question": dict(query_expansion=True, variant_weight=1.0, rerank_with="question"),
    "rewrites_rrf_0.5_ce_question": dict(query_expansion=True, variant_weight=0.5, rerank_with="question"),
    "rewrites_rrf_0.25_ce_question": dict(query_expansion=True, variant_weight=0.25, rerank_with="question"),
    "rewrites_rrf_0.5_ce_both": dict(query_expansion=True, variant_weight=0.5, rerank_with="both"),
    "rewrites_rrf_0.5_ce_normalized": dict(query_expansion=True, variant_weight=0.5, rerank_with="normalized"),
    "no_expansion_ce_normalized": dict(query_expansion=False, variant_weight=1.0, rerank_with="normalized"),
}
log = {}
for name, cfg in CONFIGS.items():
    for k, v in cfg.items():
        setattr(svc, k, v)
    by = defaultdict(list)
    for q in qs:
        ranked = svc.retrieve(q["question"])["ranked"][:10]
        s = M.retrieval_scores([facts[f] for f in q["facts"]], [rows[c] for c in ranked])
        by[q["category"]].append(s)
        by["all"].append(s)
    log[name] = {c: {k: round(float(np.mean([r[k] for r in v])), 3) for k in ("recall@3", "recall@5", "mrr@10")} for c, v in by.items()}
    print(name, {c: (m["recall@5"], m["mrr@10"]) for c, m in log[name].items()}, flush=True)
    (OUT / "tuning_expansion.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
