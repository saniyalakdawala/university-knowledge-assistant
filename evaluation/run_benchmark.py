"""Run the benchmark on one pipeline and write evaluation/results/<name>_<split>.json.

    python evaluation/run_benchmark.py --pipeline legacy   --split test
    python evaluation/run_benchmark.py --pipeline improved --split test
    python evaluation/run_benchmark.py --pipeline improved --split dev --retrieval-only --set rerank_weight=0.5

`--retrieval-only` measures ranking without any answer generation. Groq replies are cached in evaluation/results/llm_cache.json
(GROQ_API_KEY comes from the environment and is never written), so a re-run reproduces the numbers without calling the API.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation" / "results"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("LLM_DISK_CACHE", str(OUT / "llm_cache.json"))
os.environ.setdefault("GROQ_MAX_RETRIES", "8")
os.environ.setdefault("GROQ_TIMEOUT", "60")

import numpy as np

from evaluation import metrics as M


def load_questions(split):
    bm = json.loads((ROOT / "evaluation/benchmark.json").read_text(encoding="utf-8"))["questions"]
    facts = json.loads((ROOT / "evaluation/gold_facts.json").read_text(encoding="utf-8"))["facts"]
    return [q for q in bm if split == "all" or q["split"] == split], facts


def patient_post(session):
    """Benchmark only: wait out Groq's rate limit, so a 429 never lowers a pipeline's score."""
    post = session.post

    def call(*args, **kwargs):
        for _ in range(12):
            resp = post(*args, **kwargs)
            if resp.status_code != 429:
                return resp
            time.sleep(min(float(resp.headers.get("retry-after", 10)) + 1, 90))
        return resp
    return call


def wrap_legacy():
    from evaluation import legacy_rag_service as L
    svc = L.UniversityRAGService()
    svc.groq_timeout = 60
    svc.http_session.post = patient_post(svc.http_session)
    svc._rows = {i: {"text": c.text, "page": int(c.page), "end_page": int(c.page)} for i, c in enumerate(svc.chunks.itertuples())}

    def ranked(q):
        return svc.retrieve(q)["ranked"][:10]

    def answer(q):
        r = svc.ask(q)
        r["citations"] = ([{"page": r["cited_page"], "quote": r["evidence_quote"]}] if r["cited_page"] and not r["abstained"] else [])
        ret = svc.retrieve(q)
        r["context_text"] = " ".join(svc.chunks.text.iloc[c] for c in ret["ranked"][:svc.n_ctx])
        return r
    return svc, ranked, answer


def wrap_improved(overrides):
    from backend.rag_service import UniversityRAGService
    svc = UniversityRAGService(**overrides)
    svc.http_session.post = patient_post(svc.http_session)
    svc._rows = {i: {"text": c.text, "page": int(c.page), "end_page": int(c.end_page)} for i, c in enumerate(svc.chunks.itertuples())}
    by_id = {c.chunk_id: i for i, c in enumerate(svc.chunks.itertuples())}

    def ranked(q):
        return svc.retrieve(q)["ranked"][:10]

    def answer(q):
        r = svc.ask(q)
        r["context_text"] = " ".join(svc.chunks.text.iloc[by_id[c]] for c in r.get("context_ids", []))
        return r
    return svc, ranked, answer


def judge(svc, row):
    """Faithfulness of one answer, checked by a second model that did not write it."""
    resp = svc.http_session.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {svc.groq_api_key}", "Content-Type": "application/json"},
        json={"model": os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b"), "temperature": 0.0, "max_tokens": 400,
              "messages": M.judge_messages(row["question"], row["answer"], row["context"])}, timeout=60)
    return M.parse_verdict(resp.json()["choices"][0]["message"]["content"]) if resp.status_code == 200 else None


def aggregate(rows):
    keys = [k for k in rows[0] if k.startswith(("recall@", "precision@", "mrr@", "ndcg@"))] if rows else []
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    out = {"n": len(rows)}
    for k in keys:
        out[k] = round(float(np.mean([r[k] for r in answerable])), 4) if answerable else None
    if rows and "correct" in rows[0]:
        ans = [r for r in answerable]
        out["answer_accuracy"] = round(float(np.mean([r["correct"] for r in ans])), 4) if ans else None
        out["answered_share"] = round(float(np.mean([not r["abstained"] for r in ans])), 4) if ans else None
        un = [r for r in rows if r["category"] == "unanswerable"]
        if un:
            out["correct_refusal"] = round(float(np.mean([r["abstained"] for r in un])), 4)
        given = [r for r in rows if not r["abstained"]]
        cited = [r for r in ans if not r["abstained"] and r["correct"]]
        out["citation_accuracy"] = round(float(np.mean([r["citation_ok"] for r in cited])), 4) if cited else None
        scored = [r for r in given if r.get("faithful") is not None]
        out["faithfulness"] = round(float(np.mean([r["faithful"] for r in scored])), 4) if scored else None
        out["support_mean"] = round(float(np.mean([r["support"] for r in given])), 4) if given else None
        out["latency_s_mean"] = round(float(np.mean([r["seconds"] for r in rows])), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["legacy", "improved"], required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="service overrides, e.g. rerank_weight=0.5 gate=0.2 query_expansion=false")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=")
        overrides[k] = {"true": True, "false": False}.get(v, float(v) if v.replace(".", "").isdigit() else v)
        if k in ("n_ctx", "rrf_k", "rerank_candidates", "max_chunk_words"):
            overrides[k] = int(v)

    qs, facts = load_questions(a.split)
    svc, ranked, answer = wrap_legacy() if a.pipeline == "legacy" else wrap_improved(overrides)
    rows = []
    for i, q in enumerate(qs, 1):
        gold = [facts[f] for f in q["facts"]]
        row = {"id": q["id"], "category": q["category"], "question": q["question"]}
        if gold or a.retrieval_only:
            passages = [svc._rows[c] for c in ranked(q["question"])]
            row.update(M.retrieval_scores(gold, passages) if gold else {})
        if not a.retrieval_only:
            t0 = time.time()
            r = answer(q["question"])
            row.update(answer=r["answer"], abstained=bool(r["abstained"]), mode=r["mode"], confidence=r["confidence"], support=r.get("support", 0.0),
                       cited_page=r.get("cited_page"), seconds=round(time.time() - t0, 2), correct=M.answer_correct(gold, r),
                       citation_ok=(M.citation_correct(gold, r, {c.chunk_id: {"text": c.text} for c in svc.chunks.itertuples()} if a.pipeline == "improved" else None) if gold else None),
                       context=r["context_text"], faithful=None)
            if not r["abstained"] and not a.no_judge and svc.groq_api_key:
                row["faithful"] = judge(svc, row)
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(qs)}", flush=True)

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    result = {"pipeline": a.pipeline, "split": a.split, "overrides": overrides, "overall": aggregate(rows),
              "by_category": {c: aggregate(v) for c, v in sorted(by_cat.items())}}
    name = a.name or f"{a.pipeline}_{a.split}"
    (OUT / f"{name}.json").write_text(json.dumps({**result, "rows": [{k: v for k, v in r.items() if k != "context"} for r in rows]},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"overall": result["overall"]}, indent=0))


if __name__ == "__main__":
    main()
