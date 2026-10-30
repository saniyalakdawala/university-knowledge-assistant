"""Benchmark metrics. Relevance is pipeline-independent: a passage is relevant to a gold fact when it holds the fact's evidence
span and sits on its page (+/- 1). Chunks of different pipelines are therefore judged by the same rule."""
import math
import re
from typing import Dict, Iterable, List, Sequence

from backend.rag_service import canon_tokens

EVIDENCE_CONTAINMENT = 0.7


def is_relevant(fact: Dict, text: str, page: int, end_page: int = None) -> bool:
    end_page = page if end_page is None else end_page
    if not (page - 1 <= fact["page"] <= end_page + 1):
        return False
    ev = set(canon_tokens(fact["evidence"]))
    return bool(ev) and len(ev & set(canon_tokens(text))) / len(ev) >= EVIDENCE_CONTAINMENT


def retrieval_scores(facts: Sequence[Dict], passages: Sequence[Dict], ks: Iterable[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """Recall/Precision/MRR/nDCG@K for one question. passages: best first, each {text, page, end_page}."""
    rel = [[i for i, f in enumerate(facts) if is_relevant(f, p["text"], p["page"], p.get("end_page"))] for p in passages]
    out = {}
    for k in ks:
        top = rel[:k]
        covered = {i for r in top for i in r}
        out[f"recall@{k}"] = len(covered) / len(facts)
        out[f"precision@{k}"] = sum(bool(r) for r in top) / k
        first = next((rank for rank, r in enumerate(top, 1) if r), None)
        out[f"mrr@{k}"] = 1.0 / first if first else 0.0
        seen, gain = set(), []
        for r in top:                                  # a passage gains only for a fact not yet covered
            new = set(r) - seen
            gain.append(1.0 if new else 0.0)
            seen |= new
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gain))
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(facts))))
        out[f"ndcg@{k}"] = dcg / ideal
    return out


def key_in_answer(key: str, answer: str) -> bool:
    """Week-5 normalised key-fact match: any '|' alternative occurs as an ordered token subsequence of the answer."""
    toks = canon_tokens(answer)
    low = answer.lower()
    for alt in key.split("|"):
        alt = alt.strip()
        if not alt:
            continue
        if alt.lower() in low:
            return True
        want, i = canon_tokens(alt), 0
        for t in toks:
            if i < len(want) and t == want[i]:
                i += 1
        if want and i == len(want):
            return True
    return False


def answer_correct(item_facts: Sequence[Dict], result: Dict) -> bool:
    if not item_facts:
        return bool(result["abstained"])
    return not result["abstained"] and all(key_in_answer(f["key"], result["answer"]) for f in item_facts)


def citation_correct(item_facts: Sequence[Dict], result: Dict, chunks: Dict[str, Dict] = None) -> bool:
    """Every gold fact is supported by a returned citation: right page (+/- 1) and its quote or passage holds the evidence."""
    cites = result.get("citations") or ([{"page": result["cited_page"], "quote": result.get("evidence_quote", "")}] if result.get("cited_page") else [])
    for f in item_facts:
        ok = False
        for c in cites:
            text = c.get("quote", "")
            if chunks and c.get("chunk_id") in chunks:
                text += " " + chunks[c["chunk_id"]]["text"]
            if is_relevant(f, text, c["page"]):
                ok = True
                break
        if not ok:
            return False
    return True


JUDGE_SYSTEM = ("You check whether an answer is faithful to the source passages. Reply with exactly one word: "
                "SUPPORTED if every claim in the answer is stated in the passages, otherwise UNSUPPORTED.")


def judge_messages(question: str, answer: str, context: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\nVerdict:"}]


def parse_verdict(text: str) -> bool:
    return bool(re.search(r"\bSUPPORTED\b", text or "")) and "UNSUPPORTED" not in (text or "")
