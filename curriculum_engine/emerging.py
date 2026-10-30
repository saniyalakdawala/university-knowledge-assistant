"""Capability 6 - emerging-topic gaps. A programme has a gap for a topic from the cited reference list
(curriculum_engine/emerging_topics.json, verified entries only) when no unit of its courses is semantically close to it."""
import json
from pathlib import Path

import numpy as np

from curriculum_engine.data import chunks

REFERENCE = json.loads((Path(__file__).parent / "emerging_topics.json").read_text(encoding="utf-8"))


def topics(verified_only=True):
    return [t for t in REFERENCE["topics"] if t["verified"] or not verified_only]


def coverage(view, emb, tau, k=5):
    """[{topic, programme, covered, best_similarity, evidence}] for every (verified topic, relevant programme)."""
    out = []
    for t in topics():
        q = emb([f"{t['topic']}: {t['description']}"])[0]
        for p in t["programmes"]:
            if p not in view:
                continue
            units = [(c["course_name"], title, text) for f, c in sorted(view[p].items()) for title, text in chunks(c)]
            s = emb([u[2] for u in units]) @ q
            top = np.argsort(-s)[:k]
            out.append({"topic": t["topic"], "programme": p, "source": t["source"], "covered": bool(s[top[0]] >= tau),
                        "best_similarity": round(float(s[top[0]]), 3),
                        "evidence": [{"course": units[i][0], "unit": units[i][1], "similarity": round(float(s[i]), 3)} for i in top]})
    return out
