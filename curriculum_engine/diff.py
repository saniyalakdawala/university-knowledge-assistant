"""Capability 4 - compare two curricula: a programme in two academic years (or two programmes).
Output: courses retained, removed and added. Baseline (Week 2): exact course-name match. Core (Week 3): name or content match.
Ground truth for evaluation: course families (same code or same name), which neither method uses."""
from curriculum_engine import data as D


def snapshot(recs, prog, ay):
    """Courses of one programme in one academic year (latest record per family)."""
    out = {}
    for r in sorted(recs, key=lambda r: r["id"]):
        if r["academic_year"] == ay and prog in r["programmes"]:
            out[r["family"]] = r
    return out


def diff_by_name(A, B):
    na = {D.norm_name(c["course_name"]): f for f, c in A.items()}
    nb = {D.norm_name(c["course_name"]): f for f, c in B.items()}
    return {"retained": sorted((na[k], nb[k]) for k in na.keys() & nb.keys()),
            "removed": sorted(na[k] for k in na.keys() - nb.keys()), "added": sorted(nb[k] for k in nb.keys() - na.keys())}


def truth(A, B):
    return {"removed": sorted(A.keys() - B.keys()), "added": sorted(B.keys() - A.keys())}


def score(pred, true):
    """Precision / recall of the 'removed' and 'added' claims (pooled)."""
    tp = len(set(pred["removed"]) & set(true["removed"])) + len(set(pred["added"]) & set(true["added"]))
    p, t = len(pred["removed"]) + len(pred["added"]), len(true["removed"]) + len(true["added"])
    return tp, p, t


def year_pairs(recs, min_courses=10):
    """(programme, AY, next AY) with at least min_courses parsed courses in both years."""
    progs = sorted({p for r in recs for p in r["programmes"]})
    ays = sorted({r["academic_year"] for r in recs})
    out = []
    for p in progs:
        for a, b in zip(ays, ays[1:]):
            if len(snapshot(recs, p, a)) >= min_courses and len(snapshot(recs, p, b)) >= min_courses:
                out.append((p, a, b))
    return out


def diff_by_content(A, B, similar):
    """Core: match by exact name first, then pair each unmatched course of A with its most similar unmatched course of B when
    similar(a, b) >= its threshold (content MaxSim); what stays unmatched is removed / added."""
    base = diff_by_name(A, B)
    rem, add = set(base["removed"]), set(base["added"])
    retained = list(base["retained"])
    cand = sorted(((s, a, b) for a in rem for b in add for s in [similar(A[a], B[b])] if s is not None), reverse=True)
    for s, a, b in cand:
        if a in rem and b in add:
            rem.discard(a); add.discard(b); retained.append((a, b))
    return {"retained": sorted(retained), "removed": sorted(rem), "added": sorted(add)}
