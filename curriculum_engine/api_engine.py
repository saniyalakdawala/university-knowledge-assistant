"""The API-facing engine (Week 6): reference resolution, confidence values taken from the evaluation results, and the
programme-level answers. Uses prerequisite checker v2 and the overlap fusion selected on validation in Week 4.
Week 3's Engine.compare is left unchanged so that Week 3's results stay reproducible."""
import hashlib
import json
import re

import numpy as np

from curriculum_engine import RESULTS, __version__
from curriculum_engine import bloom as B
from curriculum_engine import data as D
from curriculum_engine import prereq as PQ
from curriculum_engine.engine import Engine
from curriculum_engine.overlap import chunk_vectors, maxsim, topics


def _load(path, default=None):
    f = RESULTS / f"{path}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default


def _soft(emb):
    return lambda a, b: maxsim(chunk_vectors(emb, a), chunk_vectors(emb, b), 1.0)["soft"]


def _max_unit(emb):
    def f(a, b):
        A, Bm = chunk_vectors(emb, a), chunk_vectors(emb, b)
        return float((A @ Bm.T).max()) if len(A) and len(Bm) else 0.0
    return f


class ApiEngine(Engine):
    MIN_AY_COURSES = 10

    def __init__(self, thresholds=None, recs=None):
        super().__init__(thresholds, recs)
        self.conf_model = _load("week4/confidence_model")
        self.fusion_ref = _load("week6/fusion_reference")
        self.bloom_conf = _load("week6/bloom_confidence", {})
        fp = _load("week3/flag_precision", {})
        pv = _load("week6/prerequisites_v2", {})
        self.red_precision = fp.get("redundancy", {}).get("core", {}).get("precision")
        self.gap_precision = fp.get("emerging_gap", {}).get("core", {}).get("precision")
        self.pre_precision = (pv.get("fresh_sample") or {}).get("test_v2", {}).get("precision")
        self.own_tau = pv.get("own_content_threshold_from_val")
        self.split_hash = D.split_hash(self.split)
        self.soft, self.max_unit = _soft(self.emb), _max_unit(self.emb)

    # ------------------------------------------------------------------ inputs
    def meta_block(self):
        return {"engine_version": __version__, "encoder": self.emb.name, "course_versions": len(self.recs), "split_hash": self.split_hash,
                "evaluation_note": "confidence values come from LLM-annotated reference labels (no human annotators); indicative"}

    def resolve(self, ref):
        """CourseRef -> course record: a dataset version (by id or name) or a record parsed from raw text."""
        if ref.id:
            if ref.id not in self.by_id:
                raise KeyError(f"unknown course id {ref.id}")
            return self.by_id[ref.id]
        if ref.name:
            key = D.norm_name(ref.name)
            hits = [r for r in self.recs if D.norm_name(r["course_name"]) == key and (not ref.programme or ref.programme in r["programmes"])
                    and (not ref.academic_year or r["academic_year"] == ref.academic_year)]
            if not hits:
                raise KeyError(f"no course named {ref.name!r}")
            return max(hits, key=lambda r: (r["academic_year"], r["n_source_files"], r["id"]))
        return self.from_text(ref)

    def from_text(self, ref):
        from curriculum_engine import parse as P
        rec = None
        for block in P.split_blocks(P.clean_pages([ref.text])):
            rec = P.parse_block(block, "api-input")
            if rec:
                rec.pop("_prereq_fixed", None)
                break
        if rec is None:                                   # not the NMIMS template: one unit per paragraph
            paras = [P.norm(p) for p in re.split(r"\n\s*\n", ref.text) if len(P.norm(p).split()) >= 4]
            rec = {"course_name": "input course", "code": "", "academic_year": ref.academic_year or "", "semester": "", "prerequisites": "",
                   "objectives": "", "outcomes": [], "units": [{"title": "", "text": p, "hours": None} for p in paras[:60]]}
        h = "tx" + hashlib.sha1(ref.text.encode("utf-8")).hexdigest()[:8]
        rec.update({"id": h, "family": h, "programmes": [ref.programme] if ref.programme else [], "common": False,
                    "sem": ref.semester or D.sem_min(rec.get("semester")), "n_source_files": 0})
        return rec

    # ------------------------------------------------------------------ scores
    def features(self, a, b):
        return {"tfidf_cosine": self.tfidf.cosine(a, b), "maxsim_minilm": self.soft(a, b), "max_unit_minilm": self.max_unit(a, b)}

    def confidence(self, f):
        """P(any overlap) from the Week 4 logistic model (fitted on validation labels; relative, not well calibrated)."""
        z = np.dot(self.conf_model["coef"], [f[k] for k in self.conf_model["features"]]) + self.conf_model["intercept"]
        return float(1 / (1 + np.exp(-z)))

    def fusion(self, f):
        """The Week 4 selected score: average percentile of TF-IDF and MaxSim among the validation pairs."""
        pct = lambda ref, x: float(np.searchsorted(ref, x, side="right")) / len(ref)
        return (pct(self.fusion_ref["tfidf_cosine"], f["tfidf_cosine"]) + pct(self.fusion_ref["maxsim_minilm"], f["maxsim_minilm"])) / 2

    # ------------------------------------------------------------------ answers
    def bloom_items(self, texts, tags=None):
        out = []
        for i, t in enumerate(texts):
            name = B.BLOOM[B.lexicon_level(t) or 3]       # Apply when no verb is known (the training majority)
            out.append({"outcome": t, "level": name, "confidence": self.bloom_conf.get(name), "faculty_tag": (tags[i] or None) if tags else None})
        return out

    def prereq_v2(self, c):
        if c.get("common"):
            return {"checked": False, "reason": "common first-year / all-programme course", "flags": []}
        if not c.get("sem") or c["sem"] < 2:
            return {"checked": False, "reason": "semester unknown or first semester", "flags": []}
        names = {}
        for r in sorted(self.recs, key=lambda r: r["id"]):
            names.setdefault(r["family"], r["course_name"])
        own, flags, checked = PQ.OwnChecker(self.emb, self.own_tau), [], False
        for p in c.get("programmes", []):
            if p not in self.st or not PQ.covered(self.st, self.versions, p, c["sem"]):
                continue
            checked = True
            fl = PQ.check_programme_v2(p, {p: {c["family"]: c}}, self.st, self.resolver, self.versions, self.topic_checker, own, names,
                                       families={c["family"]})
            flags += [{"programme": f["programme"], "phrase": f["phrase"], "type": f["type"], "expected_precision": self.pre_precision} for f in fl]
        return {"checked": checked, "flags": flags,
                "reason": None if checked else "no programme given, or too little parsed data for this semester (scanned booklets)"}

    def compare_courses(self, a, b, k=10):
        m = maxsim(chunk_vectors(self.emb, a), chunk_vectors(self.emb, b), self.th["unit_match"])
        t = topics(a, b, m, self.th["unit_match"], k)
        if t["common_topics"]:
            for x, n in zip(t["common_topics"], self.topic_model.assign([x["topic_a"] for x in t["common_topics"]])):
                x["topic_label"] = n
        f = self.features(a, b)
        tags = lambda c: [o["k_levels"] for o in c.get("outcomes", [])]
        return {"meta": self.meta_block(), "course_a": self.meta(a), "course_b": self.meta(b),
                "overlap": {"percent": f"{m['overlap']:.0%}", "semantic_similarity": round(m["soft"], 3),
                            "lexical_similarity": round(f["tfidf_cosine"], 3), "confidence": round(self.confidence(f), 3),
                            "redundancy_warning": bool(m["soft"] >= self.th["substantial_overlap"])},
                **t, "prerequisites_a": self.prereq_v2(a), "prerequisites_b": self.prereq_v2(b),
                "outcome_bloom_a": self.bloom_items([o["text"] for o in a.get("outcomes", [])], tags(a)),
                "outcome_bloom_b": self.bloom_items([o["text"] for o in b.get("outcomes", [])], tags(b))}

    def snapshot(self, prog, ay=None):
        from curriculum_engine import diff as DF
        if ay is None:
            ays = sorted({r["academic_year"] for r in self.recs if prog in r["programmes"]}, reverse=True)
            ay = next((y for y in ays if len(DF.snapshot(self.recs, prog, y)) >= self.MIN_AY_COURSES), ays[0] if ays else None)
        return ay, DF.snapshot(self.recs, prog, ay)

    def compare_programmes(self, pa, pb):
        from curriculum_engine import diff as DF
        ay_a, A = self.snapshot(pa.programme, pa.academic_year)
        ay_b, Bs = self.snapshot(pb.programme, pb.academic_year)
        sim = lambda x, y: (lambda v: v if v >= self.th["substantial_overlap"] else None)(self.soft(x, y))
        d = DF.diff_by_content(A, Bs, sim)
        renamed = [(A[x]["course_name"], Bs[y]["course_name"]) for x, y in d["retained"]
                   if D.norm_name(A[x]["course_name"]) != D.norm_name(Bs[y]["course_name"])]
        return {"meta": self.meta_block(), "a": {"programme": pa.programme, "academic_year": ay_a}, "b": {"programme": pb.programme, "academic_year": ay_b},
                "retained": len(d["retained"]), "renamed_or_revised": renamed,
                "removed": [{"family": f, "course_name": A[f]["course_name"]} for f in d["removed"]],
                "added": [{"family": f, "course_name": Bs[f]["course_name"]} for f in d["added"]],
                "note": "only text-layer booklets are parsed: a course 'removed' or 'added' may sit in a scanned booklet of the other year"}

    def find_overlaps(self, c, k=5, programme=None):
        cands = [r for f, r in self.reps.items() if f != c["family"] and (programme is None or programme in r["programmes"])]
        if not cands:
            return []
        T = self.tfidf.matrix([c] + cands)
        lex = (T[1:] @ T[0].T).toarray().ravel()
        hits = []
        for r, l in zip(cands, lex):
            f = {"tfidf_cosine": float(l), "maxsim_minilm": self.soft(c, r), "max_unit_minilm": self.max_unit(c, r)}
            hits.append({"course": self.meta(r), "score": round(self.fusion(f), 3), "confidence": round(self.confidence(f), 3)})
        return sorted(hits, key=lambda h: (-h["score"], -h["confidence"], h["course"]["id"]))[:k]

    def detect_redundancy(self, prog, ay=None, k=20):
        from curriculum_engine.data import chunks
        cs = ([c for f, c in sorted(self.view.get(prog, {}).items()) if not c["common"]] if ay is None
              else list(self.snapshot(prog, ay)[1].values()))
        cs = [c for c in cs if chunks(c)]
        if len(cs) < 2:
            return len(cs), []
        lab = [(i, title) for i, c in enumerate(cs) for title, _ in chunks(c)]
        U = np.vstack([chunk_vectors(self.emb, c) for c in cs])
        owner = np.array([i for i, _ in lab])
        fam = np.array([cs[i]["family"] for i in owner])
        S = U @ U.T
        S[np.tril_indices(len(S))] = -1
        S[fam[:, None] == fam[None, :]] = -1                 # versions of one course are not redundancy
        out, seen = [], set()
        for idx in np.argsort(-S, axis=None):
            a, b = np.unravel_index(idx, S.shape)
            if S[a, b] < self.th["unit_duplicate"] or len(out) >= k:
                break
            pair = tuple(sorted((int(owner[a]), int(owner[b]))))
            if pair in seen:
                continue
            seen.add(pair)
            out.append({"course_a": cs[owner[a]]["course_name"], "unit_a": lab[a][1], "course_b": cs[owner[b]]["course_name"],
                        "unit_b": lab[b][1], "similarity": round(float(S[a, b]), 3), "expected_precision": self.red_precision})
        return len(cs), out

    def emerging(self, prog):
        from curriculum_engine import emerging as EM
        # Week 7: the validation-tuned unit-match threshold replaces the prerequisite topic threshold (0.40), which made
        # almost every topic look covered (accuracy on the labelled emerging-topic decisions: results/week7/emerging_threshold.json)
        items = [g for g in EM.coverage({prog: self.view.get(prog, {})}, self.emb, self.th["unit_match"]) if g["programme"] == prog]
        return {"reference_status": EM.REFERENCE["status"], "gap_precision": self.gap_precision,
                "note": "coverage = a unit at cosine >= the validation-tuned unit-match threshold; 'covered' and 'gap' are both indicative (see results/week7)",
                "topics": [{"topic": g["topic"], "covered": g["covered"], "best_similarity": g["best_similarity"],
                            "closest_units": [f"{e['course']} - {e['unit']}" for e in g["evidence"][:3]],
                            "source": EM.REFERENCE["sources"][g["source"]]["title"]} for g in items]}
