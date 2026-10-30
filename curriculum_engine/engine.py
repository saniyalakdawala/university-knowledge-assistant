"""The reusable engine: one object that answers the committee's questions with JSON. Thresholds come from validation tuning
(results/week3/thresholds.json); nothing here looks at test labels."""
import json
from collections import Counter
from functools import cached_property

from curriculum_engine import RESULTS, __version__
from curriculum_engine import bloom as B
from curriculum_engine import data as D
from curriculum_engine import prereq as PQ
from curriculum_engine.embed import MINILM, Embedder
from curriculum_engine.overlap import chunk_vectors, maxsim, topics


def load_thresholds():
    return json.loads((RESULTS / "week3/thresholds.json").read_text(encoding="utf-8"))


class Engine:
    def __init__(self, thresholds=None, recs=None):
        self.th = thresholds or load_thresholds()
        self.recs = recs if recs is not None else D.load()
        self.by_id = {r["id"]: r for r in self.recs}
        self.emb = Embedder(MINILM)

    @cached_property
    def split(self):
        return D.split(self.recs)

    @cached_property
    def reps(self):
        return D.representatives(self.recs)

    @cached_property
    def view(self):
        return PQ.programme_view(self.recs)

    @cached_property
    def st(self):
        return D.structure(self.recs)

    @cached_property
    def versions(self):
        return Counter(p for r in self.recs for p in r["programmes"])

    @cached_property
    def tfidf(self):
        from curriculum_engine.overlap import Tfidf
        return Tfidf([c for f, c in self.reps.items() if self.split[f] == "train"])

    @cached_property
    def resolver(self):
        return PQ.EmbeddingResolver(self.recs, self.emb, self.th["prereq_resolve"])

    @cached_property
    def topic_checker(self):
        return PQ.TopicChecker(self.reps, self.emb, self.th["prereq_topic"])

    @cached_property
    def topic_model(self):
        from curriculum_engine.topics import TopicModel
        return TopicModel(self.emb).fit([c for f, c in sorted(self.reps.items()) if self.split[f] == "train"])

    # ------------------------------------------------------------------ answers
    @staticmethod
    def meta(c):
        return {k: c.get(k, "") for k in ("id", "course_name", "code", "academic_year", "semester")} | {"programmes": c.get("programmes", [])}

    def compare(self, a, b, k=10):
        m = maxsim(chunk_vectors(self.emb, a), chunk_vectors(self.emb, b), self.th["unit_match"])
        t = topics(a, b, m, self.th["unit_match"], k)
        names = self.topic_model.assign([x["topic_a"] for x in t["common_topics"]]) if t["common_topics"] else []
        for x, n in zip(t["common_topics"], names):
            x["topic_label"] = n
        return {"course_a": self.meta(a), "course_b": self.meta(b),
                "overlap": {"percent": f"{m['overlap']:.0%}", "unit_coverage_f": round(m["overlap"], 3),
                            "semantic_similarity": round(m["soft"], 3), "lexical_similarity": round(self.tfidf.cosine(a, b), 3),
                            "redundancy_warning": m["soft"] >= self.th["substantial_overlap"]},
                **t,
                "missing_prerequisites": {"a": self.check_prerequisites(a), "b": self.check_prerequisites(b)},
                "outcome_bloom": {"a": self.map_outcomes(a)[:4], "b": self.map_outcomes(b)[:4]},
                "method": f"MiniLM chunk MaxSim (unit match >= {self.th['unit_match']}), TF-IDF, programme-graph prerequisite check",
                "version": __version__}

    def check_prerequisites(self, c):
        out = []
        for p in c.get("programmes", []):
            if p in self.view and c["family"] in self.view[p]:
                fl = PQ.check_programme(p, {p: {c["family"]: c}}, self.st, self.resolver, self.versions, self.topic_checker, families={c["family"]})
                out += [{k: f[k] for k in ("programme", "phrase", "type")} for f in fl]
        return out

    @staticmethod
    def map_outcomes(c):
        out = []
        for o in c.get("outcomes", []):
            lvl = B.lexicon_level(o["text"])
            out.append({"outcome": o["text"][:90], "level": B.BLOOM.get(lvl, "unknown"), "faculty_tag": o["k_levels"] or None})
        return out
