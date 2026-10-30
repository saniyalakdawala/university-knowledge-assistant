"""Curriculum Intelligence service: runs the Curriculum Intelligence Engine's own algorithms (vendored in curriculum_engine/) on
this project's course data.

Data, best first:
  1. data/curriculum/courses.jsonl  full syllabi in the engine's format (units, outcomes, prerequisites), if present. The engine's
     parsed dataset is not public, so this file ships only if the team adds it.
  2. data/curriculum/srb_catalog.json  every programme structure printed in the SRB: real semesters, credits and course names, but
     no unit text and no prerequisites.

Every result says which of the two it rests on (`data_basis`). On SRB records the engine compares course NAMES, so its overlap,
redundancy and coverage figures are indications only, and they are never described as syllabus-content findings.
"""
import copy
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import BASE_DIR, DATA_DIR, LOW_MEMORY
from backend.curriculum.srb_catalog import save_catalog

logger = logging.getLogger(__name__)

CATALOG = DATA_DIR / "curriculum" / "srb_catalog.json"
FULL_SYLLABI = DATA_DIR / "curriculum" / "courses.jsonl"
JD_DIR = BASE_DIR / "demo" / "jds"
NAME_ONLY_NOTE = ("Based on course names and semesters printed in the SRB. The SRB has no unit-level syllabi, so this compares names, "
                  "not syllabus content.")
FULL_NOTE = "Based on full syllabus text (units, outcomes, prerequisites)."
NAME_DUPLICATE = 0.75                # name-to-name similarity: stricter than the engine's unit threshold, which was tuned on unit text
NAME_MATCH = 0.62                    # a phrase names a course when its name embedding is at least this close
ABBREVIATIONS = {
    "nlp": "Natural Language Processing", "ml": "Machine Learning", "dl": "Deep Learning", "os": "Operating Systems",
    "dbms": "Database Management Systems", "dsa": "Data Structures and Algorithms", "cn": "Computer Networks",
    "oop": "Object Oriented Programming", "oops": "Object Oriented Programming", "coa": "Computer Organization and Architecture",
    "hci": "Human Computer Interaction", "daa": "Design and Analysis of Algorithms", "se": "Software Engineering",
    "ai": "Foundations of Artificial Intelligence", "cv": "Computer Vision", "toc": "Theory of Computation",
}
PROGRAMME_ALIASES = [
    (r"\bai\s*[-&/]?\s*ml\b|\bartificial intelligence (?:and |& )?machine learning\b", "AI-ML"),
    (r"\bcse?\s*[-(]?\s*ds\b|\bcomputer science and engineering \(?data science\b", "CSE-DS"),
    (r"\bcyber ?security\b|\bcse?\s*[-(]?\s*cyber\b", "CSE-CYBER"), (r"\bcsbs\b|\bbusiness systems\b", "CSBS"),
    (r"\bartificial intelligence\b(?! and machine)|\bb ?tech ai\b", "AI"), (r"\bdata science\b|\bb ?tech ds\b", "DS"),
    (r"\bcomputer science\b|\bb ?tech cs\b", "CS"), (r"\bcomputer engineering\b|\bcomps?\b|\bce branch\b", "COMP"),
    (r"\binformation technology\b|\bit (?:branch|programme|program|students?|dept|department)\b|\bb ?tech it\b|\bmba ?tech it\b", "IT"),
    (r"\bextc\b|\belectronics (?:and|&) telecommunication\b", "EXTC"), (r"\bcivil\b", "CIVIL"),
    (r"\bmechatronics\b", "MECHATRONICS"), (r"\bmechanical\b|\bmech\b", "MECHANICAL"), (r"\bmca\b", "MCA"),
]
DEGREE_SUFFIX = [(r"\bmba ?tech\b", "MBA-TECH"), (r"\bm ?tech\b", "M-TECH"), (r"\bdiploma\b", "DIPLOMA")]
INTENTS = ["course_search", "course_info", "semester_courses", "compare_courses", "compare_programmes", "overlap", "redundancy",
           "missing_topics", "prerequisites", "bloom", "modernize"]


class LexicalEmbedder:
    """Stand-in for the sentence encoder in LOW_MEMORY mode (no PyTorch): hashed character n-grams, L2-normalised."""
    name = "hashed character n-grams (LOW_MEMORY)"

    def __init__(self, *_):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2048, norm="l2", alternate_sign=False)

    def __call__(self, texts):
        return self.vec.transform(list(texts)).toarray().astype(np.float32)

    def save(self):
        pass


def _load_records() -> Tuple[List[Dict], Dict]:
    from curriculum_engine import data as D
    if not CATALOG.exists():
        from backend.config import SRB_PDF_PATH
        CATALOG.parent.mkdir(parents=True, exist_ok=True)
        save_catalog(SRB_PDF_PATH, str(CATALOG))
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    recs = payload["courses"]
    for r in recs:
        r["calendar_year"] = r["academic_year"]
        r["academic_year"] = "2026-27"       # the SRB edition: one snapshot of each programme, whatever year a semester falls in
    if FULL_SYLLABI.exists():
        full = D.enrich([json.loads(line) for line in open(FULL_SYLLABI, encoding="utf-8")])
        for r in full:
            r["content_level"] = "full_syllabus"
        recs = recs + full
    fam = D._families(recs)
    for r in recs:
        r["family"] = fam[r["id"]]
    return recs, payload


class CurriculumService:
    def __init__(self):
        import curriculum_engine.engine as E
        from curriculum_engine import prereq as PQ
        from curriculum_engine.api_engine import ApiEngine
        PQ.MIN_VERSIONS, PQ.MIN_FAMILIES_PER_SEM = 20, 3       # the SRB lists each programme completely; the engine's rule guards scanned booklets
        if LOW_MEMORY:
            E.Embedder = LexicalEmbedder
        self.recs, self.payload = _load_records()
        self.full_data = FULL_SYLLABI.exists()
        thresholds = dict(E.load_thresholds())
        if not self.full_data:
            thresholds["unit_duplicate"] = NAME_DUPLICATE
        self.eng = ApiEngine(thresholds=thresholds, recs=self.recs)
        self.PQ = PQ
        self.names = sorted({r["course_name"] for r in self.recs})
        self.name_vecs = None
        self.addon_names = {r["course_name"] for r in self.recs if r.get("category") == "COUC"}   # Coursera credits are optional add-ons, not programme redundancy
        logger.info(f"Curriculum service: {len(self.recs)} course records, {len(self.names)} distinct names, full syllabi: {self.full_data}")

    def warm(self) -> None:
        """Embed the catalogue ahead of the first question and keep the vectors on disk (first run about a minute, then seconds)."""
        self._name_matrix()
        _ = self.eng.resolver, self.eng.topic_checker, self.eng.topic_model
        self.eng.emb.save()

    def stats(self) -> Dict:
        return {"programmes": len(self.eng.view), "course_records": len(self.recs), "distinct_courses": len(self.names),
                "full_syllabi_loaded": self.full_data, "encoder": getattr(self.eng.emb, "name", ""), "source": self.payload["source"]}

    # ----------------- entities -----------------
    def programmes(self) -> List[str]:
        return sorted(self.eng.view)

    def detect_programmes(self, text: str) -> List[str]:
        low = text.lower()
        suffix = next((s for rx, s in DEGREE_SUFFIX if re.search(rx, low)), "")
        found: List[Tuple[int, str]] = []
        taken: List[Tuple[int, int]] = []
        for rx, code in PROGRAMME_ALIASES:
            for m in re.finditer(rx, low):
                if not any(a < m.end() and m.start() < b for a, b in taken):
                    taken.append((m.start(), m.end()))
                    full = f"{code}-{suffix}" if suffix else code
                    found.append((m.start(), full if full in self.eng.view else code))
        return list(dict.fromkeys(c for _, c in sorted(found)))

    def _name_matrix(self) -> np.ndarray:
        if self.name_vecs is None:
            self.name_vecs = self.eng.emb(self.names)
        return self.name_vecs

    def resolve_name(self, phrase: str) -> Optional[str]:
        """A phrase -> a catalogue course name: abbreviation, exact, unique containment, then embedding."""
        p = re.sub(r"[?.!,]+$", "", phrase.strip())
        p = ABBREVIATIONS.get(p.lower(), p)
        key = p.lower()
        for n in self.names:
            if n.lower() == key:
                return n
        contains = [n for n in self.names if key in n.lower() and len(key) >= 4 and "lab" not in n.lower()]
        if contains:
            return min(contains, key=len)
        s = self._name_matrix() @ self.eng.emb([p])[0]
        i = int(np.argmax(s))
        return self.names[i] if s[i] >= NAME_MATCH else None

    def detect_courses(self, text: str) -> List[str]:
        """Course names mentioned in a question, in order of mention."""
        low = f" {re.sub(r'[^a-z0-9&+ ]+', ' ', text.lower())} "
        hits: List[Tuple[int, int, str]] = []
        for n in self.names:
            nl = f" {re.sub(r'[^a-z0-9&+ ]+', ' ', n.lower())} "
            pos = low.find(nl)
            if pos >= 0 and len(n) > 3:
                hits.append((pos, -len(nl), n))
        for abbr, full in ABBREVIATIONS.items():
            pos = low.find(f" {abbr} ")
            if pos >= 0 and abbr not in ("se", "ai", "cv", "os") or (pos >= 0 and abbr == "os"):
                resolved = self.resolve_name(full)
                if resolved:
                    hits.append((pos, -len(abbr) - 2, resolved))
        chosen: List[Tuple[int, int]] = []
        out: List[str] = []
        for pos, neg, n in sorted(hits, key=lambda h: (h[1], h[0])):      # longest match first; a shorter name inside it is dropped
            end = pos - neg
            if not any(pos < b and a < end for a, b in chosen):
                chosen.append((pos, end))
                out.append((pos, n))
        return list(dict.fromkeys(n for _, n in sorted(out)))

    def record(self, name: str, programme: Optional[str] = None) -> Dict:
        """The record of a course, in the given programme when it is taught there; a full syllabus is preferred."""
        key = name.lower()
        hits = [r for r in self.recs if r["course_name"].lower() == key]
        if not hits:
            raise KeyError(f"no course named {name!r}")
        scoped = [r for r in hits if programme in r["programmes"]] if programme else []
        pool = scoped or hits
        return max(pool, key=lambda r: (r.get("content_level") == "full_syllabus", r.get("category") != "COUC", r.get("degree") == "B Tech",
                                        r["programmes"][:1] == ["COMP"], r["id"]))

    def _basis(self, *courses: Dict) -> Tuple[str, str]:
        full = courses and all(c.get("content_level") == "full_syllabus" or not c.get("source_page") for c in courses)
        return ("full_syllabus", FULL_NOTE) if full else ("name_only", NAME_ONLY_NOTE)

    @staticmethod
    def _cite(c: Dict) -> Dict:
        return {"section": f"Programme structure: {c.get('program', '')}", "page": c.get("source_page")}

    @staticmethod
    def brief(c: Dict) -> Dict:
        return {"id": c["id"], "course_name": c["course_name"], "programmes": c.get("programmes", []), "degree": c.get("degree", ""),
                "semester": c.get("sem"), "credits": c.get("credits"), "L-P-T": f"{c.get('lecture_hrs', '-'):g}-{c.get('practical_hrs', '-'):g}-{c.get('tutorial_hrs', '-'):g}"
                if c.get("lecture_hrs") is not None else "", "category": c.get("category", ""), "page": c.get("source_page")}

    # ----------------- workflows -----------------
    def search(self, q: str, programme: Optional[str] = None, k: int = 10) -> Dict:
        """Course search: by name match first, then by meaning."""
        pool = [r for r in self.recs if programme is None or programme in r["programmes"]]
        names = sorted({r["course_name"] for r in pool})
        ql = q.lower().strip()
        direct = [n for n in names if ql and ql in n.lower()]
        scored: List[Tuple[float, str]] = [(1.0, n) for n in direct]
        if len(direct) < k and names:
            sims = self.eng.emb(names) @ self.eng.emb([q])[0]
            scored += [(round(float(sims[i]), 3), names[i]) for i in np.argsort(-sims)[:k * 2] if names[i] not in direct and sims[i] >= 0.35]
        hits = []
        for score, n in scored[:k]:
            rows = [r for r in pool if r["course_name"] == n]
            hits.append({**self.brief(rows[0]), "offered_in": sorted({p for r in rows for p in r["programmes"]}), "score": score})
        return {"query": q, "programme": programme, "results": hits}

    def course_info(self, name: str, programme: Optional[str] = None) -> Dict:
        rows = [r for r in self.recs if r["course_name"].lower() == name.lower()]
        return {"course_name": name, "offerings": [self.brief(r) for r in sorted(rows, key=lambda r: (r["programmes"], r["sem"] or 0))
                                                   if not programme or programme in r["programmes"]] or [self.brief(r) for r in rows]}

    def semester_courses(self, programme: str, semester: int) -> Dict:
        rows = [r for r in self.recs if programme in r["programmes"] and r.get("sem") == semester and r.get("category") != "COUC"]
        return {"programme": programme, "semester": semester, "courses": [self.brief(r) for r in rows],
                "credits_listed": sum(r.get("credits") or 0 for r in rows)}

    def compare_courses(self, a: str, b: str, programme: Optional[str] = None) -> Dict:
        ra, rb = self.record(a, programme), self.record(b, programme)
        out = self.eng.compare_courses(ra, rb)
        out["data_basis"], out["note"] = self._basis(ra, rb)
        return out

    def compare_programmes(self, a: str, b: str) -> Dict:
        from curriculum_engine.schemas import ProgrammeRef
        out = self.eng.compare_programmes(ProgrammeRef(programme=a), ProgrammeRef(programme=b))
        out["data_basis"], out["note"] = ("full_syllabus", FULL_NOTE) if self.full_data else ("name_only", NAME_ONLY_NOTE)
        return out

    def find_overlaps(self, name: str, k: int = 5, programme: Optional[str] = None) -> Dict:
        c = self.record(name, programme)
        hits = [h for h in self.eng.find_overlaps(c, 3 * k, programme)
                if not re.search(r"(?i)\blab\b", h["course"]["course_name"])][:k]
        basis, note = self._basis(c)
        return {"query": self.brief(c), "results": hits, "data_basis": basis, "note": note}

    def detect_redundancy(self, programme: str, k: int = 15) -> Dict:
        """Repeated content inside one programme. A course and its own lab are one offering, not a redundancy."""
        n, pairs = self.eng.detect_redundancy(programme, None, 3 * k)
        strip = lambda s: re.sub(r"(?i)\b(lab|laboratory|practical|i{1,3}|iv|v)\b", "", s).strip(" -:")
        pairs = [p for p in pairs if strip(p["course_a"]).lower() != strip(p["course_b"]).lower()
                 and not re.search(r"(?i)\blab\b", p["course_a"] + p["course_b"])
                 and p["course_a"] not in self.addon_names and p["course_b"] not in self.addon_names][:k]
        basis, note = ("full_syllabus", FULL_NOTE) if self.full_data else ("name_only", NAME_ONLY_NOTE)
        return {"programme": programme, "courses_checked": n, "pairs": pairs, "data_basis": basis, "note": note}

    def missing_topics(self, programme: str) -> Dict:
        out = self.eng.emerging(programme)
        basis, note = ("full_syllabus", FULL_NOTE) if self.full_data else ("name_only", NAME_ONLY_NOTE)
        return {"programme": programme, **out, "data_basis": basis, "name_note": note}

    def prerequisites(self, name: str, programme: Optional[str] = None, phrases: Optional[List[str]] = None) -> Dict:
        """The engine's checker v2. The SRB states no prerequisites, so the check runs on the syllabus field when a full syllabus exists,
        or on topics the user names ('is Linear Algebra taught before ML?'). Earlier related courses are listed as an inference."""
        c = self.record(name, programme)
        prog = programme or (c["programmes"][0] if c["programmes"] else None)
        stated = bool(c.get("prerequisites", "").strip())
        target = copy.deepcopy(c)
        if phrases:
            target["prerequisites"] = ", ".join(phrases)
        target["programmes"] = [prog] if prog else []
        check = self.eng.prereq_v2(target)
        related: List[Dict] = []
        if prog and c.get("sem"):
            before = [r for r in self.eng.view.get(prog, {}).values() if r.get("sem") and r["sem"] < c["sem"] and not r.get("elective_pool")
                      and not re.search(r"(?i)\blab\b|coursera|community|ethics|communication", r["course_name"])]
            if before:
                sims = self.eng.emb([r["course_name"] for r in before]) @ self.eng.emb([c["course_name"]])[0]
                related = [{"course_name": before[i]["course_name"], "semester": before[i]["sem"], "similarity": round(float(sims[i]), 3)}
                           for i in np.argsort(-sims)[:4] if sims[i] >= 0.4]
        basis, note = self._basis(c)
        return {"course": self.brief(c), "programme": prog, "prerequisites_stated": stated, "stated_text": c.get("prerequisites", ""),
                "checked_phrases": phrases or [], "check": check, "related_earlier_courses": related, "data_basis": basis, "note": note}

    def bloom(self, outcomes: Optional[List[str]] = None, department: Optional[str] = None) -> Dict:
        """Bloom level of each outcome (the engine's verb lexicon; Apply when no listed verb is found)."""
        source = "supplied text"
        if not outcomes:
            depts = self.payload["departments"]
            chosen = next((k for k in depts if department and department.lower() in k.lower()), None) or next(iter(depts))
            outcomes, source = depts[chosen]["outcomes"], f"{chosen}, SRB PDF page {depts[chosen]['page']}"
        items = self.eng.bloom_items(outcomes)
        return {"source": source, "outcomes": items, "distribution": {b: sum(i["level"] == b for i in items) for b in dict.fromkeys(i["level"] for i in items)}}

    def demo_jds(self) -> Dict[str, str]:
        return {p.stem: p.read_text(encoding="utf-8") for p in sorted(JD_DIR.glob("*.txt"))}

    def modernize(self, courses: List[str], jds: Optional[Dict[str, str]] = None, title: Optional[str] = None, programme: Optional[str] = None) -> Dict:
        """Industry skill-gap analysis and a traceable draft syllabus, using the engine's method. No LLM writes any of it."""
        from curriculum_engine import modernize as M
        recs = [self.record(n, programme) for n in courses]
        sample = jds is None
        jds = jds or self.demo_jds()
        out = M.run(self.eng, jds, recs, title or f"Modernised {recs[0]['course_name']}", "2026-27")
        out.pop("handoffs", None)
        basis, note = self._basis(*recs)
        return {**out, "data_basis": basis, "note": note, "jd_source": "bundled synthetic sample job descriptions" if sample else "supplied job descriptions"}

    # ----------------- chat -----------------
    def classify(self, question: str) -> str:
        q = question.lower()
        if re.search(r"\b(moderni[sz]e|skill[- ]?gap|industry (?:skills?|needs?|demand)|job descriptions?|employers?)\b", q):
            return "modernize"
        if re.search(r"\bbloom", q):
            return "bloom"
        if re.search(r"\bredundan|\bduplicat|\brepeat(?:ed|s)? (?:content|topics?|courses?)\b", q):
            return "redundancy"
        if re.search(r"\b(missing|emerging|gaps?|lacks?|not (?:taught|covered)|outdated)\b", q):
            return "missing_topics"
        if re.search(r"\bprereq|\bpre-req|\bbefore (?:taking|doing|studying)\b|\bneed to know\b|\btaught before\b", q):
            return "prerequisites"
        if re.search(r"\boverlaps?\b|\bsimilar\b", q):
            return "overlap"
        if re.search(r"\b(compare|comparison|difference|differ|versus|vs)\b", q):
            return "compare_courses"
        if re.search(r"\b(sem(?:ester)?)\s*(\d|[ivx]+)\b|\b(\d)(?:st|nd|rd|th) sem", q):
            return "semester_courses"
        if re.search(r"\b(credits?|which semester|what semester|l-?p-?t|when is .* taught)\b", q):
            return "course_info"
        return "course_search"

    @staticmethod
    def _semester(q: str) -> Optional[int]:
        m = re.search(r"\bsem(?:ester)?\s*[-:]?\s*(\d{1,2}|[ivx]{1,4})\b|\b(\d{1,2})(?:st|nd|rd|th)\s*sem", q.lower())
        if not m:
            return None
        v = m.group(1) or m.group(2)
        return int(v) if v.isdigit() else {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}.get(v)

    def answer(self, question: str, plan: Optional[Dict[str, Any]] = None, course_context: Optional[str] = None) -> Dict:
        """Chat entry point: intent + entities -> workflow -> a summary built only from the workflow's output."""
        plan = plan or {}
        intent = plan.get("intent") if plan.get("intent") in INTENTS else self.classify(question)
        courses = [n for n in ([self.resolve_name(c) for c in plan.get("courses") or []] + self.detect_courses(question)) if n]
        if course_context:
            resolved = self.resolve_name(course_context)
            courses = ([resolved] if resolved else []) + courses
        courses = list(dict.fromkeys(courses))
        programmes = [p for p in plan.get("programmes") or [] if p in self.eng.view] or self.detect_programmes(question)
        semester = plan.get("semester") or self._semester(question)
        programme = programmes[0] if programmes else None
        return self._dispatch(question, intent, courses, programmes, programme, semester)

    def _dispatch(self, question, intent, courses, programmes, programme, semester) -> Dict:
        if intent == "bloom":
            quoted = re.findall(r"[\"“]([^\"”]{8,})[\"”]", question)
            r = self.bloom(quoted or None, None if quoted else (programme and _department_of(programme)))
            items = r["outcomes"]
            text = "; ".join(f"“{i['outcome'][:80]}” → {i['level']}" for i in items[:6])
            return self._result(intent, f"Bloom levels ({r['source']}): {text}.", r, details=[{"label": k, "value": str(v)} for k, v in r["distribution"].items()],
                                basis="outcome_text", note="Verb-lexicon classifier from the Curriculum Engine (reported accuracy 77.6% against faculty tags).",
                                outcomes=items)

        if intent == "modernize":
            if not courses:
                return self._ask("Name the course or courses to modernise, for example: skill gap for Machine Learning and Deep Learning.", intent)
            r = self.modernize(courses[:3], programme=programme)
            gaps = [t["theme"] for t in r["skill_themes"] if not t["covered"]][:6]
            taught = [t["theme"] for t in r["skill_themes"] if t["covered"]][:6]
            return self._result(intent, f"Industry skill-gap check for {', '.join(courses[:3])} against {r['jd_source']}: "
                                        f"{len(r['skill_themes'])} skill themes found. Not visible in the courses: {', '.join(gaps) or 'none'}. "
                                        f"Already visible: {', '.join(taught) or 'none'}. A draft syllabus with sources is included for faculty review.",
                                r, basis=r["data_basis"], note=r["note"], details=[{"label": "Draft units", "value": str(len(r["new_curriculum"]["units"]))},
                                                                                     {"label": "Status", "value": r["new_curriculum"]["status"]}])

        if intent == "redundancy":
            if not programme:
                return self._ask("Which programme should I check for redundancy? For example: B Tech IT or MBA Tech Data Science.", intent)
            r = self.detect_redundancy(programme)
            text = "; ".join(f"{p['course_a']} / {p['course_b']} ({p['similarity']:.2f})" for p in r["pairs"][:6])
            return self._result(intent, f"Possible redundancy in {programme} across {r['courses_checked']} courses: {text or 'no similar pairs above the threshold'}.",
                                r, basis=r["data_basis"], note=r["note"], overlaps=r["pairs"])

        if intent == "missing_topics":
            if not programme:
                return self._ask("Which programme should I check for missing topics? For example: B Tech Computer Engineering.", intent)
            r = self.missing_topics(programme)
            if not r["topics"]:
                return self._result(intent, f"The engine's reference list has no verified emerging topics for {programme}.", r, basis=r["data_basis"], note=r["name_note"])
            gaps = [t for t in r["topics"] if not t["covered"]]
            cov = [t for t in r["topics"] if t["covered"]]
            return self._result(intent, f"{programme} against {len(r['topics'])} verified emerging topics: {len(cov)} visible in the programme, {len(gaps)} not visible"
                                        f"{': ' + ', '.join(t['topic'] for t in gaps[:8]) if gaps else ''}.", r, basis=r["data_basis"],
                                note=f"{r['name_note']} {r['reference_status']}", details=[{"label": t["topic"], "value": t["closest_units"][0] if t["closest_units"] else "-"} for t in cov[:5]])

        if intent == "compare_programmes" or (intent == "compare_courses" and len(courses) < 2 and len(programmes) >= 2):
            if len(programmes) < 2:
                return self._ask("Name two programmes to compare, for example: compare B Tech IT and B Tech Computer Engineering.", "compare_programmes")
            r = self.compare_programmes(programmes[0], programmes[1])
            return self._result("compare_programmes", f"{programmes[0]} vs {programmes[1]}: {r['retained']} courses shared or closely matched, "
                                f"{len(r['removed'])} only in {programmes[0]} (e.g. {', '.join(x['course_name'] for x in r['removed'][:4]) or 'none'}), "
                                f"{len(r['added'])} only in {programmes[1]} (e.g. {', '.join(x['course_name'] for x in r['added'][:4]) or 'none'}).",
                                r, basis=r["data_basis"], note=r["note"])

        if intent == "compare_courses":
            if len(courses) < 2:
                return self._ask("Name two courses to compare, for example: compare Machine Learning and Deep Learning.", intent)
            r = self.compare_courses(courses[0], courses[1], programme)
            ov = r["overlap"]
            return self._result(intent, f"{courses[0]} vs {courses[1]}: name-level similarity {ov['semantic_similarity']:.2f}, overlap confidence {ov['confidence']:.2f}"
                                f"{'; flagged as possibly redundant' if ov['redundancy_warning'] else ''}. "
                                f"{courses[0]}: semester {r['course_a'].get('semester') or '-'}. {courses[1]}: semester {r['course_b'].get('semester') or '-'}.",
                                r, basis=r["data_basis"], note=r["note"], confidence=ov["confidence"],
                                overlaps=r.get("common_topics"))

        if intent == "overlap":
            if not courses:
                return self._ask("Which course should I check for overlap?", intent)
            r = self.find_overlaps(courses[0], programme=programme)
            items = [{"course": h["course"]["course_name"], "score": h["score"], "confidence": h["confidence"]} for h in r["results"]]
            return self._result(intent, f"Courses closest to {courses[0]}: " + ", ".join(f"{i['course']} (score {i['score']:.2f})" for i in items) + ".",
                                r, basis=r["data_basis"], note=r["note"], overlaps=items, course=r["query"])

        if intent == "prerequisites":
            if not courses:
                return self._ask("Which course's prerequisites should I check?", intent)
            topic = courses[0]
            phrases = courses[1:] or None
            if phrases is None:
                m = re.search(r"(?i)\bis\s+(.+?)\s+(?:taught|covered|offered)\s+before\b", question)
                phrases = [m.group(1)] if m else None
                if m:
                    named = self.detect_courses(m.group(1))
                    if named and named[0] == topic and len(courses) == 1:
                        topic = None
            if topic is None:
                return self._ask("Name the later course as well, for example: is Linear Algebra taught before Machine Learning?", intent)
            if courses[1:] and re.search(r"(?i)\bbefore\b", question):
                topic, phrases = courses[-1], courses[:-1]
            r = self.prerequisites(topic, programme, phrases)
            return self._result(intent, _prereq_summary(topic, r), r, basis=r["data_basis"], note=r["note"], course=r["course"], prerequisites=r)

        if intent == "semester_courses":
            if programme and semester:
                r = self.semester_courses(programme, semester)
                if not r["courses"]:
                    return self._result(intent, f"The SRB lists no courses for {programme} semester {semester}.", r, basis="srb_structure", note="")
                pages = sorted({c["page"] for c in r["courses"] if c.get("page")})
                return self._result(intent, f"{programme} semester {semester} ({r['credits_listed']:g} credits listed): " +
                                    ", ".join(c["course_name"] for c in r["courses"]) + ".", r, basis="srb_structure", note="From the programme structure printed in the SRB.",
                                    citations=[{"section": f"Programme structure: {programme}", "page": p} for p in pages])
            return self._ask("Tell me the programme and semester, for example: courses in semester 4 of B Tech IT.", intent)

        if intent == "course_info" and courses:
            r = self.course_info(courses[0], programme)
            rows = r["offerings"][:6]
            text = "; ".join(f"{'/'.join(o['programmes'])} {o['degree']}: semester {o['semester']}, {o['credits']:g} credits (L-P-T {o['L-P-T']})" for o in rows)
            return self._result(intent, f"{courses[0]}: {text}.", r, basis="srb_structure", note="From the programme structure printed in the SRB.", course=rows[0],
                                citations=[{"section": f"Programme structure: {'/'.join(o['programmes'])}", "page": o["page"]} for o in rows[:3] if o.get("page")])

        r = self.search(courses[0] if courses else _topic_of(question), programme)
        if not r["results"]:
            return self._result("course_search", "I could not find a matching course in the SRB programme structures.", r, basis="srb_structure", note="", found=False)
        return self._result("course_search", "Matching courses: " + "; ".join(f"{h['course_name']} ({'/'.join(h['offered_in'][:4])}, semester {h['semester']})" for h in r["results"][:6]) + ".",
                            r, basis="srb_structure", note="From the programme structures printed in the SRB.", course=r["results"][0])

    def _ask(self, text: str, intent: str) -> Dict:
        return self._result(intent, text, {}, basis="", note="", found=False)

    @staticmethod
    def _result(intent, summary, raw, basis="", note="", confidence=None, details=None, overlaps=None, outcomes=None, course=None,
                prerequisites=None, citations=None, found=True) -> Dict:
        return {"intent": intent, "summary": summary, "raw": raw, "data_basis": basis, "note": note, "confidence": confidence, "details": details or [],
                "overlaps": overlaps, "outcomes": outcomes, "course": course, "prerequisites": prerequisites, "citations": citations or [], "found": found}


def _prereq_summary(name: str, r: Dict) -> str:
    related = ", ".join(f"{x['course_name']} (semester {x['semester']})" for x in r["related_earlier_courses"])
    if r["checked_phrases"]:
        flags = r["check"]["flags"]
        if not r["check"]["checked"]:
            return f"Prerequisite check for {name} could not run: {r['check']['reason']}."
        issue = "; ".join(f"{f['phrase']} ({f['type'].replace('_', ' ')})" for f in flags)
        return (f"Prerequisite check for {name} in {r['programme']}: {'possible gaps: ' + issue if flags else 'each named topic is covered earlier in the programme'}. "
                f"Treat flags as prompts for review; the engine's flag precision is about 50%.")
    if r["prerequisites_stated"]:
        return f"Prerequisites stated for {name}: {r['stated_text']}. Check result: {len(r['check']['flags'])} flag(s)."
    return (f"The SRB does not state prerequisites for {name}. In {r['programme']} it sits in semester {r['course'].get('semester')}"
            f"{', and the most closely related earlier courses are: ' + related if related else ''}. This is inferred from course names, not an official prerequisite list.")


def _department_of(programme: str) -> str:
    return {"IT": "Information Technology", "COMP": "Computer Engineering", "CS": "Computer Engineering", "AI": "Artificial Intelligence", "AI-ML": "Artificial Intelligence",
            "DS": "Data Science", "CSE-DS": "Data Science", "EXTC": "Electronics", "CIVIL": "Civil", "MECHANICAL": "Mechanical", "MECHATRONICS": "Mechatronics"}.get(programme.split("-")[0] if programme not in ("AI-ML", "CSE-DS") else programme, "")


_STOP = re.compile(r"(?i)\b(which|what|where|is|are|there|any|course|courses|subject|subjects|about|on|in|the|a|an|of|offered|taught|do|does|we|i|have|teach|teaches|tell|me|find|search|show|for|at|nmims|mpstme|syllabus|curriculum)\b")


def _topic_of(question: str) -> str:
    return re.sub(r"\s+", " ", _STOP.sub(" ", re.sub(r"[?.!,]", " ", question))).strip() or question


_INSTANCE = None
_LOCK = threading.Lock()


def get_curriculum_service() -> CurriculumService:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = CurriculumService()
    return _INSTANCE
