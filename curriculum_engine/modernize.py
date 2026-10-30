"""Capability 8 (Week 7): modernise overlapping syllabi from job descriptions.

    job descriptions -> requirement statements -> skill themes (clusters of MiniLM embeddings, named by class-based TF-IDF)
    selected syllabi -> overlap between them (fusion score) + groups of units taught in several courses
    themes x units   -> which demanded skills are already taught and which are gaps
    -> a new, structured syllabus: one unit per shared topic (verbatim from the most recent syllabus), the existing units that
       teach demanded skills, one new unit per gap theme (built from the JD requirement sentences). Every unit and outcome
       records its source (course id or JD id); nothing is written by an LLM, so nothing can be invented.
Hand-offs: a request in Group 11's /process-notice schema and passages in Group 4's knowledge-base chunk schema.
"""
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from curriculum_engine import bloom as B
from curriculum_engine import data as D
from curriculum_engine import prereq as PQ
from curriculum_engine.overlap import chunk_vectors

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
HEADER = re.compile(r"^(?:role|responsibilities|requirements|synthetic sample|about|location|salary)\b", re.I)


# ------------------------------------------------------------------ job descriptions
def requirements(text):
    """Requirement statements of one JD: bullet lines (or sentences); headers and short fragments are dropped."""
    out = []
    for line in (text or "").splitlines():
        line = BULLET.sub("", line).strip()
        if not line or HEADER.match(line) or line.endswith(":"):
            continue
        out += [s.strip() for s in re.split(r"(?<=[.;])\s+", line) if len(s.split()) >= 5]
    return out


GENERIC = set("experience experiences skill skills knowledge understanding background foundations foundation familiarity practice "
              "practices use cost quality rates rate release production data text model models system systems method methods technique "
              "techniques tool tools library libraries ecosystem service services team teams work way ways part parts case cases level "
              "levels range variety number lot question sequence training".split())
TOOLS = re.compile(r"(?i)\b(?:python|pytorch|tensorflow|docker|kubernetes|gpus?|cloud|rest api|api development|hugging face|git|linux|sql)\b")
LEAD_VERB = re.compile(r"^(?:build|clean|monitor|apply|design|deploy|index|align|fine-tune|evaluate|set up|handle|compress|adapt|curate|"
                       r"deduplicate|filter|track|maintain|reduce|remove|combine|improve|keep|experience|develop|create|use)\s+(?:and\s+)?", re.I)
DETS = re.compile(r"^(?:the|a|an|this|these|their|our|its|all|any|some|such|other|strong|solid|good|deep|basic|new)\s+", re.I)


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _regex_chunks(sentence):
    """Fallback when spaCy or its model is not installed: stretches between prepositions and punctuation."""
    parts = re.split(r"[,;:()]|\b(?:and|or|with|for|of|in|on|to|using|such as|including|that|which|from|by|at|across)\b", sentence)
    return [p.strip() for p in parts if p and p.strip()]


def skill_phrases(sentence):
    """Noun phrases of a requirement statement that name a skill ('parameter-efficient fine-tuning methods', 'QLoRA')."""
    try:
        chunks_ = [(ch.text, ch.root.pos_) for ch in _nlp()(sentence).noun_chunks]
    except (ImportError, OSError):
        chunks_ = [(c, "NOUN") for c in _regex_chunks(sentence)]
    out = []
    for text, pos in chunks_:
        p = re.sub(r"^[^A-Za-z]+", "", text.strip())
        while DETS.match(p) or LEAD_VERB.match(p):
            p = LEAD_VERB.sub("", DETS.sub("", p))
        words = re.findall(r"[A-Za-z][A-Za-z+\-]*", p)
        if len(p) > 2 and words and pos != "PRON" and any(w.lower() not in GENERIC for w in words):
            out.append(p)
    return out


def skill_themes(jds, emb, tau):
    """JD skill phrases grouped by meaning (cosine, average linkage, merged while similarity >= tau). A theme is named by its
    most central phrase and keeps the JDs that ask for it and the requirement sentences it came from."""
    rows = [(jid, s, p) for jid, t in sorted(jds.items()) for s in requirements(t) for p in skill_phrases(s)]
    if not rows:
        return []
    E = emb([p for *_, p in rows])
    lab = (np.zeros(1, int) if len(rows) == 1 else
           AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - tau).fit_predict(E))
    themes = []
    for k in range(lab.max() + 1):
        idx = np.flatnonzero(lab == k)
        medoid = idx[int(np.argmax(E[idx] @ E[idx].mean(0)))]
        themes.append({"theme": rows[medoid][2], "phrases": sorted({rows[i][2] for i in idx}, key=str.lower),
                       "jds": sorted({rows[i][0] for i in idx}),
                       "requirements": [{"jd": j, "text": s} for j, s in dict.fromkeys((rows[i][0], rows[i][1]) for i in idx)]})
    return sorted(themes, key=lambda t: (-len(t["jds"]), -len(t["phrases"]), t["theme"].lower()))


# ------------------------------------------------------------------ syllabi
def course_units(courses):
    return [(c, title, text) for c in courses for title, text in D.chunks(c)]


def unit_topics(courses):
    """Topic phrases of every unit: its comma / semicolon / colon separated items of 1-12 words."""
    out = []
    for c, title, text in course_units(courses):
        for t in re.split(r"[,;:.()]|\bv/s\b|\bvs\.?\b", text):
            t = t.strip(" -–")
            if len(t) > 3 and 1 <= len(t.split()) <= 12:
                out.append((c, title, t))
    return out


def coverage(themes, courses, emb, tau, k=3):
    """A theme is already taught when one of its skill phrases matches a topic phrase of some unit (cosine >= tau)."""
    tops = unit_topics(courses)
    T = emb([t for *_, t in tops]) if tops else np.zeros((0, 384))
    out = []
    for th in themes:
        best = (emb(th["phrases"]) @ T.T).max(0) if len(T) else np.zeros(0)
        seen, ev = set(), []
        for i in np.argsort(-best):
            key = (tops[i][0]["id"], tops[i][1])
            if key not in seen:
                seen.add(key)
                ev.append({"course": tops[i][0]["course_name"], "course_id": tops[i][0]["id"], "unit": tops[i][1],
                           "topic": tops[i][2], "similarity": round(float(best[i]), 3)})
            if len(ev) == k:
                break
        out.append({"theme": th["theme"], "covered": bool(ev and ev[0]["similarity"] >= tau), "closest_units": ev})
    return out


def shared_units(courses, emb, tau):
    """Groups of units taught in two or more of the selected courses (connected components over unit pairs with cosine >= tau)."""
    units = course_units(courses)
    if len(units) < 2:
        return []
    U = emb([t for *_, t in units])
    S = U @ U.T
    owner = [c["id"] for c, _, _ in units]
    parent = list(range(len(units)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if owner[i] != owner[j] and S[i, j] >= tau:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(units)):
        groups[find(i)].append(i)
    out = []
    for idx in groups.values():
        if len({owner[i] for i in idx}) < 2:
            continue
        rep = max(idx, key=lambda i: (units[i][0]["academic_year"], len(units[i][2]), units[i][0]["id"]))   # most recent syllabus
        out.append({"title": units[rep][1], "text": units[rep][2], "source": f"{units[rep][0]['course_name']} ({units[rep][0]['id']})",
                    "taught_in": sorted({f"{units[i][0]['course_name']} ({units[i][0]['id']})" for i in idx}),
                    "hours": next((u["hours"] for u in units[rep][0]["units"] if u["text"] == units[rep][2]), None)})
    return sorted(out, key=lambda g: (-len(g["taught_in"]), g["title"]))


# ------------------------------------------------------------------ the new syllabus
def _lpt(courses):
    k = Counter((c.get("lecture_hrs"), c.get("practical_hrs"), c.get("tutorial_hrs"), c.get("credits")) for c in courses
                if c.get("credits") is not None).most_common(1)
    return dict(zip(("lecture_hrs", "practical_hrs", "tutorial_hrs", "credits"), k[0][0])) if k else {}


PAGE_TAIL = re.compile(r"\s+Page(?:\s+\d+)?\s*$")              # page-footer residue at the end of a unit


def build_curriculum(title, courses, overlaps, themes, cov, shared, academic_year=None, min_jds=2, max_new=4):
    """Assemble the draft syllabus. Units: shared topics (verbatim) -> existing units that teach demanded skills -> new units
    for gap themes demanded by >= min_jds JDs (at most max_new). Outcomes reuse source outcomes or are templated from JD text."""
    demanded = [t for t in themes if len(t["jds"]) >= min_jds]
    cov_by = {c["theme"]: c for c in cov}
    units, used = [], set()
    for g in shared:
        units.append({"title": g["title"], "content": PAGE_TAIL.sub("", g["text"]), "hours": g["hours"], "kind": "shared topic (deduplicated)",
                      "sources": g["taught_in"]})
        used.add(g["text"])
    for t in demanded:
        c = cov_by[t["theme"]]
        if c["covered"]:
            u = c["closest_units"][0]
            course = next(x for x in courses if x["id"] == u["course_id"])
            text = next((tx for ti, tx in D.chunks(course) if ti == u["unit"]), "")
            if text and text not in used:
                units.append({"title": u["unit"], "content": PAGE_TAIL.sub("", text), "hours": None, "kind": "existing unit, demanded by industry",
                              "sources": [f"{u['course']} ({u['course_id']})", *t["jds"]]})
                used.add(text)
    tools = [t for t in demanded if TOOLS.search(t["theme"])]            # tools and infrastructure go to the lab, not into units
    gaps = [t for t in demanded if not cov_by[t["theme"]]["covered"] and t not in tools][:max_new]
    for t in gaps:
        units.append({"title": unit_title(t), "hours": 6, "kind": "new unit from job descriptions",
                      "content": f"Topics: {'; '.join(t['phrases'])}. Industry requirements: " + " ".join(r["text"] for r in t["requirements"]),
                      "sources": t["jds"]})
    outcomes = []
    for c in courses:
        for o in c.get("outcomes", []):
            if o["text"] not in {x["text"] for x in outcomes}:
                outcomes.append({"text": o["text"], "bloom": B.BLOOM.get(B.lexicon_level(o["text"]) or 3), "source": c["id"]})
    outcomes = outcomes[:4] + [{"text": f"Explain and apply techniques for {unit_title(t).lower()} in a lab exercise", "bloom": "Understand",
                                "source": ", ".join(t["jds"])} for t in gaps]
    pre = list(dict.fromkeys(p for c in courses for p in PQ.split_prereq_v2(c.get("prerequisites", ""))))
    for u in units:
        u["hours"] = u["hours"] or 6
    return {"title": title, "status": "DRAFT for the Board of Studies: requires faculty review before adoption",
            "academic_year": academic_year or max(c["academic_year"] for c in courses),
            "replaces": [{"id": c["id"], "course_name": c["course_name"], "programmes": c.get("programmes", []),
                          "academic_year": c["academic_year"], "semester": c.get("semester", "")} for c in courses],
            "programmes": sorted({p for c in courses for p in c.get("programmes", [])}),
            "teaching_scheme": _lpt(courses), "prerequisites": pre,
            "objectives": (f"To give students one coherent course on {title.lower()} that removes the overlap between "
                           f"{len(courses)} existing syllabi and adds the skills that the job descriptions ask for."),
            "units": units, "total_hours": sum(u["hours"] for u in units), "outcomes": outcomes,
            "lab_work": [f"Hands-on exercise: {u['title']}" for u in units if u["kind"] != "shared topic (deduplicated)"],
            "lab_environment": sorted({t["theme"] for t in tools}, key=str.lower),
            "hours_check": (f"The units add up to {sum(u['hours'] for u in units)} hours, while the current teaching scheme gives "
                            f"{(_lpt(courses).get('lecture_hrs') or 0) * 15} lecture hours per semester: the committee should trim units "
                            f"or split the course in two."),
            "assessment": "As per the programme's internal continuous assessment and term-end examination norms (not part of the parsed syllabi).",
            "overlaps_removed": overlaps, "jd_themes": [{"theme": t["theme"], "jds": t["jds"], "covered": cov_by[t["theme"]]["covered"]} for t in themes]}


def unit_title(theme):
    """The most informative phrase of a theme (most characters, at most five words), capitalised."""
    best = max((p for p in theme["phrases"] if len(p.split()) <= 5), key=lambda p: (len(p), p), default=theme["theme"])
    return best[:1].upper() + best[1:]


def traceable(doc):
    """Every unit and outcome must name a source (a course id or a JD id)."""
    return all(u["sources"] for u in doc["units"]) and all(o["source"] for o in doc["outcomes"])


def to_markdown(doc):
    ts = doc["teaching_scheme"]
    lines = [f"# {doc['title']}", f"*{doc['status']}*", "",
             f"Programmes: {', '.join(doc['programmes'])} | Academic year: {doc['academic_year']} | "
             f"L-P-T: {ts.get('lecture_hrs')}-{ts.get('practical_hrs')}-{ts.get('tutorial_hrs')} | Credits: {ts.get('credits')}",
             "", f"**Prerequisites:** {', '.join(doc['prerequisites']) or 'none listed'}", "", f"**Objectives:** {doc['objectives']}", "",
             f"**Hours:** {doc['hours_check']}", "",
             "## Units"]
    for i, u in enumerate(doc["units"], 1):
        lines += [f"{i}. **{u['title']}** ({u['hours']} h; {u['kind']}) — {u['content']}", f"   *Source: {'; '.join(u['sources'])}*"]
    lines += ["", "## Learning outcomes"] + [f"- {o['text']} ({o['bloom']}; source {o['source']})" for o in doc["outcomes"]]
    lines += ["", "## Lab work"] + [f"- {x}" for x in doc["lab_work"]] + [f"- Lab environment: {', '.join(doc['lab_environment']) or 'as per the programme'}",
              "", f"**Assessment:** {doc['assessment']}", "",
              "## Courses this syllabus replaces"] + [f"- {c['course_name']} ({', '.join(c['programmes'])}, {c['academic_year']}, {c['id']})"
                                                      for c in doc["replaces"]]
    return "\n".join(lines)


def to_docx(doc, path):
    from docx import Document
    from docx.shared import Pt
    d = Document()
    d.styles["Normal"].font.size = Pt(10)
    for line in to_markdown(doc).splitlines():
        if line.startswith("# "):
            d.add_heading(line[2:], 0)
        elif line.startswith("## "):
            d.add_heading(line[3:], 1)
        elif line.startswith("- "):
            d.add_paragraph(line[2:].replace("**", ""), style="List Bullet")
        elif line.strip():
            d.add_paragraph(line.replace("**", "").replace("*", "").strip())
    d.save(path)
    return path


# ------------------------------------------------------------------ hand-offs to other groups
def group11_request(doc):
    """Group 11 (University Communications Intelligence), POST /process-notice: {doc_id, text} (their NoticeRequest schema)."""
    names = ", ".join(sorted({c["course_name"] for c in doc["replaces"]}))
    new = [u["title"] for u in doc["units"] if u["kind"] == "new unit from job descriptions"]
    text = (f"Dear Students and Faculty Members, The Board of Studies has prepared a draft unified syllabus for the course "
            f"'{doc['title']}' for the academic year {doc['academic_year']}. It brings together the overlapping content of {names} "
            f"taught in the programmes {', '.join(doc['programmes'])}, and adds new units on {', '.join(new) or 'current industry practice'} "
            f"based on industry job descriptions. Prerequisites: {', '.join(doc['prerequisites']) or 'none listed'}. "
            f"Faculty members teaching these courses are requested to review the draft syllabus and send their comments to the "
            f"Programme Chairperson. Students will be informed about the changes to their electives before registration.")
    slug = re.sub(r"[^a-z0-9]+", "-", doc["title"].lower()).strip("-")
    return {"doc_id": f"curriculum-{slug}-{doc['academic_year']}", "text": text}


def group11_mock(request, doc):
    """A stand-in that returns Group 11's NoticeResponse schema, used only when their service is not reachable.
    Their real pipeline (classifier, NER, fine-tuned BART persona summaries) replaces this on demo day."""
    return {"doc_id": request["doc_id"], "title": f"Draft unified syllabus: {doc['title']}", "category": "academic",
            "audience": ["faculty", "student"],
            "entities": {"dates_and_deadlines": [doc["academic_year"]], "organizations": ["NMIMS", "Board of Studies"],
                         "contact_emails": [], "urls": [], "action_roles": ["Programme Chairperson"]},
            "summaries": {"student": f"A unified course '{doc['title']}' is planned for {doc['academic_year']}; you will be told about elective changes before registration.",
                          "faculty": f"Review the draft syllabus of '{doc['title']}', which merges {len(doc['replaces'])} courses, and send comments to the Programme Chairperson.",
                          "admin": "Update the course catalogue once the Board of Studies approves the syllabus."},
            "_mock": "generated by Group 9's stand-in for Group 11's /process-notice contract (not Group 11's model)"}


def group4_chunks(doc, size=100, overlap=25):
    """Group 4 (University Knowledge Assistant) knowledge-base rows: chunk_id, page, section, part, text, as 100-word windows with
    25 words of overlap (their make_chunks settings). 'page' numbers the sections of our syllabus document."""
    ts = doc["teaching_scheme"]
    sections = [("Course overview", f"The new course {doc['title']} is offered to the programmes {', '.join(doc['programmes'])} from the "
                                    f"academic year {doc['academic_year']}. {doc['objectives']}"),
                ("Credits and teaching scheme", f"The course carries {ts.get('credits')} credits, with {ts.get('lecture_hrs')} lecture hours, "
                                                f"{ts.get('practical_hrs')} practical hours and {ts.get('tutorial_hrs')} tutorial hours per week."),
                ("Prerequisites", f"The prerequisites for this course are: {', '.join(doc['prerequisites']) or 'none listed'}."),
                *[(f"Unit {i}: {u['title']}", f"{u['title']} ({u['hours']} hours). {u['content']}") for i, u in enumerate(doc["units"], 1)],
                ("Lab work", f"The course has lab work: {'; '.join(doc['lab_work']) or 'as per the programme'}. "
                             f"Lab environment: {', '.join(doc['lab_environment']) or 'as per the programme'}."),
                ("Assessment and examination", f"How the course is assessed and examined: {doc['assessment']}"),
                ("Learning outcomes", " ".join(o["text"] + "." for o in doc["outcomes"]))]
    rows = []
    for page, (section, text) in enumerate(sections, 1):
        words, start = text.split(), 0
        while True:
            rows.append({"chunk_id": f"G9C{len(rows):04d}", "page": page, "section": section,
                         "part": f"Group 9 draft syllabus: {doc['title']}", "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += size - overlap
    return rows


def group4_standin(rows, question, emb, beta=0.6):
    """Answers a question from the exported passages in the spirit of Group 4's offline mode: every sentence is scored with
    hybrid BM25 + MiniLM retrieval (beta = their keyword weight) and the best one is quoted verbatim with its source.
    A stand-in for the demo, not their code."""
    from rank_bm25 import BM25Okapi
    stem = lambda w: re.sub(r"(?:ments?|ed|ing|es|s)$", "", w) if len(w) > 4 else w      # assessed / assessment -> assess
    tok = lambda s: [stem(w) for w in re.findall(r"[a-z0-9]+", s.lower())]
    sents = [(r, s) for r in rows for s in re.split(r"(?<=[.;])\s+", r["text"]) if len(s.split()) >= 3]
    bm = BM25Okapi([tok(r["section"] + " " + s) for r, s in sents])
    kw = np.array(bm.get_scores(tok(question)))
    kw = kw / kw.max() if kw.max() > 0 else kw
    score = beta * kw + (1 - beta) * (emb([r["section"] + ". " + s for r, s in sents]) @ emb([question])[0])
    best, sent = sents[int(score.argmax())]
    return {"question": question, "answer": sent, "section": best["section"], "source": f"{best['part']} | {best['section']} | page {best['page']}",
            "confidence": round(float(score.max()), 3), "_standin": "Group 9 stand-in for Group 4's offline (extractive) mode"}


def course_overlaps(eng, courses):
    """Pairwise overlap of the selected syllabi: the Week 4 fusion score, its confidence, and the matched units."""
    from curriculum_engine.overlap import maxsim, topics
    out = []
    for i, a in enumerate(courses):
        for b in courses[i + 1:]:
            f = eng.features(a, b)
            m = maxsim(chunk_vectors(eng.emb, a), chunk_vectors(eng.emb, b), eng.th["unit_match"])
            t = topics(a, b, m, eng.th["unit_match"], k=6)
            out.append({"course_a": f"{a['course_name']} ({a['id']})", "course_b": f"{b['course_name']} ({b['id']})",
                        "overlap_percent": f"{m['overlap']:.0%}", "score": round(eng.fusion(f), 3), "confidence": round(eng.confidence(f), 3),
                        "common_topics": [f"{x['topic_a']} = {x['topic_b']}" for x in t["common_topics"]]})
    return sorted(out, key=lambda x: -x["score"])


def run(eng, jds, courses, title, academic_year=None):
    """The whole demo step: overlaps, JD themes, coverage, draft syllabus and the two hand-offs."""
    tau = eng.th["unit_match"]
    themes = skill_themes(jds, eng.emb, tau)
    cov = coverage(themes, courses, eng.emb, tau)
    shared = shared_units(courses, eng.emb, tau)
    overlaps = course_overlaps(eng, courses)
    doc = build_curriculum(title, courses, overlaps, themes, cov, shared, academic_year)
    return {"courses": [eng.meta(c) for c in courses], "overlaps": overlaps, "shared_topics": shared,
            "skill_themes": [{**t, "covered": c["covered"], "closest_units": c["closest_units"]} for t, c in zip(themes, cov)],
            "new_curriculum": doc, "traceable": traceable(doc), "markdown": to_markdown(doc),
            "handoffs": {"group11_request": group11_request(doc), "group4_chunks": group4_chunks(doc)}}
