"""Parse the real programme structures in the SRB (PDF pages 83-168) into Curriculum Engine course records.

The SRB gives programme -> semester -> course with Lecture/Practical/Tutorial/Credits and a category, plus the elective
pools and each department's outcomes. It does NOT give unit-level syllabi or prerequisites, so records built here carry
`content_level = "name_only"`: their single "unit" is the course name. Nothing is invented to fill the gap.
"""
import hashlib
import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import pymupdf

STRUCTURE_RX = re.compile(r"(?:Proposed|Program)\s+[Ss]tructure\s+of\s+(.+)", re.I)
TERM_RX = re.compile(r"Academic Year\s*-\s*(\d{4}-\d{2})(?:\s*,\s*(?:Semester|Trimester)\s*-?\s*([IVX]+|\d+))?", re.I)
ELECTIVE_RX = re.compile(r"^((?:Program|Open|Department|Professional)?\s*Electives?\s*-?\s*(?:[IVX]+|\d+)?)\s*(?:\(.*\))?$", re.I)
CATEGORY = {"COMP", "ELEC", "INTD", "COUC", "CORE", "AUDIT", "HONS", "MNR", "PROJ"}
HEADER_WORDS = {"s.", "no.", "s. no.", "s.no.", "sr.", "course", "course name", "lecture", "practical", "tutorial", "credits",
                "category", "remark", "name of the course", "course category", "sr. no.", "s.no", "total", "credit"}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

PROGRAMME_CODES = [           # first match wins; codes follow the Curriculum Engine (curriculum_engine/data.py)
    (r"artificial intelligence.*machine|aiml", "AI-ML"), (r"artificial intelligence.*data|aids", "AI-DS"),
    (r"computer science and engineering.*data|cse.*data science", "CSE-DS"),
    (r"computer science and engineering.*cyber|cybersecurity", "CSE-CYBER"),
    (r"master of computer applications", "MCA"), (r"business systems", "CSBS"), (r"artificial intelligence", "AI"), (r"data science", "DS"),
    (r"computer science", "CS"), (r"computer engineering", "COMP"), (r"information technology", "IT"),
    (r"electronics\s*&?\s*(and)?\s*telecommunication", "EXTC"), (r"civil", "CIVIL"),
    (r"mechatronics", "MECHATRONICS"), (r"mechanical", "MECHANICAL"),
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def programme_code(title: str) -> Tuple[str, str]:
    """'B Tech Information Technology' -> ('IT', 'B Tech'). The code names the discipline, the degree is kept separately."""
    degree = next((d for d in ("MBA Tech", "B Tech", "M Tech", "Diploma", "Ph.D") if d.lower() in title.lower()), "")
    low = title.lower()
    for rx, code in PROGRAMME_CODES:
        if re.search(rx, low):
            return code, degree
    return re.sub(r"[^A-Z]", "", title.title())[:8] or "OTHER", degree


def programme_key(p: Dict) -> str:
    """B Tech keeps the plain code ('IT'); the other degrees of a discipline are separate programmes ('IT-MBA-TECH')."""
    if p["degree"] in ("B Tech", ""):
        return p["code"]
    return f"{p['code']}-{re.sub(r'[^A-Z]+', '-', p['degree'].upper()).strip('-')}"


def _sem_number(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    return int(value) if value.isdigit() else ROMAN.get(value.upper())


def _rows(lines: List[str]) -> List[Dict]:
    """Rows of one table body: `<s.no> <name lines> <L> <P> <T> <credits> [category] [remark]`."""
    rows, i, n = [], 0, len(lines)
    while i < n:
        if not re.fullmatch(r"\d{1,2}", lines[i]):
            i += 1
            continue
        j, name = i + 1, []
        while j < n and not re.fullmatch(r"\d{1,2}(\.\d)?", lines[j]) and len(name) < 6:
            if lines[j].lower() in HEADER_WORDS or TERM_RX.search(lines[j]):
                break
            name.append(lines[j])
            j += 1
        nums = lines[j:j + 4]
        if name and len(nums) == 4 and all(re.fullmatch(r"\d{1,2}(\.\d)?", x) for x in nums):
            k, category, remark = j + 4, "", []
            if k < n and lines[k].upper() in CATEGORY:
                category = lines[k].upper()
                k += 1
            while k < n and re.fullmatch(r"(?i)only|ica|only ica", lines[k]):
                remark.append(lines[k])
                k += 1
            rows.append({"course_name": re.sub(r"^(?:ELEC|COMP|INTD)\s+", "", _norm(" ".join(name))), "lecture_hrs": float(nums[0]), "practical_hrs": float(nums[1]),
                         "tutorial_hrs": float(nums[2]), "credits": float(nums[3]), "category": category,
                         "only_ica": bool(remark)})
            i = k
        else:
            i += 1
    return rows


def parse_structures(pdf_path: str, first_page: int = 83, last_page: int = 163) -> List[Dict]:
    """[{programme_title, code, degree, batch, page, terms: [{academic_year, semester, page, courses, electives}]}]"""
    doc = pymupdf.open(pdf_path)
    programmes: List[Dict] = []
    state: Dict = {"current": None, "term": None, "pool": None}

    def flush(buf: List[str], pno: int) -> None:
        if buf and state["current"] is not None and state["term"] is not None:
            for r in _rows(buf):
                r["page"] = pno
                term = state["term"]
                (term["electives"].setdefault(state["pool"], []) if state["pool"] else term["courses"]).append(r)
        buf.clear()

    for pno in range(first_page, last_page + 1):
        lines = [l for l in (_norm(x) for x in doc[pno - 1].get_text().split("\n")) if l]
        if lines and lines[0].isdigit():
            lines = lines[1:]
        buf: List[str] = []
        for idx, line in enumerate(lines):
            structure, term_match, elective = STRUCTURE_RX.search(line), TERM_RX.search(line), ELECTIVE_RX.match(line)
            if structure:
                flush(buf, pno)
                title = structure.group(1)
                nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
                if not re.search(r"(?i)batch|academic year", nxt):      # a title wrapped onto a second line
                    title += " " + nxt
                code, degree = programme_code(title)
                state["current"] = {"programme_title": _norm(title), "code": code, "degree": degree, "batch": "", "page": pno, "terms": []}
                programmes.append(state["current"])
                state["term"], state["pool"] = None, None
            elif re.match(r"(?i)^batch\s*-?\s*\d{4}", line) and state["current"] is not None:
                state["current"]["batch"] = re.sub(r"(?i)^batch\s*-?\s*", "", line)
            elif term_match and state["current"] is not None:
                flush(buf, pno)
                sem = _sem_number(term_match.group(2))
                terms = state["current"]["terms"]
                if sem is None:                                          # a page whose heading omits the semester
                    sem = terms[-1]["semester"] + 1 if terms else 1
                state["term"] = {"academic_year": term_match.group(1), "semester": sem, "page": pno, "courses": [], "electives": {}}
                terms.append(state["term"])
                state["pool"] = None
            elif elective and len(line) < 60 and state["current"] is not None and "(" in line or (elective and idx + 1 < len(lines) and lines[idx + 1].lower() in HEADER_WORDS):
                flush(buf, pno)
                state["pool"] = _norm(elective.group(1))
            else:
                buf.append(line)
        flush(buf, pno)
    return programmes


def _record(p: Dict, term: Dict, row: Dict, pool: Optional[str]) -> Dict:
    key = re.sub(r"[^a-z0-9]+", "", row["course_name"].lower().replace("&", "and"))
    cid = "srb" + hashlib.sha1(f"{p['programme_title']}|{term['semester']}|{key}".encode()).hexdigest()[:8]
    name = row["course_name"]
    return {
        "id": cid, "course_name": name, "code": "", "course_key": key, "program": p["programme_title"], "semester": str(term["semester"]),
        "academic_year": term["academic_year"], "prerequisites": "", "objectives": "", "outcomes": [],
        "units": [{"title": name, "text": name, "hours": None}], "lecture_hrs": row["lecture_hrs"], "practical_hrs": row["practical_hrs"],
        "tutorial_hrs": row["tutorial_hrs"], "credits": row["credits"], "category": row["category"] or ("ELEC" if pool else ""),
        "only_ica": row["only_ica"], "elective_pool": pool or "", "programmes": [programme_key(p)], "degree": p["degree"], "batch": p["batch"],
        "sem": term["semester"], "common": False, "open_elective": bool(pool and "open" in pool.lower()), "n_source_files": 1,
        "source_files": [f"SRB 2026 p.{row['page']}"], "source_page": row["page"], "content_level": "name_only",
    }


PLACEHOLDER = re.compile(r"(?i)^(?:(?:program|open|department|professional|management)?\s*electives?\s*-?\s*(?:[ivx]+|\d+)?|coursera|total.*)$")


def records(programmes: List[Dict]) -> List[Dict]:
    """One record per (programme, degree, semester, course). Placeholder rows such as 'Open Elective - I' are dropped: the
    real elective choices are kept from their pools."""
    out, seen = [], set()
    for p in programmes:
        for term in p["terms"]:
            items = [(r, None) for r in term["courses"]] + [(r, k) for k, v in term["electives"].items() for r in v]
            for row, pool in items:
                if PLACEHOLDER.match(row["course_name"]):
                    continue
                rec = _record(p, term, row, pool)
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    out.append(rec)
    return out


OUTCOME_RX = re.compile(r"^(?:\d{1,2}[.)]|[a-l][.)]|PO\s*\d+\s*[:.-]?|PSO\s*\d+\s*[:.-]?)\s*(.+)")
OUTCOME_VERB = re.compile(r"(?i)^(an ability to|ability to|identify|apply|design|communicate|recognize|function|develop|acquire|analy[sz]e|"
                          r"use|create|understand|demonstrate|engage|graduates?|students?)\b")


def department_outcomes(pdf_path: str) -> Dict[str, Dict]:
    """Department name -> {page, outcomes}: the Programme Outcomes printed on each department's overview pages."""
    doc = pymupdf.open(pdf_path)
    out: Dict[str, Dict] = {}
    for pno in range(83, 163):
        lines = [l for l in (_norm(x) for x in doc[pno - 1].get_text().split("\n")) if l]
        heads = [l for l in lines[:8] if l.endswith("Department")]
        if not heads:
            continue
        items: List[str] = []
        for line in lines:
            m = OUTCOME_RX.match(line)
            if m:
                items.append(m.group(1))
            elif items and line[0].islower():
                items[-1] += " " + line
        items = [i for i in items if OUTCOME_VERB.match(i) and len(i.split()) >= 5]
        if items:
            out[heads[0]] = {"page": pno, "outcomes": items}
    return out


AI_LISTS = [("AI for Cybersecurity", None), ("Computing with AI", None), ("AI-Driven Software Development", None),
            ("Intelligent Data Analytics", None), ("Cloud Infrastructure", None), ("Emerging Technologies for Civil", ["CIVIL"]),
            ("Emerging Technologies for Electronics", ["EXTC"]), ("Emerging Technologies for Mechanical", ["MECHANICAL"]),
            ("Emerging Technologies for Mechatronics", ["MECHATRONICS"])]


def ai_credit_courses(pdf_path: str, codes: List[str]) -> List[Dict]:
    """SRB section 15: the Coursera 'AI and Emerging Technologies' credit courses (pages 164-168). They count as taught content
    for gap analysis, so a topic covered only here is not reported as missing."""
    doc = pymupdf.open(pdf_path)
    lines = [(t, p) for p in range(164, 169) for t in (_norm(x) for x in doc[p - 1].get_text().split("\n")) if t]
    out, spec, seen, i = [], None, set(), 0
    while i < len(lines):
        text, page = lines[i]
        hit = next((s for s in AI_LISTS if s[0].lower() in text.lower() and "applicable" in text.lower()), None)
        if hit:
            spec = hit
        m = re.fullmatch(r"(\d{4}-\d{2})", text)
        if m and spec and i + 2 < len(lines) and re.fullmatch(r"[IVX]+", lines[i + 1][0]):
            sem, j, name = ROMAN[lines[i + 1][0]], i + 2, []
            while j < len(lines) and not re.fullmatch(r"\d", lines[j][0]) and len(name) < 4:
                name.append(lines[j][0])
                j += 1
            name = _norm(" ".join(name))
            credit = float(lines[j][0]) if j < len(lines) and lines[j][0].isdigit() else 0.0
            key = re.sub(r"[^a-z0-9]+", "", name.lower())
            for code in (spec[1] or codes):
                cid = "srb" + hashlib.sha1(f"ai|{spec[0]}|{code}|{key}".encode()).hexdigest()[:8]
                if cid in seen:
                    continue
                seen.add(cid)
                row = {"course_name": name, "lecture_hrs": 0.0, "practical_hrs": 0.0, "tutorial_hrs": 0.0, "credits": credit,
                       "category": "COUC", "only_ica": True, "page": page}
                prog = {"programme_title": f"Coursera AI credits: {spec[0]}", "code": code, "degree": "B Tech", "batch": "2026"}
                rec = _record(prog, {"semester": sem, "academic_year": m.group(1)}, row, None)
                rec["id"] = cid
                out.append(rec)
            i = j
        i += 1
    return out


def save_catalog(pdf_path: str, path: str) -> Dict:
    programmes = parse_structures(pdf_path)
    recs = records(programmes)
    recs += ai_credit_courses(pdf_path, sorted({p["code"] for p in programmes if p["degree"] == "B Tech" and p["code"] not in
                                                ("CIVIL", "EXTC", "MECHANICAL", "MECHATRONICS")}))
    payload = {"source": "Student Resource Book (SRB) 2026, PDF pages 83-168",
               "programmes": [{**{k: p[k] for k in ("programme_title", "code", "degree", "batch", "page")},
                               "semesters": sorted({t["semester"] for t in p["terms"]})} for p in programmes],
               "departments": department_outcomes(pdf_path), "courses": recs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=0)
    return payload
