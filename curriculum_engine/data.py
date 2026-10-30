"""Course records -> families (all versions of one course), canonical programmes, programme structure and the split."""
import hashlib, json, re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from curriculum_engine import DATA, SEED

COURSES = DATA / "processed/courses.jsonl"
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

# zip folder name -> canonical programme. COMMON / OPEN are pools that every programme draws from.
PROGRAMMES = [
    (r"intelligence\s*&\s*data", "AI-DS"), (r"intelligence\s*&\s*machine", "AI-ML"), (r"artifi?ci?al intelligence", "AI"),
    (r"cs\s*-?\s*eds|csds|cse[\s-]*data science", "CSE-DS"), (r"cyber", "CSE-CYBER"),
    (r"csbs|business systems", "CSBS"), (r"data science", "DS"), (r"^computer$", "COMP"),
    (r"^i\s?t$", "IT"), (r"extc", "EXTC"), (r"electronic", "ELECTRONICS"), (r"ele?c?trical", "ELECTRICAL"),
    (r"mechatr|mechator", "MECHATRONICS"), (r"mechanical", "MECHANICAL"), (r"civil", "CIVIL"), (r"^ibm$", "IBM"),
    (r"open elective", "OPEN"), (r"common|all stream|semester|trim|management", "COMMON"),
]
ALL_PROGRAMMES = re.compile(r"all\s+program|all\s+branch|all\s+stream", re.I)


def programme_of(folder):
    f = folder.strip().lower()
    for rx, name in PROGRAMMES:
        if re.search(rx, f):
            return name
    return "OTHER"


def sem_min(s):
    v = [ROMAN[x] for x in re.findall(r"\b(?:VIII|VII|VI|IV|IX|V|III|II|I|X)\b", (s or "").upper())]
    v += [int(x) for x in re.findall(r"\b([1-9]|10)\b", s or "")]
    return min(v) if v else None


def norm_name(n):
    return re.sub(r"[^a-z0-9]+", "", (n or "").lower().replace("&", "and"))


def _families(recs):
    """Union-find over 'same normalised name' OR 'same course code' -> family id (smallest member id, stable)."""
    parent = {r["id"]: r["id"] for r in recs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by = defaultdict(list)
    for r in recs:
        by["n:" + r["course_key"]].append(r["id"])
        if r["code"]:
            by["c:" + r["code"]].append(r["id"])
    for ids in by.values():
        for i in ids[1:]:
            a, b = find(ids[0]), find(i)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for r in recs:
        groups[find(r["id"])].append(r["id"])
    return {i: "f" + min(ids)[2:] for ids in groups.values() for i in ids}


@lru_cache(maxsize=1)
def load():
    """All course versions, enriched with family, canonical programmes, study year and semester number."""
    return enrich([json.loads(l) for l in open(COURSES, encoding="utf-8")])


def enrich(recs):
    fam = _families(recs)
    for r in recs:
        progs, years = set(), set()
        for s in r["source_files"]:
            parts = s.split("/")
            years.add(int(parts[0][0]) if parts[0][:1].isdigit() else None)
            progs.add(programme_of(parts[1]) if len(parts) > 2 else "OTHER")
        r["family"] = fam[r["id"]]
        r["common"] = "COMMON" in progs or bool(ALL_PROGRAMMES.search(r.get("program", "")))
        r["open_elective"] = "OPEN" in progs
        r["programmes"] = sorted(progs - {"COMMON", "OPEN", "OTHER"})
        r["study_year"] = min((y for y in years if y), default=None)
        r["sem"] = sem_min(r.get("semester")) or (2 * r["study_year"] - 1 if r["study_year"] else None)
    return recs


def representatives(recs):
    """One version per family: the latest academic year, then the one found in most booklets, then the smallest id."""
    best = {}
    for r in recs:
        k = (r["academic_year"], r["n_source_files"], r["id"])
        if r["family"] not in best or k > best[r["family"]][0]:
            best[r["family"]] = (k, r)
    return {f: v[1] for f, v in sorted(best.items())}


FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def split(recs):
    """Family -> 'train' / 'val' / 'test'. Whole families (all versions of a course) stay on one side."""
    fams = sorted({r["family"] for r in recs})
    rng = np.random.default_rng(SEED)
    order = [fams[i] for i in rng.permutation(len(fams))]
    a, b = int(FRACTIONS["train"] * len(order)), int((FRACTIONS["train"] + FRACTIONS["val"]) * len(order))
    return {f: ("train" if i < a else "val" if i < b else "test") for i, f in enumerate(order)}


def split_hash(sp):
    return hashlib.sha1(json.dumps(sorted(sp.items())).encode()).hexdigest()[:10]


def structure(recs):
    """programme -> semester -> set of families taught there (any academic year). COMMON courses count for every programme."""
    progs = sorted({p for r in recs for p in r["programmes"]})
    st = {p: defaultdict(set) for p in progs}
    for r in recs:
        if not r["sem"]:
            continue
        for p in (progs if r["common"] else r["programmes"]):
            st[p][r["sem"]].add(r["family"])
    return st


def unit_text(u):
    return u["text"]


def course_text(c):
    """Lexical view of a course: objectives + outcomes + units. Name, code and prerequisites are left out (no identity leak)."""
    return " ".join([c.get("objectives", "")] + [o["text"] for o in c.get("outcomes", [])] + [u["text"] for u in c.get("units", [])])


def chunks(c):
    """The topic chunks of a course: its units; a course without units falls back to its outcomes, then its objectives."""
    us = [(u["title"] or " ".join(u["text"].split()[:8]), u["text"]) for u in c.get("units", [])]
    return us or [(o["text"][:60], o["text"]) for o in c.get("outcomes", [])] or ([("objectives", c["objectives"])] if c.get("objectives") else [])
