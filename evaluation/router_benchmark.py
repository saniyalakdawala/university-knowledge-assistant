"""Routing accuracy: the original keyword router against the new router (rules alone, and rules + Groq).
    python evaluation/router_benchmark.py          writes evaluation/results/router.json"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLM_DISK_CACHE", str(ROOT / "evaluation/results/llm_cache.json"))
os.environ.setdefault("GROQ_MAX_RETRIES", "8")
os.environ.setdefault("GROQ_TIMEOUT", "60")

CASES = [
    # university policy
    ("What is the minimum attendance required to appear for the exam?", "srb"), ("how much is the re exam fee", "srb"),
    ("got 3 KTs can i go to next year", "srb"), ("Who is the Vice Chancellor?", "srb"), ("Can I eat in the classroom?", "srb"),
    ("What happens if I lose my ID card?", "srb"), ("how long does hostel deposit refund take", "srb"), ("Is there a dress code for guest lectures?", "srb"),
    ("What is the penalty for mass copying?", "srb"), ("How do I apply for a government scholarship?", "srb"), ("how many books can i borrow", "srb"),
    ("Can I take an academic break for medical reasons?", "srb"), ("What is the grading scale for 80 to 89 percent?", "srb"),
    ("whom do i contact about a cyber security incident", "srb"), ("What is the fire brigade number?", "srb"),
    ("How do I opt out of placements?", "srb"), ("How many credits does a programme need per year?", "srb"),
    ("How many interdisciplinary courses can I take?", "srb"), ("can first years take cbcs courses", "srb"), ("What is the Dean's list?", "srb"),
    ("How long is the B Tech internship?", "srb"), ("Is the campus Wi-Fi password shared?", "srb"),
    # curriculum
    ("What are the prerequisites for Natural Language Processing?", "curriculum"), ("compare Machine Learning and Deep Learning", "curriculum"),
    ("What courses are in semester 4 of B Tech IT?", "curriculum"), ("which semester is Operating Systems taught in", "curriculum"),
    ("Does the Data Science curriculum overlap with AI?", "curriculum"), ("find redundant courses in B Tech Computer Engineering", "curriculum"),
    ("what topics are missing from the IT curriculum", "curriculum"), ("map these outcomes to Bloom's taxonomy", "curriculum"),
    ("modernize the Machine Learning syllabus for industry", "curriculum"), ("what skill gaps does our deep learning course have", "curriculum"),
    ("how many credits is Cloud Computing", "curriculum"), ("Which courses cover blockchain?", "curriculum"),
    ("Is Linear Algebra taught before Machine Learning?", "curriculum"), ("What do we study in Computer Networks?", "curriculum"),
    ("compare B Tech IT and B Tech Computer Science", "curriculum"), ("electives offered in B Tech Data Science", "curriculum"),
    ("what is the L-P-T of Database Management Systems", "curriculum"), ("is compiler design part of the COMP programme", "curriculum"),
    ("Does anyone teach quantum computing here?", "curriculum"), ("syllabus overlap between OS and computer architecture", "curriculum"),
    # both
    ("What are the prerequisites for Machine Learning, and what is the minimum attendance required?", "both"),
    ("Which semester has Deep Learning and what is the passing mark?", "both"), ("how many credits is Operating Systems and what's the re-exam fee", "both"),
    ("If I fail Data Structures can I take it again, and what comes after it in the curriculum?", "both"),
    ("What courses are in semester 5 of B Tech IT and what is the attendance rule?", "both"),
    ("Is Cloud Computing an elective, and how do I get a scholarship?", "both"), ("compare ML and DL and tell me the grace marks rule", "both"),
    ("Does the NLP course overlap with Deep Learning, and how many F grades are allowed for promotion?", "both"),
    ("What is the syllabus structure for AI and what is the fine for late library books?", "both"),
    ("what are prereqs for DBMS and what's the late fee for the re-exam", "both"),
]


def legacy_route(question):
    """determine_route exactly as it was in app.py before this change."""
    q = question.lower()
    curr = ["prerequisite", "prereq", "syllabus", "overlap", "redundancy", "course content", "units", "bloom", "curriculum", "machine learning",
            "deep learning", "nlp", "natural language processing", "operating systems", "modernize"]
    srb = ["attendance", "grade", "grading", "cgpa", "exam", "re-exam", "re-examination", "fee", "fees", "refund", "scholarship", "library", "dean",
           "council", "atkt", "passing standard", "rules", "conduct", "ragging", "disciplinary", "leave", "deferment", "break", "convocation",
           "portal", "credit", "minimum"]
    has_curr, has_srb = any(k in q for k in curr), any(k in q for k in srb)
    return "both" if has_curr and has_srb else "curriculum" if has_curr else "srb"


def report(name, fn):
    rows = [(q, want, fn(q)) for q, want in CASES]
    right = Counter(w for _, w, g in rows if w == g)
    total = Counter(w for _, w, _ in rows)
    misses = [{"question": q, "expected": w, "got": g} for q, w, g in rows if w != g]
    return {"accuracy": round(sum(right.values()) / len(rows), 4), "by_route": {k: f"{right[k]}/{total[k]}" for k in total}, "misses": misses}


def main():
    from backend.curriculum.service import get_curriculum_service
    from backend.rag_service import UniversityRAGService
    from backend.router import route_question
    detect = get_curriculum_service().detect_courses
    out = {"n": len(CASES), "legacy": report("legacy", legacy_route),
           "rules": report("rules", lambda q: route_question(q, "auto", detect, None)["route"])}
    rag = UniversityRAGService()
    if rag.groq_api_key:
        llm = lambda m, t: rag._chat(m, t, json_mode=True)
        out["rules_plus_groq"] = report("groq", lambda q: route_question(q, "auto", detect, llm)["route"])
    (ROOT / "evaluation/results/router.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: (v["accuracy"], v["by_route"]) for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
