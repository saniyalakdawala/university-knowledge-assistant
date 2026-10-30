"""Curriculum Intelligence: the vendored algorithms on the engine's own syllabus fixture, and the service on the real SRB catalogue."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.curriculum.service import get_curriculum_service
from backend.curriculum.srb_catalog import parse_structures
from backend.config import SRB_PDF_PATH
from curriculum_engine import data as D
from curriculum_engine import prereq as PQ
from curriculum_engine.api_engine import ApiEngine
from curriculum_engine.schemas import CourseRef

FIX = Path(__file__).parent / "fixtures/courses_small.jsonl"


@pytest.fixture(scope="module")
def engine():
    return ApiEngine(recs=D.enrich([json.loads(l) for l in open(FIX, encoding="utf-8")]))


@pytest.fixture(scope="module")
def svc():
    return get_curriculum_service()


@pytest.fixture
def small_programme(monkeypatch):            # the fixture is far smaller than the engine's coverage rule
    monkeypatch.setattr(PQ, "MIN_VERSIONS", 1)
    monkeypatch.setattr(PQ, "MIN_FAMILIES_PER_SEM", 1)


def test_full_syllabus_overlap_uses_unit_text(engine):
    a, b = engine.resolve(CourseRef(name="Deep Learning")), engine.resolve(CourseRef(name="Machine Learning"))
    r = engine.compare_courses(a, b)
    assert 0 <= r["overlap"]["semantic_similarity"] <= 1 and r["overlap"]["confidence"] > 0
    assert r["common_topics"] or r["unique_to_a"]


def test_full_syllabus_prerequisite_flag(engine, small_programme):
    flagged = [c for c in engine.recs if c.get("prerequisites") and engine.prereq_v2(c)["flags"]]
    assert flagged, "the fixture contains a prerequisite that is not offered earlier"


def test_full_syllabus_bloom_uses_verb_lexicon(engine):
    items = engine.bloom_items(["Design a network protocol", "Define the term packet"])
    assert [i["level"] for i in items] == ["Create", "Remember"]


def test_catalogue_matches_the_srb_tables():
    programmes = {p["programme_title"]: p for p in parse_structures(SRB_PDF_PATH)}
    it = programmes["B Tech Information Technology"]
    sem4 = next(t for t in it["terms"] if t["semester"] == 4)
    os_row = next(c for c in sem4["courses"] if c["course_name"] == "Operating Systems")
    assert (os_row["lecture_hrs"], os_row["practical_hrs"], os_row["credits"]) == (3, 2, 4)
    assert len(programmes) == 25


def test_no_invented_prerequisites(svc):
    r = svc.answer("What are the prerequisites for Machine Learning?")
    assert r["intent"] == "prerequisites" and r["data_basis"] == "name_only"
    assert "does not state prerequisites" in r["summary"]


def test_semester_listing_cites_the_page(svc):
    r = svc.answer("courses in semester 4 of B Tech IT")
    assert "Operating Systems" in r["summary"] and r["citations"][0]["page"] == 85


def test_named_topic_checked_against_earlier_semesters(svc):
    r = svc.answer("is Linear Algebra taught before Machine Learning in B Tech IT?")
    assert r["intent"] == "prerequisites" and "covered earlier" in r["summary"]


def test_compare_programmes(svc):
    r = svc.answer("compare B Tech IT and B Tech Computer Engineering")
    assert r["intent"] == "compare_programmes" and "IT vs COMP" in r["summary"]


def test_redundancy_excludes_course_and_its_lab(svc):
    pairs = svc.detect_redundancy("IT")["pairs"]
    assert not [p for p in pairs if "Lab" in p["course_a"] + p["course_b"]]


def test_missing_topics_are_labelled_name_level(svc):
    r = svc.answer("what topics are missing in B Tech Computer Engineering")
    assert r["intent"] == "missing_topics" and r["data_basis"] == "name_only" and "names" in r["note"]


def test_modernize_is_traceable_and_labels_sample_jds(svc):
    r = svc.modernize(["Machine Learning"])
    assert r["traceable"] and "synthetic" in r["jd_source"]


def test_unknown_course_gets_a_question_not_a_guess(svc):
    r = svc.answer("compare")
    assert r["found"] is False


def test_rest_endpoints():
    from app import app
    c = TestClient(app)
    assert c.get("/api/curriculum/courses", params={"q": "Cloud Computing"}).json()["results"][0]["course_name"] == "Cloud Computing"
    assert c.post("/api/curriculum/compare-courses", json={"course_a": "Machine Learning", "course_b": "Deep Learning"}).status_code == 200
    assert c.post("/api/curriculum/overlaps", json={"course": "no such course zzzz qqq"}).status_code == 404
    assert c.post("/api/curriculum/redundancy", json={"programme": "NOPE"}).status_code == 404
    assert c.post("/api/curriculum/bloom", json={"outcomes": ["Analyze a circuit"]}).json()["outcomes"][0]["level"] == "Analyze"
