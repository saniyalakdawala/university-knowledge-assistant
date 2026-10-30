"""Group 9 -> Group 4 hand-off: load the new course's syllabus passages and answer the demo-day questions."""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, determine_route
from backend.course_kb import get_course_kb

client = TestClient(app)
SAMPLE = Path(__file__).parent / "fixtures" / "sample_group9_chunks.csv"


@pytest.fixture(scope="module", autouse=True)
def load_sample_course():
    rows = pd.read_csv(SAMPLE).to_dict("records")
    r = client.post("/api/knowledge/course", json={"rows": rows})
    assert r.status_code == 200, r.text
    yield


def test_ingest_status():
    s = client.get("/api/knowledge/course").json()
    assert s["loaded"] is True
    assert s["passages"] == len(pd.read_csv(SAMPLE))
    assert s["title"] == "Natural Language Processing and Large Language Models"


def test_ingest_accepts_group9_modernize_response():
    rows = pd.read_csv(SAMPLE).to_dict("records")
    r = client.post("/api/knowledge/course", json={"meta": {}, "courses": [], "handoffs": {"group4_chunks": rows}})
    assert r.status_code == 200
    assert r.json()["loaded"] is True


def test_ingest_rejects_bad_rows():
    r = client.post("/api/knowledge/course", json={"rows": []})
    assert r.status_code == 422
    r = client.post("/api/knowledge/course", json={"handoffs": {"group4_chunks": [{"text": "no other columns"}]}})
    assert r.status_code == 422


@pytest.mark.parametrize("question,route", [
    ("What are the prerequisites for the new course?", "course"),
    ("Does the course have lab work?", "course"),
    ("How many credits does the course carry?", "course"),
    ("What are the prerequisites for Natural Language Processing and Large Language Models?", "course"),
    ("What are the prerequisites for the new course, and how many credits must I complete to graduate?", "course+srb"),
    ("What is the minimum attendance required to appear for the end-semester exam?", "srb"),
    ("What are the prerequisites for Machine Learning?", "curriculum"),
    ("What are the prerequisites for Natural Language Processing?", "curriculum"),   # existing course, not the new one
    ("Is there an end-semester exam in the new NLP and LLM course?", "course"),
])
def test_routing(question, route):
    assert determine_route(question) == route


@pytest.mark.parametrize("question,must", [
    ("What are the prerequisites for the new course?", "machine learning"),
    ("Does the course have lab work?", "lab"),
    ("How many credits does the course carry?", "4 credits"),
])
def test_group9_demo_questions(question, must):
    data = client.post("/api/chat", json={"question": question}).json()
    assert data["route_taken"] == "course"
    c = data["course"]
    assert not c["abstained"]
    assert must in (c["answer"] + " " + c["evidence_quote"]).lower()
    assert c["document"].startswith("Group 9 draft syllabus")
    assert "srb" not in c["source"].lower()          # a syllabus passage is never cited as SRB policy


def test_combined_new_course_and_policy():
    q = "What are the prerequisites for the new course, and what is the minimum attendance required?"
    data = client.post("/api/chat", json={"question": q}).json()
    assert data["route_taken"] == "course+srb"
    assert data["course"] is not None and data["srb"] is not None
    assert "New course" in data["synthesized_answer"] and "Official University Policy (SRB 2026)" in data["synthesized_answer"]


def test_unknown_course_is_not_guessed():
    from backend.curriculum_client import CurriculumClient
    res = CurriculumClient(base_url="http://invalid-host:9999").query("What are the units of this subject?")
    assert "which course" in res.summary.lower()
    assert "natural language processing" not in res.summary.lower()
