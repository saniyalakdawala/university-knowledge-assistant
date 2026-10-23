import pytest
from fastapi.testclient import TestClient
from app import app
from backend.curriculum_client import CurriculumClient

client = TestClient(app)

def test_srb_policy_question():
    """Test standard university policy query over SRB (Group 4 only)."""
    question = "What is the minimum attendance required to appear for the end-semester exam?"
    response = client.post("/api/chat", json={"question": question, "route": "srb"})
    assert response.status_code == 200
    data = response.json()
    assert data["route_taken"] == "srb"
    assert data["srb"] is not None
    srb = data["srb"]
    assert not srb["abstained"]
    assert "attendance" in srb["answer"].lower() or "80" in srb["answer"] or "75" in srb["answer"]
    assert "srb 2026" in srb["source"].lower()
    assert srb["cited_page"] is not None
    assert srb["confidence"] >= 0.46
    assert len(srb["evidence_quote"]) > 0

def test_curriculum_question_with_fallback():
    """Test curriculum intelligence query for prerequisites."""
    question = "What are the prerequisites for Natural Language Processing?"
    response = client.post("/api/chat", json={"question": question, "route": "curriculum"})
    assert response.status_code == 200
    data = response.json()
    assert data["route_taken"] == "curriculum"
    assert data["curriculum"] is not None
    curr = data["curriculum"]
    # Either live or reference stand-in fallback
    assert curr["summary"] != ""
    assert "prerequisite" in curr["summary"].lower()
    if curr["is_fallback"]:
        assert "Group 9" in curr["status_note"]

def test_cross_group_combined_question():
    """Test combined question requiring both Group 4 policy and Group 9 curriculum."""
    question = "What are the prerequisites for Machine Learning, and what is the minimum attendance and passing grade required?"
    response = client.post("/api/chat", json={"question": question, "route": "both"})
    assert response.status_code == 200
    data = response.json()
    assert data["route_taken"] == "both"
    assert data["srb"] is not None
    assert data["curriculum"] is not None
    assert len(data["synthesized_answer"]) > 50
    # Both sources are represented
    assert "official university policy" in data["synthesized_answer"].lower()
    assert "curriculum intelligence" in data["synthesized_answer"].lower()

def test_out_of_scope_refusal():
    """Test that out-of-scope question triggers proper refusal and abstention."""
    question = "How can I book a badminton court on campus?"
    response = client.post("/api/chat", json={"question": question, "route": "srb"})
    assert response.status_code == 200
    data = response.json()
    srb = data["srb"]
    assert srb["abstained"] is True
    assert "could not find" in srb["answer"].lower() or "not_found" in srb["answer"].lower()
    assert srb["confidence"] < 0.46

def test_group9_unreachable_fallback(monkeypatch):
    """Test Group 9 client gracefully handles connection timeout or server downtime."""
    def mock_post(*args, **kwargs):
        raise Exception("Mocked connection failure: Group 9 service port 8000 closed")

    import httpx
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    client_g9 = CurriculumClient(base_url="http://invalid-host:9999")
    res = client_g9.check_prerequisites("Operating Systems")
    assert res.connected is False
    assert res.is_fallback is True
    assert "Operating Systems" in res.summary
    assert "Group 9 service unavailable" in res.status_note

def test_group9_invalid_response_handling(monkeypatch):
    """Test Group 9 client handles unexpected 500 error responses."""
    class MockBadResponse:
        status_code = 500
        text = "Internal Server Error: Database failure"

    def mock_bad_post(*args, **kwargs):
        return MockBadResponse()

    import httpx
    monkeypatch.setattr(httpx.Client, "post", mock_bad_post)

    client_g9 = CurriculumClient()
    res = client_g9.check_prerequisites("Machine Learning")
    assert res.connected is False
    assert res.is_fallback is True
    assert "HTTP 500" in res.status_note

def test_week6_regression_cases():
    """Regression test against representative Week 6 benchmark questions."""
    cases = [
        {
            "question": "What is the minimum attendance required to appear for the end-semester exam?",
            "must_contain": ["80", "%", "attendance"],
            "expect_abstained": False
        },
        {
            "question": "Are smart watches allowed in the examination hall?",
            "must_contain": ["forbidden", "not", "smart"],
            "expect_abstained": False
        },
        {
            "question": "Who is the Vice-Chancellor of NMIMS?",
            "must_contain": ["dr", "vice-chancellor"],
            "expect_abstained": False
        }
    ]

    for c in cases:
        res = client.post("/api/chat", json={"question": c["question"], "route": "srb"})
        assert res.status_code == 200
        srb = res.json()["srb"]
        assert srb["abstained"] == c["expect_abstained"]
        if not c["expect_abstained"]:
            combined_text = (srb["answer"] + " " + srb["evidence_quote"]).lower()
            assert any(word in combined_text for word in c["must_contain"]), f"Failed on: {c['question']}"

def test_no_secrets_in_payload():
    """Verify secrets or private paths are not returned to the frontend."""
    res = client.post("/api/chat", json={"question": "What is the attendance rule?"})
    assert res.status_code == 200
    text = res.text.lower()
    assert "gsk_" not in text
    assert "groq_api_key" not in text
    assert "secret" not in text
    assert "c:\\users\\" not in text

def test_groq_fallback_when_unavailable(monkeypatch):
    """Test that when Groq API encounters network failure, it falls back to verified extractive answer gracefully."""
    import requests
    from backend.rag_service import get_rag_service
    rag_inst = get_rag_service()

    def mock_post(*args, **kwargs):
        raise requests.ConnectionError("Mocked Groq service unreachable")

    monkeypatch.setattr(rag_inst.http_session, "post", mock_post)
    res = rag_inst.ask("What is the minimum attendance required to appear for the end-semester exam?")
    assert not res["abstained"]
    assert "fallback-extractive" in res["mode"]
    assert "attendance" in res["answer"].lower() or "80" in res["answer"]
    assert res["cited_page"] is not None
