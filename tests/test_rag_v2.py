"""Query processing, chunking, routing and the benchmark metrics. No Groq call is needed."""
import pandas as pd

from backend.chunking import make_section_chunks
from backend.query_processing import normalize_query, parse_expansion
from backend.rag_service import split_sentences_v2
from backend.router import route_question, rule_route
from evaluation import metrics as M


def test_normalize_adds_srb_terms_and_keeps_the_question():
    q = normalize_query("bro got 3 KTs can i go to next yr")
    assert q.startswith("got 3 KTs") and "ATKT" in q and "F grade" in q and "year" in q


def test_expansion_rejects_malformed_output():
    assert parse_expansion("not json", "q?")["sub_questions"] == ["q?"]
    assert parse_expansion('{"standalone": 3}', "q?")["standalone"] == "q?"
    plan = parse_expansion('{"standalone": "A formal question?", "variants": ["one variant"], "sub_questions": ["part one?", "part two?"]}', "q?")
    assert plan["standalone"] == "A formal question?" and len(plan["sub_questions"]) == 2


def test_chunks_stay_inside_sections_and_keep_sentence_pages():
    text = " ".join(f"3.{i} Rule number {i} says that students must follow this clause carefully every day of the term." for i in range(1, 30))
    seg = pd.DataFrame([{"page": 5, "section": "A", "part": "P", "text": text}, {"page": 6, "section": "A", "part": "P", "text": "More of section A continues here on the next page of the book."},
                        {"page": 6, "section": "B", "part": "P", "text": "Section B starts here with its own first sentence."}])
    chunks = make_section_chunks(seg, split_sentences_v2, max_words=60, min_words=30)
    assert set(chunks[chunks.section == "B"].text.str.contains("Section B")) == {True}
    assert not chunks[chunks.section == "A"].text.str.contains("Section B").any()
    assert chunks.n_words.max() <= 75
    assert chunks.iloc[-2].sent_pages[-1] == 6                # the continuation on page 6 keeps its own page


def test_router_rules():
    assert rule_route("What is the re-exam fee?")["route"] == "srb"
    assert rule_route("what are the prerequisites for DBMS")["route"] == "curriculum"
    assert rule_route("prerequisites for ML and the minimum attendance")["route"] == "both"
    assert route_question("anything", "curriculum")["route"] == "curriculum"


def test_router_survives_a_failing_llm():
    def boom(*_):
        raise RuntimeError("down")
    assert route_question("prerequisites for ML and the minimum attendance", "auto", None, boom)["route"] == "both"


def test_router_splits_a_two_part_question():
    r = route_question("What are the prerequisites for Machine Learning, and what is the minimum attendance required?", "auto")
    assert "attendance" in r["srb_question"] and "prerequisites" in r["curriculum_question"]


def test_retrieval_metrics_by_hand():
    fact = {"evidence": "a fine of rs 3 per day per book", "page": 30}
    other = {"evidence": "two books for ten days", "page": 30}
    hit_a = {"text": "A fine of Rs 3 per day per book is levied", "page": 30}
    hit_b = {"text": "Borrowing entitlement: two books for ten days", "page": 30}
    miss = {"text": "unrelated text about examinations", "page": 12}
    s = M.retrieval_scores([fact, other], [miss, hit_a, hit_b], ks=(3,))
    assert s["recall@3"] == 1 and round(s["precision@3"], 3) == 0.667 and s["mrr@3"] == 0.5
    assert 0 < s["ndcg@3"] < 1


def test_key_match_and_refusal_scoring():
    assert M.key_in_answer("mumbai court|mumbai", "Only Mumbai courts apply")
    assert not M.key_in_answer("3 days", "within 5 days")
    assert M.answer_correct([], {"abstained": True, "answer": ""})
    assert not M.answer_correct([], {"abstained": False, "answer": "yes"})
