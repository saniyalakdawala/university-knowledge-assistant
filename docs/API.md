# API reference: University Knowledge Assistant (Group 4)

Base URL: `http://localhost:8000` when run locally (see README). Interactive docs: `GET /docs` (Swagger UI).
All requests and responses are JSON. No authentication.

| Endpoint | Method | What it does |
|---|---|---|
| `/api/chat` | POST | Answer a question from the SRB 2026, from the new course's syllabus, or from Group 9's live API |
| `/api/knowledge/course` | POST | Load the new course's syllabus passages handed over by Group 9 |
| `/api/knowledge/course` | GET | Which new course is loaded (title, number of passages) |
| `/api/health` | GET | Status of both indexes, retrieval mode, Groq and Group 9 connection |

---

## POST /api/chat

### Request

| Field | Type | Required | Meaning |
|---|---|---|---|
| `question` | string | yes | The student's question (1 or more characters) |
| `route` | string | no | `auto` (default), `srb`, `course`, `course+srb`, `curriculum`, `both` |
| `course_context` | string | no | Course name to send to Group 9's API when `route` is `curriculum` |
| `session_id` | string | no | Accepted but not used yet (no chat memory in the API) |

What `route` means:

| Route | Knowledge source |
|---|---|
| `srb` | Student Resource Book 2026 (university and MPSTME policy) |
| `course` | The new course's draft syllabus from Group 9 (needs `POST /api/knowledge/course` first) |
| `course+srb` | Both of the above, kept as two separate answers (for example, the new course's prerequisites *and* the credits needed to graduate) |
| `curriculum` | Group 9's live Curriculum Intelligence API for an existing named course |
| `both` | SRB plus Group 9's live API |

With `auto`, a question that mentions "the new course", "this course" or the loaded course's title goes to `course`. A question that also asks about a university rule (graduation, attendance, ATKT, passing, fees and similar) goes to `course+srb`. A named existing course with "prerequisite" or "overlap" goes to `curriculum`. Everything else goes to `srb`.

### Response

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The question as received |
| `route_taken` | string | Which route answered |
| `synthesized_answer` | string | The text to show the user |
| `srb` | object or null | SRB answer (fields below) |
| `course` | object or null | New-course answer (same fields as `srb`) |
| `curriculum` | object or null | Group 9 API result (`connected`, `is_fallback`, `status_note`, `summary`, `course`, `prerequisites`, `overlaps`) |
| `timestamp` | string | UTC ISO time |

Fields of `srb` and `course`:

| Field | Meaning |
|---|---|
| `answer` | Short answer, or `"I could not find this in ..."` when it refuses |
| `abstained` | `true` when the system refused (retrieval confidence below 0.24, or the LLM found no answer in the passages) |
| `document` | `Student Resource Book (SRB) 2026` or `Group 9 draft syllabus: <title>` |
| `section`, `cited_page` | Section name and page (SRB: PDF page; new course: section number) |
| `source` | One-line citation, e.g. `SRB 2026 \| Part II (MPSTME) \| MPSTME 4. Passing Standards \| PDF page 77` |
| `evidence_quote` | A sentence copied word for word from the cited passage |
| `confidence` | Retrieval confidence 0 to 1: mean of keyword coverage, cosine similarity and cross-encoder probability. Computed before the LLM is called |
| `mode` | `generated` (LLM answer passed the grounding check), `fallback-extractive (...)` (LLM unavailable or ungrounded, so the SRB sentence is returned), `refused` |
| `numbers_grounded`, `support` | Grounding check: every number in the answer appears in the passages, and the share of answer words found in them |
| `lexical_coverage`, `semantic_similarity`, `ce_prob` | The three signals behind `confidence` |

### Example: SRB policy question

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum attendance required to appear for the end-semester exam?"}'
```

```python
import requests
r = requests.post("http://localhost:8000/api/chat",
                  json={"question": "What is the minimum attendance required to appear for the end-semester exam?"})
data = r.json()
print(data["synthesized_answer"], data["srb"]["source"], data["srb"]["confidence"])
```

Response shape (values come from the live system):

```json
{
  "question": "What is the minimum attendance required to appear for the end-semester exam?",
  "route_taken": "srb",
  "synthesized_answer": "<answer>",
  "srb": {
    "answer": "<answer>",
    "source": "SRB 2026 | <part> | <section> | PDF page <n>",
    "document": "Student Resource Book (SRB) 2026",
    "section": "<section>",
    "confidence": 0.0,
    "evidence_quote": "<sentence copied from the SRB>",
    "cited_page": 0,
    "abstained": false,
    "refusal_reason": "",
    "mode": "generated",
    "numbers_grounded": true,
    "support": 0.0,
    "lexical_coverage": 0.0,
    "semantic_similarity": 0.0,
    "ce_prob": 0.0
  },
  "course": null,
  "curriculum": null,
  "timestamp": "2026-10-31T04:00:00+00:00"
}
```

### Example: out-of-scope question (refusal)

```bash
curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "How can I book a badminton court on campus?", "route": "srb"}'
```

Expected: `srb.abstained = true`, `srb.answer = "I could not find this in the Student Resource Book."`.

---

## POST /api/knowledge/course

Loads the new course's syllabus that Group 9's Curriculum Intelligence Engine produces. It replaces any course loaded before. The rows use our own passage schema (Group 9 builds them with our chunking settings: 100-word windows, 25-word overlap).

Body, either of:

```json
{"rows": [{"chunk_id": "G9C0000", "page": 1, "section": "Course overview", "part": "Group 9 draft syllabus: <title>", "text": "..."}]}
```

or Group 9's whole `POST /modernize-curriculum` response (we read `handoffs.group4_chunks`):

```bash
# 1) Group 9 produces the new course
curl -s -X POST $GROUP9_URL/modernize-curriculum -H "Content-Type: application/json" -d @modernize_request.json > g9.json
# 2) Hand it to Group 4
curl -s -X POST http://localhost:8000/api/knowledge/course -H "Content-Type: application/json" -d @g9.json
```

Response: `{"loaded": true, "title": "<course title>", "passages": 12, "source": "POST /api/knowledge/course"}`.
Error 422 if there are no rows, or if a row is missing `chunk_id`, `page`, `section`, `part` or `text`.

The server also loads `data/group9/group4_chunks.csv` at start-up if the file exists (path set by `COURSE_PASSAGES_CSV`).

Then ask:

```bash
curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "How many credits does the new course carry?"}'
```

The answer is in `course` and is cited as `Group 9 draft syllabus: <title> | <section>`. It is never cited as SRB policy.

---

## GET /api/health

```json
{
  "status": "ok",
  "group4": {"status": "ready", "chunks_count": 424, "gate": 0.24, "retrieval_mode": "full: BM25 + MiniLM + cross-encoder (evaluated)",
             "groq_configured": true, "groq_model": "qwen/qwen3.8-27b", "new_course": {"loaded": true, "title": "...", "passages": 12}},
  "group9": {"status": "online", "connected": true, "endpoint": "http://localhost:8001"}
}
```

`retrieval_mode` is `lite: BM25 only (LOW_MEMORY)` on the free Render deployment. Our reported metrics are for the full mode only.

## Errors

| Code | When |
|---|---|
| 422 | Empty question, unknown `route`, or bad passage rows |
| 200 with `abstained: true` | The question is not answered by the source; this is the expected refusal, not an error |
| 200 with `mode: fallback-extractive (...)` | Groq is unavailable (no key, timeout, HTTP 429): the answer is the best-matching SRB sentence, quoted word for word |
