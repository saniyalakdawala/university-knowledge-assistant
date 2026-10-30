# Demo script: 31 October (15 minutes)

Run everything from one laptop with the **full** local version (`/api/health` shows `retrieval_mode: full`).
Do not demo from the Render link: it runs the lite keyword-only mode.

## Before class (checklist)

- [ ] `.env` has `GROQ_API_KEY`, `GROQ_MODEL=qwen/qwen3.8-27b`, `CONFIDENCE_GATE=0.24`, `TOP_K_PASSAGES=2`
- [ ] `data/group9/group4_chunks.csv` from Group 9 is in place (or Group 9 will POST it live)
- [ ] `uvicorn app:app --port 8000` started once already (models downloaded and cached)
- [ ] `python -m pytest tests/ -q`: all passed; screenshot of the result
- [ ] Every question below tried once; note the actual answers
- [ ] Backup: a screen recording of this whole script, in case the Wi-Fi or Groq fails
- [ ] Second terminal open for the `curl` calls; browser zoom 125%

## Running order

| Time | Who | What to show | What to say |
|---|---|---|---|
| 0:00 to 1:30 | Speaker 1 | Slides 1 to 3 | Problem, users, what the asset returns |
| 1:30 to 3:00 | Speaker 2 | Slides 4 to 6 | Data (SRB 2026, 424 chunks, 182 test questions) and the pipeline |
| 3:00 to 4:00 | Speaker 3 | Slide 7 | Baseline 48.8% to final 89.5%; hallucination 20% to 0% |
| 4:00 to 6:30 | Speaker 3 | Browser, policy questions | Q1 and Q2 below. Point at the page, evidence quote and confidence |
| 6:30 to 9:30 | Speaker 4 | Group 9 hand-off | Q3 to Q6 below: the new course from Group 9 |
| 9:30 to 11:00 | Speaker 4 | Terminal, `curl` | Reuse by another group: Q7 |
| 11:00 to 12:30 | Speaker 5 | Failure case | Q8 and Q9 below |
| 12:30 to 15:00 | Speaker 5 | Slides 9 to 10, questions | Reusable asset, limitations, future work |

## Questions to ask

Policy (SRB):
1. "What is the minimum attendance required to appear for the end-semester exam?"
2. "What is the passing criterion for a course assessed only through internal continuous assessment?"
   (This is also what Group 6's student asks about.)

The new course from Group 9 (the class demo chain: Group 9 designs the course, then Group 4 answers students):
3. "What are the prerequisites for the new course?"
4. "Does the course have lab work?"
5. "How many credits does the course carry?"
6. "What are the prerequisites for the new course, and how many credits must I complete to graduate?"
   Show the two separate answers: the new course (Group 9 syllabus) and the graduation rule (SRB).

If Group 9 is running live, have them send their output straight to us:

```bash
curl -X POST http://localhost:8000/api/knowledge/course -H "Content-Type: application/json" -d @g9.json
curl http://localhost:8000/api/knowledge/course
```

Reuse by another group:

7. ```bash
   curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
     -d '{"question": "Can I appear for a re-examination if I fail the end-semester exam?"}'
   ```
   Say: "Group 6 or Group 10 call exactly this. JSON in, JSON out, with a citation and a confidence score."

Failure (required by the brief, Part G):

8. Out of scope: "How can I book a badminton court on campus?" The system refuses. That is the correct behaviour.
9. A known failure, for example: "Who is the head of the Data Science department at MPSTME?"
   Explain: the staff table is flattened by PDF extraction, so a title can attach to the wrong name (error case 1 in `docs/evaluation.md`).
   Check this one before class; if it now answers correctly, use error case 6 or 8 instead.

## If something breaks

- Groq down or rate-limited: answers switch to quoting the SRB sentence (`mode: fallback-extractive`). Say so; it is designed behaviour.
- Group 9 not reachable: the CSV in `data/group9/` still answers Q3 to Q6.
- Laptop fails: play the backup recording.
