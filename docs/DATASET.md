# Dataset documentation: University Knowledge Assistant (Group 4)

## 1. Source

| Item | Value |
|---|---|
| Document | Student Resource Book (SRB) 2026, SVKM's NMIMS, Mukesh Patel School of Technology Management & Engineering (MPSTME) |
| Type | Public university handbook, PDF, 172 pages, English |
| Where | `data/SRB_2026.pdf` in the repository |
| Second source (integration) | The new course's draft syllabus produced by Group 9 (Curriculum Intelligence Engine), received as passage rows (`handoffs.group4_chunks` / `data/group9/group4_chunks.csv`) |
| Questions | Written by the team from the SRB (no student data) |

## 2. Size

| Item | Count |
|---|---|
| PDF pages | 172 |
| Pages indexed (policy text) | 60 |
| Pages excluded | 112: cover and title pages (1, 2, 67), contents (4, 69), blank application forms (48 to 66), curriculum and course-list tables (83 to 168), image-only academic calendar (169, 170) |
| Named sections | 41 (e.g. "3. Attendance Rules", "MPSTME 4. Passing Standards & Progression (ATKT)") |
| Policy text | about 30,900 words |
| Chunks (passages) | 424: 324 from Part I (University), 100 from Part II (MPSTME) |
| Chunk size | 100 words with 25 words of overlap (mean 93.8 words; shortest 26) |
| Sentences embedded | 2,267 (384-dimensional MiniLM vectors) |

## 3. Data format

`data/cache/chunks.pkl` is a pandas DataFrame with one row per passage:

| Column | Meaning | Example |
|---|---|---|
| `chunk_id` | Stable id | `C0042` |
| `page` | PDF page number (1-based) | `11` |
| `section` | Section name from the SRB's headings | `3. Attendance Rules` |
| `part` | `Part I (University)` or `Part II (MPSTME)` | |
| `text` | The passage text after cleaning | |
| `n_words` | Word count | `100` |

`data/cache/sent_emb.npy` holds the sentence embeddings (2,267 x 384, float32, L2-normalised), in the same order as the sentence split of the chunks.
Group 9 passages use the same columns (`chunk_id, page, section, part, text`); there `page` numbers the sections of their syllabus document.

## 4. Labels and splits (question sets)

Each answerable question has a reference answer, an answer key (the key fact that must appear), a verbatim evidence span and the gold PDF page. Out-of-scope questions are flagged and must be refused.

| Set | Answerable | Out-of-scope | Use |
|---|---|---|---|
| Development | 36 | 19 | All tuning: fusion weight beta, re-rank weight alpha, passages k, refusal gate, generator choice |
| Test A (Week 2 held-out) | 54 | 6 | Comparable across Weeks 2 to 5 |
| Test B (blind, Week 5) | 108 | 14 | New SRB facts not in Test A; 4 near-miss out-of-scope questions |
| Conversation | 16 follow-ups in 12 dialogues | | Follow-up handling (notebook only) |
| Robustness | 240 perturbed questions | 9 | Typos, informal style, keyword-only, hand-written paraphrases of Test A |
| Adversarial | 18 probes | | Prompt injection, false premise, near-miss, off-domain, multi-part |

The question sets are embedded in `notebooks/Group4_Week6.ipynb` (lists `QA`, `QA_B`, `EXTRA_OOS`, `ADVERSARIAL`, `CONV_DEV`). A notebook check confirms that no test question was used for tuning.

## 5. Preprocessing

1. Extract text page by page with PyMuPDF.
2. Normalise Unicode (NFKC), curly quotes and dashes; drop page-number lines, dotted leaders and underscores; rejoin hyphenated words.
3. Drop the 112 non-policy pages listed above.
4. Split each page at the 41 known section headings; tag Part I or Part II.
5. Chunk each section into 100-word windows with 25 words of overlap (chunks never cross a section).
6. Split chunks into sentences (abbreviation-safe) for embeddings and evidence quotes; each sentence is embedded as `section: sentence`.
7. For BM25: lowercase, alphanumeric tokens, English stop-words removed, Porter stemming, section title added to the chunk.
8. For the grounding check: numbers and phrases normalised ("two" = 2, "per cent" = %, "Rs. 1,000" = rs 1000, "at least" = minimum).

## 6. Data quality issues

- PDF extraction flattens tables: staff lists, committee lists and penalty tables lose their row structure (cause of error cases 1, 6 and 8 in `docs/evaluation.md`).
- A table row can be cut at a chunk boundary.
- Part I (University) and Part II (MPSTME) sometimes state different rules on the same topic (e.g. community service credits).
- Students use words the SRB never uses ("backlog", "KT", "bunk"), so purely lexical search misses them.
- The question sets were written by the team; answer keys were checked by hand, but a few keys may be too strict (1 of 17 Week 5 failures was a correct answer missed by the key).

## 7. Privacy

- The SRB is a public university document. It names university officials in their official roles; no student records are used.
- No student names, roll numbers, grades or messages are in the data or the question sets.
- At runtime the server stores no chat history and does not log question text. The Groq API receives only the question and two SRB passages.

## 8. Licensing and usage

- The SRB is published by SVKM's NMIMS for its students. We use it for a non-commercial academic course project only. Copyright remains with the university; the PDF in the repository should not be reused outside the course.
- Group 9's syllabus passages belong to Group 9 and are used with their hand-off for the class integration demo.
- Models: all-MiniLM-L6-v2 and ms-marco-MiniLM-L-6-v2 (Apache 2.0); Groq-hosted `qwen/qwen3.8-27b` under Groq's terms of service.

## 9. Limitations

- One document, one edition (2026), English only. A new SRB edition needs the index rebuilt.
- Curriculum tables (pages 83 to 168) are excluded, so course-level questions are answered through Group 9, not from the SRB.
- Test sets are modest in size (162 answerable, 20 out-of-scope): 95% intervals are about plus or minus 5 points.
