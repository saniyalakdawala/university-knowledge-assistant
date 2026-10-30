# Evaluation: University Knowledge Assistant (Group 4)

Source of all numbers: `notebooks/Group4_Week6.ipynb` (Week 5 evaluation, reproduced exactly in Week 6: 182 of 182 answers identical).

## 1. Data

| Item | Value |
|---|---|
| Source document | Student Resource Book (SRB) 2026, SVKM's NMIMS, MPSTME (public PDF, 172 pages) |
| Pages indexed | 60 policy pages (cover pages, contents, blank forms, curriculum tables and image-only calendar pages excluded) |
| Chunks | 424 passages of 100 words with 25-word overlap; each keeps part, section and PDF page |
| Development set | 36 answerable + 19 out-of-scope questions (all tuning) |
| Test A | 54 answerable + 6 out-of-scope (Week 2 held-out set) |
| Test B | 108 answerable + 14 out-of-scope, written blind on SRB facts not in Test A |
| Extra suites | 12 dialogues (16 follow-ups), 240 perturbed questions (typos, informal, keyword-only, paraphrase), 18 adversarial probes |
| Labels | Key fact + gold page per question; out-of-scope flag |

## 2. Metrics

- **Answer accuracy**: answerable questions whose answer contains the annotated key fact (numbers and phrasing normalised; "two" = 2, "per cent" = %).
- **Citation accuracy**: the cited PDF page is the gold page.
- **Hallucination rate**: out-of-scope questions that got an answer instead of a refusal.
- **End-to-end accuracy**: answerable questions answered correctly plus out-of-scope questions refused, over all questions.
- 95% Wilson intervals; paired comparisons with the exact McNemar test.

## 3. Baseline vs improved

| System | Test A | Test B | A + B [95% CI] | Hallucination | End-to-end |
|---|---|---|---|---|---|
| Baseline (W2): BM25 + extractive sentence | 37.0% | 54.6% | 48.8% [41, 56] | 20.0% | 52.2% |
| v1 (W3): hybrid BM25 + MiniLM | 61.1% | 65.7% | 64.2% [57, 71] | 25.0% | 65.4% |
| v2 (W4): + cross-encoder re-rank | 70.4% | 69.4% | 69.8% [62, 76] | 25.0% | 70.3% |
| v3-local: Qwen2.5-1.5B on CPU | 75.9% | 79.6% | 78.4% [71, 84] | 25.0% | 78.0% |
| **v3 (final): Groq LLM + grounding + refusal** | **92.6%** | **88.0%** | **89.5% [84, 93]** | **0.0%** | **90.7%** |

- v3 vs v2: Test A 12 fixed / 0 broken (p = 0.0005); Test B 21 fixed / 1 broken (p < 0.0001).
- Citation accuracy: 90.1% (v2) to 94.4% (v3). False refusal on answerable questions: 5.6%.
- Latency: about 0.7 s per answer with the hosted LLM; about 21 s with the local 1.5B model on a laptop CPU.

## 4. Ablation (Test A + B)

| Variant | Answer accuracy | Hallucination | End-to-end |
|---|---|---|---|
| v3 (full) | 89.5% | 0% | 90.7% |
| Without LLM refusal (retrieval gate only) | 90.7% | 80% | 83.0% |
| Without any refusal | 90.7% | 100% | 80.8% |
| Without generation (v2 sentence + v3 refusals) | 69.8% | 0% | 73.1% |

Generation explains the accuracy gain; the LLM's NOT_FOUND explains the hallucination gain.

## 5. Confidence score

Confidence = mean of keyword coverage, cosine similarity and cross-encoder probability (computed before the LLM call).
High-confidence answers are 96% correct, medium 91%. AUROC for separating correct from wrong answers: 0.943.

## 6. Robustness

| Suite | Retrieval Hit@3 | v2 accuracy (sample) | v3 accuracy (sample) |
|---|---|---|---|
| Original | 98.1% | 100% | 100% |
| Typos | 87.0% | 50.0% | 83.3% |
| Keyword-only | 90.7% | 83.3% | 75.0% |
| Paraphrase (hand-written) | 90.7% | 66.7% | 91.7% |

Adversarial: 16 of 18 probes passed, including all 5 prompt injections. Both failures were refusals, not wrong answers.
Follow-up questions (notebook only, not in the API): 62.5% (10 of 16).

## 7. Where answers fail (162 answerable test questions)

| Outcome | Baseline | v1 | v2 | v3 |
|---|---|---|---|---|
| Correct | 79 | 104 | 113 | 145 |
| Retrieval miss | 30 | 29 | 21 | 6 |
| Reader error | 35 | 26 | 23 | 2 |
| Refused by retrieval gate | 18 | 3 | 5 | 0 |
| Refused by LLM (NOT_FOUND) | 0 | 0 | 0 | 9 |

## 8. Error cases (v3)

| # | Question | Expected | Output | Root cause |
|---|---|---|---|---|
| 1 | Who is the head of the Data Science department at MPSTME? | Dr. Siba Panda | Dr. Pintu Shah | Staff table flattened by PDF extraction; title attached to the wrong name |
| 2 | Who chairs the University's Women Grievance Redressal Cell? | Prof. Amita Vaidya | Dr. Meena Galliara | Wrong committee list read; correct list ranked 3rd |
| 3 | How many re-examinations are held for a course in a year? | Only one | "two re-exam attempts" | Neighbouring M.Tech/MCA rule retrieved; correct passage ranked 4th |
| 4 | Is the community service course a credit course? | No (MPSTME: non-credit) | "Yes, 1 credit" | SRB ambiguity: Part I and Part II disagree |
| 5 | Who chairs the MPSTME Woman Grievance Redressal Committee? | Dr. Archana Bhise | "The Chairperson / Head of department" | Generic Part I rule outranked the MPSTME member list |
| 6 | Penalty for revealing identity in the answer book? | Annulment of performance in that subject | "treated as unfair means" | Penalty-table row ranked 3rd, outside the 2 passages read |
| 7 | Punishment for trying to bribe an examiner? | Exclusion from four additional exams | Refused | Safe refusal: passages describe the offence, not the penalty |
| 8 | Punishment for a mobile phone in the exam hall? | Annulment of performance in full | Refused | Chunk boundary cut the table row before the penalty |
| 9 | Follow-up "How many times can I take it in the same year?" | Only once | Answered about academic breaks | Follow-up rewrite resolved "it" to the wrong topic |
| 10 | "Why is the library late fine Rs. 50 per day?" | Correct the premise: Rs. 3 per day | Refused | False premise refused instead of corrected |

Patterns: 10 of 17 failures had the right passage ranked 3rd or 4th (just below the 2-passage cut-off); 6 come from tables and lists flattened by PDF extraction.

## 9. Week 8 final testing

Run on the final code before the demo (`python -m pytest tests/ -v`) and paste the real output here.

- Full mode (local, all models): `<paste pytest summary line>`
- Lite mode (`LOW_MEMORY=true`, as on the free Render deployment): 23 passed, 4 failed. The 4 failures are SRB answers
  (attendance rule missed, badminton question answered instead of refused). The same 4 tests also fail on the Week 7 code
  in lite mode. Lite mode is therefore not used for the demo and is not covered by the metrics above.
- Group 9 hand-off (sample passages in Group 9's format): routing and loading tests pass in both modes.
