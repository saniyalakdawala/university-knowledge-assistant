# Responsible AI Statement: University Knowledge Assistant (Group 4)

The assistant gives information about university rules from the Student Resource Book (SRB) 2026. It never decides anything about a student. It is built to make every answer checkable and to refuse rather than guess.

**Privacy.** No login, no student IDs and no chat history. The server logs only the length of each question and the route taken, never the question text. The LLM provider (Groq) receives the question and two SRB passages, nothing else. Students are told not to type personal details. The Groq API key is kept in a server-side `.env` file that is excluded from git and never sent to the browser. The data is a public university handbook; no student records were used to build or test the system.

**Hallucination.** Answers come only from the top two SRB passages. The prompt forbids outside knowledge and requires the reply NOT_FOUND when the passages lack the answer. A grounding check rejects any answer containing a number, amount or percentage that is not in those passages and returns the verified SRB sentence instead. The system also refuses when retrieval confidence is below 0.24. On the held-out test sets, it answered 0 of 20 out-of-scope questions and ignored all 5 prompt-injection probes. Remaining risk: an answer can be on topic but incomplete, so the evidence quote is always shown.

**Bias and fairness.** The SRB applies to all students equally, but the assistant does not work equally well on every question type. Names and contacts (70.6% accuracy) and MPSTME-specific rules (79.3%) are weaker than university-level rules (91.7%). Accuracy falls for keyword-only queries (75.0%), and the system works in English only. This can disadvantage students who are less comfortable writing full English questions.

**Explainability.** Every answer shows the document, section and PDF page, a word-for-word quote from the SRB, and a confidence score built from three visible signals (keyword coverage, cosine similarity, cross-encoder probability). Students can open the SRB at that page and check.

**Security.** The prompt tells the LLM to ignore any instruction inside the question, and answers are limited to 100 tokens. Group 12's malicious-input module can screen questions before they reach the assistant.

**Human oversight.** Answers are informational only. Decisions on attendance shortfall, ATKT, re-examination, fees or discipline stay with faculty and the examination office. A refusal points the student back to the office. Group 9's syllabus answers are labelled as a draft for the Board of Studies, never as university policy.

**Limitations.** Confidence is a retrieval score, not a probability of being correct. The free cloud deployment runs a lite keyword-only mode that was not evaluated and fails 4 of our tests; it must not be used for real advice. Tables in the PDF lose their structure, and the SRB contains a few conflicting rules between Part I and Part II. The system covers the 2026 edition only and must be re-indexed when the SRB changes.
