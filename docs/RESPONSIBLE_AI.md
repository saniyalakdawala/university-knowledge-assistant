# Responsible AI Analysis: University Knowledge Assistant & Curriculum Intelligence

**Course / Component:** SNLP Project — Group 4 (University Knowledge Assistant) × Group 9 (Curriculum Intelligence Engine)  
**Document Date:** October 2026  
**System Scope:** Student Resource Book (SRB) 2026 & NMIMS Syllabus Intelligence  

---

## 1. Executive Summary & Core Principle

The AI University Assistant integrates two distinct artificial intelligence workflows:
1. **Group 4:** High-stakes retrieval-augmented generation over binding university administrative policies (attendance requirements, disciplinary codes, grade calculations, fee refund timelines, de-registration).
2. **Group 9:** Advisory curriculum analysis over engineering syllabi (prerequisite dependency graphs, cross-course overlaps, Bloom taxonomy mapping).

> [!CAUTION]
> **Fundamental Responsible AI Disclaimer:**  
> A successful HTTP 200 API response does **NOT** prove that an answer is factually correct or legally binding. Academic decisions involving student eligibility, grade progression (ATKT), scholarships, or course exemptions must always be confirmed through official university channels and academic deans.

---

## 2. Groundedness and Hallucination Risks

### 2.1 The Specific Danger in University Settings
In an academic institution, hallucinations have severe real-world ramifications:
- If an AI incorrectly states that *70% attendance is sufficient* when policy requires *80%*, a student relying on this advice risks course de-registration or being debarred from final examinations.
- If an AI invents refund percentages or deferment deadlines, students incur financial hardship.

### 2.2 Group 4 Mitigation Architecture
To prevent hallucination, Group 4 implements a three-tier defensive pipeline:
1. **Verbatim Evidence Quoting:** Every answer extracts and highlights an exact sentence quotation directly from the underlying SRB PDF passage (`evidence_quote`).
2. **Strict Grounding Checks:** Before returning any generated response, `grounding_check()` parses all numerical figures, monetary amounts (e.g. `Rs. 500`), and percentages (e.g. `80 %`). If any number in the output was not present in the cited source context, the generation is flagged as ungrounded and immediately rejected in favor of the verbatim extractive sentence.
3. **Calibrated Confidence Gating ($\ge 0.46$):** The system computes a composite confidence score:
   $$\text{Confidence} = \frac{\text{LexicalCoverage} + \text{SemanticSimilarity} + \sigma(\text{CrossEncoderLogit})}{3}$$
   Queries scoring below $0.46$ trigger an automatic refusal: `"I could not find this in the Student Resource Book."`

---

## 3. Disentangling Official Policy from Curriculum Advisory

A vital responsible-AI design choice in our Week 7 web architecture is the **strict visual and semantic separation** of sources:

| Dimension | Group 4 (SRB Knowledge Assistant) | Group 9 (Curriculum Intelligence Engine) |
|---|---|---|
| **Authoritative Status** | **Official Policy:** Binding rules enacted by SVKM's NMIMS Academic Council. | **Advisory Analysis:** Machine-learning inferences from syllabus texts and reference job descriptions. |
| **Ground Truth Source** | Student Resource Book 2026 (PDF text layer). | NMIMS syllabus archive + LLM-annotated reference labels. |
| **Human Supervision** | Published university regulations. | Draft syllabus suggestions intended for curriculum committees; **no automated syllabus pruning**. |
| **Frontend Representation** | Blue Card with official citations (Part, Section, Page). | Purple Card explicitly labeled with `"Curriculum Engine"` and `"Live API"` or `"Reference Stand-in"`. |

Group 9's predictions (e.g., missing prerequisites or overlap percentages) are **never** presented as ratified university rules.

---

## 4. Limitations of Retrieval Confidence Scores

Confidence scores must not instill a false sense of security:
- **Heuristic Nature:** While our dev set tuned the refusal threshold at $0.46$, semantic cosine similarities can occasionally score high on plausible-sounding but irrelevant university clauses.
- **Lexical Outliers:** Acronyms or unique phrases outside the vocabulary may reduce keyword coverage even when the relevant passage is found.
- **Uncalibrated Output:** As documented by Group 9 in their Week 4 evaluation, logistic confidence models over TF-IDF and MaxSim are relative and require cautious interpretation rather than absolute probabilistic reliance.

---

## 5. Handling Unsupported and Out-of-Scope Questions

A responsible system must know what it does **not** know.
- **Demonstrated Safety:** During Week 6 blind evaluation, our system answered **0 out of 20** out-of-scope questions (100% rejection rate on queries such as sports facility booking, campus cafeteria menus, or hostel room allocations).
- **Graceful Refusal:** When a question lacks sufficient grounding in the SRB, the interface displays a clear refusal badge explaining that the query falls outside documented university policy.

---

## 6. Privacy, Anonymity, and Data Minimization

- **No Student PII Collection:** The assistant does not require login, student roll numbers (SAP IDs), grades, or email addresses.
- **Zero Logging of Sensitive Queries:** All interactions remain session-scoped in memory.
- **Zero Credential Leakage:** API keys (such as `GROQ_API_KEY`) reside exclusively in server-side environment variables and are never transmitted to JavaScript clients or emitted in error traces.

---

## 7. Model Failures, API Downtime, and Fallback Behavior

When external services fail, graceful degradation preserves usability without deception:
1. **LLM Downtime / Rate-Limiting:** If the generative LLM experiences latency spikes or token quota limits (e.g. Groq 429), the backend transitions seamlessly to **Extractive Verified Mode**, serving the exact reranked passage sentence.
2. **Group 9 Service Unreachable:** If Group 9's local server (`http://localhost:8000`) is offline, the client does not crash or fabricate mock data. Instead, it activates the **Documented Reference Stand-in Cache**, clearly displaying an advisory banner:  
   *`"Group 9 service unavailable. Using documented reference syllabus data."`*

---

## 8. Bias, Accessibility, and Inclusivity

- **Accessible Interface Design:** High-contrast text palettes adhering to WCAG 2.1 AA standards, semantic HTML headings, and responsive layouts across viewport sizes.
- **Assistive Technology Support:** Keyboard-operable input stream with Enter-to-submit, distinct status announcements, and screen-reader readable badges.
- **Linguistic Fairness:** Text normalization handles spelling variations, British vs. American English, hyphenations, and common student abbreviations (`re-exam`, `ICA`, `ATKT`, `CGPA`).

---

## 9. Demo-Level Integration vs. Production University Deployment

| Feature | Week 7 Integration Demo | Future Production University Architecture |
|---|---|---|
| **Authentication** | Anonymous demonstration session. | SSO via NMIMS Student Portal with role-based access (Student, Faculty, Dean). |
| **Data Scope** | Text-layer SRB 2026 PDF + 4 course syllabi. | Complete campus repository including OCR over scanned circulars and program booklets. |
| **Auditing** | Console and pytest logging. | Immutable audit log of advice provided to students for compliance tracking. |
| **Model Hosting** | Local CPU inference + cached embeddings. | Enterprise self-hosted LLM with low-latency GPU cluster and redundancy. |
| **Dispute Redressal** | None (Demo). | Direct escalation button connecting students to the Academic Registrar. |
