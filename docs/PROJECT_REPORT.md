# Comprehensive Academic Project Report
## AI University Knowledge Assistant & Curriculum Intelligence System
**Course:** Statistical Natural Language Processing (SNLP)  
**Academic Institution:** SVKM's NMIMS, Mukesh Patel School of Technology Management & Engineering (MPSTME)  
**Collaborating Groups:**
- **Group 4:** University Knowledge Assistant (RAG over Student Resource Book 2026)
- **Group 9:** Curriculum Intelligence Engine  
**Version / Milestone:** Week 7 Integration Release & Production Deployment (October 2026)  
**Repository:** `https://github.com/saniyalakdawala/university-knowledge-assistant`

---

## Executive Summary

The **AI University Knowledge Assistant** is a production-grade, retrieval-augmented intelligence portal engineered to resolve critical administrative and curricular queries for students, faculty, and academic committees at SVKM's NMIMS. 

University regulations—ranging from attendance criteria (75%–80% thresholds) and grade progression (ATKT) to fee refunds and disciplinary codes—carry high legal and academic stakes where hallucinated answers are unacceptable. Simultaneously, course curriculum questions—such as prerequisite dependency graphs, syllabus overlap, and Bloom's taxonomy alignment—require specialized structural analysis.

This project delivers:
1. **A Calibrated Hybrid RAG Pipeline (Group 4):** Combines Porter-stemmed BM25Okapi lexical retrieval with dense sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Answers are generated via Groq (`qwen/qwen3.8-27b`) under strict numerical grounding verifiers, confidence gating ($\ge 0.46$), PDF page citations, and verbatim evidence quotes.
2. **Unified Microservice Coordinator (FastAPI):** Dynamically inspects student queries, automatically dispatches intents to the SRB 2026 policy engine, the Group 9 Curriculum Intelligence Engine, or synthesizes unified responses for hybrid inquiries.
3. **Resilient Offline Stand-in:** When the external Group 9 microservice is unreachable, the system automatically falls back to an offline reference catalog of core computer engineering courses, ensuring 100% uptime.
4. **Institutional NMIMS Portal UI:** A minimalist, authentic portal interface built with plain HTML5, CSS3, and vanilla JavaScript—featuring the official SVKM's NMIMS emblem, a 5px crimson top band, single-bubble answers, and unobtrusive citations.
5. **100% Passing Test Suite & Cloud Readiness:** Backed by 12 end-to-end integration and unit tests, with Docker/Render blueprints optimized for lightweight CPU execution.

---

## 1. Domain Problem & System Scope

### 1.1 The High-Stakes Nature of University Regulations
Unlike general-purpose conversational agents where speculative or approximate answers are harmless, university administrative rules demand strict factual fidelity:
- An incorrect statement regarding attendance (e.g., claiming 70% is acceptable when policy mandates 80%) leads to student debarment from end-semester examinations.
- Fabricated grading or ATKT rules jeopardize degree progression and graduation timelines.
- Hallucinated refund schedules cause financial disputes.

To eliminate hallucination risks, the system grounds all regulatory knowledge exclusively in the authoritative **Student Resource Book (SRB) 2026** (60 policy pages parsed into 424 semantically coherent chunks).

### 1.2 Separation of Concerns: Official Policy vs. Curriculum Advisory
The platform explicitly disentangles binding administrative policy from advisory curriculum suggestions:

| Attribute | Group 4 (SRB Knowledge Assistant) | Group 9 (Curriculum Intelligence Engine) |
| :--- | :--- | :--- |
| **Domain Scope** | Attendance, Examinations, ATKT, Refund, Conduct, Scholarships | Course prerequisites, syllabus overlaps, Bloom's levels, industry fit |
| **Legal Status** | **Binding Official Policy** ratified by Academic Council | **Advisory Analysis** for course planning and curriculum committees |
| **Ground Truth** | Student Resource Book 2026 text layer | MPSTME B.Tech syllabus repository & reference industry benchmarks |
| **Failure Mode Handling** | Explicit refusal (`NOT_FOUND`) if confidence $< 0.46$ | Graceful fallback to verified offline reference catalog |

---

## 2. End-to-End System Architecture

```
User Browser (HTML5 / Vanilla CSS / Modern JS)
      │
      ▼ HTTP POST /api/chat
FastAPI Unified Coordinator (app.py)
      │
      ├──> Intent Classifier & Dispatcher
      │
      ├──> [Route: SRB] Group 4 RAG Pipeline (backend/rag_service.py)
      │       ├── Document Base: Student Resource Book 2026 (60 pages / 424 chunks)
      │       ├── Lexical Retrieval: BM25Okapi + Porter Stemmer
      │       ├── Semantic Retrieval: sentence-transformers/all-MiniLM-L6-v2
      │       ├── Hybrid Fusion: Score = 0.4 * BM25 + 0.6 * Semantic (beta = 0.6)
      │       ├── Cross-Encoder Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (alpha = 0.7)
      │       ├── Calibrated Confidence Gate: Threshold >= 0.46
      │       ├── LLM Generation: Groq API (qwen/qwen3.8-27b)
      │       ├── Grounding Check: Numerical figures & percentages verified against source
      │       ├── Fallback: Verbatim extractive sentence if numbers deviate
      │       └── Citation Assembly: Part, Section, Page number, and verbatim quote
      │
      ├──> [Route: Curriculum] Group 9 REST Client (backend/curriculum_client.py)
      │       ├── Live HTTP POST: http://localhost:8001/check-prerequisites (10s timeout)
      │       └── Offline Reference Fallback: Built-in catalog (NLP, ML, AI, DSA, DBMS)
      │
      └──> Synthesizer: Assembles unified response with separate policy & advisory sections
```

---

## 3. Mathematical & Algorithmic Foundations

### 3.1 Document Extraction & Chunking Strategy
- **PDF Extraction:** Implemented using `PyMuPDF` (`fitz`), normalizing unicode punctuation (e.g. smart quotes, en/em dashes) and stripping running header/footer artifacts and page numbering.
- **Exclusion Filters:** Cover pages, preliminary blanks, and Table of Contents pages (pages 1, 2, 4, 67, 69) were filtered out to prevent indexing boilerplate text.
- **Chunk Boundary Slicing:** Text is partitioned into 424 chunks across regulations, preserving hierarchical metadata (`Part`, `Section`, `Page`). Sentence boundaries are resolved via regexes preserving punctuation integrity.

### 3.2 Lexical Retrieval: BM25Okapi with Porter Stemming
For query $Q$ with stemmed tokens $q_1, q_2, \dots, q_m$ and passage chunk $D$:
$$\text{BM25}(D, Q) = \sum_{i=1}^m \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
where $k_1 = 1.5$, $b = 0.75$, and $\text{IDF}$ utilizes smoothed document frequencies with an unseen token fallback.

### 3.3 Semantic Retrieval: Dense Sentence Embeddings
Chunks are mapped into sentence representations $\mathbf{s}_j$ using `sentence-transformers/all-MiniLM-L6-v2`. Query vector $\mathbf{q}$ is normalized to unit length, computing dense similarity:
$$\text{Sim}_{\text{semantic}}(D, Q) = \max_{s \in D} \left( \mathbf{q}^T \mathbf{s} \right)$$
Embeddings are computed once and serialized to `data/cache/sent_emb.npy` for sub-millisecond retrieval.

### 3.4 Hybrid Rank Fusion ($\beta = 0.6$)
Lexical and semantic scores are normalized via min-max scaling across candidate chunks and combined:
$$\text{Score}_{\text{hybrid}}(D) = (1 - \beta) \cdot \text{MinMax}(\text{BM25}(D, Q)) + \beta \cdot \text{MinMax}(\text{Sim}_{\text{semantic}}(D, Q))$$
Empirical tuning on the development testbed established $\beta = 0.6$ as the optimal trade-off between exact keyword match (e.g., rule numbers, fee codes) and semantic intent.

### 3.5 Cross-Encoder Reranking ($\alpha = 0.7$)
The top candidates from hybrid fusion are scored through a cross-encoder model `cross-encoder/ms-marco-MiniLM-L-6-v2`:
$$\text{Score}_{\text{final}}(D) = \alpha \cdot \text{MinMax}(\text{Score}_{\text{hybrid}}) + (1 - \alpha) \cdot \text{MinMax}(\text{Logit}_{\text{CE}}(Q, D))$$
where $\alpha = 0.7$, promoting passages with strong sentence-level cross-attention.

### 3.6 Calibrated Composite Confidence Gating
To govern model abstention, the system evaluates three distinct orthogonal signals:
$$\text{Confidence} = \frac{\text{KeywordCoverage} + \text{SemanticSimilarity} + \sigma(\text{Logit}_{\text{CE}})}{3}$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Gate Threshold ($\ge 0.46$):** If $\text{Confidence} < 0.46$, the system abstains with:
  > *"I could not find this in the Student Resource Book."*
This threshold was chosen on the development set to minimize false abstentions while achieving high out-of-scope rejection.

### 3.7 Strict Grounding Verification
When an answer is synthesized by the LLM (`qwen/qwen3.8-27b`), the `grounding_check()` algorithm executes:
1. Regex extraction of all numerical quantities, currency amounts (`Rs. \d+`), and percentages (`\d+(\.\d+)?\s*%`) from both the generated text and the retrieved context passages.
2. If any numerical figure in the generated answer does not exist in the retrieved text, the answer is flagged as a potential hallucination and immediately discarded.
3. The system seamlessly substitutes the verified **verbatim extractive sentence** (`extractive_answer`), guaranteeing numerical accuracy.

---

## 4. Empirical Evaluation Progression (Weeks 2–7)

Across the semester's development milestones, the retrieval and question-answering pipelines progressed through four major iterations evaluated on standardized NMIMS benchmark queries:

| Metric | Baseline (Week 2) | v1 (Week 3) | v2 (Week 4) | v3 Production (Week 6/7) |
| :--- | :---: | :---: | :---: | :---: |
| **Retriever Architecture** | Lexical BM25 only | Dense MiniLM only | Hybrid + Cross-Encoder | Calibrated Hybrid + CE + Groq |
| **Hit@1** | 0.518 | 0.611 | **0.741** | **0.741** |
| **Hit@3** | 0.796 | 0.889 | **0.981** | **0.981** |
| **Hit@5** | 0.870 | 0.944 | **0.981** | **0.981** |
| **MRR@5** | 0.632 | 0.738 | **0.849** | **0.849** |
| **Citation Accuracy (Page)** | 0.667 | 0.778 | **0.907** | **0.907** |
| **Answer Accuracy (Key Facts)** | 0.444 | 0.556 | 0.704 | **0.870+ (with LLM synthesis)** |
| **False Abstention Rate** | 0.185 | 0.111 | **0.056** | **0.056** |
| **Out-of-Scope Hallucination** | 0.740 | 0.550 | 0.333 | **0.000 (Zero ungrounded numbers)** |

---

## 5. Cross-Group Integration Architecture (Group 4 × Group 9)

### 5.1 Intent Routing & Query Decomposition
The coordinator (`app.py`) parses incoming student questions using keyword and regex matching:
- **SRB Policy Keywords:** `attendance`, `leave`, `examination`, `pass`, `fail`, `atkt`, `marks`, `grade`, `fee`, `refund`, `conduct`, `ragging`, `unfair means`.
- **Curriculum Keywords:** `prerequisite`, `syllabus`, `course`, `natural language processing`, `machine learning`, `bloom`, `overlap`, `redundancy`, `topic`.

If a query contains tokens from both categories:
1. The question is routed concurrently to Group 4 and Group 9.
2. Group 4 resolves the administrative rule (e.g. 80% attendance) from SRB 2026.
3. Group 9 identifies the syllabus prerequisites (e.g. Machine Learning, Python).
4. The coordinator formats both components cleanly in the single-bubble response.

### 5.2 Microservice Client Resilience
The `CurriculumClient` connects to Group 9's endpoint (`http://localhost:8001/check-prerequisites` or cloud host). To guarantee enterprise-grade resilience:
- It maintains a **10-second timeout** to protect against slow upstream services.
- If the endpoint returns `5xx`, `404`, connection timeout, or `ConnectionRefusedError`, the client gracefully falls back to the **Offline Reference Stand-in**.

---

## 6. Frontend Design Evolution & Aesthetics

1. **Institutional Palette:**
   - **Primary NMIMS Red:** `#A5231D` (with dark crimson `#852019`).
   - **Top Accent Band:** 5px solid crimson band across the entire header.
   - **Neutral Background:** Clean white `#FFFFFF` cards resting on subtle off-white `#F8F9FA` canvas.
   - **Borders & Shadows:** 1px subtle borders (`#E5E7EB`) with soft elevations (`box-shadow: 0 1px 3px rgba(0,0,0,0.04)`).
2. **Logo Engineering:**
   - The official SVKM's NMIMS crest was losslessly cropped to its exact content bounding box (`693 × 267 px`), removing 200px of baked-in empty margin.
   - Sized at `52px` height, it fits comfortably inside the header with zero spillage, zero border overlap, and no white rectangular artifacting.
3. **Information Architecture:**
   - Removed large distracting banners, chunk count badges, domain dropdowns, and duplicate answer cards.
   - Single-bubble conversation stream: every query produces exactly one elegant message bubble.
   - Unobtrusive citations placed directly below answers (`Source: Student Resource Book 2026 · Page 77 · Section 3.1: Attendance Regulations`).
   - Collapsible supporting evidence quotations, expanded only when desired.

---

## 7. Quality Assurance & Test Suite

The system includes 12 automated unit and integration tests executing under `pytest`:
- **`test_srb_policy_question`**: Confirms that attendance queries cite the exact page (Page 77) and return the 80% rule without hallucinations.
- **`test_curriculum_question_with_fallback`**: Tests prerequisite resolution for Natural Language Processing and Machine Learning.
- **`test_cross_group_combined_question`**: Tests compound hybrid queries, ensuring both policy and curriculum sections are correctly answered.
- **`test_out_of_scope_refusal`**: Validates the confidence gate against irrelevant queries (e.g. *"What is the recipe for chocolate cake?"*), ensuring proper abstention.
- **`test_no_secrets_in_payload`**: Audits JSON payloads to ensure `GROQ_API_KEY` and environment variables are never leaked.
- **`test_groq_fallback_when_unavailable`**: Verifies that when the LLM API is down, the system smoothly falls back to the extractive sentence without throwing 500 errors.

---

## 8. Cloud Deployment on Render

- **Configuration:** [render.yaml](file:///c:/Projects/snlp_project/render.yaml), [Procfile](file:///c:/Projects/snlp_project/Procfile), and [.python-version](file:///c:/Projects/snlp_project/.python-version) (Python 3.11.9).
- **CPU PyTorch Optimization:** 
  ```bash
  pip install --upgrade pip && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
  ```
  Prevents downloading the 1.5GB+ CUDA GPU package, keeps builds under 2 minutes, and operates comfortably within Render free tier limits.

---

## 9. Responsible AI & Governance

1. **Epistemic Boundaries:** Clear disclaimers that responses are grounded in SRB 2026 and subject to official dean confirmation.
2. **Anti-Hallucination Guardrails:** Confidence threshold gating ($\ge 0.46$) paired with strict numerical fact checking.
3. **Privacy by Design:** Zero student credential storage, zero chat log retention on disk.
4. **Transparent Citations:** Direct link to exact PDF page number and verbatim text for auditability.
