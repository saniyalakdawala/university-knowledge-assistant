# AI University Assistant: Group 4 × Group 9 Integration

**Course:** Natural Language Processing (SNLP)  
**Collaborating Groups:**  
- **Group 4:** University Knowledge Assistant (RAG) over the Student Resource Book (SRB) 2026  
- **Group 9:** Curriculum Intelligence Engine  
**Release:** Week 7 Integration Demo & Deployment (October 2026)

---

## 1. Project Overview & Responsibilities

The **AI University Assistant** is a unified full-stack web application designed for students and academic committees of SVKM's NMIMS (MPSTME). It brings together two specialized intelligent systems:

- **Group 4 (University Knowledge Assistant):**
  - Answers questions regarding official university policies, attendance rules, examination regulations, passing standards (ATKT), fee refunds, scholarships, and codes of conduct.
  - Queries the **Student Resource Book (SRB) 2026** using a hybrid retrieval-augmented generation (RAG) pipeline with strict grounding checks, refusal gating, PDF page citations, and verbatim evidence quotes.
- **Group 9 (Curriculum Intelligence Engine):**
  - Analyzes engineering course syllabi, flags missing prerequisites, detects curriculum redundancies and overlaps, maps learning outcomes to Bloom's taxonomy, and modernizes curricula against industry job descriptions.
  - Integrates through its documented REST API (`curriculum_engine.api`) with an intelligent offline reference fallback.

---

## 2. System Architecture

```
User (Browser)
      │
      ▼
FastAPI Unified Coordinator (app.py) ── [Serves Plain HTML/CSS/JS Frontend]
      │
      ├──> Intent Analysis & Route Dispatch (/api/chat)
      │
      ├──> [Route: SRB] Group 4 RAG Service (backend/rag_service.py)
      │       ├── Lexical Retrieval: BM25Okapi (Porter Stemmed)
      │       ├── Semantic Retrieval: sentence-transformers/all-MiniLM-L6-v2
      │       ├── Cross-Encoder Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
      │       ├── Refusal Gate: Confidence threshold >= 0.46
      │       └── Grounding Verifier & Extractive Evidence Quoting
      │
      ├──> [Route: Curriculum] Group 9 API Client (backend/curriculum_client.py)
      │       ├── Live HTTP REST: http://localhost:8000/check-prerequisites
      │       └── Offline Reference Stand-in Fallback
      │
      └──> Synthesized Response Assembly (Strict Policy vs. Curriculum Advisory)
```

For full architecture details and Mermaid diagrams, see [`docs/ARCHITECTURE.md`](file:///c:/Projects/snlp_project/docs/ARCHITECTURE.md).

---

## 3. Project Directory Structure

```
snlp_project/
├── app.py                      # FastAPI application & coordinator
├── requirements.txt            # Pinned dependencies
├── .env.example                # Configuration template
├── .gitignore                  # Git exclusion rules
├── README.md                   # Complete documentation & runbook
├── data/
│   ├── SRB_2026.pdf            # Authoritative Student Resource Book 2026
│   └── cache/                  # Serialized chunks and sentence embeddings
├── backend/
│   ├── __init__.py
│   ├── config.py               # Path and model configuration
│   ├── schemas.py              # Pydantic v2 data models
│   ├── rag_service.py          # Group 4 RAG pipeline
│   └── curriculum_client.py    # Group 9 REST client with offline fallback
├── frontend/
│   ├── index.html              # Plain HTML5 user interface
│   ├── style.css               # Modern vanilla CSS design system
│   └── script.js               # Responsive vanilla JavaScript client
├── tests/
│   ├── test_api.py             # Endpoint and health-check tests
│   └── test_integration.py     # Cross-group routing and regression tests
├── notebooks/
│   └── Group4_Week6.ipynb      # Preserved Week 6 RAG reference notebook
└── docs/
    ├── ARCHITECTURE.md         # System architecture and data flow
    ├── RESPONSIBLE_AI.md       # Responsible AI, safety, and ethics analysis
    └── WEEK7_DEMO.md           # Step-by-step classroom demonstration sequence
```

---

## 4. Local Setup & Execution Guide

### Prerequisites
- Python 3.11, 3.12, or 3.13 on Windows, macOS, or Linux
- PowerShell (on Windows) or Terminal (on Linux/macOS)

### Windows PowerShell Quickstart
Open PowerShell in the project directory and execute:

```powershell
# 1. Clone/navigate to project
cd c:\Projects\snlp_project

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (copy template)
Copy-Item .env.example .env

# Set your Groq API Key and Model in .env:
# GROQ_API_KEY=gsk_your_actual_key_here
# GROQ_MODEL=qwen/qwen3.8-27b
# GROUP9_API_URL=http://localhost:8001

# 5. Start the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The application will start immediately at:
👉 **`http://localhost:8000`**

Open your browser to interact with the web interface. Both the API documentation (`/docs`) and the web client (`/`) are hosted from this single endpoint.

---

## 5. Group 9 Integration Setup

To run a **live cross-group integration**:

1. Clone and install Group 9's repository in a separate directory:
   ```bash
   git clone https://github.com/jainamsethia/Curriculum-Intelligence-Engine
   cd Curriculum-Intelligence-Engine
   pip install -r requirements.txt
   uvicorn curriculum_engine.api:app --port 8001
   ```
2. Configure Group 4 to target Group 9's port in `.env`:
   ```dotenv
   GROUP9_API_URL=http://localhost:8001
   ```
3. **If Group 9 is offline:**
   The backend automatically activates the **Documented Reference Stand-in Cache**. The client continues to answer curriculum questions with a transparent `Reference Stand-in` label without crashing.

---

## 6. Running Tests

Run the comprehensive pytest suite covering health checks, citations, routing, fallbacks, regression questions, and security:

```powershell
python -m pytest tests/ -v
```

Expected output:
```
tests/test_api.py::test_health_endpoint PASSED                           [  8%]
tests/test_api.py::test_static_frontend_routes PASSED                    [ 16%]
tests/test_api.py::test_invalid_chat_request PASSED                      [ 25%]
tests/test_integration.py::test_srb_policy_question PASSED               [ 33%]
tests/test_integration.py::test_curriculum_question_with_fallback PASSED [ 41%]
tests/test_integration.py::test_cross_group_combined_question PASSED     [ 50%]
tests/test_integration.py::test_out_of_scope_refusal PASSED              [ 58%]
tests/test_integration.py::test_group9_unreachable_fallback PASSED       [ 66%]
tests/test_integration.py::test_group9_invalid_response_handling PASSED  [ 75%]
tests/test_integration.py::test_week6_regression_cases PASSED            [ 83%]
tests/test_integration.py::test_no_secrets_in_payload PASSED             [ 91%]
tests/test_integration.py::test_groq_fallback_when_unavailable PASSED    [100%]
============================== 12 passed in 15.2s ===============================
```

---

## 7. Cloud Deployment (Render / Railway / Hugging Face Spaces)

This application is packaged as a standard self-contained Python web service:

### Render Deployment Instructions
1. Connect your repository to **Render** as a **Web Service**.
2. **Environment:** Python
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables:**
   - `PORT`: Set automatically by Render
   - `SRB_PDF_PATH`: `data/SRB_2026.pdf`
   - `GROUP9_API_URL`: URL of Group 9 service (default: `http://localhost:8001`)
   - `GROQ_API_KEY`: Your Groq API key (compulsory for natural-language generation)
   - `GROQ_MODEL`: `qwen/qwen3.8-27b`
6. **Health Check Path:** `/api/health`

*Note on Memory Constraints:* The application uses `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` with cached NumPy embeddings, requiring only ~450 MB of RAM. It fits comfortably within Render's free tier (512 MB).

---

## 8. Responsible AI & Historical Benchmarks

- **Answer Accuracy (Week 6 Benchmark):** 89.5%
- **Citation Accuracy (Week 6 Benchmark):** 94.4%
- **Out-of-Scope Questions Answered:** 0 out of 20 tested (100% rejection rate)
- For in-depth safety, grounding checks, and ethical analysis, refer to [`docs/RESPONSIBLE_AI.md`](file:///c:/Projects/snlp_project/docs/RESPONSIBLE_AI.md).
