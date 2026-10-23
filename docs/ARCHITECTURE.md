# Group 4 × Group 9 Architecture & Integration Specification

**Component:** Group 4 (University Knowledge Assistant) × Group 9 (Curriculum Intelligence Engine)  
**System:** AI University Application  
**Target Release:** Week 7 Integration Demo (October 2026)

---

## 1. High-Level Architecture Overview

The system bridges the **Student Resource Book (SRB) 2026** knowledge base (Group 4) with the **Curriculum Intelligence Engine** (Group 9), offering a unified entry point for university students and curriculum committee members. 

Students can ask:
1. **University Policy & General Guidelines:** Rules, attendance requirements, examination policies, passing standards (ATKT), fee refunds, code of conduct, and disciplinary policies directly verified from SRB 2026.
2. **Curriculum Intelligence:** Course prerequisite trees, syllabus overlaps, Bloom's taxonomy mapping, and emerging topic coverage.
3. **Cross-Group Inquiries:** Hybrid questions that combine curriculum selection with administrative/university rules (e.g., taking an advanced course like Machine Learning and understanding the prerequisite requirements alongside the mandatory attendance and grading regulations).

---

## 2. Mermaid System Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientLayer["Frontend Layer (Client Browser)"]
        UI["Plain HTML5 / Vanilla CSS / Modern JS UI"]
        ChatStream["Chat Stream & Dual-Card Result View"]
        Pills["System Status & Health Indicators"]
        UI --> ChatStream
        UI --> Pills
    end

    subgraph CoordinatorLayer["Unified Backend Coordinator (FastAPI)"]
        Router["Smart Query Router\n(/api/chat)"]
        HealthEndpoint["Health Check Service\n(/api/health)"]
        StaticServer["Static Asset Server\n(/, /style.css, /script.js)"]
        Synth["Response Synthesizer &\nSecurity Filter"]
    end

    subgraph Group4Service["Group 4: University Knowledge Assistant (RAG)"]
        SRB["Student Resource Book (SRB) 2026\n(60 kept pages / 424 chunks)"]
        BM25["Lexical Retriever\n(BM25Okapi + Porter Stemmer)"]
        MiniLM["Semantic Retriever\n(all-MiniLM-L6-v2 Embeddings)"]
        Hybrid["Hybrid Rank Fusion\n(beta = 0.6)"]
        Reranker["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2, alpha = 0.7)"]
        Gate["Calibrated Gate Filter\n(Confidence >= 0.46)"]
        Verifier["Grounding Verifier &\nExtractive Fallback"]
        Citations["SRB Citation Generator\n(Part, Section, Page, Verbatim Evidence)"]

        SRB --> BM25
        SRB --> MiniLM
        BM25 --> Hybrid
        MiniLM --> Hybrid
        Hybrid --> Reranker
        Reranker --> Gate
        Gate --> Verifier
        Verifier --> Citations
    end

    subgraph Group9Service["Group 9: Curriculum Intelligence Engine"]
        G9Client["Curriculum HTTP Client\n(curriculum_client.py)"]
        G9API["Group 9 FastAPI Service\n(http://localhost:8000)"]
        G9Fallback["Curriculum Reference Stand-in\n(Documented Offline Cache)"]
        
        G9Client -->|HTTP POST| G9API
        G9Client -.->|On Timeout / Offline| G9Fallback
    end

    %% Connections
    UI -->|HTTP POST /api/chat| Router
    UI -->|HTTP GET /api/health| HealthEndpoint
    UI -->|HTTP GET Static Assets| StaticServer

    Router -->|SRB / Policy Intent| Group4Service
    Router -->|Curriculum Intent| G9Client

    Citations --> Synth
    G9Client --> Synth
    Synth -->|JSON ChatResponse| UI
```

---

## 3. Data Flow and Component Interactions

### 3.1 Step-by-Step Request Lifecycle
1. **Student Query Entry:**
   The student enters a question via the browser (e.g., *"What are the prerequisites for Machine Learning, and what is the minimum attendance required?"*).
2. **Intent Analysis & Route Dispatch:**
   `determine_route()` in `app.py` examines the query tokens:
   - Contains curriculum tokens (*"machine learning"*, *"prerequisites"*) -> Flag `curriculum`.
   - Contains policy tokens (*"attendance"*, *"minimum"*) -> Flag `srb`.
   - Both flagged -> Route set to `both`.
3. **Group 4 RAG Execution:**
   - **Lexical Search:** BM25 retrieves top matching passages using Porter-stemmed tokens and stop-word filtering.
   - **Semantic Search:** `sentence-transformers/all-MiniLM-L6-v2` queries the cached sentence-level embeddings.
   - **Hybrid Fusion:** Combines scores using $\beta = 0.6$.
   - **Cross-Encoder Reranking:** Top passages are reranked via `cross-encoder/ms-marco-MiniLM-L-6-v2` ($\alpha = 0.7$).
   - **Confidence Gating:** If retrieval confidence $< 0.46$, the system abstains with `NOT_FOUND` ("I could not find this in the Student Resource Book.").
   - **Grounding Verification:** Numbers and percentages in any generated text must strictly match the passage; otherwise, it falls back immediately to the verified verbatim sentence.
   - **Citation Assembly:** Attaches part, section name, PDF page number, and verbatim evidence quote.
4. **Group 9 Curriculum Client Execution:**
   - Client sends structured JSON payload to Group 9's documented endpoint (`/check-prerequisites`, `/find-overlaps`, or `/compare-curricula`).
   - If Group 9's local server is live, the live engine output is returned with `connected=True`.
   - If the endpoint is unreachable or times out, the client falls back to the documented reference stand-in cache, returning `is_fallback=True` with a transparent advisory message.
5. **Synthesis & Response Delivery:**
   `app.py` synthesizes the response, strictly isolating verified official university policy from curriculum advisory intelligence. The client UI renders both cards with clear badges and evidence callouts.

---

## 4. Cold-Start and Resource Optimization

1. **Precomputed Chunk and Embedding Cache:**
   On initial boot, the application parses `SRB_2026.pdf` (424 chunks, 1,514 sentences), computes sentence embeddings with `all-MiniLM-L6-v2`, and serializes the resulting structures to:
   - `data/cache/chunks.pkl` (276 KB)
   - `data/cache/sent_emb.npy` (2.3 MB)
   On subsequent reloads, these structures load from disk in **under 0.3 seconds**, preventing redundant PyMuPDF parsing and embedding computation.
2. **Single-Process Lightweight Server:**
   The entire application runs as a single FastAPI process using Uvicorn on a unified origin. No frontend build tools, Node runtime, or complex container meshes are required.
