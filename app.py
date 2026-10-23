import os
import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import BASE_DIR, PORT, HOST, GROQ_API_KEY, GROQ_MODEL, GROUP9_API_URL
from backend.schemas import ChatRequest, ChatResponse, HealthStatus, SRBResult, CurriculumResult
from backend.rag_service import get_rag_service, UniversityRAGService
from backend.curriculum_client import CurriculumClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("university_assistant")

curriculum_client = CurriculumClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up University AI Assistant server...")
    if not GROQ_API_KEY:
        logger.warning(
            "STARTUP CONFIGURATION WARNING: GROQ_API_KEY is not set in environment or .env. "
            "Groq API is mandatory for natural language answers. Extractive fallback will be used."
        )
    else:
        logger.info(f"Groq API configured with model: {GROQ_MODEL}")
    logger.info(f"Group 9 Curriculum Intelligence Engine configured endpoint: {GROUP9_API_URL}")

    # Preload RAG service in memory
    try:
        rag = get_rag_service()
        logger.info(f"Group 4 RAG Service loaded with {len(rag.chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error initializing RAG Service: {e}")
    yield
    logger.info("Shutting down University AI Assistant server...")

app = FastAPI(
    title="AI University Assistant API",
    description="Group 4 (University Knowledge Assistant) × Group 9 (Curriculum Intelligence Engine) Integration API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for open accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"

# ----------------- Router Logic -----------------
def determine_route(question: str, requested_route: str = "auto") -> str:
    """Intelligently route the question to 'srb', 'curriculum', or 'both'."""
    if requested_route in ("srb", "curriculum", "both"):
        return requested_route

    q = question.lower()
    
    # Curriculum keywords
    curr_keywords = [
        "prerequisite", "prereq", "syllabus", "overlap", "redundancy", "course content",
        "units", "bloom", "curriculum", "machine learning", "deep learning", "nlp",
        "natural language processing", "operating systems", "modernize"
    ]
    # SRB policy keywords
    srb_keywords = [
        "attendance", "grade", "grading", "cgpa", "exam", "re-exam", "re-examination",
        "fee", "fees", "refund", "scholarship", "library", "dean", "council", "atkt",
        "passing standard", "rules", "conduct", "ragging", "disciplinary", "leave",
        "deferment", "break", "convocation", "portal", "credit", "minimum"
    ]

    has_curr = any(kw in q for kw in curr_keywords)
    has_srb = any(kw in q for kw in srb_keywords)

    if has_curr and has_srb:
        return "both"
    elif has_curr:
        return "curriculum"
    else:
        # Default to SRB (Group 4 Knowledge base)
        return "srb"


# ----------------- API Endpoints -----------------
@app.get("/api/health", response_model=HealthStatus)
def health_check():
    """Health check endpoint providing status of both Group 4 and Group 9 components."""
    # Check Group 4
    try:
        rag = get_rag_service()
        group4_info = {
            "status": "ready",
            "pdf_path": rag.pdf_path,
            "chunks_count": len(rag.chunks),
            "gate": rag.gate,
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "groq_configured": bool(rag.groq_api_key),
            "groq_model": rag.llm_model,
        }
    except Exception as e:
        group4_info = {"status": "error", "error": str(e)}

    # Check Group 9
    group9_health = curriculum_client.check_health()
    group9_info = {
        "status": "online" if group9_health["connected"] else "stand-in (offline fallback)",
        "connected": group9_health["connected"],
        "endpoint": curriculum_client.base_url,
        "details": group9_health.get("details"),
        "error": group9_health.get("error")
    }

    return {
        "status": "ok",
        "service": "AI University Assistant (Group 4 × Group 9)",
        "group4": group4_info,
        "group9": group9_info
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Chat endpoint supporting SRB policy questions, curriculum intelligence, and cross-group queries."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    route = determine_route(question, req.route or "auto")
    logger.info(f"Received query: '{question}' | Route: {route}")

    srb_res: Optional[SRBResult] = None
    curriculum_res: Optional[CurriculumResult] = None
    synth_parts = []

    # 1. Execute Group 4 RAG query if route is 'srb' or 'both'
    if route in ("srb", "both"):
        try:
            rag = get_rag_service()
            raw_srb = rag.ask(question)
            # If composite query in both mode abstains, try targeting the policy-specific clause
            if raw_srb.get("abstained") and route == "both":
                parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", question, flags=re.I)
                srb_kws = ["attendance", "grade", "exam", "passing", "rule", "fee", "cgpa", "atkt", "minimum", "credit"]
                srb_parts = [p for p in parts if any(k in p.lower() for k in srb_kws)]
                if srb_parts:
                    sub_q = " ".join(srb_parts)
                    sub_res = rag.ask(sub_q)
                    if not sub_res.get("abstained"):
                        raw_srb = sub_res

            srb_res = SRBResult(**raw_srb)
            if not srb_res.abstained:
                synth_parts.append(f"Official University Policy (SRB 2026): {srb_res.answer}")
            else:
                synth_parts.append(f"Official University Policy: {srb_res.answer}")
        except Exception as e:
            logger.error(f"Error during Group 4 RAG retrieval: {e}")
            srb_res = SRBResult(
                answer="Error querying Student Resource Book.",
                source="System Error",
                confidence=0.0,
                abstained=True,
                refusal_reason=str(e),
                mode="error"
            )

    # 2. Execute Group 9 Curriculum query if route is 'curriculum' or 'both'
    if route in ("curriculum", "both"):
        try:
            curriculum_res = curriculum_client.query(question, req.course_context)
            if curriculum_res.summary:
                synth_parts.append(f"Curriculum Intelligence (Group 9): {curriculum_res.summary}")
        except Exception as e:
            logger.error(f"Error querying Group 9 Curriculum Engine: {e}")
            curriculum_res = CurriculumResult(
                connected=False,
                is_fallback=True,
                status_note=f"Error contacting Group 9: {e}",
                summary="Curriculum details could not be processed at this time."
            )

    # 3. Formulate synthesized answer
    if route == "both":
        synthesized_text = " \n\n".join(synth_parts)
    elif route == "curriculum" and curriculum_res:
        synthesized_text = curriculum_res.summary
    elif srb_res:
        synthesized_text = srb_res.answer
    else:
        synthesized_text = "No answer could be generated for this request."

    return ChatResponse(
        question=question,
        route_taken=route,  # type: ignore
        synthesized_answer=synthesized_text,
        srb=srb_res,
        curriculum=curriculum_res,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ----------------- Static Frontend Hosting -----------------
# Serve frontend assets from the static folder
if FRONTEND_DIR.exists():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/style.css")
    def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @app.get("/script.js")
    def serve_js():
        return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
