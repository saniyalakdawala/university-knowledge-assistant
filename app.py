import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import BASE_DIR, PORT, HOST, GROQ_API_KEY, GROQ_MODEL, GROUP9_API_URL
from backend.config import LOW_MEMORY
from backend.schemas import ChatRequest, ChatResponse, HealthStatus, SRBResult, CurriculumResult, CourseIngestRequest
from backend.rag_service import get_rag_service, UniversityRAGService
from backend.curriculum_client import CurriculumClient
from backend.course_kb import get_course_kb

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
        get_course_kb().load_default_if_present()
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
# Curriculum keywords -> Group 9 live API (named existing courses: prerequisites, overlaps)
CURR_KEYWORDS = [
    "prerequisite", "prereq", "syllabus", "overlap", "redundancy", "course content",
    "units", "bloom", "curriculum", "machine learning", "deep learning", "nlp",
    "natural language processing", "operating systems", "modernize"
]
# SRB policy keywords -> Group 4 RAG over the Student Resource Book
SRB_KEYWORDS = [
    "attendance", "grade", "grading", "cgpa", "exam", "re-exam", "re-examination",
    "fee", "fees", "refund", "scholarship", "library", "dean", "council", "atkt",
    "passing standard", "rules", "conduct", "ragging", "disciplinary", "leave",
    "deferment", "break", "convocation", "portal", "credit", "minimum"
]
# University-wide policy words that still need the SRB when the question is about the new course
POLICY_KEYWORDS = [
    "graduat", "degree", "attendance", "atkt", "passing", "cgpa", "fee", "refund", "re-exam", "scholarship",
    "convocation", "promotion", "detention", "ragging", "conduct", "library"
]
# "new/this/proposed/draft course" = the course Group 9 just designed
NEW_COURSE_RX = re.compile(r"\b(?:new|newly offered|this|proposed|draft|modernized|modernised|unified)\b(?:\s+[\w-]+){0,4}?\s+(?:course|subject|syllabus|elective|paper)\b", re.I)
GENERIC_COURSE_RX = re.compile(r"\b(the|that)\s+(course|subject|syllabus)\b", re.I)


def split_clauses(question: str) -> List[str]:
    return [p.strip() for p in re.split(r"[?;]|,\s*(?:and\s+)?|\s+and\s+(?=(?:what|how|which|when|is|are|do|does|can|will)\b)", question, flags=re.I) if p and p.strip()]


def determine_route(question: str, requested_route: str = "auto") -> str:
    """Route to 'srb', 'curriculum', 'both', 'course' (Group 9 new-course passages) or 'course+srb'."""
    if requested_route in ("srb", "curriculum", "both", "course", "course+srb"):
        return requested_route

    q = question.lower()
    kb = get_course_kb()
    if kb.loaded:
        strong = bool(NEW_COURSE_RX.search(q)) or kb.mentions_title(question)
        generic = bool(GENERIC_COURSE_RX.search(q))
        policy = any(k in q for k in POLICY_KEYWORDS)
        if strong or (generic and not any(k in q for k in SRB_KEYWORDS if k not in ("credit", "exam"))):
            return "course+srb" if policy else "course"

    has_curr = any(kw in q for kw in CURR_KEYWORDS)
    has_srb = any(kw in q for kw in SRB_KEYWORDS)
    if has_curr and has_srb:
        return "both"
    elif has_curr:
        return "curriculum"
    return "srb"  # default: Group 4 knowledge base


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
            "retrieval_mode": "lite: BM25 only (LOW_MEMORY)" if LOW_MEMORY else "full: BM25 + MiniLM + cross-encoder (evaluated)",
            "new_course": get_course_kb().status(),
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


@app.get("/api/knowledge/course")
def course_status():
    """Is a Group 9 new-course syllabus loaded? (title, number of passages, where it came from)"""
    return get_course_kb().status()


@app.post("/api/knowledge/course")
def load_course(req: CourseIngestRequest):
    """Load the new course's syllabus passages handed over by Group 9.

    Body: {"rows": [{chunk_id, page, section, part, text}, ...]} or Group 9's whole POST /modernize-curriculum response
    (we read handoffs.group4_chunks). Replaces any previously loaded course."""
    rows = [r.model_dump() for r in req.rows] if req.rows else (req.handoffs or {}).get("group4_chunks")
    if not rows:
        raise HTTPException(status_code=422, detail="Send rows=[...] or a Group 9 response with handoffs.group4_chunks.")
    try:
        return get_course_kb().load_rows(rows, source="POST /api/knowledge/course")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _ask(service: UniversityRAGService, question: str, clauses: Optional[List[str]] = None) -> dict:
    """Ask the full question; if the index abstains, retry with only the clauses meant for this index."""
    res = service.ask(question)
    if res.get("abstained") and clauses and " ".join(clauses) != question:
        sub = service.ask(" ".join(clauses))
        if not sub.get("abstained"):
            res = sub
    return res


def _error_result(msg: str, doc: str) -> SRBResult:
    return SRBResult(answer=msg, source="System Error", document=doc, confidence=0.0, abstained=True, refusal_reason=msg, mode="error")


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Chat endpoint: SRB policy questions, the new course from Group 9, curriculum intelligence, and combinations."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    route = determine_route(question, req.route or "auto")
    if route in ("course", "course+srb") and not get_course_kb().loaded:
        route = "srb" if route == "course+srb" else "curriculum"
    logger.info(f"Received query ({len(question)} chars) | Route: {route}")  # question text is not logged (privacy)

    srb_res: Optional[SRBResult] = None
    course_res: Optional[SRBResult] = None
    curriculum_res: Optional[CurriculumResult] = None
    synth_parts = []
    clauses = split_clauses(question)

    # 1. New course (Group 9 draft syllabus passages)
    if route in ("course", "course+srb"):
        try:
            course_clauses = [c for c in clauses if not any(k in c.lower() for k in POLICY_KEYWORDS)]
            course_res = SRBResult(**_ask(get_course_kb().service, question, course_clauses))
            synth_parts.append(f"New course ({get_course_kb().title}, Group 9 draft syllabus): {course_res.answer}")
        except Exception as e:
            logger.error(f"Error querying new-course passages: {e}")
            course_res = _error_result("Error querying the new course syllabus.", "Group 9 draft syllabus")
        if route == "course" and course_res.abstained:
            route = "course+srb"  # not in the syllabus: also try the SRB before giving up

    # 2. Group 4 RAG over the SRB
    if route in ("srb", "both", "course+srb"):
        try:
            policy_kw = POLICY_KEYWORDS if route == "course+srb" else SRB_KEYWORDS
            srb_clauses = [c for c in clauses if any(k in c.lower() for k in policy_kw)]
            srb_res = SRBResult(**_ask(get_rag_service(), question, srb_clauses if route != "srb" else None))
            synth_parts.append(f"Official University Policy (SRB 2026): {srb_res.answer}")
        except Exception as e:
            logger.error(f"Error during Group 4 RAG retrieval: {e}")
            srb_res = _error_result("Error querying Student Resource Book.", "Student Resource Book (SRB) 2026")

    # 3. Group 9 live curriculum API
    if route in ("curriculum", "both"):
        try:
            curriculum_res = curriculum_client.query(question, req.course_context)
            if curriculum_res.summary:
                synth_parts.append(f"Curriculum Intelligence (Group 9): {curriculum_res.summary}")
        except Exception as e:
            logger.error(f"Error querying Group 9 Curriculum Engine: {e}")
            curriculum_res = CurriculumResult(connected=False, is_fallback=True, status_note=f"Error contacting Group 9: {e}",
                                              summary="Curriculum details could not be processed at this time.")

    if route == "curriculum" and curriculum_res:
        synthesized_text = curriculum_res.summary
    elif route == "srb" and srb_res:
        synthesized_text = srb_res.answer
    elif route == "course" and course_res:
        synthesized_text = course_res.answer
    elif synth_parts:
        synthesized_text = " \n\n".join(synth_parts)
    else:
        synthesized_text = "No answer could be generated for this request."

    return ChatResponse(
        question=question,
        route_taken=route,  # type: ignore
        synthesized_answer=synthesized_text,
        srb=srb_res,
        course=course_res,
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
