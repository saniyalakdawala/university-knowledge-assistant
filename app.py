import os
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("TORCH_THREADS", "4"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("TORCH_THREADS", "4"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
except ImportError:
    pass

import re
import time
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import BASE_DIR, PORT, HOST, GROQ_API_KEY, GROQ_MODEL, GROUP9_API_URL, CURRICULUM_BACKEND, EMB_MODEL_NAME, CE_MODEL_NAME
from backend.schemas import (ChatRequest, ChatResponse, HealthStatus, SRBResult, CurriculumResult, CourseCompareRequest, ProgrammeCompareRequest,
                             OverlapRequest, ProgrammeRequest, PrerequisiteRequest, BloomRequest, ModernizeRequest)
from backend.rag_service import get_rag_service, UniversityRAGService
from backend.curriculum_client import CurriculumClient
from backend.curriculum.service import get_curriculum_service
from backend.router import route_question

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
    try:
        get_curriculum_service().warm()
        logger.info("Curriculum Intelligence service ready.")
    except Exception as e:
        logger.error(f"Error initializing Curriculum service: {e}")
    yield
    logger.info("Shutting down University AI Assistant server...")

app = FastAPI(
    title="AI University Assistant API",
    description="University Knowledge RAG and Curriculum Intelligence behind one chat API",
    version="2.0.0",
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

# ----------------- Conversation memory -----------------
_SESSIONS: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
MAX_SESSIONS, MAX_TURNS = 500, 6


def session_history(session_id: Optional[str]) -> List[Dict[str, str]]:
    return list(_SESSIONS.get(session_id, [])) if session_id else []


def remember(session_id: Optional[str], question: str, answer: str) -> None:
    if not session_id:
        return
    turns = _SESSIONS.setdefault(session_id, [])
    turns.append({"question": question, "answer": answer})
    del turns[:-MAX_TURNS]
    _SESSIONS.move_to_end(session_id)
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)


def groq_for_routing():
    """The RAG service's Groq client, when a key is configured."""
    try:
        rag = get_rag_service()
    except Exception:
        return None
    return (lambda messages, max_tokens: rag._chat(messages, max_tokens, json_mode=True)) if rag.groq_api_key else None


def determine_route(question: str, requested_route: str = "auto") -> Dict:
    """Decide which service answers: rules for clear questions, Groq for ambiguous ones."""
    try:
        detect = get_curriculum_service().detect_courses
    except Exception as e:
        logger.warning(f"Curriculum service unavailable for routing: {e}")
        detect = None
    return route_question(question, requested_route, detect, groq_for_routing())


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
            "embedder": EMB_MODEL_NAME,
            "reranker": CE_MODEL_NAME,
            "groq_configured": bool(rag.groq_api_key),
            "groq_model": rag.llm_model,
        }
    except Exception as e:
        group4_info = {"status": "error", "error": str(e)}

    # Curriculum Intelligence: built in, with an optional remote engine
    try:
        group9_info = {"status": "ready", "backend": CURRICULUM_BACKEND, "connected": True, **get_curriculum_service().stats()}
    except Exception as e:
        group9_info = {"status": "error", "backend": CURRICULUM_BACKEND, "connected": False, "error": str(e)}
    if CURRICULUM_BACKEND == "remote":
        remote = curriculum_client.check_health()
        group9_info.update(remote_connected=remote["connected"], endpoint=curriculum_client.base_url, error=remote.get("error"))

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

    routing = determine_route(question, req.route or "auto")
    route = routing["route"]
    history = session_history(req.session_id)
    logger.info(f"Received query | route: {route} ({routing['decided_by']})")

    srb_res: Optional[SRBResult] = None
    curriculum_res: Optional[CurriculumResult] = None
    synth_parts = []

    # 1. University Knowledge RAG
    if route in ("srb", "both"):
        try:
            raw_srb = get_rag_service().ask(routing["srb_question"] if route == "both" else question, history)
            srb_res = SRBResult(**{k: v for k, v in raw_srb.items() if k in SRBResult.model_fields})
            prefix = "Official University Policy (SRB 2026)" if not srb_res.abstained else "Official University Policy"
            synth_parts.append(f"{prefix}: {srb_res.answer}")
        except Exception as e:
            logger.error(f"Error during RAG retrieval: {e}")
            srb_res = SRBResult(
                answer="Error querying Student Resource Book.",
                source="System Error",
                confidence=0.0,
                abstained=True,
                refusal_reason="The knowledge base could not be queried.",
                mode="error"
            )

    # 2. Curriculum Intelligence
    if route in ("curriculum", "both"):
        try:
            asked = routing["curriculum_question"] if route == "both" else question
            curriculum_res = curriculum_client.query(asked, req.course_context, routing["curriculum_plan"])
            if curriculum_res.summary:
                synth_parts.append(f"Curriculum Intelligence: {curriculum_res.summary}")
        except Exception as e:
            logger.error(f"Error querying Curriculum Intelligence: {e}")
            curriculum_res = CurriculumResult(
                connected=False,
                is_fallback=True,
                status_note="Curriculum Intelligence could not process this request.",
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

    remember(req.session_id, question, synthesized_text)
    return ChatResponse(
        question=question,
        route_taken=route,  # type: ignore
        synthesized_answer=synthesized_text,
        srb=srb_res,
        curriculum=curriculum_res,
        timestamp=datetime.now(timezone.utc).isoformat(),
        routed_by=routing["decided_by"],
        session_id=req.session_id,
    )


# ----------------- Curriculum Intelligence API -----------------
def curriculum_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


def known_course(name: str) -> str:
    resolved = get_curriculum_service().resolve_name(name)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"no course matching {name!r}")
    return resolved


def known_programme(code: str) -> str:
    svc = get_curriculum_service()
    if code in svc.eng.view:
        return code
    found = svc.detect_programmes(code)
    if found:
        return found[0]
    raise HTTPException(status_code=404, detail=f"unknown programme {code!r}; known: {svc.programmes()}")


@app.get("/api/curriculum/programmes")
def curriculum_programmes():
    return {"programmes": get_curriculum_service().payload["programmes"], "codes": get_curriculum_service().programmes()}


@app.get("/api/curriculum/courses")
def curriculum_search(q: str = "", programme: Optional[str] = None, k: int = 10):
    return get_curriculum_service().search(q, known_programme(programme) if programme else None, min(max(k, 1), 50))


@app.post("/api/curriculum/compare-courses")
def curriculum_compare_courses(req: CourseCompareRequest):
    return curriculum_call(get_curriculum_service().compare_courses, known_course(req.course_a), known_course(req.course_b), req.programme)


@app.post("/api/curriculum/compare-programmes")
def curriculum_compare_programmes(req: ProgrammeCompareRequest):
    return get_curriculum_service().compare_programmes(known_programme(req.programme_a), known_programme(req.programme_b))


@app.post("/api/curriculum/overlaps")
def curriculum_overlaps(req: OverlapRequest):
    return curriculum_call(get_curriculum_service().find_overlaps, known_course(req.course), req.k, req.programme)


@app.post("/api/curriculum/redundancy")
def curriculum_redundancy(req: ProgrammeRequest):
    return get_curriculum_service().detect_redundancy(known_programme(req.programme))


@app.post("/api/curriculum/missing-topics")
def curriculum_missing_topics(req: ProgrammeRequest):
    return get_curriculum_service().missing_topics(known_programme(req.programme))


@app.post("/api/curriculum/prerequisites")
def curriculum_prerequisites(req: PrerequisiteRequest):
    return curriculum_call(get_curriculum_service().prerequisites, known_course(req.course), req.programme, req.topics)


@app.post("/api/curriculum/bloom")
def curriculum_bloom(req: BloomRequest):
    return get_curriculum_service().bloom(req.outcomes, req.department)


@app.post("/api/curriculum/modernize")
def curriculum_modernize(req: ModernizeRequest):
    return curriculum_call(get_curriculum_service().modernize, [known_course(c) for c in req.courses], req.job_descriptions, req.title, req.programme)


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
