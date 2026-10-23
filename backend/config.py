import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# SRB PDF path detection
def get_srb_pdf_path() -> str:
    env_path = os.getenv("SRB_PDF_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    candidates = [
        DATA_DIR / "SRB_2026.pdf",
        BASE_DIR / "Student Resource Book (SRB) 2026.pdf",
        DATA_DIR / "Student Resource Book (SRB) 2026.pdf",
        BASE_DIR / "Student_Resource_Book__SRB__2026.pdf",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(DATA_DIR / "SRB_2026.pdf")

SRB_PDF_PATH = get_srb_pdf_path()

# Model configurations (Week 6 frozen specifications)
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CE_MODEL_NAME = os.getenv("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval & Gate hyperparameters
RETRIEVAL_BETA = float(os.getenv("RETRIEVAL_BETA", "0.6"))     # Hybrid lexical vs semantic weight
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.7"))         # Cross-encoder rank weight
CONFIDENCE_GATE = float(os.getenv("CONFIDENCE_GATE", "0.46"))  # Dev-tuned refusal threshold
TOP_K_PASSAGES = int(os.getenv("TOP_K_PASSAGES", "3"))         # Top-k passages fed to answerer

# Mandatory Groq LLM Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
DEFAULT_LLM_MODEL = GROQ_MODEL
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "10.0"))
LLM_CACHE_FILE = str(BASE_DIR / "week5_outputs" / "llm_cache.json")

# Group 9 Integration Settings
GROUP9_API_URL = os.getenv("GROUP9_API_URL", "http://localhost:8001")
GROUP9_TIMEOUT_SECONDS = float(os.getenv("GROUP9_TIMEOUT", "10.0"))

# Server & Memory Optimization Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Auto-detect Render cloud container to guarantee safety under 512MB RAM limit
IS_RENDER = (
    os.getenv("RENDER") is not None
    or os.getenv("RENDER_SERVICE_ID") is not None
    or (os.getenv("PORT") is not None and os.getenv("PORT") != "8000")
)
LOW_MEMORY = os.getenv("LOW_MEMORY", "true" if IS_RENDER else "false").lower() in ("true", "1", "yes")
