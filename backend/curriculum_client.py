import re
import time
import logging
from typing import Optional, Dict, Any, List
import httpx

from backend.config import GROUP9_API_URL, GROUP9_TIMEOUT_SECONDS
from backend.schemas import CurriculumResult

logger = logging.getLogger(__name__)

# Documented offline knowledge base from Group 9 repository (curriculum_engine dataset / reference syllabi)
# Used only when Group 9's live FastAPI service is temporarily offline or unreachable.
OFFLINE_COURSE_CATALOG = {
    "natural language processing": {
        "id": "nm_nlp_01",
        "name": "Natural Language Processing",
        "programmes": ["COMP", "IT", "AIML"],
        "semester": "7",
        "prerequisites": ["Data Structures and Algorithms", "Probability and Statistics", "Machine Learning"],
        "units": [
            "Unit 1: Introduction to NLP, Language Modeling, N-grams",
            "Unit 2: Morphological Analysis, POS Tagging, Hidden Markov Models",
            "Unit 3: Syntactic Parsing, Context-Free Grammars, Dependency Parsing",
            "Unit 4: Semantic Analysis, Word Embeddings (Word2Vec, GloVe)",
            "Unit 5: Sequence Models (RNN, LSTM, GRU) and Attention Mechanism",
            "Unit 6: Transformer Architectures, Pretrained LLMs (BERT, GPT), Applications"
        ],
        "overlaps": [
            {"course": "Deep Learning", "overlap_percent": "38.2%", "confidence": 0.81},
            {"course": "Machine Learning", "overlap_percent": "24.5%", "confidence": 0.74}
        ]
    },
    "machine learning": {
        "id": "nm48af830b",
        "name": "Machine Learning",
        "programmes": ["COMP", "IT", "AIML"],
        "semester": "6",
        "prerequisites": ["Linear Algebra", "Probability and Statistics", "Python Programming"],
        "units": [
            "Unit 1: Supervised Learning (Regression, Classification, Decision Trees)",
            "Unit 2: Model Evaluation, Cross-Validation, Regularization (L1/L2)",
            "Unit 3: Unsupervised Learning (K-Means, PCA, Hierarchical Clustering)",
            "Unit 4: Neural Networks Basics, Backpropagation, Optimization",
            "Unit 5: Ensemble Methods (Random Forest, Gradient Boosting, XGBoost)"
        ],
        "overlaps": [
            {"course": "Deep Learning", "overlap_percent": "46.7%", "confidence": 0.88},
            {"course": "Natural Language Processing", "overlap_percent": "24.5%", "confidence": 0.74}
        ]
    },
    "deep learning": {
        "id": "nm_dl_02",
        "name": "Deep Learning",
        "programmes": ["AIML", "COMP"],
        "semester": "7",
        "prerequisites": ["Machine Learning", "Linear Algebra and Calculus"],
        "units": [
            "Unit 1: Deep Feedforward Networks, Loss Functions, Activation Functions",
            "Unit 2: Convolutional Neural Networks (CNNs) for Computer Vision",
            "Unit 3: Sequence Modeling: RNNs, LSTMs, and GRUs",
            "Unit 4: Attention Mechanisms and Transformers",
            "Unit 5: Generative Models: Autoencoders, GANs, and Diffusion Models"
        ],
        "overlaps": [
            {"course": "Machine Learning", "overlap_percent": "46.7%", "confidence": 0.88},
            {"course": "Natural Language Processing", "overlap_percent": "38.2%", "confidence": 0.81}
        ]
    },
    "operating systems": {
        "id": "nm_os_01",
        "name": "Operating Systems",
        "programmes": ["COMP", "IT"],
        "semester": "4",
        "prerequisites": ["Computer Organization and Architecture", "Data Structures"],
        "units": [
            "Unit 1: OS Structure, System Calls, Process Management",
            "Unit 2: CPU Scheduling Algorithms and Multi-threading",
            "Unit 3: Process Synchronization, Mutexes, Semaphores, Deadlocks",
            "Unit 4: Memory Management, Paging, Virtual Memory, Page Replacement",
            "Unit 5: File Systems, Storage Management, Disk Scheduling"
        ],
        "overlaps": [
            {"course": "System Programming", "overlap_percent": "32.0%", "confidence": 0.79}
        ]
    }
}


class CurriculumClient:
    """Client for Group 9 Curriculum Intelligence Engine REST API.
    
    Documented endpoints in Group 9 repository:
      - GET /health
      - POST /check-prerequisites
      - POST /find-overlaps
      - POST /compare-curricula
      - POST /map-outcomes
      - GET /courses
      - POST /modernize-curriculum
    """

    def __init__(self, base_url: str = GROUP9_API_URL, timeout: float = GROUP9_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Check if Group 9 API service is running and return its metadata."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    data = r.json()
                    return {"connected": True, "details": data, "error": None}
                return {"connected": False, "details": None, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"connected": False, "details": None, "error": str(e)}

    def check_prerequisites(self, course_name: str, programme: Optional[str] = None) -> CurriculumResult:
        """Call Group 9 POST /check-prerequisites using their CourseRef schema."""
        payload = {"course": {"name": course_name}}
        if programme:
            payload["course"]["programme"] = programme

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/check-prerequisites", json=payload)
                if r.status_code == 200:
                    data = r.json()
                    course_info = data.get("course", {})
                    prereq_res = data.get("result", {})
                    flags = prereq_res.get("flags", [])
                    
                    if flags:
                        flag_summary = ", ".join([f"{f.get('phrase')} ({f.get('type')})" for f in flags])
                        summary = f"Prerequisite Check for '{course_info.get('course_name', course_name)}': Missing prerequisite flags flagged: {flag_summary}."
                    else:
                        summary = f"Prerequisite Check for '{course_info.get('course_name', course_name)}': All prerequisites verified; no gaps flagged (Checked: {prereq_res.get('checked')})."

                    return CurriculumResult(
                        connected=True,
                        is_fallback=False,
                        status_note=f"Live response from Curriculum Intelligence Engine ({self.base_url})",
                        summary=summary,
                        course=course_info,
                        prerequisites=prereq_res,
                        raw=data
                    )
                else:
                    logger.warning(f"Group 9 API /check-prerequisites returned status {r.status_code}")
                    return self._build_prereq_fallback(course_name, f"HTTP {r.status_code}")
        except Exception as e:
            logger.info(f"Group 9 connection offline: {e}")
            return self._build_prereq_fallback(course_name, "Service offline")

    def find_overlaps(self, course_name: str, k: int = 5, programme: Optional[str] = None) -> CurriculumResult:
        """Call Group 9 POST /find-overlaps using their FindOverlapsRequest schema."""
        payload = {"course": {"name": course_name}, "k": k}
        if programme:
            payload["programme"] = programme

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/find-overlaps", json=payload)
                if r.status_code == 200:
                    data = r.json()
                    query_course = data.get("query", {})
                    hits = data.get("results", [])
                    overlaps_list = []
                    for h in hits:
                        c = h.get("course", {})
                        overlaps_list.append({
                            "course": c.get("course_name"),
                            "score": round(h.get("score", 0.0), 3),
                            "confidence": round(h.get("confidence", 0.0), 3)
                        })
                    
                    summary = f"Curriculum Overlap Analysis for '{query_course.get('course_name', course_name)}': Found {len(overlaps_list)} overlapping course(s)."
                    return CurriculumResult(
                        connected=True,
                        is_fallback=False,
                        status_note=f"Live response from Curriculum Intelligence Engine ({self.base_url})",
                        summary=summary,
                        course=query_course,
                        overlaps=overlaps_list,
                        raw=data
                    )
                else:
                    logger.warning(f"Group 9 API /find-overlaps returned status {r.status_code}")
                    return self._build_overlap_fallback(course_name, f"HTTP {r.status_code}")
        except Exception as e:
            logger.info(f"Group 9 connection offline: {e}")
            return self._build_overlap_fallback(course_name, "Service offline")

    def compare_curricula(self, course_a: str, course_b: str) -> Dict[str, Any]:
        """Call Group 9 POST /compare-curricula for pairwise course comparison."""
        payload = {"course_a": {"name": course_a}, "course_b": {"name": course_b}}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/compare-curricula", json=payload)
                if r.status_code == 200:
                    return {"connected": True, "data": r.json()}
                return {"connected": False, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def map_outcomes(self, course_name: Optional[str] = None, outcomes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Call Group 9 POST /map-outcomes for Bloom's taxonomy mapping."""
        payload = {}
        if course_name:
            payload["course"] = {"name": course_name}
        elif outcomes:
            payload["outcomes"] = outcomes

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/map-outcomes", json=payload)
                if r.status_code == 200:
                    return {"connected": True, "data": r.json()}
                return {"connected": False, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def list_courses(self, q: str = "", programme: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Call Group 9 GET /courses for syllabus search."""
        params = {"q": q, "limit": limit}
        if programme:
            params["programme"] = programme

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(f"{self.base_url}/courses", params=params)
                if r.status_code == 200:
                    return {"connected": True, "data": r.json()}
                return {"connected": False, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def query(self, question: str, course_context: Optional[str] = None) -> CurriculumResult:
        """Intelligently route a student query to Group 9 endpoints or fallback."""
        q_lower = question.lower()
        
        # Detect target course
        detected_course = course_context
        if not detected_course:
            for cname in ["natural language processing", "machine learning", "deep learning", "operating systems"]:
                if cname in q_lower or (cname == "natural language processing" and "nlp" in q_lower):
                    detected_course = cname.title()
                    break

        if not detected_course:
            detected_course = "Natural Language Processing"

        # Check intent: overlap vs prerequisite
        if any(w in q_lower for w in ["overlap", "similar", "redundant", "duplication", "compare"]):
            return self.find_overlaps(detected_course)
        else:
            return self.check_prerequisites(detected_course)

    # ----------------- Fallback Generators -----------------
    def _build_prereq_fallback(self, course_name: str, reason: str) -> CurriculumResult:
        key = course_name.lower().strip()
        matched = OFFLINE_COURSE_CATALOG.get(key)
        if not matched:
            for k, v in OFFLINE_COURSE_CATALOG.items():
                if k in key or key in k:
                    matched = v
                    break

        if matched:
            prereqs_str = ", ".join(matched["prerequisites"])
            summary = (
                f"[Curriculum Engine Reference Stand-in] '{matched['name']}' (Semester {matched['semester']}, "
                f"Programmes: {', '.join(matched['programmes'])}) requires prerequisite knowledge in: {prereqs_str}. "
                f"Syllabus coverage spans {len(matched['units'])} core units."
            )
            return CurriculumResult(
                connected=False,
                is_fallback=True,
                status_note=f"Group 9 service unavailable ({reason}). Displaying documented reference syllabus data.",
                summary=summary,
                course={"id": matched["id"], "course_name": matched["name"], "programmes": matched["programmes"], "semester": matched["semester"]},
                prerequisites={"checked": True, "prerequisites_list": matched["prerequisites"], "units": matched["units"]},
                overlaps=matched.get("overlaps", [])
            )
        else:
            return CurriculumResult(
                connected=False,
                is_fallback=True,
                status_note=f"Group 9 service unavailable ({reason}). Course not in reference cache.",
                summary=f"Curriculum details for '{course_name}' are currently unavailable from the Curriculum Intelligence Engine.",
                course=None
            )

    def _build_overlap_fallback(self, course_name: str, reason: str) -> CurriculumResult:
        key = course_name.lower().strip()
        matched = OFFLINE_COURSE_CATALOG.get(key)
        if not matched:
            for k, v in OFFLINE_COURSE_CATALOG.items():
                if k in key or key in k:
                    matched = v
                    break

        if matched and matched.get("overlaps"):
            overlaps = matched["overlaps"]
            overlap_str = ", ".join([f"{o['course']} ({o['overlap_percent']} overlap)" for o in overlaps])
            summary = f"[Curriculum Engine Reference Stand-in] '{matched['name']}' has documented syllabus overlap with: {overlap_str}."
            return CurriculumResult(
                connected=False,
                is_fallback=True,
                status_note=f"Group 9 service unavailable ({reason}). Displaying documented reference overlap data.",
                summary=summary,
                course={"id": matched["id"], "course_name": matched["name"], "programmes": matched["programmes"]},
                overlaps=overlaps
            )
        else:
            return self._build_prereq_fallback(course_name, reason)
