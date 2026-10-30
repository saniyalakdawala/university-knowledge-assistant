import re
import time
import logging
from typing import Optional, Dict, Any, List
import httpx

from backend.config import CURRICULUM_BACKEND, GROUP9_API_URL, GROUP9_TIMEOUT_SECONDS
from backend.curriculum.service import get_curriculum_service
from backend.schemas import CurriculumResult

logger = logging.getLogger(__name__)

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

    def query(self, question: str, course_context: Optional[str] = None, plan: Optional[Dict[str, Any]] = None) -> CurriculumResult:
        """Answer from the local engine, or from a running remote engine first when CURRICULUM_BACKEND=remote."""
        if CURRICULUM_BACKEND == "remote":
            q_lower = question.lower()
            names = (plan or {}).get("courses") or get_curriculum_service().detect_courses(question)
            course = course_context or (names[0] if names else None)
            if course and any(w in q_lower for w in ("overlap", "similar", "redundant", "duplication", "compare")):
                res = self.find_overlaps(course)
            elif course:
                res = self.check_prerequisites(course)
            else:
                res = None
            if res is not None and res.connected:
                return res
        return self.local(question, course_context, plan)

    def local(self, question: str, course_context: Optional[str] = None, plan: Optional[Dict[str, Any]] = None) -> CurriculumResult:
        r = get_curriculum_service().answer(question, plan, course_context)
        fields = {k: r[k] for k in ("summary", "intent", "data_basis", "note", "details", "citations", "confidence", "course", "prerequisites", "raw")}
        overlaps = r["overlaps"]
        return CurriculumResult(connected=True, is_fallback=False, status_note="Curriculum Intelligence (built into this application)",
                                overlaps=overlaps, outcomes=r["outcomes"], **fields)

    # ----------------- Fallback when the remote engine cannot be reached: real SRB data, clearly labelled -----------------
    def _build_prereq_fallback(self, course_name: str, reason: str) -> CurriculumResult:
        svc = get_curriculum_service()
        name = svc.resolve_name(course_name)
        note = f"Group 9 service unavailable ({reason}). Showing SRB programme data from this application."
        if not name:
            return CurriculumResult(connected=False, is_fallback=True, status_note=note,
                                    summary=f"Curriculum details for '{course_name}' are currently unavailable: no such course in the SRB programme structures.")
        r = svc.answer(f"prerequisites for {name}", {"intent": "prerequisites", "courses": [name]})
        return CurriculumResult(connected=False, is_fallback=True, status_note=note, summary=r["summary"], course=r["course"], prerequisites=r["prerequisites"],
                                intent="prerequisites", data_basis=r["data_basis"], note=r["note"])

    def _build_overlap_fallback(self, course_name: str, reason: str) -> CurriculumResult:
        svc = get_curriculum_service()
        name = svc.resolve_name(course_name)
        if not name:
            return self._build_prereq_fallback(course_name, reason)
        r = svc.answer(f"overlap with {name}", {"intent": "overlap", "courses": [name]})
        return CurriculumResult(connected=False, is_fallback=True, status_note=f"Group 9 service unavailable ({reason}). Showing SRB programme data from this application.",
                                summary=r["summary"], course=r["course"], overlaps=r["overlaps"], intent="overlap", data_basis=r["data_basis"], note=r["note"])
