from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Student question")
    session_id: Optional[str] = Field(None, description="Client session identifier for multi-turn conversations")
    route: Optional[Literal["auto", "srb", "curriculum", "both", "course", "course+srb"]] = Field(
        "auto", description="auto (default) | srb (policy) | curriculum (Group 9 live API) | both | course (new-course syllabus from Group 9) | course+srb")
    course_context: Optional[str] = Field(None, description="Optional specific course name or code to target in curriculum queries")

class SRBResult(BaseModel):
    answer: str
    source: str
    document: str = "Student Resource Book (SRB) 2026"
    section: str = ""
    confidence: float
    evidence_quote: str = ""
    cited_page: Optional[int] = None
    abstained: bool = False
    refusal_reason: str = ""
    mode: str = "generated"
    numbers_grounded: bool = True
    support: float = 0.0
    lexical_coverage: float = 0.0
    semantic_similarity: float = 0.0
    ce_prob: float = 0.0

class CurriculumResult(BaseModel):
    connected: bool = False
    is_fallback: bool = False
    status_note: str = ""
    summary: str = ""
    course: Optional[Dict[str, Any]] = None
    prerequisites: Optional[Dict[str, Any]] = None
    overlaps: Optional[List[Dict[str, Any]]] = None
    outcomes: Optional[List[Dict[str, Any]]] = None
    raw: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    question: str
    route_taken: Literal["srb", "curriculum", "both", "course", "course+srb"]
    synthesized_answer: str
    srb: Optional[SRBResult] = None
    course: Optional[SRBResult] = Field(None, description="Answer from the new course's draft syllabus (Group 9 hand-off), same fields as srb")
    curriculum: Optional[CurriculumResult] = None
    timestamp: str

class HealthStatus(BaseModel):
    status: str
    service: str = "Group 4 University Knowledge Assistant"
    group4: Dict[str, Any]
    group9: Dict[str, Any]


class PassageRow(BaseModel):
    chunk_id: str
    page: int
    section: str
    part: str
    text: str

class CourseIngestRequest(BaseModel):
    """Either rows=[...] or Group 9's whole /modernize-curriculum response (handoffs.group4_chunks is read from it)."""
    rows: Optional[List[PassageRow]] = None
    handoffs: Optional[Dict[str, Any]] = None
