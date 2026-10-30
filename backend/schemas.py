from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Student question")
    session_id: Optional[str] = Field(None, description="Client session identifier for multi-turn conversations")
    route: Optional[Literal["auto", "srb", "curriculum", "both"]] = Field("auto", description="Routing preference")
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
    standalone_question: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="One entry per passage used: part, section, page, verbatim quote")

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
    intent: str = ""
    data_basis: str = Field("", description="full_syllabus, name_only, srb_structure or outcome_text")
    note: str = ""
    confidence: Optional[float] = None
    details: List[Dict[str, str]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    question: str
    route_taken: Literal["srb", "curriculum", "both"]
    synthesized_answer: str
    srb: Optional[SRBResult] = None
    curriculum: Optional[CurriculumResult] = None
    timestamp: str
    routed_by: str = Field("", description="request, rules or groq")
    session_id: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    service: str = "Group 4 University Knowledge Assistant"
    group4: Dict[str, Any]
    group9: Dict[str, Any]


class CourseCompareRequest(BaseModel):
    course_a: str
    course_b: str
    programme: Optional[str] = None

class ProgrammeCompareRequest(BaseModel):
    programme_a: str
    programme_b: str

class OverlapRequest(BaseModel):
    course: str
    k: int = Field(5, ge=1, le=20)
    programme: Optional[str] = None

class ProgrammeRequest(BaseModel):
    programme: str

class PrerequisiteRequest(BaseModel):
    course: str
    programme: Optional[str] = None
    topics: Optional[List[str]] = Field(None, description="Topics or courses assumed by the course; checked against earlier semesters")

class BloomRequest(BaseModel):
    outcomes: Optional[List[str]] = None
    department: Optional[str] = None

class ModernizeRequest(BaseModel):
    courses: List[str] = Field(..., min_length=1, max_length=5)
    job_descriptions: Optional[Dict[str, str]] = Field(None, description="id -> text; omit to use the bundled synthetic samples")
    title: Optional[str] = None
    programme: Optional[str] = None
