"""Request / response schemas of the Curriculum Intelligence API (Pydantic v2). Every response carries `meta` (engine version,
encoder, data snapshot) and confidence fields whose meaning is stated in the field description."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------ inputs
class CourseRef(BaseModel):
    """A course: an id from the parsed NMIMS dataset, a course name (latest version is used), or raw syllabus text."""
    id: Optional[str] = Field(None, examples=["nm48af830b"])
    name: Optional[str] = Field(None, examples=["Machine Learning"])
    text: Optional[str] = Field(None, description="Syllabus text in the NMIMS template, or any text with one topic per paragraph")
    programme: Optional[str] = Field(None, description="Programme code, needed for prerequisite checks on raw text", examples=["COMP"])
    semester: Optional[int] = Field(None, ge=1, le=10)
    academic_year: Optional[str] = Field(None, examples=["2025-26"])

    @model_validator(mode="after")
    def one_source(self):
        if sum(x is not None for x in (self.id, self.name, self.text)) != 1:
            raise ValueError("give exactly one of id, name or text")
        return self


class ProgrammeRef(BaseModel):
    programme: str = Field(..., examples=["COMP"])
    academic_year: Optional[str] = Field(None, description="Default: the latest year with enough parsed courses", examples=["2025-26"])


class CompareRequest(BaseModel):
    course_a: Optional[CourseRef] = None
    course_b: Optional[CourseRef] = None
    programme_a: Optional[ProgrammeRef] = None
    programme_b: Optional[ProgrammeRef] = None

    @model_validator(mode="after")
    def pair(self):
        courses, progs = self.course_a and self.course_b, self.programme_a and self.programme_b
        if bool(courses) == bool(progs):
            raise ValueError("give either course_a + course_b, or programme_a + programme_b")
        return self


class OutcomesRequest(BaseModel):
    course: Optional[CourseRef] = None
    outcomes: Optional[list[str]] = Field(None, examples=[["Design a relational database schema", "Explain normal forms"]])

    @model_validator(mode="after")
    def one(self):
        if (self.course is None) == (self.outcomes is None):
            raise ValueError("give either course or outcomes")
        return self


class FindOverlapsRequest(BaseModel):
    course: CourseRef
    k: int = Field(5, ge=1, le=50)
    programme: Optional[str] = Field(None, description="Only search courses of this programme")


class PrerequisitesRequest(BaseModel):
    course: CourseRef


class ProgrammeRequest(BaseModel):
    programme: str = Field(..., examples=["COMP"])
    academic_year: Optional[str] = None
    k: int = Field(20, ge=1, le=200)


# ------------------------------------------------------------------ outputs
class Meta(BaseModel):
    engine_version: str
    encoder: str
    course_versions: int
    split_hash: str
    evaluation_note: str


class CourseInfo(BaseModel):
    id: str
    course_name: str
    code: str = ""
    academic_year: str = ""
    semester: str = ""
    programmes: list[str] = []


class TopicMatch(BaseModel):
    topic_a: str
    topic_b: str
    similarity: float
    topic_label: Optional[str] = None


class PrereqFlag(BaseModel):
    programme: str
    phrase: str
    type: Literal["not_offered_earlier", "assumed_topic_not_taught"]
    expected_precision: float = Field(..., description="Share of such flags judged correct on LLM-annotated reference labels (test split)")


class PrereqResult(BaseModel):
    checked: bool
    reason: Optional[str] = Field(None, description="Why the course was not checked (coverage, common course, first semester)")
    flags: list[PrereqFlag] = []


class BloomItem(BaseModel):
    outcome: str
    level: str
    confidence: Optional[float] = Field(None, description="Precision of this predicted level against faculty K/L tags (test split)")
    faculty_tag: Optional[list[int]] = None


class Overlap(BaseModel):
    percent: str = Field(..., description="Share of matched units (F-measure of unit coverage)")
    semantic_similarity: float
    lexical_similarity: float
    confidence: float = Field(..., description="P(any overlap) from a logistic model fitted on validation labels; relative, not well calibrated")
    redundancy_warning: bool


class CompareCoursesResponse(BaseModel):
    meta: Meta
    course_a: CourseInfo
    course_b: CourseInfo
    overlap: Overlap
    common_topics: list[TopicMatch]
    unique_to_a: list[str]
    unique_to_b: list[str]
    prerequisites_a: PrereqResult
    prerequisites_b: PrereqResult
    outcome_bloom_a: list[BloomItem]
    outcome_bloom_b: list[BloomItem]


class DiffCourse(BaseModel):
    family: str
    course_name: str


class CompareProgrammesResponse(BaseModel):
    meta: Meta
    a: ProgrammeRef
    b: ProgrammeRef
    retained: int
    renamed_or_revised: list[tuple[str, str]] = Field(..., description="Matched by content although the name differs")
    removed: list[DiffCourse]
    added: list[DiffCourse]
    note: str


class OutcomesResponse(BaseModel):
    meta: Meta
    outcomes: list[BloomItem]


class OverlapHit(BaseModel):
    course: CourseInfo
    score: float = Field(..., description="Rank-average of TF-IDF and MaxSim percentiles among validation pairs (0-1)")
    confidence: float


class FindOverlapsResponse(BaseModel):
    meta: Meta
    query: CourseInfo
    results: list[OverlapHit]


class PrerequisitesResponse(BaseModel):
    meta: Meta
    course: CourseInfo
    result: PrereqResult


class RedundantPair(BaseModel):
    course_a: str
    unit_a: str
    course_b: str
    unit_b: str
    similarity: float
    expected_precision: float = Field(..., description="Precision of top redundancy flags on LLM-annotated reference labels (test split)")


class RedundancyResponse(BaseModel):
    meta: Meta
    programme: str
    academic_year: Optional[str]
    courses_checked: int
    pairs: list[RedundantPair]


class EmergingItem(BaseModel):
    topic: str
    covered: bool
    best_similarity: float
    closest_units: list[str]
    source: str


class EmergingResponse(BaseModel):
    meta: Meta
    programme: str
    reference_status: str
    gap_precision: Optional[float] = Field(None, description="Precision of flagged gaps on reference labels; gap recall is low (see note)")
    note: str
    topics: list[EmergingItem]


class Health(BaseModel):
    status: str
    meta: Meta
    endpoints: list[str]


# ------------------------------------------------------------------ Week 7
class JobDescription(BaseModel):
    id: str = Field(..., examples=["JD1"])
    text: str = Field(..., min_length=20)


class ModernizeRequest(BaseModel):
    """Job descriptions + the current syllabi to modernise -> overlaps, demanded skills, a traceable draft syllabus, hand-offs."""
    jds: list[JobDescription] = Field(..., min_length=1, max_length=10)
    courses: list[CourseRef] = Field(..., min_length=2, max_length=10)
    title: str = Field("Natural Language Processing and Large Language Models", min_length=3)
    academic_year: Optional[str] = None


class ModernizeResponse(BaseModel):
    meta: Meta
    courses: list[CourseInfo]
    overlaps: list[dict] = Field(..., description="Pairwise overlap of the selected syllabi (Week 4 fusion score and confidence)")
    shared_topics: list[dict] = Field(..., description="Units taught in two or more of the selected syllabi")
    skill_themes: list[dict] = Field(..., description="JD skill themes, the JDs that ask for them, and whether a unit already teaches them")
    new_curriculum: dict = Field(..., description="Draft syllabus; every unit and outcome names its source (course id or JD id)")
    traceable: bool
    markdown: str
    handoffs: dict = Field(..., description="group11_request (Group 11 /process-notice body) and group4_chunks (Group 4 passage rows)")


class CourseListResponse(BaseModel):
    meta: Meta
    courses: list[CourseInfo]
