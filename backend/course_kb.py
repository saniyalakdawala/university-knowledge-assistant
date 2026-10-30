"""Second knowledge base: the new course's draft syllabus produced by Group 9.

Group 9 (Curriculum Intelligence Engine) returns, from POST /modernize-curriculum, a hand-off for us in our own passage
schema: handoffs.group4_chunks = [{chunk_id, page, section, part, text}, ...] (100-word windows, 25-word overlap, the
same make_chunks settings as our SRB index). They also export it as data/week7/group4_chunks.csv.

We index those rows with exactly the same pipeline as the SRB (BM25 + MiniLM + cross-encoder + Groq + grounding check +
refusal gate), but in a separate index, so a syllabus passage can never be cited as university policy and vice versa.
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import COURSE_PASSAGES_CSV
from backend.rag_service import UniversityRAGService, get_rag_service, canon_tokens

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["chunk_id", "page", "section", "part", "text"]
PART_PREFIX = re.compile(r"^\s*group 9 draft syllabus:\s*", re.I)


class CourseKnowledgeBase:
    def __init__(self):
        self.service: Optional[UniversityRAGService] = None
        self.title: str = ""
        self.source: str = ""
        self.n_rows: int = 0

    @property
    def loaded(self) -> bool:
        return self.service is not None

    def load_rows(self, rows: List[Dict[str, Any]], source: str = "API upload") -> Dict[str, Any]:
        if not rows:
            raise ValueError("no passage rows given")
        df = pd.DataFrame(rows)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"passage rows are missing columns {missing}; expected {REQUIRED_COLUMNS}")
        df = df[REQUIRED_COLUMNS].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0]
        if df.empty:
            raise ValueError("all passage rows have empty text")
        df["page"] = pd.to_numeric(df["page"], errors="coerce").fillna(0).astype(int)
        df["section"] = df["section"].astype(str)
        df["part"] = df["part"].astype(str)
        df["n_words"] = df["text"].str.split().str.len()

        title = PART_PREFIX.sub("", str(df["part"].iloc[0])).strip() or "New course"
        srb = get_rag_service()  # reuse the already-loaded embedder and reranker (no second model load)
        self.service = UniversityRAGService(
            chunks=df,
            doc_label=f"Group 9 draft syllabus: {title}",
            short_label="New course syllabus (Group 9)",
            not_found="I could not find this in the new course's draft syllabus.",
            shared_models=(srb.embedder, srb.reranker),
        )
        self.title, self.source, self.n_rows = title, source, len(df)
        logger.info(f"Loaded {len(df)} Group 9 passages for '{title}' from {source}.")
        return self.status()

    def load_csv(self, path: str) -> Dict[str, Any]:
        df = pd.read_csv(path)
        return self.load_rows(df.to_dict("records"), source=str(path))

    def load_default_if_present(self) -> None:
        p = Path(COURSE_PASSAGES_CSV)
        if p.exists():
            try:
                self.load_csv(str(p))
            except Exception as e:  # never block the SRB assistant because of a bad hand-off file
                logger.error(f"Could not load Group 9 passages from {p}: {e}")
        else:
            logger.info(f"No Group 9 passage file at {p}; POST /api/knowledge/course to load one.")

    def mentions_title(self, question: str) -> bool:
        """True when most content words of the course title appear in the question."""
        if not self.title:
            return False
        if self.title.lower() in question.lower():
            return True
        t = {w for w in canon_tokens(self.title) if len(w) > 2}
        if len(t) < 2:
            return False
        return len(t & set(canon_tokens(question))) / len(t) >= 0.8

    def status(self) -> Dict[str, Any]:
        return {"loaded": self.loaded, "title": self.title, "passages": self.n_rows, "source": self.source}


_COURSE_KB = CourseKnowledgeBase()


def get_course_kb() -> CourseKnowledgeBase:
    return _COURSE_KB
