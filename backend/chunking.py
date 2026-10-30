"""Section-aware chunking: a passage never crosses a section, starts at a numbered clause when it can, and keeps the page of
every sentence, so a citation points at the page the quoted sentence is really on.

The fixed 100-word windows of the first pipeline cut clauses and table rows in half, and a clause running across two PDF
pages was stored in two unrelated pieces.
"""
import re
from typing import Callable, Dict, List

import pandas as pd

CLAUSE_START = re.compile(r"^\(?\d{1,2}(?:\.\d{1,2}){0,3}[.)]?\s|^[a-z]\)\s|^\([ivx]{1,4}\)\s")


def _pieces(sentence: str, max_words: int) -> List[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    step = max(1, max_words - 15)                           # flattened tables: overlapping windows keep a row whole
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words) - 15, step)]


def make_section_chunks(segments: pd.DataFrame, split_sentences: Callable[[str], List[str]],
                        max_words: int = 130, min_words: int = 40) -> pd.DataFrame:
    """segments: rows of (page, section, part, text) in document order. Returns one row per chunk, with `sent_pages`."""
    rows: List[Dict] = []
    blocks: List[List] = []
    for page, section, part, text in segments[["page", "section", "part", "text"]].itertuples(index=False):
        if blocks and blocks[-1][0] == section and blocks[-1][1] == part:
            blocks[-1][2].append((page, text))
        else:
            blocks.append([section, part, [(page, text)]])

    for section, part, parts in blocks:
        sentences = [(p, piece) for page, text in parts for s in split_sentences(text) for p, piece in
                     ((page, x) for x in _pieces(s, max_words))]
        current: List = []
        count = 0

        def emit():
            rows.append({"page": current[0][0], "end_page": current[-1][0], "section": section, "part": part,
                         "text": " ".join(s for _, s in current), "sent_pages": [p for p, _ in current],
                         "sentences": [s for _, s in current]})

        for page, sentence in sentences:
            n = len(sentence.split())
            at_clause = bool(CLAUSE_START.match(sentence))
            if current and (count + n > max_words or (at_clause and count >= min_words)):
                emit()
                tail = current[-1]                          # one sentence of overlap, unless a new clause starts cleanly
                current, count = ([tail], len(tail[1].split())) if not at_clause and count + n > max_words else ([], 0)
            current.append((page, sentence))
            count += n
        if current:
            emit()

    chunks = pd.DataFrame(rows)
    chunks.insert(0, "chunk_id", [f"S{i:04d}" for i in range(len(chunks))])
    chunks["n_words"] = chunks.text.str.split().str.len()
    return chunks
