"""Capability 5 - learning outcome -> Bloom's taxonomy level. Ground truth: the faculty's own K/L tag printed in the syllabus.
Baseline (Week 2): hand-written verb lexicon. Improved (Week 4): learned verb map + embedding classifier fallback."""
import re
from collections import Counter, defaultdict

BLOOM = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}
BLOOM_VERBS = {
    1: "define list recall recognize recognise name state label memorize repeat retrieve enumerate cite reproduce know mention recite tell identify",
    2: "explain describe summarize summarise discuss interpret classify illustrate paraphrase understand comprehend distinguish outline review translate express relate convert compare",
    3: "apply use implement solve execute demonstrate compute calculate operate perform employ utilize utilise practice show simulate install configure write draw sketch program select measure",
    4: "analyze analyse examine contrast categorize categorise investigate deduce infer diagnose experiment organize organise survey decompose debug troubleshoot differentiate test model",
    5: "evaluate assess justify critique judge defend recommend appraise validate verify argue support prioritize prioritise rate optimize optimise conclude decide select",
    6: "design develop create formulate construct build compose devise plan propose synthesize synthesise invent generate integrate assemble architect engineer author produce",
}
VERB_LEVEL = {}
for _lvl, _words in BLOOM_VERBS.items():
    for _w in _words.split():
        VERB_LEVEL.setdefault(_w, _lvl)          # a verb listed at several levels keeps its lowest
SKIP_WORDS = set("to be able will would can could students student the a an and or of in on for with also should".split())


def first_word(text):
    """The action verb an outcome starts with ('Students will be able to effectively design ...' -> 'design')."""
    for w in re.findall(r"[a-z]+", text.lower())[:7]:
        if w not in SKIP_WORDS and not w.endswith("ly"):
            return w
    return None


def lexicon_level(text):
    """Level of the first listed Bloom verb among the first 5 words; None if none matches."""
    for w in re.findall(r"[a-z]+", text.lower())[:5]:
        if w in VERB_LEVEL:
            return VERB_LEVEL[w]
    return None


def tagged_outcomes(recs, split_of):
    """{split: [(text, level, family)]} for outcomes with exactly one faculty tag. Identical texts are kept once, and a text seen
    in a training course is dropped from val/test (syllabi are re-used, so the same sentence would otherwise leak)."""
    seen, out = {}, defaultdict(list)
    for r in sorted(recs, key=lambda r: ({"train": 0, "val": 1, "test": 2}[split_of[r["family"]]], r["id"])):
        s = split_of[r["family"]]
        for o in r["outcomes"]:
            if len(o["k_levels"]) == 1 and o["text"] not in seen:
                seen[o["text"]] = s
                out[s].append((o["text"], o["k_levels"][0], r["family"]))
    return out


def learn_verb_map(texts, levels, min_count=2):
    by = defaultdict(Counter)
    for t, l in zip(texts, levels):
        w = first_word(t)
        if w:
            by[w][l] += 1
    return {w: c.most_common(1)[0][0] for w, c in by.items() if sum(c.values()) >= min_count}
