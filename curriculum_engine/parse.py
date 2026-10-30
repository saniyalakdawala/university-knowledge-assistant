"""Information extraction: NMIMS B.Tech semester syllabus booklets (PDF) -> structured course records.

Reads the text-layer syllabus PDFs straight from the zip (no need to unpack 16 GB) and writes
    data/processed/courses.jsonl         one record per distinct course version
    results/dataset/parse_stats.json
Page text is cached in data/interim/nmims_pages.jsonl so that re-running after a parser change takes seconds.

Scanned PDFs (all past exam papers, most pre-2020 syllabi) have no text layer and are skipped: no OCR engine here.
Records from before AY 2020-21 are dropped: they come from OCR'd scans with heavy character noise.

    python -m curriculum_engine.parse [path/to/B TECH.zip] [--rebuild-cache]     (default: $NMIMS_ZIP or ./B TECH.zip)
"""
import hashlib, json, re, sys, zipfile
from collections import Counter, defaultdict

from curriculum_engine import DATA, RESULTS, ZIP

OUT, RES = DATA / "processed/courses.jsonl", RESULTS / "dataset"
CACHE = DATA / "interim/nmims_pages.jsonl"
SKIP_NAME = re.compile(r"exam|question|paper", re.I)          # past papers: scanned, and not syllabi anyway
MIN_AY = 2020                                                   # first academic year kept (start year)
NOISE = re.compile(r"SVKM|Narsee Monjee|Mukesh Patel School|Prepared by|Approved by|Head of the Dep|^\s*Signature|"
                   r"^_{3,}|^\s*Page \d+|Technology Management Department|^\s*A\.?\s?Y\.?\s*20\d\d|"
                   r"^\s*\((?:Dean|Chairperson|Program Chairperson|Head of Department|HOD)[^)]*\)\s*$|^\s*w\.e\.f\.?", re.I)
# fallback footer patterns for booklets too short to detect repeated footers (inline: they can sit inside a text line)
FOOTER = re.compile(r"(?i)(?:MBA\s*\(?\s*Tech\)?|B\.?\s?Tech\.?)[^\n]{0,80}?(?:page|pg)\s*[|:.]?\s*\d+"
                    r"|[A-Za-z&. ]{5,50}\(\s*20\d\d\s*[-–]\s*20\d\d\s*\)"
                    r"|B\.?\s?TECH[/A-Z .()]*/\d{4}[-–]\d{2}/SEM\s*[IVX]+/\d+"
                    r"|MPSTME-[A-Z]+-\d+[^\n]{0,15}"
                    # 'B. Tech. AI / Semester-III / AY 2023-24 / Page', 'B. Tech /DS/AY 2022-23/SEM V and VI', '2022-23 / Page 6', 'of 14 / Page 4'
                    r"|B\.?\s?Tech\.?[^\n]{0,60}?/\s*(?:Semester|SEM)[-\s]*[IVX]+(?:\s*(?:and|&|/)\s*[IVX]+)?(?:\s*/?\s*(?:A\.?Y\.?[-\s]*)?20\d\d[-–]\d\d\s*/?)?(?:\s*Page\s*\d*)?"
                    r"|(?:Electronics\s*(?:&|&amp;|and)\s*Telecommunication|Information Technology|Computer Engineering|Mechanical Engineering|Civil Engineering|Electrical Engineering)\s+Department|_{3,}|\s+of\s+\d{1,3}\s*$"
                    r"|(?:\bof\s+\d+|20\d\d\s*[-–]\s*\d\d)\s*/\s*Page\s*\d*|\bA\.?Y\.?[-\s]*20\d\d[-–]\d\d|(?<=[\w.,;])\s*/\s*Page\s*\d+(?:\s*of\s*\d+)?")
# the pattern before the fix, kept only to count how many records the fix changed (evidence for the report)
OLD_PRE = re.compile(r"(?is)pre\s*[-–]?\s*requisites?\s*[:\-–]?\s*(.*?)(?=\n\s*(?:course\s+)?objectives?|\n\s*(?:course\s+)?outcomes?)")
AY = re.compile(r"(20\d\d)\s*[-–/_]\s*(?:20)?(\d\d)\b")
# Bloom tags on outcomes: "(K3)", "(K3, K4)", "(L4 - Analyzing)", "(L2)"; K/L both mean Bloom level 1-6
KLEVEL = re.compile(r"\(\s*([KL]\s?[1-6](?:\s*[,&/]\s*[KL]?\s?[1-6])*)(?:\s*[-–:]\s*[A-Za-z]+)?\s*\)", re.I)
HEAD = {  # section -> heading text (matched at the start of a line); 'stop' headings just end the previous section
    "objectives": r"(?:course\s+)?objectives?",
    "outcomes": r"(?:course\s+|learning\s+)?outcomes?",
    "syllabus": r"detailed\s+syllabus.*",
    "experiments": r"list\s+of\s+(?:experiments|practicals?|laboratory\s+experiments?)",
    "books": r"(?:prescribed\s+)?text\s?books?(?:\s+and\s+reference\s+books?)?",
    "refs": r"reference\s+books?|references?",
    "lab": r"laboratory\s*/?\s*(?:tutorial\s+)?work|tutorial\s+work",
    "stop": r"any\s+other\s+information|internet\s+references?|nptel.*|online\s+(?:resources|courses?).*|e-?resources.*|term\s*work\b.*|details?\s+of\s+(?:test|term).*|total\s+marks.*|"
            r"distribution\s+of\s+ica.*|in\s+order\s+to\s+complete.*|internal\s+continuous\s+assessment.*|note\s*:.*",
}


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def key(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower().replace("&", "and"))


ELECTIVE = re.compile(r"^(?:professional|open|department(?:al)?|dept\.?|institute|program(?:me)?)?\s*elective(?:\s+course)?\s*[-–]?\s*[0-9IVX]*\s*\((.+)\)\s*(?:\(\s*\))?$", re.I)


def clean_name(n):
    """'Deep Learning (Department Elective III)' -> 'Deep Learning'; 'Professional Elective Course - II (Ground Improvement)' -> 'Ground Improvement'."""
    n = norm(re.split(r"\s+(?:module\s+|course\s+)?code\s*[:\-–]", n, flags=re.I)[0])
    m = ELECTIVE.match(n)
    if m:
        n = m.group(1)
    return norm(re.sub(r"\(\s*(?:[^)]*(?:elective|technical pool)[^)]*)?\)", "", n, flags=re.I))


def ay_of(text):
    m = AY.search(text)
    if m and int(m.group(2)) == (int(m.group(1)) + 1) % 100:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def line_key(l):
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", l.strip().lower()))


def clean_pages(pages):
    """Drop page headers/footers. Footers are found per booklet: a line containing a number that repeats (digit-normalised)
    at the top/bottom of >= 2 pages. The same text is then also removed where it sits inside another line."""
    protected = re.compile(r"(?i)\s*(program|semester|course|code|module)")
    cand = {}
    for pg in pages:
        ls = [l.strip() for l in pg.split("\n") if l.strip()]
        for k, l in {line_key(l): l for l in ls[:4] + ls[-4:]}.items():
            if re.search(r"\d", l) and len(k) > 8 and not re.fullmatch(r"[#\W]*", k) and not protected.match(l):
                cand.setdefault(k, [l, 0])[1] += 1
    pats = []
    for l, c in cand.values():
        if c >= 2:
            rx = r"\d+".join(re.escape(p) for p in re.split(r"\d+", l))
            pats.append(re.compile(re.sub(r"(?:\\ )+", r"\\s+", rx), re.I))
    lines = []
    for pg in pages:
        ay = ay_of("\n".join(pg.split("\n")[:25]))
        for l in pg.split("\n"):
            for p in pats:
                l = p.sub(" ", l)
            l = FOOTER.sub(" ", l)
            if l.strip() and not NOISE.search(l):
                lines.append((l.rstrip(), ay))
    return lines


def split_blocks(lines):
    """A course starts at a capitalised 'Program' line with 'Semester' and 'Course' lines right after it."""
    starts = [i for i, (l, _) in enumerate(lines)
              if re.match(r"\s*Program(?:me)?\b\s*[:\-–]?\s*(?:[A-Z]|\()", l)
              and any(re.match(r"\s*Semester\b", x, re.I) for x, _ in lines[i:i + 10])
              and any(re.match(r"\s*Course\b", x, re.I) for x, _ in lines[i:i + 14])]
    return [lines[a:b] for a, b in zip(starts, starts[1:] + [len(lines)])]


def section_spans(text):
    """Cut a block into named sections at heading lines. A heading is alone on its line ('Course Outcomes'),
    or is followed by ':' / '-' and text ('Objectives: ...'); a sentence that merely starts with the word does not count."""
    text = re.sub(r"(?i)(?<=\S)[ \t]+((?:text\s?books?|reference\s+books?|laboratory\s*/?\s*(?:tutorial\s+)?work|tutorial\s+work|nptel[^:\n]{0,20})[ \t]*:)", r"\n\1", text)
    hits = []
    for name, rx in HEAD.items():
        for m in re.finditer(rf"(?im)^[ \t]*(?:{rx})[ \t]*[:\-–]?[ \t]*$|^[ \t]*(?:{rx})[ \t]*[:\-–][ \t]*(?=\S)", text):
            hits.append((m.start(), m.end(), name))
    hits.sort()
    first, kept = set(), []
    for h in hits:                        # only the first heading of each kind cuts the text; 'stop' headings always do
        if h[2] == "stop" or h[2] not in first:
            kept.append(h); first.add(h[2])
    out = {}
    for (s, e, n), nxt in zip(kept, kept[1:] + [(len(text), len(text), None)]):
        if n != "stop":
            out[n] = text[e:nxt[0]]
    return out


def parse_units(txt):
    """Numbered units, each with a duration in hours. Handles 'N' / 'N.' alone on a line as well as 'N. Title ...' inline.
    A bare number equal to the next unit number is really a duration when the next line also looks like that unit start."""
    txt = re.split(r"(?im)^\s*total\b", txt)[0]
    junk = re.compile(r"(?i)^(unit|description|duration|topics?|hrs?\.?|hours?|no\.?)(\s+(description|duration|hrs?\.?|hours?))*\s*$")
    lines = [l.strip() for l in txt.split("\n") if l.strip() and not junk.match(l.strip())]
    own = lambda l, n: re.fullmatch(rf"{n}\s*[.)]?", l) is not None
    strict = any(own(l, 1) for l in lines) and any(own(l, 2) for l in lines)       # unit numbers sit on their own lines

    def is_start(l, n):
        if strict:
            return own(l, n)
        return own(l, n) or re.match(rf"^{n}\s*[.)]\s*\S", l) is not None or re.match(rf"^{n}\s+[A-Z]", l) is not None

    units, cur, want = [], None, 1
    for i, l in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if is_start(l, want) and not (cur is not None and re.fullmatch(r"\d{1,2}", l) and is_start(nxt, want)):
            if cur is not None:
                units.append(cur)
            rest = re.sub(r"^\d{1,2}\s*[.)]?\s*", "", l)
            cur, want = ([rest] if rest else []), want + 1
        elif cur is not None:
            while True:      # the next unit can start mid-line, right after a sentence end: '... dashboards. 6. Calculus Basic concept ...'
                m = re.search(rf"(?<=[.;])\s+{want}[.)]\s+(?=[A-Z])", l)
                if not m:
                    break
                cur.append(l[:m.start()])
                units.append(cur)
                cur, l, want = [], l[m.end():], want + 1
            cur.append(l)
    if cur is not None:
        units.append(cur)
    res = []
    for u in units:
        hours, body_lines = None, []
        for l in u:
            if re.fullmatch(r"\d{1,2}", l):                       # a duration cell that drifted into the text
                hours = hours or int(l)
                continue
            m = re.search(r"(?<![\w.,/()-])0([1-9])(?![\w.,%/()-])", l)   # '... continuous 06 streams' : zero-padded duration
            if m:
                hours = hours or int(m.group(1))
                l = (l[:m.start()] + " " + l[m.end():]).strip()
            body_lines.append(l)
        body = norm(" ".join(body_lines))
        if not body:
            continue
        first = norm(body_lines[0]) if body_lines else ""
        title = ""
        if ":" in first[:90] and len(first.split(":")[0].split()) <= 12:
            title = first.split(":")[0]
        elif len(first.split()) <= 10 and len(body_lines) > 1 and not first.endswith((",", ";", "-")) and re.match(r"[A-Z\"(‘']", norm(body_lines[1])):
            title = first
        res.append({"title": title.strip(" :.-–"), "text": body, "hours": hours})
    return res


STEM = re.compile(r"(?is)^\s*(?:after|upon|on|at the end|by the end|the students?|students?)\b[^:\n]{0,140}?\b(?:able to|will be|would be|following)\b[^\n:]*[:\-–]?[ \t]*")
OUTCOME_VERBS = set("""define list recall recognize recognise name state label memorize repeat retrieve enumerate identify explain describe summarize
summarise discuss interpret classify illustrate paraphrase understand comprehend distinguish outline review translate express relate convert
compare apply use implement solve execute demonstrate compute calculate operate perform employ utilize utilise practice show simulate install
configure write draw sketch program select measure analyze analyse examine contrast categorize categorise investigate deduce infer diagnose
experiment organize organise survey decompose debug troubleshoot differentiate test model evaluate assess justify critique judge defend
recommend appraise validate verify argue support prioritize optimize conclude decide design develop create formulate construct build compose
devise plan propose synthesize synthesise invent generate integrate assemble produce work summarise communicate function handle choose
formulate make prepare interpret determine estimate predict select comprehend recognize acquire gain learn apply demonstrate effectively""".split())


def split_outcomes(txt):
    """Numbered ('1.' / '1)' at line start, or inline '1 Implement ..., 2 Design ...'); otherwise a new outcome starts at
    each line that begins with a capitalised action verb."""
    txt = re.split(r"(?im)^\s*(?:unit\s+description|syllabus\b)", txt)[0]      # an unheaded syllabus table can follow the outcomes
    txt = STEM.sub("", txt, count=1)
    items = re.split(r"(?m)^\s*(?:CO\s?)?\d{1,2}\s*[.):]\s*", txt)
    if len(items) > 1:
        return [i for i in items if norm(i)]
    inline = [m for m in re.finditer(r"(?:(?<=[\s,;.])|^)(\d{1,2})[.)]?\s+(?=[A-Z][a-z]+)", txt)]
    seq, want = [], 1
    for m in inline:                                              # accept only a 1,2,3,... run
        if int(m.group(1)) == want:
            seq.append(m); want += 1
    if len(seq) >= 2 and seq[0].start() <= 2:
        cuts = [m.start() for m in seq] + [len(txt)]
        return [txt[m.end():c] for m, c in zip(seq, cuts[1:])]
    lines = []
    for line in re.sub("[●••▪◦]", "\n", txt).split("\n"):       # bullet glyphs, or bullets fused into one line by 2+ spaces
        lines += re.split(r"[ \t]{2,}(?=(?:" + "|".join(w.capitalize() for w in OUTCOME_VERBS) + r")\b)", line)
    out = []
    for line in lines:
        first = re.match(r"\s*([A-Z][a-z]+)", line)
        if first and first.group(1).lower() in OUTCOME_VERBS or not out:
            out.append(line)
        else:
            out[-1] += " " + line
    return out


def parse_outcomes(txt):
    out = []
    for it in split_outcomes(txt):
        t = norm(it)
        levels = sorted({int(x) for x in re.findall(r"\d", " ".join(KLEVEL.findall(t)))})
        t = norm(KLEVEL.sub("", t)).rstrip(",;")
        if len(t.split()) >= 3 and not re.match(r"(?i)^(after|upon|on|at the end|by the end|the student|students?)\b.*\b(able to|will be|would be|following)\W*$", t) \
                and not re.search(r"(?i)\b(able to|learn to|will be|would be|the following|is to)\W*$", t):
            out.append({"text": t, "k_levels": levels})
    return out


def tidy(s):
    """Book / lab text: drop separator lines, and page numbers left at the end."""
    s = norm(re.sub(r"_{3,}", " ", s or ""))
    return re.sub(r"(?:\s+\d{1,2})+$", "", s).strip()


def parse_block(block, src):
    text = "\n".join(l for l, _ in block)
    ays = Counter(a for _, a in block if a)
    if not ays:
        a = ay_of(src.rsplit("/", 1)[-1].replace("_", " "))
        ays = Counter({a: 1}) if a else ays
    head = text[:900]
    prog = re.search(r"(?is)program(?:me)?\s*[:\-–]?\s*(.*?)\n\s*semester", head)
    sem = re.search(r"(?i)semester\s*[:\-–]?[ \t]*([^\n]*(?:\n\s*[/&,][^\n]*)?)", head)
    name = re.search(r"(?is)\n\s*course(?:\s*/\s*module|\s+name|\s+title)?\s*[:\-–]?\s*(?!objectives?|outcomes?)(\S.*?)(?=\n\s*(?:module\s+|course\s+)?code|\n\s*teaching)", head)
    code = re.search(r"(?i)(?:module\s+|course\s+)?code[ \t]*[:\-–]?[ \t]*((?:[A-Z0-9][ \t]?){6,14})", head)
    if not name:
        return None
    cname = clean_name(name.group(1)) or norm(name.group(1))
    body = text[name.end():]
    sec = section_spans(body)
    units = parse_units(sec.get("syllabus", ""))
    if not units and sec.get("experiments"):              # lab courses list experiments instead of units
        units = [{"title": "Experiments", "text": norm(sec["experiments"]), "hours": None}]
    outs = parse_outcomes(sec.get("outcomes", ""))
    objectives = norm(sec.get("objectives", ""))
    if not (units or (outs and objectives)):
        return None
    ts = re.search(r"(?s)(\d)\s*\n\s*(\d)\s*\n\s*(\d)\s*\n\s*(\d{1,2})\s*\n\s*(?:Marks|-|\d)", body[:1500])
    # [ \t]* (not \s*) after the colon: \s* ate the newline of an empty field, so 'Course Objective' text became the prerequisite
    pre = re.search(r"(?is)pre\s*[-–]?\s*requisites?[ \t]*[:\-–]?[ \t]*(.*?)(?=\n\s*(?:course\s+)?objectives?|\n\s*(?:course\s+)?outcomes?)", body)
    code_s = re.sub(r"\s+", "", code.group(1)).upper() if code else ""
    return {
        "course_name": cname, "code": code_s if re.search(r"\d", code_s) else "", "course_key": key(cname),
        "program": norm(prog.group(1)) if prog else "", "semester": norm(re.sub(r"(?i)[^IVX0-9/&,\- to]|^\W+", "", sem.group(1))) if sem else "",
        "academic_year": ays.most_common(1)[0][0] if ays else "",
        "lecture_hrs": int(ts.group(1)) if ts else None, "practical_hrs": int(ts.group(2)) if ts else None,
        "tutorial_hrs": int(ts.group(3)) if ts else None, "credits": int(ts.group(4)) if ts else None,
        "prerequisites": norm(pre.group(1)) if pre else "", "objectives": objectives,
        "_prereq_fixed": bool(pre) and norm(OLD_PRE.search(body).group(1)) != norm(pre.group(1)),
        "outcomes": outs, "units": units, "text_books": tidy(sec.get("books", "")),
        "reference_books": tidy(sec.get("refs", "")), "lab_work": tidy(sec.get("lab", "")), "source": src,
    }


def read_pages(zip_path, rebuild):
    """{pdf path: [page text]} for every text-layer, non-exam PDF (from the cache if present)."""
    if CACHE.exists() and not rebuild:
        sf = CACHE.with_suffix(".stats.json")
        return {d["path"]: d["pages"] for d in map(json.loads, open(CACHE, encoding="utf-8"))}, Counter(json.loads(sf.read_text()) if sf.exists() else {})
    import fitz
    z = zipfile.ZipFile(zip_path)
    entries = [i for i in z.infolist() if i.filename.lower().endswith(".pdf") and not SKIP_NAME.search(i.filename)]
    stats, out = Counter({"pdfs_considered": len(entries)}), {}
    for n, info in enumerate(entries, 1):
        try:
            pages = [p.get_text() for p in fitz.open(stream=z.read(info), filetype="pdf")]
        except Exception:
            stats["unreadable_pdf"] += 1
            continue
        if sum(map(len, pages[:5])) < 200:
            stats["scanned_pdf_skipped"] += 1
            continue
        stats["text_pdf"] += 1
        out["/".join(p.strip() for p in info.filename.split("/"))] = pages
        if n % 300 == 0:
            print(f"read {n}/{len(entries)} pdfs", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        for p, pages in out.items():
            f.write(json.dumps({"path": p, "pages": pages}, ensure_ascii=False) + "\n")
    CACHE.with_suffix(".stats.json").write_text(json.dumps(dict(stats)))      # read-stage counts, reused when the cache is used
    return out, stats


def main(zip_path, rebuild=False):
    pdfs, stats = read_pages(zip_path, rebuild)
    seen, sources, raw = {}, defaultdict(list), {}
    for src, pages in sorted(pdfs.items()):          # sorted: the kept copy of a duplicated syllabus must not depend on zip order
        for block in split_blocks(clean_pages(pages)):
            stats["blocks_seen"] += 1
            rec = parse_block(block, src)
            if not rec:
                stats["blocks_unparsed"] += 1
                continue
            if not rec["academic_year"] or int(rec["academic_year"][:4]) < MIN_AY:
                stats["dropped_legacy_or_unknown_year"] += 1
                continue
            h = hashlib.md5(json.dumps([rec["course_key"], rec["code"], rec["academic_year"], rec["program"], rec["semester"],
                                       [u["text"] for u in rec["units"]], rec["objectives"]], sort_keys=True).encode()).hexdigest()
            sources[h].append(src)
            if h in seen:
                stats["exact_duplicates"] += 1
                continue
            rec["id"] = "nm" + h[:8]                 # content-derived, so ids stay valid when files are re-read in another order
            assert all(r["id"] != rec["id"] for r in seen.values()), "id collision"
            raw[rec["id"]] = "\n".join(l for l, _ in block)
            seen[h] = rec
    recs, fixed = [], sum(r.pop("_prereq_fixed") for r in seen.values())
    for h, rec in seen.items():
        rec["source_files"] = sorted(set(sources[h]))          # all of them: programme membership comes from these
        rec["n_source_files"] = len(set(sources[h]))
        recs.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / "interim/nmims_raw_blocks.jsonl", "w", encoding="utf-8") as f:   # audit aid, not committed
        for r in recs:
            f.write(json.dumps({"id": r["id"], "raw": raw[r["id"]]}, ensure_ascii=False) + "\n")
    fam = Counter(r["course_key"] for r in recs)
    st = {**stats, "courses": len(recs), "distinct_course_names": len(fam), "names_in_>=2_versions": sum(c >= 2 for c in fam.values()),
          "by_academic_year": dict(sorted(Counter(r["academic_year"] for r in recs).items())),
          "with_prerequisites_text": sum(bool(r["prerequisites"]) for r in recs),
          "prerequisite_fix_changed": fixed,
          "with_objectives": sum(bool(r["objectives"]) for r in recs),
          "with_outcomes": sum(bool(r["outcomes"]) for r in recs), "outcomes_total": sum(len(r["outcomes"]) for r in recs),
          "outcomes_with_k_level": sum(bool(o["k_levels"]) for r in recs for o in r["outcomes"]),
          "units_total": sum(len(r["units"]) for r in recs),
          "units_with_title": sum(bool(u["title"]) for r in recs for u in r["units"]),
          "units_with_hours": sum(u["hours"] is not None for r in recs for u in r["units"]),
          "with_code": sum(bool(r["code"]) for r in recs), "with_credits": sum(r["credits"] is not None for r in recs)}
    (RES / "parse_stats.json").write_text(json.dumps(st, indent=2))
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0] if a else ZIP, "--rebuild-cache" in sys.argv)
