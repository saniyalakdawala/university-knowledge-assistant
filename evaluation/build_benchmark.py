"""Build evaluation/benchmark.json from the gold facts annotated in Weeks 2-5.

Every answerable question points at one or more gold facts (answer key, verbatim evidence span, PDF page), so relevance is
judged against the SRB text and not against any pipeline's chunks. The questions below were written by hand; none of them
reuses the wording of the gold fact's original question unless the category is `direct`.

The split is by fact: a fact in `dev` never appears in `test`, in any wording.    python evaluation/build_benchmark.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 42

PARAPHRASED = {
    "Q003": "What is the time limit for flagging an error once the monthly record of my class presence comes out?",
    "Q004": "At what point does the college notify my family about how regularly I come to lectures?",
    "Q005": "I was unwell and have just rejoined; what is the cut-off for handing in my doctor's note?",
    "Q006": "What is the consequence of falling short of the required presence in a course?",
    "Q008": "What is the largest concession on missed lectures for illness or for sports and cultural participation?",
    "Q010": "What is the pass mark for subjects evaluated purely through continuous internal work?",
    "Q012": "Will bonus marks ever be added to continuous assessment scores?",
    "Q014": "Having plagiarised coursework and been permitted to redo it, what is the ceiling on my score?",
    "Q017": "For what period can I use the free online learning subscription from the university?",
    "Q019": "How long before the paper starts do I need to be in the examination room?",
    "Q020": "Will I be let in if I turn up over half an hour after the paper has begun?",
    "Q023": "What is the sanction if somebody sits a paper on behalf of another person?",
    "Q026": "What additional writing duration do candidates who use a writer receive?",
    "Q029": "Is there any mark concession for learners diagnosed with reading difficulties?",
    "Q030": "What is the window after results to request a copy of my evaluated script?",
    "Q033": "My rechecked script differs from the original score by over a fifth. What follows?",
    "Q037": "What share of a cohort earns the academic honour roll?",
    "Q041": "Which website do I use for central government study funding?",
    "Q042": "Do female engineering students have a dedicated funding scheme?",
    "Q046": "What is the cap on study gaps I can take during my degree?",
    "Q050": "What is the result of leaving tuition dues unpaid past the penalty window?",
    "Q052": "Which refreshments are okay in the lecture room?",
    "Q055": "Who is the final authority when I contest a misconduct ruling?",
    "Q057": "Must I get clearance before leaving town in the middle of a semester?",
    "Q062": "What is the daily penalty on overdue loans from the university reading room?",
    "Q064": "Can outstanding dues to the reading room delay my graduation certificate?",
    "Q065": "What cloud drive allocation does each learner receive?",
    "Q066": "What notice does tech support need for additional network capacity during a function?",
    "Q067": "What login ID am I given by default on the learning management system?",
    "Q079": "Up to how many failed subjects can a BTech student carry and still get promoted?",
    "Q081": "How much do I pay per paper to reappear?",
    "Q085": "How is a no-show at the final paper recorded in my results?",
    "Q087": "What is the procedure for skipping on-campus recruitment?",
    "B005": "What is the consequence of being found intoxicated within college grounds?",
    "B007": "May I request a professor to push back a due date?",
    "B016": "What is the minimum class strength for running an optional subject?",
    "B029": "What sanction applies when a whole room cheats together?",
    "B047": "What is the longest leave of absence from studies permitted?",
    "B051": "What if I vanish from college for three weeks with no notice?",
    "B060": "I noticed a hacking attempt; whom do I write to?",
    "B076": "Which retired judge handles student complaints as an independent authority?",
    "B081": "In a blaze, is the elevator safe to take?",
    "B089": "What is the upper limit in years to finish an engineering degree?",
    "B055": "May I carry journals and consultation-only volumes out of the reading room?",
    "B026": "Could I be turned away from a paper for not having my university identification?",
}

INFORMAL = {
    "Q001": "bro how much attendance do i need to sit for finals",
    "Q002": "attendance sheet says im absent but i was there wht do i do",
    "Q007": "got a reexam today, do i get marked present for my normal lectures",
    "Q011": "is there reexam for ica only subjects",
    "Q021": "can i wear my apple watch in the exam",
    "Q025": "tried to bribe a prof for marks what happens",
    "Q027": "need a writer for my exam how early to sort that out",
    "Q032": "can i get reval without taking photocopy",
    "Q035": "how many grace marks can i get if i fail",
    "Q038": "our batch is 30 ppl how many prizes do we get",
    "Q045": "cant afford fees can i take a yr off",
    "Q047": "readmission fees kitna hai",
    "Q049": "how many times can i redo the same yr",
    "Q053": "forgot my id today what do i do",
    "Q063": "used my phone in library what happens",
    "Q068": "pendrive in smartboard allowed?",
    "Q069": "site is blocked on college wifi how to unblock",
    "Q073": "hostel deposit refund how long",
    "Q078": "whats the passing marks for a subject",
    "Q080": "got 3 KTs can i go to next year",
    "Q082": "missed reexam form deadline how much extra",
    "Q086": "how long is the btech internship",
    "Q089": "when do we do community service thing",
    "Q090": "how do they pick the CR",
    "B002": "how often r we supposed to check the portal",
    "B003": "lost my id card now what",
    "B004": "can i record lectures on my phone",
    "B013": "mid sem min duration?",
    "B018": "failed an only ica subject can i still give my other finals",
    "B020": "caught cheating in ica, can i still make deans list",
    "B025": "how much time do profs give for coursera per week",
    "B027": "can i leave exam in the last 10 mins",
    "B028": "caught with phone in exam, how screwed am i",
    "B032": "caught cheating a second time whats the punishment",
    "B035": "last date for reval?",
    "B044": "will my scholarship carry over if i defer admission",
    "B053": "can i eat chips in the library",
    "B054": "lost a library book what do i pay",
    "B065": "forgot portal password help",
    "B067": "do profs know who gave them bad feedback",
    "B077": "how long does the ombudsman take",
    "B093": "can CR cancel a lecture",
    "B099": "can i improve my ica marks somehow",
    "B102": "mtech year back students get how many reexam tries",
    "B104": "dvoc guy with a year back can he join btech fy",
}

MULTI = [
    ("Q001", "Q006", "What is the minimum attendance to write the exam, and what happens if I go below it?"),
    ("Q009", "B013", "What are the minimum durations of term-end and mid-term exams?"),
    ("Q010", "Q011", "What's the pass mark for an only-ICA course and can I take a re-exam if I fail it?"),
    ("Q019", "Q020", "How early must I reach the hall, and will I be allowed in if I'm more than half an hour late?"),
    ("Q023", "Q024", "What is the punishment for impersonation, and what happens to someone caught using unfair means more than twice?"),
    ("Q030", "B035", "What are the deadlines to apply for a photocopy of my script and for revaluation?"),
    ("Q046", "B047", "How many academic breaks can I take and what is the maximum length of one?"),
    ("Q047", "Q049", "What is the re-admission fee and how many times can I take re-admission in the same year?"),
    ("Q061", "Q062", "How many books can I borrow from the library, and what is the fine if I return them late?"),
    ("Q079", "Q080", "How many F grades can I carry into the next year, and what happens if I have more than that?"),
    ("Q081", "Q082", "What is the re-exam fee per subject and what is the late fee if I miss the deadline?"),
    ("B094", "B095", "How long are the MBA Tech technical internship and the management internship?"),
    ("Q038", "B039", "How many prizes are given for a batch of 30 students, and how many for a batch above 60?"),
    ("B069", "Q071", "What travel reimbursement do Grade A contest participants get, and what is the condition for Grade B?"),
    ("B072", "B073", "How many representatives does each school send to NUSC, and what are its executive positions?"),
    ("B086", "B087", "Who is the Controller of Examinations and who is the University Librarian?"),
    ("B091", "B092", "Who is the convener of the MPSTME disciplinary committee and who chairs the Woman Grievance Redressal Committee?"),
    ("B107", "B108", "Who is the Dean of MPSTME, and who heads the Data Science department?"),
    ("Q065", "B059", "How much OneDrive space do I get and how long should my password be?"),
    ("B063", "B065", "What is the student portal address and what do I do if I forget the password?"),
    ("B080", "B081", "What number do I dial for the fire brigade, and should I use the lift during a fire?"),
    ("Q073", "Q074", "How long does a hostel deposit refund take, and how soon must activity expenses be settled?"),
    ("Q012", "Q035", "Can grace marks be added to ICA, and how many grace marks can a failing student get?"),
    ("Q086", "B094", "How long is the B Tech internship and how long is the MBA Tech technical internship?"),
    ("B097", "B098", "When are remedial classes held before re-exams, and how many re-exams are held per course in a year?"),
    ("Q016", "Q015", "Can first-year undergraduates take CBCS courses, and how many interdisciplinary courses can I take?"),
    ("B043", "Q044", "Who can apply for admission deferment, and can a second-year student get it?"),
    ("Q031", "Q033", "What is the retotalling deadline, and what happens if revaluation changes my marks by more than 20%?"),
    ("B058", "B057", "Which database does the finance lab use and which plagiarism software does NMIMS subscribe to?"),
    ("B084", "Q069", "Is gambling content blocked on campus, and how do I get a blocked site opened?"),
]

UNANSWERABLE_EXTRA = [
    "How can I book a badminton court on campus?", "Does the university provide a laundry service?",
    "What is on the cafeteria menu this week?", "Is there an ATM inside the campus?", "Which cab services operate near the campus?",
    "What is the NAAC grade of the university?", "What is the university's QS world ranking?", "Is there a swimming pool on campus?",
    "What is the monthly bus fare to Andheri?", "What are the opening hours of the stationery shop?", "Can I bring my pet to campus?",
    "Who is the football coach of the university team?", "What is the fee for a duplicate marksheet?",
    "where can i get good biryani near college", "whats the last local train after evening lectures",
    "does college give free laptops to students", "any good pg near campus under 15k",
    "who won the inter college cricket tournament", "what is the salary of the dean",
    "How many solar panels are installed on the MPSTME building?", "bro is there a nap room",
    "Which bank gives the lowest interest for students of NMIMS?",
]


def main():
    gold = json.loads((HERE / "gold_facts.json").read_text(encoding="utf-8"))
    facts = gold["facts"]
    used = set(PARAPHRASED) | set(INFORMAL) | {f for a, b, _ in MULTI for f in (a, b)}
    rng = random.Random(SEED)
    direct_pool = sorted(set(facts) - used)
    direct = rng.sample(direct_pool, min(45, len(direct_pool)))

    # Facts are grouped so that those joined by a multi-condition question stay on one side of the split.
    parent = {f: f for f in facts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in MULTI:
        parent[find(a)] = find(b)
    groups = sorted({find(f) for f in facts})
    rng.shuffle(groups)
    dev_groups = set(groups[: int(0.35 * len(groups))])
    split = lambda fid: "dev" if find(fid) in dev_groups else "test"

    items = []
    for fid in direct:
        items.append({"category": "direct", "question": facts[fid]["question"], "facts": [fid], "split": split(fid)})
    for fid, q in PARAPHRASED.items():
        items.append({"category": "paraphrased", "question": q, "facts": [fid], "split": split(fid)})
    for fid, q in INFORMAL.items():
        items.append({"category": "informal", "question": q, "facts": [fid], "split": split(fid)})
    for a, b, q in MULTI:
        items.append({"category": "multi_condition", "question": q, "facts": [a, b], "split": split(a)})
    unanswerable = gold["out_of_scope"] + UNANSWERABLE_EXTRA
    rng.shuffle(unanswerable)
    cut = int(0.35 * len(unanswerable))
    for i, q in enumerate(unanswerable):
        items.append({"category": "unanswerable", "question": q, "facts": [], "split": "dev" if i < cut else "test"})
    for i, it in enumerate(items):
        it["id"] = f"BM{i:03d}"
    (HERE / "benchmark.json").write_text(json.dumps({"seed": SEED, "questions": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(Counter((i["split"], i["category"]) for i in items))


if __name__ == "__main__":
    main()
