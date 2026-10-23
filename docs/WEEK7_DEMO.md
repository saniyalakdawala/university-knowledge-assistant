# Week 7 Integration Demo Script: Classroom Presentation Sequence

**Teams:** Group 4 (University Knowledge Assistant) × Group 9 (Curriculum Intelligence Engine)  
**Date:** October 2026  
**Audience:** Evaluation Committee & Classmates  

---

## Preparation Checklist Before Presenting
1. Open Windows PowerShell in the project directory:
   ```powershell
   cd c:\Projects\snlp_project
   uvicorn app:app --port 8000 --reload
   ```
2. Open your web browser at:
   ```
   http://localhost:8000
   ```
3. Verify that the top header shows:
   - `Group 4 RAG (SRB 2026): Ready (424 chunks)`
   - `Group 9 CE: Connected (Live API)` or `Group 9 CE: Stand-in (Offline Cache)`

---

## 7-Step Presentation Walkthrough

### Step 1: Architecture & Integration Introduction (1 Minute)
- **What to say:**
  > *"Good morning committee and classmates. Today Group 4 and Group 9 present our integrated AI University Application. Group 4 provides an authoritative Knowledge Assistant over the Student Resource Book 2026 for binding administrative and academic policies. Group 9 provides the Curriculum Intelligence Engine for analyzing syllabus prerequisites, overlaps, and course modernizations. Our unified interface coordinates between both systems seamlessly."*
- **What to show:**
  - Point out the clean dual-status indicators in the top right.
  - Highlight the architecture cards on the right sidebar showing the 4-stage query resolution pipeline.

---

### Step 2: SRB University Policy Question (1.5 Minutes)
- **Action:** Click the quick prompt chip **"📋 Attendance Rule"** or type:
  > *"What is the minimum attendance required to appear for the end-semester exam?"*
- **What to show:**
  - **Synthesized Answer:** *"Each student is required to maintain a minimum attendance of 80% in aggregate..."*
  - **Source Citation:** `SRB 2026 | Part I (University) | 3. Attendance Rules | PDF page 10`
  - **Evidence Quote:** Highlight the highlighted green callout displaying the verbatim sentence from the SRB.
  - **Verification Badges:** Confidence score (e.g. `0.78`), and `✓ Numbers Grounded`.
- **Key Takeaway:**
  > *"Notice that Group 4 does not hallucinate arbitrary percentages: our grounding verifier confirms that '80%' and 'aggregate' exist in the source document, and cites the exact PDF page for student verification."*

---

### Step 3: Group 9 Curriculum Intelligence Question (1.5 Minutes)
- **Action:** Click the quick prompt chip **"🧠 NLP Prerequisites"** or type:
  > *"What are the prerequisites for Natural Language Processing?"*
- **What to show:**
  - The router automatically classifies this as a curriculum query.
  - The interface highlights the **Group 9 Curriculum Intelligence** card.
  - The result details the prerequisite knowledge required: *Data Structures and Algorithms, Probability and Statistics, Machine Learning*.
  - Displays the 6 core units of the syllabus.
  - Highlights overlapping courses (e.g. *Deep Learning: 38.2% overlap*).
- **Key Takeaway:**
  > *"This query retrieves curriculum intelligence from Group 9's course graph. It clearly identifies prerequisite pathways before a student registers for advanced electives."*

---

### Step 4: Cross-Group Combined Interaction (2 Minutes)
- **Action:** Click the quick prompt chip **"🔗 ML + University Policy"** or type:
  > *"What are the prerequisites for Machine Learning, and what is the minimum attendance and passing grade required?"*
- **What to show:**
  - The unified coordinator detects that this question spans both domains:
    - **Route Taken:** `BOTH`
  - The **Synthesized Answer** integrates both dimensions.
  - **Left Card (Group 4):** Provides the official passing standard (e.g., 40% aggregate / grade requirements) and the mandatory 80% attendance rule from the SRB.
  - **Right Card (Group 9):** Provides the prerequisite subjects (*Linear Algebra, Probability, Python*) and syllabus units from the Curriculum Engine.
- **Key Takeaway:**
  > *"This demonstrates the core objective of Week 7: an integrated experience where students receive complete academic guidance without confusing advisory syllabus suggestions with binding administrative regulations."*

---

### Step 5: Unsupported / Out-of-Scope Safe Refusal (1 Minute)
- **Action:** Click the quick prompt chip **"🚫 Out-of-Scope Test"** or type:
  > *"How can I book a badminton court on campus?"*
- **What to show:**
  - Retrieval confidence falls to $\approx 0.12$, well below the dev-tuned gate of $0.46$.
  - The system displays a prominent red warning banner:
    *`⚠️ Refusal / Out-of-Scope: Retrieval confidence below policy gate (0.46). I could not find this in the Student Resource Book.`*
- **Key Takeaway:**
  > *"In our Week 6 benchmark evaluation, our system refused 20 out of 20 out-of-scope questions with 0 hallucinations. We preserve this strict safety standard in the production web interface."*

---

### Step 6: Fault Tolerance & Group 9 Offline Stand-in (1 Minute)
- **Explanation:**
  - Demonstrate what happens when Group 9's external service is stopped or in maintenance.
  - Show that the backend does not crash or return a generic 500 error.
  - Show the **Reference Stand-in** badge with transparent messaging explaining that cached reference syllabus data is displayed.
- **Key Takeaway:**
  > *"Our integration client is designed for resilience: failure of a partner API never brings down the university knowledge portal."*

---

### Step 7: Responsible AI Summary & Q&A (1 Minute)
- **Closing remarks:**
  > *"To summarize: Group 4 provides verifiable, grounded answers over official university policy with page citations and evidence quotes. Group 9 provides advisory curriculum intelligence. The interface enforces clear visual boundaries so students always know the authority and source of their information."*
- Invite questions from the committee.

---

## Live vs. Stand-in Verification Status

| Step / Feature | Verification Status | Notes |
|---|---|---|
| SRB PDF Extraction & Parsing | **Live Verified** | 424 chunks parsed from `data/SRB_2026.pdf`. |
| BM25 + MiniLM Hybrid Retrieval | **Live Verified** | Passes all 11 unit & regression tests. |
| Cross-Encoder Reranking | **Live Verified** | `cross-encoder/ms-marco-MiniLM-L-6-v2` active. |
| Grounding Check & Refusal Gate | **Live Verified** | Rejects ungrounded numbers & out-of-scope queries. |
| Group 9 Local REST Call | **Documented Contract** | Connects to `POST /check-prerequisites` & `/find-overlaps` on port 8000 when active. |
| Group 9 Reference Fallback | **Live Verified** | Activated whenever Group 9 server is offline/unreachable. |
