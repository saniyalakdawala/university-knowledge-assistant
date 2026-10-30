// ==========================================================================
// University Knowledge Assistant — Minimalist Frontend Logic (Vanilla JS)
// Single-bubble answers, verified citations, direct Groq grounded responses
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  const chatStream = document.getElementById("chat-stream");
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const submitBtn = document.getElementById("submit-btn");

  // Auto-resize input textarea and handle Enter to submit
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!submitBtn.disabled && questionInput.value.trim()) {
        chatForm.requestSubmit();
      }
    }
  });



  // Handle Form Submission
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    // Append User Message Bubble
    appendUserMessage(question);
    questionInput.value = "";
    submitBtn.disabled = true;

    // Append Loading Placeholder
    const loadingElem = appendLoadingPlaceholder();
    scrollToLatest();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          route: "auto"
        })
      });

      loadingElem.remove();

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: "Server error" }));
        appendErrorMessage(
          errData.detail || "Unable to complete request. Please verify connection."
        );
        return;
      }

      const data = await res.json();
      appendAssistantResponse(data);
    } catch (err) {
      loadingElem.remove();
      appendErrorMessage(
        `Connection error: Could not reach the University Assistant service (${err.message}).`
      );
    } finally {
      submitBtn.disabled = false;
      questionInput.focus();
      scrollToLatest();
    }
  });

  // ----------------- Message Rendering -----------------

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    chatStream.appendChild(row);
  }

  function appendLoadingPlaceholder() {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    row.innerHTML = `
      <div class="assistant-bubble">
        <div class="loading-box">
          <div class="shimmer-line" style="width: 45%;"></div>
          <div class="shimmer-line" style="width: 85%;"></div>
          <div class="shimmer-line" style="width: 65%;"></div>
        </div>
      </div>
    `;
    chatStream.appendChild(row);
    return row;
  }

  // Display each answer ONLY ONCE in a clean chat bubble
  function appendAssistantResponse(data) {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    const bubble = document.createElement("div");
    bubble.className = "assistant-bubble";

    const srb = data.srb;
    const course = data.course;
    const curr = data.curriculum;
    const ragResults = [course, srb].filter(Boolean);
    const answered = ragResults.filter((r) => !r.abstained);

    // 1. Direct answer text
    const answerDiv = document.createElement("div");
    answerDiv.className = "bubble-answer";
    let mainText = data.synthesized_answer || "No answer available.";
    if (ragResults.length && !answered.length && !curr) {
      mainText = course
        ? "I could not find this in the new course's syllabus or in the Student Resource Book 2026."
        : "I could not find this information in the Student Resource Book 2026.";
    }
    answerDiv.textContent = mainText;
    bubble.appendChild(answerDiv);

    // 2. Evidence quote(s) and one rubric card per knowledge base that answered
    answered.forEach((r) => {
      const quoteText = (r.evidence_quote || "").trim();
      if (quoteText.length > 20 && quoteText !== mainText.trim()) {
        const quoteDiv = document.createElement("blockquote");
        quoteDiv.className = "evidence-quote";
        quoteDiv.textContent = `“${quoteText}”`;
        bubble.appendChild(quoteDiv);
      }
      const score = typeof r.confidence === "number" ? r.confidence : 0;
      const level = score >= 0.7 ? "high" : score >= 0.46 ? "medium" : "low";
      const label = level === "high" ? "High Confidence" : level === "medium" ? "Moderate Confidence" : "Low Confidence";
      const page = r.cited_page ? (course === r ? ` · Section ${r.cited_page}` : ` · Page ${r.cited_page}`) : "";
      bubble.appendChild(rubricCard(`${r.document}${page}`, r.section || "", `${score.toFixed(2)} · ${label}`, level));
    });

    if (ragResults.length && !answered.length && !curr) {
      const r = ragResults[ragResults.length - 1];
      const score = typeof r.confidence === "number" ? r.confidence : 0;
      bubble.appendChild(rubricCard(r.document, r.section ? `${r.section} (closest match)` : "Not located", `${score.toFixed(2)} · Below policy threshold`, "low"));
    }

    // 3. Group 9 live API / stand-in: no invented confidence
    if (curr) {
      const name = curr.course ? (curr.course.course_name || curr.course.name || "") : "";
      const docText = curr.is_fallback ? "Curriculum reference stand-in (Group 9 offline)" : "Group 9 Curriculum Intelligence API (live)";
      bubble.appendChild(rubricCard(docText, name || "Course catalogue", curr.is_fallback ? "n/a · stand-in, not model output" : "see Group 9 response", curr.is_fallback ? "low" : "medium"));
    }

    row.appendChild(bubble);
    chatStream.appendChild(row);
  }

  function rubricCard(docText, sectionText, scoreDisplay, scoreLevel) {
    const card = document.createElement("div");
    card.className = "retrieval-rubric-card";
    card.innerHTML = `
      <div class="rubric-grid">
        <div class="rubric-item">
          <div class="rubric-label">Supporting Document</div>
          <div class="rubric-val">${escapeHtml(docText)}</div>
        </div>
        <div class="rubric-item">
          <div class="rubric-label">Source Section</div>
          <div class="rubric-val">${escapeHtml(sectionText)}</div>
        </div>
        <div class="rubric-item">
          <div class="rubric-label">Confidence / Retrieval Score</div>
          <div class="rubric-val">
            <span class="score-pill score-${scoreLevel}">${escapeHtml(scoreDisplay)}</span>
          </div>
        </div>
      </div>
    `;
    return card;
  }

  function appendErrorMessage(msg) {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    row.innerHTML = `
      <div class="assistant-bubble">
        <div class="bubble-answer" style="color: #991b1b;">
          ${escapeHtml(msg)}
        </div>
      </div>
    `;
    chatStream.appendChild(row);
  }

  function scrollToLatest() {
    window.scrollTo({
      top: document.body.scrollHeight,
      behavior: "smooth"
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
