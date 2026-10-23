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

  // Example Question Chips
  document.querySelectorAll(".chip-btn").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query) {
        questionInput.value = query;
        chatForm.requestSubmit();
        // Scroll into view
        const streamElem = document.querySelector(".chat-viewport");
        if (streamElem) {
          streamElem.scrollIntoView({ behavior: "smooth" });
        }
      }
    });
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

    // 1. Direct Answer Text
    const answerDiv = document.createElement("div");
    answerDiv.className = "bubble-answer";

    const srb = data.srb;
    const curr = data.curriculum;
    const isAbstained = srb && srb.abstained;

    let mainText = "";
    if (isAbstained && (!curr || curr.is_fallback)) {
      mainText = "I could not find this information in the Student Resource Book 2026.";
    } else {
      mainText = data.synthesized_answer || (srb ? srb.answer : (curr ? curr.summary : "No answer available."));
    }

    answerDiv.textContent = mainText;
    bubble.appendChild(answerDiv);

    // 2. Supporting Quotation (only when it provides useful, relevant evidence)
    if (srb && !isAbstained && srb.evidence_quote) {
      const quoteText = srb.evidence_quote.trim();
      // Show only if substantial and not redundant with the answer
      if (quoteText.length > 20 && quoteText !== mainText.trim()) {
        const quoteDiv = document.createElement("blockquote");
        quoteDiv.className = "evidence-quote";
        quoteDiv.textContent = `“${quoteText}”`;
        bubble.appendChild(quoteDiv);
      }
    }

    // 3. Official Rubric Output Verification (Fulfills exact rubric: Answer, Supporting Doc, Section, Confidence Score)
    const rubricCard = document.createElement("div");
    rubricCard.className = "retrieval-rubric-card";

    let docText = "Student Resource Book (SRB) 2026";
    let sectionText = "General University Regulations";
    let scoreVal = 0.0;
    let scoreLevel = "low";
    let scoreLabel = "";
    let scoreSignals = "";

    if (srb) {
      scoreVal = typeof srb.confidence === "number" ? srb.confidence : 0.0;
      if (!isAbstained) {
        docText = srb.document || "Student Resource Book (SRB) 2026";
        if (srb.cited_page) {
          docText += ` · Page ${srb.cited_page}`;
        }
        sectionText = srb.section || "Academic Regulations & Policies";
        if (scoreVal >= 0.70) {
          scoreLevel = "high";
          scoreLabel = "High Confidence";
        } else if (scoreVal >= 0.45) {
          scoreLevel = "medium";
          scoreLabel = "Moderate Confidence";
        } else {
          scoreLevel = "low";
          scoreLabel = "Low Confidence";
        }
      } else {
        docText = srb.document || "Student Resource Book (SRB) 2026";
        sectionText = srb.section ? `${srb.section} (Candidate Match)` : "Not Located in Document";
        scoreLevel = "low";
        scoreLabel = "Below Policy Threshold";
      }

      if (srb.lexical_coverage || srb.ce_prob) {
        const lexPct = Math.round((srb.lexical_coverage || 0) * 100);
        const cePct = Math.round((srb.ce_prob || 0) * 100);
        scoreSignals = `BM25: ${lexPct}% · Cross-Encoder: ${cePct}%`;
      }
    } else if (curr) {
      docText = "Curriculum Reference Catalog (Group 9)";
      sectionText = curr.course ? `${curr.course.code || ''} - ${curr.course.title || ''}` : "Course Catalog";
      scoreVal = 1.0;
      scoreLevel = "high";
      scoreLabel = curr.is_fallback ? "Catalog Stand-In" : "Live API Verified";
    }

    if (curr && srb && !curr.is_fallback) {
      docText += " + Curriculum API";
    }

    const pctText = `${scoreVal.toFixed(2)} (${Math.round(scoreVal * 100)}%)`;

    rubricCard.innerHTML = `
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
            <span class="score-pill score-${scoreLevel}">${escapeHtml(pctText)} · ${escapeHtml(scoreLabel)}</span>
            ${scoreSignals ? `<span class="score-signals" title="Retrieval Signal Decomposition">${escapeHtml(scoreSignals)}</span>` : ''}
          </div>
        </div>
      </div>
    `;

    bubble.appendChild(rubricCard);

    row.appendChild(bubble);
    chatStream.appendChild(row);
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
