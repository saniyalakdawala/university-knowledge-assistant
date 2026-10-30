# Results

All numbers come from `notebooks/Group4_Week6.ipynb` (Week 5 evaluation, re-run unchanged in Week 6).
Test A = 54 answerable + 6 out-of-scope questions (Week 2 held-out set). Test B = 108 answerable + 14 out-of-scope (blind set).
Every setting was tuned on the development set only (36 answerable + 19 out-of-scope).

- `metrics_testAB.csv`: baseline vs improved systems on Test A and Test B.
- Details, ablation, robustness and error cases: `docs/evaluation.md`.

The final API (`app.py`) uses the v3 configuration that was evaluated: hybrid retrieval (beta 0.6), cross-encoder (alpha 0.7),
2 passages to the LLM, confidence = mean of the three retrieval signals, refusal gate 0.24, grounding min support 0.5,
temperature 0. Chat memory (follow-up rewriting) was evaluated in the notebook but is not exposed in the API.
