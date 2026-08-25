# Customer Support AI Employee

A chat-based Tier-1 support assistant for a fictional SaaS product. It classifies
incoming messages into **Billing / Technical / Account Access**, answers FAQs by
**retrieving grounded answers** from a small knowledge base, and **escalates to a
human agent with a visible, specific reason** whenever it isn't confident enough
to answer safely.

Live demo: _(add your deployed Streamlit URL here after deployment)_

---

## 1. Architecture

```
User message
     │
     ▼
┌─────────────────────┐   keyword-weighted, deterministic
│   classifier.py      │   → category (Billing / Technical / Account Access)
└─────────┬────────────┘     + confidence score + which words matched
          │
          ▼
┌─────────────────────┐   TF-IDF + cosine similarity over knowledge_base.json
│   retrieval.py        │   → top-k matching KB articles + similarity scores
└─────────┬────────────┘     (extractive — the answer is always the literal
          │                   KB text, so it cannot contain invented facts)
          ▼
┌─────────────────────┐   plain if/else rules, fully logged as a trace
│  decision_engine.py   │   → ANSWER (grounded) / ESCALATE (with reason) / REJECT
└─────────┬────────────┘     (this is the "inspectable reasoning" layer —
          │                   nothing here is hidden inside an LLM prompt)
          ▼
┌─────────────────────┐   optional: only rewords the already-grounded
│  llm_enhancer.py      │   answer via Groq's free API — never adds new facts.
└─────────┬────────────┘   Disabled by default (no key needed to run/demo).
          │
          ▼
┌─────────────────────┐
│      app.py            Streamlit chat UI: shows the reply, a status badge
│  (Streamlit chat UI)    (ANSWERED/ESCALATED/REJECTED), source attribution,
└─────────────────────┘   and an expandable "inspect reasoning trace" panel.
```

**Why this design (documented assumptions):**

1. **No LLM is required to run the app at all.** Classification is a transparent
   keyword-weighted rule engine; retrieval is TF-IDF cosine similarity over the
   knowledge base; answers are the literal KB text (extractive, not generated).
   This guarantees ₹0 cost, zero external dependencies/API keys, zero
   hallucination risk, and 100% inspectable logic — which is explicitly what the
   brief asks for ("make the reasoning/decision logic inspectable rather than
   hiding everything inside one LLM prompt"). The brief's RAG requirement
   ("answer FAQs using retrieval over the provided knowledge base") is satisfied
   by the retrieval step; it does not require a generative model.
2. **An LLM is still supported as a pure enhancement.** If a free Groq API key
   is present in the environment, the final grounded answer is reworded for a
   more natural tone. The LLM is only ever allowed to reword text already
   selected by retrieval — it cannot introduce new claims — and any failure
   (no key, network error, rate limit) silently falls back to the raw grounded
   answer, so the app never breaks because of it.
3. **Escalation is deliberately conservative.** If the classifier finds *zero*
   keyword overlap with any of the three categories, the message is treated as
   out-of-scope and escalated — even if TF-IDF happens to find a superficial
   text overlap with some KB article (this was caught by testing: "help me plan
   a birthday party" shares the word "plan" with a subscription-plan FAQ and
   would otherwise be answered incorrectly). See `decision_engine.py` for the
   exact, commented rule order.
4. **Single-file Streamlit app**, no separate frontend/backend, no database, no
   vector DB, no auth, no payments — per the brief's "avoid over-engineering"
   rule for a 15-article knowledge base.

---

## 2. Project structure

```
supervity-support-ai/
├── app.py                    # Streamlit chat UI (entry point)
├── classifier.py              # Rule-based ticket classifier
├── retrieval.py                # TF-IDF retrieval over the knowledge base
├── decision_engine.py           # Answer / escalate / reject decision logic
├── llm_enhancer.py               # Optional Groq-based answer rephrasing
├── data/
│   ├── knowledge_base.json        # 15 synthetic FAQ articles (5 per category)
│   └── mock_tickets.json           # 10 synthetic test tickets incl. edge cases
├── tests/
│   └── test_pipeline.py             # 16 unit/integration tests (pytest)
├── .streamlit/config.toml            # Headless server + theme config
├── requirements.txt                   # Pinned, verified dependency versions
├── .env.example                        # GROQ_API_KEY (optional) template
├── .gitignore
└── README.md
```

---

## 3. Run it locally

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd supervity-support-ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. No API key is required — it runs
fully grounded/extractive out of the box.

### Optional: enable natural-language rephrasing (still free)
1. Create a free key at https://console.groq.com/keys (no billing required).
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY=your-key`, **or** export
   it directly: `export GROQ_API_KEY=your-key` (macOS/Linux) /
   `set GROQ_API_KEY=your-key` (Windows).
3. Re-run `streamlit run app.py`. The sidebar will show "LLM rephrasing:
   enabled (Groq)".

### Run the tests

```bash
pytest tests/ -v
```

All 16 tests should pass. They cover: all three categories, low-confidence
input, out-of-scope input, empty/whitespace input, overly long input, and a
simulated knowledge-base/retrieval failure.

---

