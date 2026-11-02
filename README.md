# Revio AI Copilot — AI opportunity PoC for Cynbit

> AI/ML Engineer Intern technical assessment · Shibli NK · November 2026

**Opportunity:** add an AI copilot to **Revio**, Cynbit's review and reputation product, that triages every review by sentiment, topic and risk, escalates crises instantly, and drafts on-brand replies for a manager to approve.

| Deliverable | Where |
|---|---|
| AI Opportunity Report | [`docs/AI_Opportunity_Report.md`](docs/AI_Opportunity_Report.md) |
| Technical Approach Report | [`docs/Technical_Approach_Report.md`](docs/Technical_Approach_Report.md) |
| Proof of concept (Streamlit) | [`app.py`](app.py), [`revio/`](revio) |
| Evaluation and tests | [`scripts/evaluate.py`](scripts/evaluate.py), [`tests/`](tests) |

## What the demo does
- **Single review:** paste a review and star rating → sentiment with confidence, topics, crisis flag, 0–100 priority, and an editable reply draft.
- **Location dashboard:** upload a CSV (or use the bundled 72-review sample across four Cynbit client brands) → negative share, crisis alerts, sentiment by location, top complaint topics, a priority-sorted response queue, and a CSV export.
- **Reply engine:** a guard-railed LLM prompt when `OPENAI_API_KEY` is set; otherwise a free template engine, so the demo always runs.

## Quick start
```bash
git clone https://github.com/Squash2453/cynbit-revio-ai-copilot.git
cd cynbit-revio-ai-copilot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Optional: `cp .env.example .env`, add your OpenAI key, export it, and toggle **Use LLM** in the sidebar.

## Results
```
python scripts/evaluate.py
5-fold CV (text only)            accuracy 0.958 · macro-F1 0.962
Out-of-distribution hold-out     text only 0.473 → text + star prior 0.903 macro-F1
Crisis detection                 recall 1.00 · precision 1.00 (12/12)
pytest                           7 passed
```
The hold-out gap is discussed openly in the Technical Approach Report, along with the plan to fix it (a fine-tuned multilingual transformer on real Revio data).

## Project layout
```
app.py                    Streamlit UI
revio/analyzer.py         sentiment, aspects, crisis, priority
revio/replies.py          LLM prompt + template reply engine
scripts/build_dataset.py  builds data/sample_reviews.csv
scripts/evaluate.py       CV, hold-out and crisis metrics
data/                     sample and hold-out reviews (synthetic, labelled)
tests/                    pytest suite
docs/                     the two reports
```

## Notes
- The review data is synthetic and built around Cynbit's public client names for realism. No real customer data is used.
- Classification is CPU-only and runs in milliseconds; only reply drafting uses an LLM, and only when it's opted in.
