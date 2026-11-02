# Technical Approach Report: Revio AI Copilot

**Candidate:** Shibli NK — AI/ML Engineer Intern assessment
**Date:** 2 November 2026

---

## 1. Goal
Turn a raw stream of customer reviews into (a) a prioritised response queue with crisis alerts, (b) reply drafts a manager can approve in one click, and (c) per-location insight, all cheaply enough to run for every Revio customer.

## 2. Architecture

```
 Google / Instagram / Facebook / Justdial
              │  (Revio sync connectors, existing)
              ▼
   ┌──────────────────────┐
   │ Ingest & normalise   │  dedupe, language detect, strip PII
   └─────────┬────────────┘
             ▼
   ┌──────────────────────┐   CPU, ~ms per review
   │ Analyzer             │──► sentiment (TF-IDF + LR, blended with star prior)
   │ (revio/analyzer.py)  │──► aspects   (taxonomy, later a multi-label model)
   │                      │──► crisis    (high-recall regex rules, later a classifier)
   │                      │──► priority  (0–100)
   └─────────┬────────────┘
             ▼
   ┌──────────────────────┐   LLM only here
   │ Reply drafter        │──► gpt-4o-mini / Gemini Flash with a guard-railed prompt
   │ (revio/replies.py)   │──► template fallback (free, deterministic)
   └─────────┬────────────┘
             ▼
   Human approval queue (crisis = mandatory)  →  publish via platform API
             ▼
   Location dashboard + weekly insight email
```

**Design choice:** a small classical model for *classification* and an LLM only for *generation*. Classification runs on every review, so it must be fast, cheap and auditable. Generation quality is where an LLM clearly beats templates.

## 3. Models

| Task | PoC model | Why | Production upgrade |
|---|---|---|---|
| Sentiment | TF-IDF (1–2-grams, sublinear TF) + Logistic Regression, class-balanced; probabilities blended 65/35 with a star-rating prior | Trains in < 1 s, interpretable weights, no GPU | Fine-tune `distilbert-base-multilingual` or `xlm-roberta-base` on labelled Revio reviews (Hindi and Hinglish support) |
| Aspects | Keyword taxonomy per vertical | Zero training data needed; transparent to clients | Multi-label classifier on sentence embeddings (e.g. `multilingual-e5-small`) + LLM labelling to bootstrap |
| Crisis | Regex patterns across four risk categories; overrides sentiment to negative and priority to ≥ 90 | Recall matters most; rules are easy to audit | Rules **OR** a fine-tuned classifier (keep rules as a safety net) |
| Priority | Weighted score: sentiment × confidence, low stars, number of topics, crisis | Simple and explainable to managers | Learn weights from which reviews managers actually answer first |
| Reply | `gpt-4o-mini` (configurable), temperature 0.4 | Low cost, strong tone control | A/B test against Gemini Flash; cache per-brand style examples |

## 4. Data pipeline

1. **Ingest:** Revio's existing connectors pull reviews (ID, platform, location, rating, text, timestamp) into Postgres.
2. **Normalise:** Unicode cleanup, language detection, dedupe across platforms, and PII masking (phone numbers, emails) before anything reaches the LLM.
3. **Analyse:** batch job (or a queue worker) writes sentiment, aspects, crisis, priority and model version per review.
4. **Label loop:** every manager edit or override (e.g. changing sentiment, dismissing a crisis) is stored as a label. Low-confidence predictions (< 0.6) are sampled for labelling (active learning).
5. **Retrain:** weekly, with a per-vertical evaluation gate. Deploy only if macro-F1 does not drop and crisis recall stays ≥ 0.95.
6. **Serve:** FastAPI endpoint `/analyze` and `/draft-reply`. The Streamlit app in this repo is the demo front end.

**PoC data:** `scripts/build_dataset.py` generates 72 labelled reviews for four real Cynbit client brands (clinic, retail, restaurant, SaaS) from hand-written fragments, and `data/holdout_reviews.csv` holds 18 reviews written in *different* wording to test generalisation. These are synthetic, because a candidate has no access to real Revio data.

## 5. Prompt engineering

The reply prompt (`revio/replies.py`, `SYSTEM_PROMPT`) is built from analyzer output rather than raw text alone:

- **Grounding:** the prompt receives sentiment, aspects and crisis reasons, so the reply addresses the *specific* complaints.
- **Constraints:** 40–80 words, no emojis, no invented discounts or promises, one apology, and the conversation moves offline to the brand's contact.
- **Crisis mode:** no admission of liability and no medical or legal detail in public; a senior manager follows up today. Crisis drafts are never auto-published.
- **Brand voice:** business name and contact are injected. In production, add 3–5 approved past replies per brand as few-shot examples.
- **Fallback:** any API error (quota, network) falls back to the template engine, so the product never blocks.
- **Evaluation:** an LLM-as-judge rubric (specificity, tone, policy compliance) on 100 sampled replies per release, plus manager edit-distance as an online metric.

## 6. Results (PoC)

From `python scripts/evaluate.py`:

| Test | Result |
|---|---|
| Sentiment, 5-fold CV on sample (text only) | accuracy 0.958, macro-F1 0.962 |
| Sentiment, out-of-distribution hold-out, text only | macro-F1 0.473 |
| Sentiment, out-of-distribution hold-out, **text + star prior** | **macro-F1 0.903** |
| Crisis detection (sample and hold-out) | recall 1.00, precision 1.00 (12/12) |
| Unit tests | 7/7 passing |

**Honest reading:** the cross-validation score is inflated because the sample is built from a limited fragment pool. The hold-out shows the text-only model does **not** generalise to new wording yet (0.47), which is exactly why the PoC blends in the star rating (0.90) and why the production plan replaces TF-IDF with a fine-tuned multilingual transformer trained on real Revio reviews. Crisis rules work on these examples, but they will need expansion (e.g. "police" can appear in harmless reviews), so they stay paired with a model plus human review.

## 7. Execution plan

| Phase | Weeks | Deliverable | Exit metric |
|---|---|---|---|
| 0. PoC (this repo) | — | Streamlit demo, eval, tests | Stakeholder sign-off |
| 1. Data | 1–2 | Export about 5k historical Revio reviews; label 1.5k with LLM pre-labels + human check | Labelled set with κ ≥ 0.75 |
| 2. Models v1 | 3–4 | Fine-tuned multilingual sentiment + aspect model; crisis rules + classifier | Macro-F1 ≥ 0.85 on real hold-out; crisis recall ≥ 0.95 |
| 3. Integration | 5–6 | FastAPI service, approval queue UI in Revio, crisis alerts by WhatsApp/email | p95 latency < 300 ms (excluding LLM) |
| 4. Pilot | 7–8 | 2–3 clients (e.g. a clinic and a restaurant chain) | Reply rate > 90%; median reply time < 4 h |
| 5. Scale | 9+ | Weekly insight digest, auto-publish for 5★ positives (opt-in), Hindi/Hinglish | Paid add-on conversion |

**Team:** 1 ML engineer, 1 backend engineer (part-time), 1 PM/QA (part-time).
**Cost estimate:** classification on existing CPU; LLM replies about ₹0.05–0.15 each on gpt-4o-mini-class models.

## 8. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py              # demo
python scripts/evaluate.py        # metrics
pytest -q                         # tests
# optional LLM replies
export OPENAI_API_KEY=...         # then toggle "Use LLM" in the sidebar
```
