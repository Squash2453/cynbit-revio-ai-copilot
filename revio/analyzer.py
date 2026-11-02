"""Core review-intelligence pipeline for the Revio AI Copilot PoC.

For every incoming review we produce:
  * sentiment  - positive / neutral / negative (TF-IDF + Logistic Regression,
                 blended with the star rating as a prior)
  * aspects    - what the customer is talking about (staff, wait_time, price ...)
  * crisis     - True when the review signals legal, safety, health, fraud or
                 harassment risk and must reach a human immediately
  * priority   - 0-100 score used to order the response queue

The model is deliberately small: it trains in < 1 s on CPU, needs no GPU and
no paid API, so it can run inside Revio's existing backend. The LLM layer
(revio/replies.py) is only used for writing the reply, where generation
quality actually matters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_reviews.csv"

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "staff": ["staff", "rude", "polite", "manager", "team", "receptionist", "counter",
              "front desk", "behaviour", "behavior", "attitude", "doctor", "waiter"],
    "wait_time": ["wait", "waited", "waiting", "delay", "late", "slow", "quick", "on time",
                  "hours", "forever", "queue"],
    "price": ["price", "expensive", "overcharg", "cheap", "value for money", "bill",
              "hidden charge", "costly", "refund"],
    "quality": ["quality", "stale", "fresh", "taste", "material", "fell apart", "defect",
                "cold", "well made", "top notch"],
    "cleanliness": ["clean", "dirty", "hygien", "washroom", "smell", "dust", "spotless"],
    "app_experience": ["app", "website", "login", "crash", "checkout", "payment failed",
                       "bug", "seamless", "booking"],
    "delivery": ["deliver", "shipping", "courier", "package", "box", "arrived", "order"],
}

CRISIS_PATTERNS: dict[str, list[str]] = {
    "health_safety": [r"food poisoning", r"wrong medicine", r"severe reaction", r"injur",
                      r"hospitali[sz]ed", r"allerg"],
    "legal": [r"legal action", r"consumer court", r"police", r"lawyer", r"\bfir\b", r"sue\b"],
    "fraud_privacy": [r"scam", r"fraud", r"charged .*twice", r"leaked", r"data breach",
                      r"stole"],
    "harassment": [r"harass", r"abuse", r"assault", r"threaten"],
}


@dataclass
class ReviewInsight:
    text: str
    rating: int | None
    sentiment: str
    confidence: float
    aspects: list[str] = field(default_factory=list)
    crisis: bool = False
    crisis_reasons: list[str] = field(default_factory=list)
    priority: int = 0


def extract_aspects(text: str) -> list[str]:
    t = text.lower()
    found = [a for a, kws in ASPECT_KEYWORDS.items() if any(k in t for k in kws)]
    return found or ["general"]


def detect_crisis(text: str) -> list[str]:
    t = text.lower()
    return [cat for cat, pats in CRISIS_PATTERNS.items() if any(re.search(p, t) for p in pats)]


class ReviewAnalyzer:
    """Train once, then call .analyze(text, rating) per review."""

    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, data_path: Path = DATA_PATH, rating_weight: float = 0.35):
        self.rating_weight = rating_weight
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        df = pd.read_csv(data_path)
        self.fit(df["text"].tolist(), df["label_sentiment"].tolist())

    def fit(self, texts: list[str], labels: list[str]) -> "ReviewAnalyzer":
        self.model.fit(texts, labels)
        return self

    @staticmethod
    def _rating_prior(rating: int | None) -> dict[str, float] | None:
        if rating is None:
            return None
        table = {1: (0.85, 0.1, 0.05), 2: (0.7, 0.2, 0.1), 3: (0.2, 0.6, 0.2),
                 4: (0.05, 0.25, 0.7), 5: (0.03, 0.07, 0.9)}
        neg, neu, pos = table.get(int(rating), (1 / 3, 1 / 3, 1 / 3))
        return {"negative": neg, "neutral": neu, "positive": pos}

    def sentiment(self, text: str, rating: int | None = None) -> tuple[str, float]:
        probs = dict(zip(self.model.classes_, self.model.predict_proba([text])[0]))
        prior = self._rating_prior(rating)
        if prior:
            w = self.rating_weight
            probs = {k: (1 - w) * probs.get(k, 0) + w * prior[k] for k in self.LABELS}
        label = max(probs, key=probs.get)
        return label, round(float(probs[label]), 3)

    def analyze(self, text: str, rating: int | None = None) -> ReviewInsight:
        label, conf = self.sentiment(text, rating)
        reasons = detect_crisis(text)
        if reasons:
            label = "negative"
        aspects = extract_aspects(text)
        priority = self._priority(label, conf, rating, reasons, aspects)
        return ReviewInsight(text, rating, label, conf, aspects, bool(reasons), reasons, priority)

    @staticmethod
    def _priority(label, conf, rating, reasons, aspects) -> int:
        score = {"negative": 55, "neutral": 25, "positive": 10}[label]
        score += int(15 * conf) if label == "negative" else 0
        if rating is not None and rating <= 2:
            score += 10
        score += 3 * max(0, len([a for a in aspects if a != "general"]) - 1)
        if reasons:
            score = max(score, 90) + 3 * (len(reasons) - 1)
        return min(score, 100)

    def analyze_frame(self, df: pd.DataFrame, text_col="text", rating_col="rating") -> pd.DataFrame:
        out = []
        for _, r in df.iterrows():
            rating = int(r[rating_col]) if rating_col in df and pd.notna(r[rating_col]) else None
            ins = self.analyze(str(r[text_col]), rating)
            out.append({"sentiment": ins.sentiment, "confidence": ins.confidence,
                        "aspects": ", ".join(ins.aspects), "crisis": ins.crisis,
                        "crisis_reasons": ", ".join(ins.crisis_reasons),
                        "priority": ins.priority})
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)
