"""Build a small, labelled multi-business review dataset for the PoC.

Revio serves multi-location brands (clinics, retail, restaurants, SaaS), so the
dataset mixes those verticals. Reviews are composed from hand-written fragments
so every row carries a known sentiment, aspect and crisis label, which lets us
evaluate the pipeline honestly. In production this would be replaced by real
reviews pulled from Google Business Profile / social APIs and labelled through
an active-learning loop (see docs/Technical_Approach_Report.md).

Run:  python scripts/build_dataset.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

BUSINESSES = {
    "Pragya Healths": ("clinic", ["Jaipur - Malviya Nagar", "Jaipur - Vaishali Nagar"]),
    "Urban Wardrobe": ("retail", ["Jaipur - WTP Mall", "Delhi - Saket", "Online Store"]),
    "Wild Basil": ("restaurant", ["Jaipur - C Scheme", "Jaipur - Raja Park"]),
    "Evento": ("saas", ["Web App", "Android App"]),
}

# (aspect, sentiment) -> fragments
FRAGMENTS = {
    ("staff", "positive"): [
        "The staff were warm and really helpful",
        "Everyone at the front desk was polite and patient",
        "The team went out of their way to help me",
    ],
    ("staff", "negative"): [
        "The staff were rude and dismissive",
        "Nobody at the counter bothered to help",
        "The manager was arrogant and refused to listen",
    ],
    ("wait_time", "positive"): [
        "No waiting at all, I was attended within five minutes",
        "Service was quick and on time",
    ],
    ("wait_time", "negative"): [
        "I waited almost two hours past my slot",
        "The delay was ridiculous and nobody updated us",
        "Took forever, extremely slow service",
    ],
    ("price", "positive"): [
        "Prices are fair for the quality",
        "Great value for money",
    ],
    ("price", "negative"): [
        "Way too expensive for what you get",
        "They overcharged me and added hidden charges to the bill",
    ],
    ("quality", "positive"): [
        "The quality is excellent",
        "Everything was fresh and well made",
        "Product quality is top notch",
    ],
    ("quality", "negative"): [
        "The quality was poor and it fell apart in a week",
        "The food was stale and cold",
        "Cheap material, very disappointing quality",
    ],
    ("cleanliness", "positive"): [
        "The place was spotless and hygienic",
        "Very clean and well maintained",
    ],
    ("cleanliness", "negative"): [
        "The washroom was dirty and smelled bad",
        "Unhygienic place, dust everywhere",
    ],
    ("app_experience", "positive"): [
        "The app is smooth and easy to use",
        "Booking through the website was seamless",
    ],
    ("app_experience", "negative"): [
        "The app keeps crashing at checkout",
        "Login is broken and the website is painfully slow",
        "Payment failed twice on the app",
    ],
    ("delivery", "positive"): [
        "Delivery arrived a day early",
        "Order was packed well and delivered on time",
    ],
    ("delivery", "negative"): [
        "My order arrived late and the box was damaged",
        "Still waiting for my delivery after ten days",
    ],
}

NEUTRAL = [
    "It was okay, nothing special",
    "Average experience overall",
    "Visited once, it was fine I guess",
    "Decent but could be better in some areas",
]

CRISIS = [
    "I got food poisoning after eating here",
    "They charged my card twice and refused a refund, this is a scam",
    "A staff member harassed my sister, I am filing a police complaint",
    "The doctor gave the wrong medicine and my father had a severe reaction",
    "My card details were leaked after using the app, this is fraud",
    "I will take legal action and go to the consumer court",
]

CLOSERS_POS = ["Highly recommend!", "Will definitely come back.", "Loved it.", ""]
CLOSERS_NEG = ["Never coming back.", "Very disappointed.", "Avoid.", ""]


def compose(sentiment: str) -> tuple[str, list[str], int]:
    aspects = [a for (a, s) in FRAGMENTS if s == sentiment]
    chosen = random.sample(aspects, k=random.choice([1, 1, 2]))
    parts = [random.choice(FRAGMENTS[(a, sentiment)]) for a in chosen]
    closer = random.choice(CLOSERS_POS if sentiment == "positive" else CLOSERS_NEG)
    text = ". ".join(parts) + ". " + closer
    rating = random.choice([4, 5]) if sentiment == "positive" else random.choice([1, 2])
    return text.strip(), chosen, rating


def main() -> None:
    rows = []
    rid = 1
    for business, (vertical, locations) in BUSINESSES.items():
        for _ in range(16):
            sentiment = random.choice(["positive", "positive", "negative", "negative", "neutral"])
            if sentiment == "neutral":
                text, aspects, rating = random.choice(NEUTRAL) + ".", ["general"], 3
            else:
                text, aspects, rating = compose(sentiment)
            rows.append([rid, business, vertical, random.choice(locations),
                         random.choice(["Google", "Instagram", "Facebook", "Justdial"]),
                         rating, text, sentiment, "|".join(aspects), 0])
            rid += 1
        # two crisis reviews per business
        for _ in range(2):
            text = random.choice(CRISIS) + ". " + random.choice(CLOSERS_NEG)
            rows.append([rid, business, vertical, random.choice(locations), "Google",
                         1, text.strip(), "negative", "general", 1])
            rid += 1

    out = Path(__file__).resolve().parents[1] / "data" / "sample_reviews.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "business", "vertical", "location", "platform", "rating",
                    "text", "label_sentiment", "label_aspects", "label_crisis"])
        w.writerows(rows)
    print(f"wrote {len(rows)} reviews -> {out}")


if __name__ == "__main__":
    main()
