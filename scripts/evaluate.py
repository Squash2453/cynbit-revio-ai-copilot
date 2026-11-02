"""Evaluate the analyzer.

1. 5-fold stratified cross-validation on the bundled dataset (text only).
2. A hand-written out-of-distribution set (data/holdout_reviews.csv) written in
   different wording from the training fragments, to check generalisation.
3. Crisis-detection recall/precision on both.

Run:  python scripts/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revio import ReviewAnalyzer, detect_crisis  # noqa: E402


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "sample_reviews.csv")
    an = ReviewAnalyzer()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = cross_val_predict(an.model, df.text, df.label_sentiment, cv=cv)
    print("== 5-fold CV, text-only sentiment ==")
    print(classification_report(df.label_sentiment, pred, digits=3, zero_division=0))

    ho = pd.read_csv(ROOT / "data" / "holdout_reviews.csv")
    text_only = [an.sentiment(t)[0] for t in ho.text]
    blended = [an.analyze(t, r).sentiment for t, r in zip(ho.text, ho.rating)]
    print("== Out-of-distribution hold-out (n=%d) ==" % len(ho))
    print("text-only macro F1 : %.3f" % f1_score(ho.label_sentiment, text_only, average="macro"))
    print("text+rating macro F1: %.3f" % f1_score(ho.label_sentiment, blended, average="macro"))

    for name, frame in [("train set", df), ("hold-out", ho)]:
        flag = frame.text.map(lambda t: bool(detect_crisis(t)))
        truth = frame.label_crisis.astype(bool)
        tp = int((flag & truth).sum())
        rec = tp / max(int(truth.sum()), 1)
        prec = tp / max(int(flag.sum()), 1)
        print(f"crisis detection on {name}: recall {rec:.2f}, precision {prec:.2f} "
              f"({tp}/{int(truth.sum())} caught, {int(flag.sum())} flagged)")


if __name__ == "__main__":
    main()
