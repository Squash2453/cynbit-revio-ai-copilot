"""Revio AI Copilot - Streamlit demo.

Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from revio import ReviewAnalyzer, draft_reply

st.set_page_config(page_title="Revio AI Copilot | Cynbit", page_icon="⭐", layout="wide")


@st.cache_resource
def get_analyzer() -> ReviewAnalyzer:
    return ReviewAnalyzer()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv("data/sample_reviews.csv")


analyzer = get_analyzer()

st.title("Revio AI Copilot")
st.caption("Proof of concept for Cynbit's Revio: triage every review by sentiment, topic and risk, "
           "then draft an on-brand reply for a human to approve.")

with st.sidebar:
    st.header("Brand settings")
    business = st.text_input("Business name", "Wild Basil")
    contact = st.text_input("Support contact", "care@wildbasil.in")
    use_llm = st.toggle("Use LLM for replies (needs OPENAI_API_KEY)", value=False)
    st.divider()
    st.markdown("**How it works**\n\n1. TF-IDF + Logistic Regression sentiment, blended with the star rating\n"
                "2. Aspect tagging against a business taxonomy\n3. Rule-based crisis detection "
                "(health, legal, fraud, harassment)\n4. Priority score orders the response queue\n"
                "5. Reply draft: LLM with guard-railed prompt, or template fallback")

tab_single, tab_batch = st.tabs(["Single review", "Location dashboard"])

with tab_single:
    c1, c2 = st.columns([3, 1])
    text = c1.text_area("Paste a customer review",
                        "Waited over an hour for our table and the food came cold. "
                        "The manager was rude when we complained.", height=120)
    rating = c2.select_slider("Star rating", options=[1, 2, 3, 4, 5], value=2)
    customer = c2.text_input("Customer first name (optional)", "")
    if st.button("Analyze and draft reply", type="primary"):
        ins = analyzer.analyze(text, rating)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sentiment", ins.sentiment.title(), f"{ins.confidence:.0%} confidence")
        m2.metric("Priority", f"{ins.priority}/100")
        m3.metric("Crisis", "YES" if ins.crisis else "No")
        m4.metric("Topics", len([a for a in ins.aspects if a != "general"]) or 1)
        st.write("**Topics:**", ", ".join(a.replace("_", " ") for a in ins.aspects))
        if ins.crisis:
            st.error(f"Escalate to a manager now: {', '.join(r.replace('_', ' ') for r in ins.crisis_reasons)}. "
                     "This reply will not be auto-published.")
        reply, engine = draft_reply(ins, business, contact, customer, use_llm=use_llm)
        st.subheader("Suggested reply")
        st.text_area("Edit before publishing", reply, height=140)
        st.caption(f"Engine: {engine}")

with tab_batch:
    up = st.file_uploader("Upload reviews CSV (columns: text, rating, optional location/business)",
                          type="csv")
    df = pd.read_csv(up) if up else load_sample()
    st.caption("Showing the bundled sample of 72 reviews across 4 Cynbit client brands." if not up else "")
    res = analyzer.analyze_frame(df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reviews", len(res))
    k2.metric("Negative share", f"{(res.sentiment == 'negative').mean():.0%}")
    k3.metric("Crisis alerts", int(res.crisis.sum()))
    k4.metric("Avg rating", f"{res['rating'].mean():.2f}" if "rating" in res else "-")

    group_col = "location" if "location" in res else None
    if group_col:
        st.subheader("Sentiment by location")
        piv = (res.pivot_table(index=group_col, columns="sentiment", values="text",
                               aggfunc="count", fill_value=0))
        st.bar_chart(piv)

    st.subheader("Top complaint topics")
    neg = res[res.sentiment == "negative"]["aspects"].str.split(", ").explode()
    st.bar_chart(neg[neg != "general"].value_counts())

    st.subheader("Response queue (highest priority first)")
    cols = [c for c in ["priority", "crisis", "business", "location", "platform", "rating",
                        "sentiment", "aspects", "text"] if c in res]
    st.dataframe(res.sort_values("priority", ascending=False)[cols], use_container_width=True,
                 hide_index=True)
    st.download_button("Download triaged CSV", res.to_csv(index=False), "revio_triaged.csv")
