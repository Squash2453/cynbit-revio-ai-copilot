"""Reply drafting for Revio.

Two engines behind one function:
  * LLM engine   - used when OPENAI_API_KEY is set. A structured prompt grounds
                   the model in the analyzer's output (sentiment, aspects,
                   crisis flags) and the brand's voice rules.
  * Template engine - deterministic fallback that needs no API, so the demo
                   always works and costs nothing.

Crisis reviews are never auto-published: the draft is marked for human
approval and avoids admitting liability.
"""
from __future__ import annotations

import os

from .analyzer import ReviewInsight

ASPECT_PHRASES = {
    "staff": "the way our team treated you",
    "wait_time": "the waiting time",
    "price": "the pricing and billing",
    "quality": "the quality you received",
    "cleanliness": "the cleanliness of our premises",
    "app_experience": "the trouble with our app and website",
    "delivery": "the delivery of your order",
    "general": "your experience",
}

SYSTEM_PROMPT = """You are Revio, the reputation assistant for {business}.
Write a public reply to a customer review.
Rules:
- 40 to 80 words, warm, specific, professional. No emojis, no hashtags.
- Mention the specific issues the customer raised: {aspects}.
- Never invent facts, discounts, or promises the brand has not approved.
- Negative review: apologise once, take ownership, and move the conversation
  offline using the contact {contact}.
- Positive review: thank them and reference one detail they enjoyed.
- If CRISIS is true ({crisis_reasons}), do not admit legal liability, do not
  discuss medical or legal details in public, express concern, and ask them to
  contact {contact} so a senior manager can follow up today.
Return only the reply text."""


def _join(items: list[str]) -> str:
    items = [ASPECT_PHRASES.get(a, a) for a in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def template_reply(ins: ReviewInsight, business: str, contact: str, customer: str = "") -> str:
    hi = f"Hi {customer}," if customer else "Hi,"
    if ins.crisis:
        return (f"{hi} we're very concerned to read this and we take it extremely seriously. "
                f"We'd like a senior member of the {business} team to look into it personally today. "
                f"Please reach us at {contact} so we can understand exactly what happened and make it right.")
    if ins.sentiment == "negative":
        return (f"{hi} thank you for telling us, and we're sorry about {_join(ins.aspects)}. "
                f"That isn't the standard we hold ourselves to at {business}. "
                f"We've shared your feedback with the team. Please write to us at {contact} "
                f"so we can fix this for you directly.")
    if ins.sentiment == "neutral":
        return (f"{hi} thanks for sharing your feedback on {_join(ins.aspects)}. "
                f"We're always working to improve at {business}, and we'd love to hear what would "
                f"have made your visit better. You can reach us anytime at {contact}.")
    return (f"{hi} thank you so much for the kind words about {_join(ins.aspects)}! "
            f"It means a lot to the whole {business} team, and we look forward to welcoming you back.")


def llm_reply(ins: ReviewInsight, business: str, contact: str, model: str | None = None) -> str:
    from openai import OpenAI  # imported lazily so the dependency stays optional

    client = OpenAI()
    system = SYSTEM_PROMPT.format(
        business=business, contact=contact, aspects=", ".join(ins.aspects),
        crisis_reasons=", ".join(ins.crisis_reasons) or "none")
    user = (f"REVIEW ({ins.rating or 'no'} stars): {ins.text}\n"
            f"SENTIMENT: {ins.sentiment}\nCRISIS: {ins.crisis}")
    resp = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.4,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def draft_reply(ins: ReviewInsight, business: str, contact: str, customer: str = "",
                use_llm: bool | None = None) -> tuple[str, str]:
    """Return (reply_text, engine_name). Falls back to templates on any LLM error."""
    if use_llm is None:
        use_llm = bool(os.getenv("OPENAI_API_KEY"))
    if use_llm:
        try:
            return llm_reply(ins, business, contact), "llm"
        except Exception as exc:  # network, quota, missing package
            return template_reply(ins, business, contact, customer) + "", f"template (LLM error: {type(exc).__name__})"
    return template_reply(ins, business, contact, customer), "template"
