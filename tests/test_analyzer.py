import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from revio import ReviewAnalyzer, detect_crisis, draft_reply, extract_aspects  # noqa: E402


@pytest.fixture(scope="module")
def an():
    return ReviewAnalyzer()


def test_positive(an):
    assert an.analyze("The staff were lovely and the food was excellent", 5).sentiment == "positive"


def test_negative(an):
    ins = an.analyze("Rude staff and we waited two hours", 1)
    assert ins.sentiment == "negative"
    assert {"staff", "wait_time"} <= set(ins.aspects)


def test_crisis_forces_negative_and_top_priority(an):
    ins = an.analyze("Got food poisoning here, taking legal action", 3)
    assert ins.crisis and ins.sentiment == "negative"
    assert set(ins.crisis_reasons) == {"health_safety", "legal"}
    assert ins.priority >= 90


def test_no_false_crisis():
    assert detect_crisis("Lovely evening, will come again") == []


def test_aspect_fallback():
    assert extract_aspects("It was fine") == ["general"]


def test_template_reply_never_needs_api(an, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ins = an.analyze("App crashes at checkout", 1)
    reply, engine = draft_reply(ins, "Evento", "support@evento.app")
    assert engine == "template"
    assert "support@evento.app" in reply and "app" in reply.lower()


def test_crisis_reply_avoids_liability(an):
    ins = an.analyze("They charged my card twice, this is fraud", 1)
    reply, _ = draft_reply(ins, "Urban Wardrobe", "care@uw.in", use_llm=False)
    assert "senior" in reply and "fault" not in reply.lower()
