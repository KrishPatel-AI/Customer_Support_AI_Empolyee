import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier import classify
from retrieval import KnowledgeBase
from decision_engine import triage

kb = KnowledgeBase()


def test_classify_billing():
    r = classify("I was charged twice for my subscription this month")
    assert r.category == "Billing"
    assert not r.is_low_confidence


def test_classify_technical():
    r = classify("The app keeps crashing and my integration sync is broken")
    assert r.category == "Technical"


def test_classify_account_access():
    r = classify("I forgot my password and I'm locked out of my account")
    assert r.category == "Account Access"


def test_classify_empty():
    r = classify("")
    assert r.category is None
    assert r.confidence == 0.0


def test_classify_gibberish_low_confidence():
    r = classify("asdkjaslkdj qqweoiu")
    assert r.category is None or r.is_low_confidence


def test_retrieval_returns_relevant_doc():
    results = kb.search("how do I reset my password")
    assert len(results) > 0
    assert results[0].category == "Account Access"


def test_retrieval_empty_query():
    assert kb.search("") == []


def test_triage_answers_clear_billing_question():
    result = triage("Where can I download my invoice from last month?", kb)
    assert result.status == "answered"
    assert result.category.category == "Billing"
    assert result.answer is not None


def test_triage_escalates_out_of_scope():
    result = triage("What's the weather like in Mumbai today?", kb)
    assert result.status == "escalated"
    assert result.escalation_reason is not None


def test_triage_rejects_empty_input():
    result = triage("", kb)
    assert result.status == "rejected"


def test_triage_rejects_whitespace_input():
    result = triage("     ", kb)
    assert result.status == "rejected"


def test_triage_handles_all_three_categories():
    cases = {
        "I need to reset my two factor authentication": "Account Access",
        "My Zapier integration stopped syncing yesterday": "Technical",
        "Can I get a refund for my annual plan?": "Billing",
    }
    for msg, expected_cat in cases.items():
        result = triage(msg, kb)
        assert result.category.category == expected_cat, f"Failed for: {msg}"


def test_triage_gibberish_escalates_or_rejects():
    result = triage("zxjkqwpoiuqwe", kb)
    assert result.status in ("escalated", "answered")  # never crashes
    # Should not confidently answer nonsense
    if result.status == "answered":
        assert False, "Gibberish should not be confidently answered"


def test_triage_unrelated_topic_never_answered_on_word_overlap():
    # Regression test: "plan a birthday party" shares the word "plan" with
    # a Billing FAQ ("subscription plan") and previously produced a false
    # positive confident answer. It must always escalate as out-of-scope.
    result = triage("Can you help me plan a birthday party for my dog?", kb)
    assert result.status == "escalated"
    assert "out-of-scope" in result.escalation_reason.lower()


def test_triage_very_long_input_is_truncated_not_crashed():
    long_msg = "billing invoice payment " * 500
    result = triage(long_msg, kb)
    assert result.status in ("answered", "escalated", "rejected")


def test_retrieval_failure_is_handled(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated KB outage")

    monkeypatch.setattr(kb, "search", boom)
    result = triage("I forgot my password", kb)
    assert result.status == "escalated"
    assert "retrieval error" in result.escalation_reason.lower()
