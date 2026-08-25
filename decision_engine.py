from __future__ import annotations
from dataclasses import dataclass, field

from classifier import classify, ClassificationResult
from retrieval import KnowledgeBase, RetrievedDoc, RETRIEVAL_CONFIDENCE_THRESHOLD

MAX_MESSAGE_LEN = 2000


@dataclass
class TriageResult:
    status: str  # "answered" | "escalated" | "rejected"
    category: ClassificationResult
    top_matches: list[RetrievedDoc]
    answer: str | None
    escalation_reason: str | None
    trace: list[str] = field(default_factory=list)


def _log(trace: list[str], msg: str) -> None:
    trace.append(msg)


def triage(message: str, kb: KnowledgeBase) -> TriageResult:
    trace: list[str] = []

    # ---- Input validation / edge cases -------------------------------
    if message is None or not message.strip():
        _log(trace, "Input rejected: empty or whitespace-only message.")
        return TriageResult(
            status="rejected",
            category=ClassificationResult(category=None, confidence=0.0),
            top_matches=[],
            answer=None,
            escalation_reason=None,
            trace=trace,
        )

    if len(message) > MAX_MESSAGE_LEN:
        _log(trace, f"Message truncated from {len(message)} to {MAX_MESSAGE_LEN} chars.")
        message = message[:MAX_MESSAGE_LEN]

    # ---- Step 1: classify ---------------------------------------------
    cls = classify(message)
    _log(trace, f"Classifier: category={cls.category}, confidence={cls.confidence:.2f} "
                f"(threshold=0.35), matched_keywords={cls.matched_keywords}")

    # ---- Step 2: retrieve grounded candidates -------------------------
    try:
        matches = kb.search(message, top_k=3, category=cls.category)
    except Exception as exc:  # retrieval failure handling
        _log(trace, f"Retrieval failed with error: {exc!r}. Escalating.")
        return TriageResult(
            status="escalated",
            category=cls,
            top_matches=[],
            answer=None,
            escalation_reason="Internal retrieval error - could not search the knowledge base.",
            trace=trace,
        )

    top_score = matches[0].score if matches else 0.0
    _log(trace, f"Retrieval: top_match={matches[0].id if matches else None}, "
                f"score={top_score:.3f} (threshold={RETRIEVAL_CONFIDENCE_THRESHOLD})")

    if cls.category is None:
        reason = ("Out-of-scope: the message does not contain any keywords "
                  "associated with Billing, Technical, or Account Access, "
                  "so it is treated as outside the supported ticket scope "
                  "even if a knowledge-base article superficially matched.")
        _log(trace, f"Decision: ESCALATE - {reason}")
        return TriageResult("escalated", cls, matches, None, reason, trace)

    if cls.is_low_confidence and top_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        reason = ("Low confidence: classifier confidence "
                  f"({cls.confidence:.2f}) is below the 0.35 threshold and the "
                  f"best knowledge-base match ({top_score:.2f}) is below the "
                  f"{RETRIEVAL_CONFIDENCE_THRESHOLD} retrieval threshold.")
        _log(trace, f"Decision: ESCALATE - {reason}")
        return TriageResult("escalated", cls, matches, None, reason, trace)

    if top_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        reason = (f"No knowledge-base article is a confident match "
                  f"(best score {top_score:.2f} < {RETRIEVAL_CONFIDENCE_THRESHOLD}), "
                  "even though the ticket category was identified. Answering "
                  "would risk an ungrounded response.")
        _log(trace, f"Decision: ESCALATE - {reason}")
        return TriageResult("escalated", cls, matches, None, reason, trace)

    # ---- Step 4: answer, grounded in the top retrieved document -------
    best = matches[0]
    _log(trace, f"Decision: ANSWER using source {best.id} ({best.title}), score={best.score:.3f}")
    return TriageResult("answered", cls, matches, best.answer, None, trace)
