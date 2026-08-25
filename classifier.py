
from __future__ import annotations
from dataclasses import dataclass, field
import re

CATEGORIES = ["Billing", "Technical", "Account Access"]

CATEGORY_KEYWORDS: dict[str, dict[str, float]] = {
    "Billing": {
        "invoice": 1.0, "billing": 1.0, "charge": 1.0, "charged": 1.0,
        "refund": 1.2, "payment": 1.0, "credit card": 1.3, "card": 0.6,
        "subscription": 0.9, "subscription plan": 0.9, "upgrade": 0.7, "downgrade": 0.7,
        "receipt": 1.0, "price": 0.8, "pricing": 0.8, "renew": 0.8,
        "renewal": 0.8, "double charge": 1.4, "charged twice": 1.4,
        "money back": 1.0, "bill": 0.8, "cost": 0.5,
    },
    "Technical": {
        "bug": 1.2, "error": 1.0, "crash": 1.2, "crashed": 1.2,
        "freezing": 1.1, "freeze": 1.1, "slow": 1.0, "lag": 0.9,
        "not working": 1.1, "broken": 1.0, "integration": 1.1,
        "api": 1.0, "sync": 1.0, "webhook": 1.1, "export": 0.8,
        "csv": 0.7, "loading": 0.7, "performance": 0.9, "glitch": 1.0,
        "dark mode": 0.9, "mobile app": 0.8, "feature": 0.5,
    },
    "Account Access": {
        "password": 1.2, "login": 1.0, "log in": 1.0, "locked": 1.2,
        "locked out": 1.4, "2fa": 1.2, "two factor": 1.3,
        "authenticator": 1.1, "reset": 0.7, "access": 0.6,
        "account locked": 1.4, "sign in": 0.9, "username": 0.9,
        "verify email": 1.0, "email address": 0.6, "permissions": 0.8,
        "team member": 0.8, "otp": 1.0, "suspended": 1.0,
    },
}

LOW_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class ClassificationResult:
    category: str | None
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)
    is_low_confidence: bool = True

    def as_trace(self) -> dict:
        return {
            "predicted_category": self.category,
            "confidence": round(self.confidence, 3),
            "raw_scores": {k: round(v, 3) for k, v in self.scores.items()},
            "matched_keywords": self.matched_keywords,
            "low_confidence": self.is_low_confidence,
        }


def _stem(word: str) -> str:
    """Very small heuristic stemmer so 'crash'/'crashing'/'crashed' are
    treated as the same signal, without pulling in an NLP dependency."""
    for suffix in ("ing", "ed", "es"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("s") and not word.endswith("ss") and len(word) >= 4:
        return word[:-1]
    return word


def classify(message: str) -> ClassificationResult:
    if not message or not message.strip():
        return ClassificationResult(category=None, confidence=0.0, scores={}, matched_keywords={})

    text = message.lower().strip()
    tokens = re.findall(r"[a-z0-9]+", text)
    stemmed_tokens = {_stem(t) for t in tokens}

    raw_scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        hits = []
        for kw, weight in keywords.items():
            if " " in kw:
                
                if kw in text:
                    score += weight
                    hits.append(kw)
            else:
                
                if _stem(kw) in stemmed_tokens:
                    score += weight
                    hits.append(kw)
        raw_scores[category] = score
        matched[category] = hits

    total = sum(raw_scores.values())
    if total == 0:
       
        return ClassificationResult(
            category=None, confidence=0.0, scores=raw_scores, matched_keywords=matched,
            is_low_confidence=True,
        )

    normalized = {k: v / total for k, v in raw_scores.items()}
    best_category = max(normalized, key=normalized.get)
    best_confidence = normalized[best_category]

    return ClassificationResult(
        category=best_category,
        confidence=best_confidence,
        scores=normalized,
        matched_keywords=matched,
        is_low_confidence=best_confidence < LOW_CONFIDENCE_THRESHOLD,
    )
