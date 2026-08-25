from __future__ import annotations
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
_ENABLED = bool(GROQ_API_KEY)


def is_enabled() -> bool:
    return _ENABLED


def rephrase(user_question: str, grounded_answer: str, source_title: str) -> str | None:
    """Best-effort rephrase. Returns None on any failure so callers can
    fall back to the raw grounded answer without ever breaking the flow."""
    if not _ENABLED:
        return None

    try:
        from groq import Groq  # imported lazily so it's not a hard dependency

        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            "You are a support assistant. Rewrite the ANSWER below in a "
            "warm, natural tone for the CUSTOMER_QUESTION. "
            "Do not add any information that is not already in the ANSWER. "
            "Do not remove any instructions or steps. Keep it concise.\n\n"
            f"CUSTOMER_QUESTION: {user_question}\n"
            f"ANSWER (source: {source_title}): {grounded_answer}"
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        text = completion.choices[0].message.content.strip()
        return text if text else None
    except Exception:
        # Any failure (missing package, network, bad key, rate limit) ->
        # silently fall back to the grounded extractive answer.
        return None
