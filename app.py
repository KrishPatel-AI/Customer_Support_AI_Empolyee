import streamlit as st

from retrieval import KnowledgeBase
from decision_engine import triage
import llm_enhancer

st.set_page_config(page_title="Customer Support AI Employee", layout="wide")


@st.cache_resource
def load_kb() -> KnowledgeBase:
    return KnowledgeBase()


kb = load_kb()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "meta": {...}}

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Customer Support AI Employee")

    st.subheader("Status")
    st.write(f"Knowledge base articles: **{len(kb.docs)}**")
    st.write(f"LLM rephrasing: **{'enabled (Groq)' if llm_enhancer.is_enabled() else 'disabled (extractive mode)'}**")

    with st.expander("View knowledge base"):
        for d in kb.docs:
            st.markdown(f"**[{d['id']}] {d['title']}** _(​{d['category']})_")

    st.subheader("Try these")
    samples = [
        "I was charged twice this month, why?",
        "The app keeps freezing on large reports",
        "I forgot my password",
        "What's the weather in Mumbai?",
        "asdkjaslkdj",
    ]
    for s in samples:
        if st.button(s, use_container_width=True):
            st.session_state.pending_input = s

    if st.button(" Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("How it works")
    st.markdown(
        "1. **Classify** the message into Billing / Technical / Account Access "
        "using a transparent keyword-weighted rule engine.\n"
        "2. **Retrieve** the closest knowledge-base article with TF-IDF cosine "
        "similarity.\n"
        "3. **Decide**: answer (grounded, extractive) or **escalate** with a "
        "stated reason, based on explicit confidence thresholds.\n"
        "4. *(Optional)* If a `GROQ_API_KEY` is set, rephrase the grounded "
        "answer naturally — never adding new facts."
    )

# ---------------------------------------------------------------- Main chat
st.title("Customer Support Chat")
st.caption("synthetic knowledge base, no real customer data.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            badge_color = {"answered": "green", "escalated": "orange", "rejected": "red"}.get(meta["status"], "gray")
            st.markdown(f":{badge_color}[**{meta['status'].upper()}**] — category: `{meta['category']}`, "
                        f"confidence: `{meta['confidence']:.2f}`")
            if meta["status"] == "escalated":
                st.warning(f"Escalated to a human agent.\n\n**Reason:** {meta['reason']}")
            if meta.get("source"):
                st.info(f"Source: **{meta['source']}**")
            with st.expander("Inspect reasoning trace"):
                for line in meta["trace"]:
                    st.code(line, language=None)
                if meta.get("top_matches"):
                    st.write("Top retrieved candidates:")
                    st.table(meta["top_matches"])

user_input = st.chat_input("Describe your issue...")
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input is not None:
    st.session_state.messages.append({"role": "user", "content": user_input if user_input.strip() else "*(empty message)*"})

    result = triage(user_input, kb)

    if result.status == "rejected":
        reply = "It looks like your message was empty. Could you tell me a bit more about the issue you're facing?"
        meta = None
    elif result.status == "escalated":
        reply = ("I'm not confident I can answer this accurately from our knowledge base, "
                  "so I've escalated it to a human support agent who will follow up with you shortly.")
        meta = {
            "status": "escalated",
            "category": result.category.category or "Unclassified",
            "confidence": result.category.confidence,
            "reason": result.escalation_reason,
            "trace": result.trace,
            "top_matches": [{"id": m.id, "title": m.title, "score": round(m.score, 3)} for m in result.top_matches],
            "source": None,
        }
    else:  # answered
        source_doc = result.top_matches[0]
        final_answer = result.answer
        rephrased = llm_enhancer.rephrase(user_input, result.answer, source_doc.title)
        if rephrased:
            reply = rephrased
        else:
            reply = result.answer
        meta = {
            "status": "answered",
            "category": result.category.category,
            "confidence": result.category.confidence,
            "reason": None,
            "trace": result.trace,
            "top_matches": [{"id": m.id, "title": m.title, "score": round(m.score, 3)} for m in result.top_matches],
            "source": f"[{source_doc.id}] {source_doc.title}",
        }

    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
    st.rerun()
