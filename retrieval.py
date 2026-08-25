from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).parent / "data" / "knowledge_base.json"

# Below this cosine similarity, we don't trust the top retrieval hit
# well enough to answer confidently -> triggers escalation.
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.18


@dataclass
class RetrievedDoc:
    id: str
    title: str
    category: str
    answer: str
    score: float


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.docs: list[dict] = json.load(f)

        # Index on question + title + keywords for best lexical recall.
        corpus = [
            f"{d['title']} {d['question']} {' '.join(d.get('keywords', []))}"
            for d in self.docs
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3, category: str | None = None) -> list[RetrievedDoc]:
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix)[0]

        ranked_idx = sims.argsort()[::-1]

        results = []
        for idx in ranked_idx:
            doc = self.docs[idx]
            if category and doc["category"] != category:
                continue
            results.append(
                RetrievedDoc(
                    id=doc["id"],
                    title=doc["title"],
                    category=doc["category"],
                    answer=doc["answer"],
                    score=float(sims[idx]),
                )
            )
            if len(results) >= top_k:
                break

        # If category filtering produced nothing (e.g. mismatch), fall back
        # to an unfiltered search so we still surface the best global match.
        if category and not results:
            return self.search(query, top_k=top_k, category=None)

        return results
