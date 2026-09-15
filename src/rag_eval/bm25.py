from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import pandas as pd

from .common import tokenize


def rank_bm25(questions: pd.DataFrame, corpus: pd.DataFrame, *, top_k: int = 4,
              k1: float = 1.2, b: float = 0.75) -> pd.DataFrame:
    """Robertson BM25 over supplied-document candidates with global unit IDF."""
    corpus = corpus.copy()
    corpus["tokens"] = corpus.text.map(tokenize)
    corpus["length"] = corpus.tokens.map(len)
    n_units = len(corpus)
    average_length = float(corpus.length.mean()) if n_units else 0.0
    document_frequency = Counter()
    for tokens in corpus.tokens:
        document_frequency.update(set(tokens))
    postings = defaultdict(list)
    for row in corpus.itertuples(index=False):
        counts = Counter(row.tokens)
        postings[str(row.doc_id)].append((row, counts))

    output = []
    for question in questions.itertuples(index=False):
        query_tokens = list(dict.fromkeys(tokenize(question.question)))
        allowed = json.loads(question.allowed_doc_ids_json)
        candidates = []
        for doc_id in allowed:
            for unit, counts in postings.get(str(doc_id), []):
                score = 0.0
                for token in query_tokens:
                    tf = counts.get(token, 0)
                    if not tf:
                        continue
                    df = document_frequency[token]
                    idf = max(math.log((n_units - df + 0.5) / (df + 0.5)), 0.0)
                    norm = tf + k1 * (1.0 - b + b * unit.length / average_length)
                    score += idf * tf * (k1 + 1.0) / norm
                candidates.append((score, str(unit.unit_id), unit))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        excerpts = []
        for rank, (score, _, unit) in enumerate(candidates[:top_k], 1):
            excerpt = {"rank": rank, "unit_id": str(unit.unit_id),
                       "doc_id": str(unit.doc_id), "text": str(unit.text),
                       "score": float(score)}
            for column in ("paragraph_idx", "subparagraph_idx"):
                if hasattr(unit, column) and pd.notna(getattr(unit, column)):
                    excerpt[column] = int(getattr(unit, column))
            excerpts.append(excerpt)
        row = question._asdict()
        row.update(retrieval_method="bm25", bm25_k1=k1, bm25_b=b, top_k=top_k,
                   retrieved_count=len(excerpts),
                   retrieved_context="\n\n".join(item["text"] for item in excerpts),
                   retrieved_excerpts_json=json.dumps(excerpts, ensure_ascii=False),
                   retrieval_error="")
        for rank in range(1, top_k + 1):
            excerpt = excerpts[rank - 1] if len(excerpts) >= rank else {}
            row[f"excerpt_{rank}"] = excerpt.get("text", "")
            row[f"excerpt_{rank}_unit_id"] = excerpt.get("unit_id", "")
            row[f"excerpt_{rank}_doc_id"] = excerpt.get("doc_id", "")
            row[f"excerpt_{rank}_score"] = excerpt.get("score")
        output.append(row)
    return pd.DataFrame(output)
