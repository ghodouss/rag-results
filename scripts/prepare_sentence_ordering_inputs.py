#!/usr/bin/env python3
"""Build matched honest sentence-window inputs in two presentation orders."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag_eval.common import atomic_parquet, normalize, relaxed_bigram_match  # noqa: E402

CACHE = ROOT / "data/contractnli/sentence-source"
DEFAULT_TEMPLATE = (
    ROOT / "artifacts/experiments/contractnli-top2-preceding-evidence-1000/inputs/"
    "sturdy-top2-preceding.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/experiments/contractnli-sentence-a4-r2-ordering-evidence-1000/inputs"
)
OUTPUT_COLUMNS = [
    "dataset", "query_id", "split", "source_row", "question", "golden_answer",
    "allowed_doc_ids_json", "choice", "retrieval_method", "retrieval_level",
    "retrieved_count", "retrieved_context", "retrieval_error",
]


def sentence_id(unit: dict) -> tuple[str, int]:
    return str(unit["doc_id"]), int(unit["sentence_ordinal"])


def sentence_unit_id(unit: dict) -> str:
    return (
        f"{unit['doc_id']}|{int(unit['paragraph_idx'])}|{int(unit['sentence_idx'])}"
    )


def render_source_units(units: list[dict]) -> str:
    """Render source-ordered sentences, marking omitted source gaps."""
    parts: list[str] = []
    previous = None
    for unit in units:
        current = sentence_id(unit)
        if previous is not None and (
            current[0] != previous[0] or current[1] != previous[1] + 1
        ):
            parts.append("\n[context gap]\n")
        parts.append(str(unit["text"]))
        previous = current
    return "".join(parts).strip()


def retrieve_windows(
    query_anchors: pd.DataFrame, snapshots: pd.DataFrame, radius: int = 2,
    top_k: int = 4,
) -> dict:
    """Expand true ranked anchors and deduplicate later windows cumulatively."""
    ranked = query_anchors.sort_values("rank").head(top_k)
    expected_ranks = list(range(1, top_k + 1))
    if ranked["rank"].astype(int).tolist() != expected_ranks:
        raise ValueError(f"expected true sentence-anchor ranks {expected_ranks}")
    documents = set(ranked.doc_id.astype(str))
    if len(documents) != 1:
        raise ValueError("sentence anchors cross documents")
    doc_id = next(iter(documents))
    source = snapshots[snapshots.doc_id.astype(str).eq(doc_id)].sort_values(
        "sentence_ordinal"
    ).to_dict("records")
    positions = {
        (int(unit["paragraph_idx"]), int(unit["sentence_idx"])): index
        for index, unit in enumerate(source)
    }
    seen: set[str] = set()
    windows, deduplicated = [], 0
    for anchor in ranked.itertuples(index=False):
        key = (int(anchor.paragraph_idx), int(anchor.sentence_idx))
        if key not in positions:
            raise ValueError(f"missing sentence anchor {doc_id}|{key[0]}|{key[1]}")
        position = positions[key]
        local = source[max(0, position - radius):position + radius + 1]
        fresh = []
        for unit in local:
            unit_id = sentence_unit_id(unit)
            if unit_id in seen:
                deduplicated += 1
                continue
            seen.add(unit_id)
            fresh.append(unit)
        windows.append(fresh)
    union = sorted(
        [unit for window in windows for unit in window], key=sentence_id
    )
    if len(union) != len(seen):
        raise ValueError("deduplicated sentence union is not unique")
    return {
        "doc_id": doc_id,
        "windows": windows,
        "union": union,
        "unit_ids": frozenset(seen),
        "deduplicated": deduplicated,
    }


def build_inputs(
    template: pd.DataFrame, questions: pd.DataFrame,
    anchors: pd.DataFrame, snapshots: pd.DataFrame,
    top_k: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(template) != 1_000 or template.query_id.astype(str).duplicated().any():
        raise ValueError("template must contain 1,000 unique query IDs")
    reference = template[["query_id", "source_row", "question", "choice"]].merge(
        questions[["query_id", "source_row", "question", "choice", "golden_answer"]],
        on=["query_id", "source_row"], suffixes=("", "_reference"),
        how="inner", validate="one_to_one",
    )
    if len(reference) != 1_000:
        raise ValueError("template identities do not map one-to-one to ContractNLI questions")
    if not reference.question.eq(reference.question_reference).all() \
            or not reference.choice.eq(reference.choice_reference).all():
        raise ValueError("template question or choice differs from ContractNLI source")
    evidence_by_id = reference.set_index("query_id").golden_answer.to_dict()
    sample_ids = set(template.query_id.astype(str))
    anchors = anchors[anchors.query_id.astype(str).isin(sample_ids)].copy()
    anchors["query_id"] = anchors.query_id.astype(str)
    anchors["doc_id"] = anchors.doc_id.astype(str)
    snapshots = snapshots.copy()
    snapshots["doc_id"] = snapshots.doc_id.astype(str)
    grouped = {query_id: group for query_id, group in anchors.groupby("query_id")}
    if set(grouped) != sample_ids:
        raise ValueError("template and sentence-anchor query IDs differ")

    retrievals, audit = {}, []
    for row in template.sort_values("source_row").itertuples(index=False):
        query_id = str(row.query_id)
        retrieval = retrieve_windows(grouped[query_id], snapshots, top_k=top_k)
        allowed = {str(value) for value in json.loads(row.allowed_doc_ids_json)}
        if retrieval["doc_id"] not in allowed:
            raise ValueError(f"{query_id}: anchor document is not allowed")
        ranked_context = "\n\n".join(
            text for text in map(render_source_units, retrieval["windows"]) if text
        )
        global_context = render_source_units(retrieval["union"])
        canonical_context = global_context
        payload_chars = sum(len(str(unit["text"])) for unit in retrieval["union"])
        retrievals[query_id] = {
            **retrieval, "ranked_context": ranked_context,
            "global_context": global_context,
        }
        audit.append({
            "query_id": query_id,
            "source_row": int(row.source_row),
            "sentence_count": len(retrieval["union"]),
            "deduplicated_sentences": retrieval["deduplicated"],
            "payload_chars": payload_chars,
            "exact": normalize(evidence_by_id[query_id]) in normalize(canonical_context),
            "relaxed": relaxed_bigram_match(evidence_by_id[query_id], canonical_context),
        })

    def make_variant(method: str, context_key: str) -> pd.DataFrame:
        output = template[[
            "dataset", "query_id", "split", "source_row", "question",
            "allowed_doc_ids_json", "choice",
        ]].copy().sort_values("source_row").reset_index(drop=True)
        output["golden_answer"] = output.choice
        output["retrieval_method"] = method
        output["retrieval_level"] = "sentence"
        output["retrieved_count"] = top_k
        output["retrieved_context"] = output.query_id.map(
            lambda query_id: retrievals[str(query_id)][context_key]
        )
        output["retrieval_error"] = ""
        return output[OUTPUT_COLUMNS]

    ranked = make_variant(f"sturdy-sentence-a{top_k}-r2-ranked-windows", "ranked_context")
    source = make_variant(f"sturdy-sentence-a{top_k}-r2-global-source-order", "global_context")
    return ranked, source, pd.DataFrame(audit).sort_values("source_row").reset_index(drop=True)


def ordered_context_hash(frame: pd.DataFrame) -> str:
    rows = [
        f"{row.query_id}\t{hashlib.sha256(row.retrieved_context.encode()).hexdigest()}"
        for row in frame.sort_values("source_row").itertuples(index=False)
    ]
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def sorted_value_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(set(values.astype(str)))).encode()).hexdigest()


def validate_pair(ranked: pd.DataFrame, source: pd.DataFrame, audit: pd.DataFrame) -> None:
    identity = ["dataset", "query_id", "split", "source_row", "question", "golden_answer",
                "allowed_doc_ids_json", "choice", "retrieved_count", "retrieval_error"]
    if not ranked[identity].equals(source[identity]):
        raise ValueError("ordering variants differ outside presentation metadata")
    for name, frame in (("ranked", ranked), ("source", source)):
        if len(frame) != 1_000 or frame.retrieved_context.str.len().eq(0).any():
            raise ValueError(f"{name}: expected 1,000 nonempty contexts")
        if not frame.golden_answer.eq(frame.choice).all():
            raise ValueError(f"{name}: golden answer is not ContractNLI choice")
    if len(audit) != 1_000:
        raise ValueError("audit does not cover every paired row")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ranked, source, audit = build_inputs(
        pd.read_parquet(args.template),
        pd.read_parquet(ROOT / "data/contractnli/questions.parquet"),
        pd.read_parquet(args.cache / "sentence-anchors.parquet"),
        pd.read_parquet(args.cache / "sentence-snapshot.parquet"),
    )
    validate_pair(ranked, source, audit)
    atomic_parquet(ranked, args.output_dir / "ranked-windows.parquet")
    atomic_parquet(source, args.output_dir / "global-source-order.parquet")
    summary = {
        "rows": len(ranked),
        "sorted_query_id_hash": sorted_value_hash(ranked.query_id),
        "sorted_source_row_hash": sorted_value_hash(ranked.source_row),
        "exact": audit.exact.mean(), "relaxed": audit.relaxed.mean(),
        "mean_sentences": audit.sentence_count.mean(),
        "mean_deduplicated_sentences": audit.deduplicated_sentences.mean(),
        "mean_payload_chars": audit.payload_chars.mean(),
        "ranked_mean_chars": ranked.retrieved_context.str.len().mean(),
        "ranked_mean_tokens": ranked.retrieved_context.map(
            lambda value: len(normalize(value).split())
        ).mean(),
        "source_mean_chars": source.retrieved_context.str.len().mean(),
        "source_mean_tokens": source.retrieved_context.map(
            lambda value: len(normalize(value).split())
        ).mean(),
        "identical_prompt_rows": int(ranked.retrieved_context.eq(source.retrieved_context).sum()),
        "ranked_ordered_context_hash": ordered_context_hash(ranked),
        "source_ordered_context_hash": ordered_context_hash(source),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
