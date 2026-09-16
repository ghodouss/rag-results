#!/usr/bin/env python3
"""Run E5/OpenAI generation and independent judging on all three datasets."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/e2e/embedding-retrievals"
MODELS = ("google/gemini-2.5-flash", "openai/gpt-4o-mini")
JUDGES = (
    ("anthropic/claude-haiku-4.5", "binary"),
    ("anthropic/claude-sonnet-4", "scale-1-5"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_contractnli_full_dual_label_matrix as dual  # noqa: E402
import run_contractnli_full_prompt_matrix as matrix  # noqa: E402
import run_contractnli_prompt_pilot as pilot  # noqa: E402
from rag_eval.common import atomic_parquet  # noqa: E402
from rag_eval.llm import model_slug, text_sha256  # noqa: E402


def contract_frame(source: pd.DataFrame, method: str) -> pd.DataFrame:
    source = source.sort_values(["source_row", "query_id"], kind="stable").reset_index(drop=True)
    source["selection_rank"] = range(1, len(source) + 1)
    source["selection_version"] = "embedding-retrievals-full-v1"
    source["selection_key"] = source.query_id.astype(str)
    source["retrieval_method"] = method
    source["retrieval_context_sha256"] = source.retrieved_context.astype(str).map(text_sha256)
    source["retrieval_error"] = ""
    if "reference_evidence" not in source:
        source["reference_evidence"] = ""
    return source[[
        "selection_rank", "selection_version", "selection_key", "dataset",
        "query_id", "split", "source_row", "question", "golden_answer",
        "reference_evidence", "retrieval_method", "retrieved_context",
        "retrieval_context_sha256", "retrieval_error",
    ]].copy()


def prepare_contract() -> dict[str, pd.DataFrame]:
    e5 = pd.read_parquet(ROOT / "results/contractnli/e5.parquet")
    e5 = e5.sort_values(["source_row", "query_id"], kind="stable").reset_index(drop=True)
    raw_openai = pd.read_parquet(
        ROOT / "data/contractnli/retrieval/openai-embedding-model-unspecified.parquet"
    ).reset_index(drop=True)
    if len(e5) != 6_173 or len(raw_openai) != len(e5):
        raise ValueError("ContractNLI embedding inputs must contain 6,173 rows")
    suffix = raw_openai.query_id.astype(str).str.extract(r"(\d+)$")[0].astype(int)
    if not suffix.equals(pd.Series(range(len(raw_openai)))):
        raise ValueError("ContractNLI OpenAI rows are not in source order")
    if not raw_openai.question.astype(str).equals(e5.question.astype(str)):
        raise ValueError("ContractNLI OpenAI questions do not align")
    openai = e5.copy()
    openai["retrieved_context"] = raw_openai.retrieved_context.astype(str).to_numpy()
    frames = {
        "e5-small-v2": contract_frame(e5, "intfloat/e5-small-v2"),
        "openai": contract_frame(openai, "openai/embedding-model-unspecified"),
    }
    for condition, frame in frames.items():
        if frame.retrieved_context.fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"ContractNLI {condition} contains empty contexts")
        atomic_parquet(frame, OUTPUT / "contractnli/input" / f"{condition}.parquet")
    return frames


def prepare_qa(dataset: str) -> Path:
    if dataset == "bioasq":
        frames = [
            pd.read_parquet(ROOT / f"results/{dataset}/{method}.parquet")
            for method in ("e5", "openai")
        ]
    else:
        questions = pd.read_parquet(ROOT / f"data/{dataset}/questions.parquet")
        questions = questions.sort_values(["source_row", "query_id"], kind="stable").reset_index(drop=True)
        frames = []
        for filename in ("e5-small-v2.parquet", "openai-embedding-model-unspecified.parquet"):
            raw = pd.read_parquet(ROOT / f"data/{dataset}/retrieval/{filename}").reset_index(drop=True)
            if len(raw) != len(questions):
                raise ValueError(f"{dataset}/{filename}: row-count mismatch")
            if not raw.question.astype(str).equals(questions.question.astype(str)):
                raise ValueError(f"{dataset}/{filename}: question-order mismatch")
            frame = questions.copy()
            frame["retrieval_method"] = raw.retrieval_method.astype(str).to_numpy()
            frame["retrieved_context"] = raw.retrieved_context.astype(str).to_numpy()
            frames.append(frame)
    required = [
        "dataset", "query_id", "question", "golden_answer",
        "retrieval_method", "retrieved_context",
    ]
    combined = pd.concat((frame[required] for frame in frames), ignore_index=True)
    expected = {"bioasq": 8_774, "finqa": 12_502}[dataset]
    if len(combined) != expected:
        raise ValueError(f"{dataset}: expected {expected} rows, found {len(combined)}")
    if combined.duplicated(["query_id", "retrieval_method"]).any():
        raise ValueError(f"{dataset}: duplicate query/retrieval pairs")
    if combined.retrieved_context.fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"{dataset}: empty contexts")
    target = OUTPUT / dataset / "input/embeddings.parquet"
    atomic_parquet(combined, target)
    return target


async def command(*args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable, *args, cwd=str(ROOT)
    )
    code = await process.wait()
    if code:
        raise RuntimeError(f"command failed ({code}): {' '.join(args)}")


async def settle(label: str, awaitable):
    """Let every parallel cell checkpoint before reporting stage failures."""
    try:
        return await awaitable
    except (SystemExit, RuntimeError) as error:
        print(f"CELL FAILED [{label}]: {error}", flush=True)
        return error


async def qa_generation(dataset: str, source: Path, model: str,
                        concurrency: int, checkpoint_every: int) -> Path:
    target = OUTPUT / dataset / "generations" / f"{model_slug(model)}.parquet"
    await command(
        "scripts/generate_answers.py", dataset,
        "--model", model, "--reasoning-effort", "none",
        "--input", str(source), "--output", str(target),
        "--concurrency", str(concurrency),
        "--checkpoint-every", str(checkpoint_every),
    )
    return target


async def qa_judgment(dataset: str, generation: Path, generator: str,
                      judge: str, mode: str, concurrency: int,
                      checkpoint_every: int) -> Path:
    target = (
        OUTPUT / dataset / "judgments"
        / f"{model_slug(generator)}__judge-{model_slug(judge)}.parquet"
    )
    await command(
        "scripts/judge_answers.py", dataset,
        "--generator-model", generator, "--judge-model", judge,
        "--judge-mode", mode, "--reasoning-effort", "none",
        "--input", str(generation), "--output", str(target),
        "--concurrency", str(concurrency),
        "--checkpoint-every", str(checkpoint_every),
    )
    return target


def prune_retryable_qa_sonnet_errors() -> None:
    """Keep failed Sonnet rows retryable until a complete final file exists."""
    judge = "anthropic/claude-sonnet-4"
    for dataset in ("bioasq", "finqa"):
        for generator in MODELS:
            target = (
                OUTPUT / dataset / "judgments"
                / f"{model_slug(generator)}__judge-{model_slug(judge)}.parquet"
            )
            checkpoint = target.parent / ".checkpoints" / target.name
            if target.exists() or not checkpoint.exists():
                continue
            frame = pd.read_parquet(checkpoint)
            failed = frame.judge_error.fillna("").astype(str).ne("")
            if failed.any():
                print(f"retrying {int(failed.sum()):,} prior errors from {checkpoint}")
                atomic_parquet(frame.loc[~failed].copy(), checkpoint)


async def run(concurrency: int, checkpoint_every: int) -> None:
    contract = prepare_contract()
    qa_inputs = {dataset: prepare_qa(dataset) for dataset in ("bioasq", "finqa")}
    provenance = {
        "scope": "E5 and OpenAI embedding retrieval E2E",
        "datasets": {"contractnli": 6173, "bioasq": 4387, "finqa": 6251},
        "generators": list(MODELS),
        "judges": {judge: mode for judge, mode in JUDGES},
        "concurrency_per_cell": concurrency,
        "contractnli_prompt_version": dual.GENERATOR_PROMPT_VERSION,
        "qa_prompt_version": "grounded-answer-v2-openrouter",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    jobs = []
    metadata = []
    for condition, frame in contract.items():
        for model in MODELS:
            root = OUTPUT / "contractnli" / condition
            jobs.append(settle(
                f"generate/contractnli/{condition}/{model}",
                pilot.run_generation(
                    frame, model, dual.GENERATOR_PROMPT_VERSION,
                    dual.GENERATOR_SYSTEM_PROMPT, root,
                    concurrency, checkpoint_every,
                ),
            ))
            metadata.append(("contractnli", condition, model, root))
    for dataset, source in qa_inputs.items():
        for model in MODELS:
            jobs.append(settle(
                f"generate/{dataset}/{model}",
                qa_generation(dataset, source, model, concurrency, checkpoint_every),
            ))
            metadata.append((dataset, "embeddings", model, None))
    generations = await asyncio.gather(*jobs)
    failures = [value for value in generations if isinstance(value, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} generation cells need another resume")

    matrix.SONNET_MODEL = "anthropic/claude-sonnet-4"
    matrix.GENERATOR_PROMPT_VERSION = dual.GENERATOR_PROMPT_VERSION
    prune_retryable_qa_sonnet_errors()
    jobs = []
    for path, (dataset, condition, model, root) in zip(generations, metadata):
        if dataset == "contractnli":
            jobs.extend((
                settle(
                    f"judge/contractnli/{condition}/{model}/haiku",
                    pilot.run_judgment(
                        path, matrix.HAIKU_MODEL, matrix.HAIKU_PROMPT_VERSION,
                        matrix.HAIKU_SYSTEM_PROMPT, root,
                        concurrency, checkpoint_every,
                    ),
                ),
                settle(
                    f"judge/contractnli/{condition}/{model}/sonnet4",
                    matrix.run_scale_judgment(
                        path,
                        matrix.sonnet_path(OUTPUT / "contractnli", condition, model),
                        concurrency, checkpoint_every,
                    ),
                ),
            ))
        else:
            for judge, mode in JUDGES:
                jobs.append(settle(
                    f"judge/{dataset}/{model}/{judge}",
                    qa_judgment(
                        dataset, path, model, judge, mode,
                        concurrency, checkpoint_every,
                    ),
                ))
    judgments = await asyncio.gather(*jobs)
    failures = [value for value in judgments if isinstance(value, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} judgment cells need another resume")
    print(f"complete: {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    args = parser.parse_args()
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")
    asyncio.run(run(args.concurrency, args.checkpoint_every))


if __name__ == "__main__":
    main()
