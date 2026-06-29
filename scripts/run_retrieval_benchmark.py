#!/usr/bin/env python3
"""
Run a retrieval benchmark against candidate and golden QA datasets.

Indexes paragraph chunks from cleaned transcripts, embeds queries with a
multilingual sentence-transformer, and scores Hit@K and MRR against
timestamp-grounded ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "cleaned_transcripts"
CANDIDATE_CSV = PROJECT_ROOT / "candidate_questions.csv"
GOLDEN_CSV = PROJECT_ROOT / "golden_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
JSON_OUTPUT = REPORTS_DIR / "benchmark_results.json"
MD_OUTPUT = REPORTS_DIR / "benchmark_results.md"

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_VALUES = (1, 3, 5)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit: a timestamped transcript paragraph."""

    chunk_id: int
    video_id: str
    title: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class EvalRow:
    """One benchmark question with ground-truth citation."""

    question: str
    video: str
    timestamp_seconds: float
    difficulty: str
    query_type: str


def parse_timestamp(value: str) -> float:
    """Convert HH:MM:SS to seconds."""
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS timestamp, got {value!r}")

    hours, minutes, seconds = (int(part) for part in parts)
    return float(hours * 3600 + minutes * 60 + seconds)


def query_type_for_difficulty(difficulty: str) -> str:
    """Map difficulty to the benchmark query-type bucket."""
    normalized = difficulty.strip().lower()
    if normalized == "easy":
        return "factual"
    if normalized == "hard":
        return "multi-hop/synthesis"
    return "medium"


def load_chunks(transcripts_dir: Path) -> list[Chunk]:
    """Load all paragraph chunks from cleaned transcript JSON files."""
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Cleaned transcript directory not found: {transcripts_dir}")

    chunks: list[Chunk] = []
    chunk_id = 0

    for path in sorted(transcripts_dir.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)

        video_id = str(payload.get("video_id", path.stem))
        title = str(payload.get("title", ""))
        paragraphs = payload.get("paragraphs", [])

        if not isinstance(paragraphs, list):
            raise ValueError(f"Invalid paragraphs in {path.name}")

        for paragraph in paragraphs:
            text = str(paragraph.get("text", "")).strip()
            if not text:
                continue

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    video_id=video_id,
                    title=title,
                    start=float(paragraph["start"]),
                    end=float(paragraph["end"]),
                    text=text,
                )
            )
            chunk_id += 1

    if not chunks:
        raise FileNotFoundError(f"No paragraph chunks found in {transcripts_dir}")

    return chunks


def load_eval_rows(csv_path: Path) -> list[EvalRow]:
    """Load benchmark questions from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    rows: list[EvalRow] = []
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            rows.append(
                EvalRow(
                    question=str(record["Question"]).strip(),
                    video=str(record["Video"]).strip(),
                    timestamp_seconds=parse_timestamp(str(record["Timestamp"])),
                    difficulty=str(record["Difficulty"]).strip(),
                    query_type=query_type_for_difficulty(str(record["Difficulty"])),
                )
            )

    if not rows:
        raise ValueError(f"No evaluation rows found in {csv_path}")

    return rows


def is_correct_chunk(chunk: Chunk, row: EvalRow) -> bool:
    """Return True when a chunk matches the ground-truth citation."""
    return (
        chunk.title == row.video
        and chunk.start <= row.timestamp_seconds <= chunk.end
    )


def retrieve_top_k(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    k: int,
) -> list[int]:
    """Return indices of the top-k most similar chunks for one query."""
    scores = chunk_embeddings @ query_embedding
    if k >= len(scores):
        return np.argsort(-scores).tolist()

    top_indices = np.argpartition(-scores, k - 1)[:k]
    return top_indices[np.argsort(-scores[top_indices])].tolist()


def score_retrieval(
    rows: list[EvalRow],
    chunks: list[Chunk],
    query_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    top_k_values: tuple[int, ...] = TOP_K_VALUES,
) -> dict[str, Any]:
    """Compute Hit@K, MRR, and per-group breakdowns for one dataset."""
    per_question: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    hits_by_k: dict[int, list[bool]] = {k: [] for k in top_k_values}

    for row_index, row in enumerate(rows):
        ranked_indices = retrieve_top_k(
            query_embeddings[row_index],
            chunk_embeddings,
            max(top_k_values),
        )

        first_hit_rank: int | None = None
        for rank, chunk_index in enumerate(ranked_indices, start=1):
            if is_correct_chunk(chunks[chunk_index], row):
                first_hit_rank = rank
                break

        reciprocal_ranks.append(0.0 if first_hit_rank is None else 1.0 / first_hit_rank)

        question_hits: dict[str, bool] = {}
        for k in top_k_values:
            hit = any(
                is_correct_chunk(chunks[chunk_index], row)
                for chunk_index in ranked_indices[:k]
            )
            hits_by_k[k].append(hit)
            question_hits[f"hit@{k}"] = hit

        per_question.append(
            {
                "question": row.question,
                "video": row.video,
                "timestamp_seconds": row.timestamp_seconds,
                "difficulty": row.difficulty,
                "query_type": row.query_type,
                "first_hit_rank": first_hit_rank,
                "reciprocal_rank": reciprocal_ranks[-1],
                **question_hits,
            }
        )

    def summarize_group(predicate) -> dict[str, Any]:
        indices = [index for index, row in enumerate(rows) if predicate(row)]
        if not indices:
            return {"count": 0}

        return {
            "count": len(indices),
            "mrr": round(float(np.mean([reciprocal_ranks[i] for i in indices])), 4),
            **{
                f"hit@{k}": round(
                    float(np.mean([hits_by_k[k][i] for i in indices])),
                    4,
                )
                for k in top_k_values
            },
        }

    overall = {
        "count": len(rows),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        **{f"hit@{k}": round(float(np.mean(hits_by_k[k])), 4) for k in top_k_values},
    }

    by_difficulty = {
        difficulty: summarize_group(lambda row, d=difficulty: row.difficulty == d)
        for difficulty in sorted({row.difficulty for row in rows})
    }
    by_query_type = {
        query_type: summarize_group(lambda row, q=query_type: row.query_type == q)
        for query_type in sorted({row.query_type for row in rows})
    }

    return {
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_query_type": by_query_type,
        "per_question": per_question,
    }


def format_percent(value: float) -> str:
    """Format a ratio as a percentage string."""
    return f"{value * 100:.1f}%"


def build_markdown_report(payload: dict[str, Any]) -> str:
    """Render a human-readable markdown summary of benchmark results."""
    lines = [
        "# Retrieval Benchmark Results",
        "",
        f"Generated: {payload['generated_at']}",
        f"Embedding model: `{payload['embedding_model']}`",
        f"Corpus chunks: {payload['chunk_count']}",
        "",
        "## Overall",
        "",
        "| Dataset | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---------|-----------|-------|-------|-------|-----|",
    ]

    for dataset_name in ("candidate_questions", "golden_dataset"):
        overall = payload["datasets"][dataset_name]["overall"]
        lines.append(
            "| {name} | {count} | {hit1} | {hit3} | {hit5} | {mrr} |".format(
                name=dataset_name.replace("_", " "),
                count=overall["count"],
                hit1=format_percent(overall["hit@1"]),
                hit3=format_percent(overall["hit@3"]),
                hit5=format_percent(overall["hit@5"]),
                mrr=f"{overall['mrr']:.3f}",
            )
        )

    for dataset_name in ("candidate_questions", "golden_dataset"):
        dataset = payload["datasets"][dataset_name]
        lines.extend(
            [
                "",
                f"## {dataset_name.replace('_', ' ').title()} — By Difficulty",
                "",
                "| Difficulty | Count | Hit@1 | Hit@3 | Hit@5 | MRR |",
                "|------------|-------|-------|-------|-------|-----|",
            ]
        )
        for difficulty, stats in dataset["by_difficulty"].items():
            if stats["count"] == 0:
                continue
            lines.append(
                "| {difficulty} | {count} | {hit1} | {hit3} | {hit5} | {mrr} |".format(
                    difficulty=difficulty,
                    count=stats["count"],
                    hit1=format_percent(stats["hit@1"]),
                    hit3=format_percent(stats["hit@3"]),
                    hit5=format_percent(stats["hit@5"]),
                    mrr=f"{stats['mrr']:.3f}",
                )
            )

        lines.extend(
            [
                "",
                f"## {dataset_name.replace('_', ' ').title()} — By Query Type",
                "",
                "| Query Type | Count | Hit@1 | Hit@3 | Hit@5 | MRR |",
                "|------------|-------|-------|-------|-------|-----|",
            ]
        )
        for query_type, stats in dataset["by_query_type"].items():
            if stats["count"] == 0:
                continue
            lines.append(
                "| {query_type} | {count} | {hit1} | {hit3} | {hit5} | {mrr} |".format(
                    query_type=query_type,
                    count=stats["count"],
                    hit1=format_percent(stats["hit@1"]),
                    hit3=format_percent(stats["hit@3"]),
                    hit5=format_percent(stats["hit@5"]),
                    mrr=f"{stats['mrr']:.3f}",
                )
            )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python3 scripts/run_retrieval_benchmark.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(model_name: str) -> dict[str, Any]:
    """Index corpus, embed queries, and score both evaluation datasets."""
    chunks = load_chunks(TRANSCRIPTS_DIR)
    candidate_rows = load_eval_rows(CANDIDATE_CSV)
    golden_rows = load_eval_rows(GOLDEN_CSV)

    logger.info("Loading embedding model '%s'", model_name)
    model = SentenceTransformer(model_name)

    chunk_texts = [chunk.text for chunk in chunks]
    logger.info("Encoding %d paragraph chunk(s)", len(chunk_texts))
    chunk_embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    all_rows = candidate_rows + golden_rows
    query_texts = [row.question for row in all_rows]
    logger.info("Encoding %d question(s)", len(query_texts))
    query_embeddings = model.encode(
        query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    candidate_embeddings = query_embeddings[: len(candidate_rows)]
    golden_embeddings = query_embeddings[len(candidate_rows) :]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_name,
        "chunk_count": len(chunks),
        "datasets": {
            "candidate_questions": score_retrieval(
                candidate_rows,
                chunks,
                candidate_embeddings,
                chunk_embeddings,
            ),
            "golden_dataset": score_retrieval(
                golden_rows,
                chunks,
                golden_embeddings,
                chunk_embeddings,
            ),
        },
    }


def save_reports(payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and Markdown benchmark reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    MD_OUTPUT.write_text(build_markdown_report(payload), encoding="utf-8")
    return JSON_OUTPUT, MD_OUTPUT


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run retrieval benchmark on QA datasets.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformers model name (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def main() -> int:
    """Run the retrieval benchmark and write reports."""
    args = parse_args()

    try:
        payload = run_benchmark(args.model)
        json_path, md_path = save_reports(payload)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1

    candidate = payload["datasets"]["candidate_questions"]["overall"]
    golden = payload["datasets"]["golden_dataset"]["overall"]
    logger.info(
        "Candidate set: Hit@1=%s Hit@3=%s Hit@5=%s MRR=%.3f",
        format_percent(candidate["hit@1"]),
        format_percent(candidate["hit@3"]),
        format_percent(candidate["hit@5"]),
        candidate["mrr"],
    )
    logger.info(
        "Golden set: Hit@1=%s Hit@3=%s Hit@5=%s MRR=%.3f",
        format_percent(golden["hit@1"]),
        format_percent(golden["hit@3"]),
        format_percent(golden["hit@5"]),
        golden["mrr"],
    )
    logger.info("Wrote %s and %s", json_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
