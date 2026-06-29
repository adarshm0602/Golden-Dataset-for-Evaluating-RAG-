#!/usr/bin/env python3
"""
Generate candidate question-answer pairs for RAG evaluation.

Reads Markdown summaries from data/summaries/ and cleaned transcripts from
data/cleaned_transcripts/, then uses Claude to produce approximately 20 grounded
QA pairs with difficulty, topic, and retrieval-usefulness notes.

Output: candidate_questions.csv at the project root.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

from anthropic import Anthropic, APIError
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_DIR = PROJECT_ROOT / "data" / "summaries"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "cleaned_transcripts"
OUTPUT_CSV = PROJECT_ROOT / "candidate_questions.csv"

TARGET_QUESTION_COUNT = 20
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8192

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

Difficulty = Literal["Easy", "Medium", "Hard"]


class CandidateQuestion(BaseModel):
    """One grounded QA pair for RAG evaluation."""

    question: str = Field(description="A natural-language question a user might ask.")
    answer: str = Field(
        description="A concise, factually correct answer grounded only in the source material."
    )
    video: str = Field(description="Exact video title the answer comes from.")
    timestamp: str = Field(
        description="Start timestamp (HH:MM:SS) in the video where the answer is supported."
    )
    difficulty: Difficulty = Field(description="Easy, Medium, or Hard.")
    topic: str = Field(description="Short label for the subject area of the question.")
    retrieval_usefulness: str = Field(
        description="Why this question is useful for evaluating retrieval quality."
    )


class CandidateQuestionBatch(BaseModel):
    """Structured batch of candidate questions returned by the model."""

    questions: list[CandidateQuestion] = Field(
        description="Approximately 20 diverse candidate QA pairs across all videos."
    )


SYSTEM_PROMPT = """\
You are building a golden evaluation dataset for Retrieval-Augmented Generation (RAG) systems.

Given summaries and timestamped transcripts from educational videos, generate candidate
question-answer pairs that will later be used to benchmark retrieval quality.

Rules:
- Ground every answer strictly in the provided source material; do not invent facts.
- Each question must be answerable from a specific passage in one video.
- Use the exact video title from the source metadata for the "video" field.
- Timestamps must point to where the supporting content appears (use the paragraph timestamps).
- Format timestamps as HH:MM:SS (e.g. 00:04:32).
- Spread questions across all provided videos when possible (~5 per video for 4 videos).
- Vary difficulty: Easy (direct fact/definition), Medium (conceptual explanation),
  Hard (synthesis, comparison, or multi-step reasoning).
- Vary topics to cover different themes in each video.
- Write questions in the same language as the dominant language of the source video.
- For retrieval_usefulness, explain what retrieval skill the question tests, e.g.:
  exact terminology lookup, semantic paraphrase matching, disambiguation among similar concepts,
  locating a specific example, or retrieving a definition buried in a long passage.
- Prefer questions that would fail if retrieval returns the wrong video or an adjacent topic.
- Do not duplicate or near-duplicate questions.
"""


PLACEHOLDER_KEYS = {
    "your-api-key-here",
    "your_api_key_here",
    "sk-ant-api03-...your-real-key...",
}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS for CSV output."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_anthropic_client() -> tuple[Anthropic, str, int]:
    """Initialize the Anthropic client from environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment or to a .env file in the project root."
        )

    if api_key.strip() in PLACEHOLDER_KEYS:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is still the placeholder value in .env. "
            "Replace it with your real key from https://console.anthropic.com/settings/keys"
        )

    api_key = api_key.strip()
    if api_key.startswith(("'", '"')) or api_key.endswith(("'", '"')):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY appears to be wrapped in quotes. Remove the quotes in .env."
        )

    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    return Anthropic(api_key=api_key), model, max_tokens


def load_cleaned_transcripts(transcripts_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all cleaned transcript JSON files keyed by video_id."""
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Cleaned transcript directory not found: {transcripts_dir}")

    transcripts: dict[str, dict[str, Any]] = {}
    for path in sorted(transcripts_dir.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        video_id = str(data.get("video_id", path.stem))
        transcripts[video_id] = data

    if not transcripts:
        raise FileNotFoundError(f"No cleaned transcript JSON files found in {transcripts_dir}")

    return transcripts


def load_summaries(summaries_dir: Path) -> dict[str, str]:
    """Load Markdown summaries keyed by video_id. Missing summaries are omitted."""
    summaries: dict[str, str] = {}
    if not summaries_dir.exists():
        return summaries

    for path in sorted(summaries_dir.glob("*.md")):
        summaries[path.stem] = path.read_text(encoding="utf-8")

    return summaries


def build_transcript_text(cleaned: dict[str, Any]) -> str:
    """Flatten timestamped paragraphs into a single text block for the LLM."""
    paragraphs = cleaned.get("paragraphs", [])
    lines: list[str] = []

    for index, paragraph in enumerate(paragraphs, start=1):
        start = float(paragraph["start"])
        end = float(paragraph["end"])
        text = str(paragraph.get("text", "")).strip()
        if not text:
            continue
        timestamp = format_timestamp(start)
        end_timestamp = format_timestamp(end)
        lines.append(f"[Paragraph {index} | {timestamp} - {end_timestamp}]\n{text}")

    return "\n\n".join(lines)


def build_video_context(
    video_id: str,
    cleaned: dict[str, Any],
    summary: str | None,
) -> str:
    """Assemble one video's summary and transcript for the generation prompt."""
    title = cleaned.get("title", "Unknown")
    channel = cleaned.get("channel", "Unknown")
    url = cleaned.get("url", "")
    transcript_text = build_transcript_text(cleaned)

    sections = [
        f"### Video: {title}",
        f"Video ID: {video_id}",
        f"Channel: {channel}",
        f"URL: {url}",
    ]

    if summary:
        sections.extend(["", "#### Summary", summary.strip()])
    else:
        sections.append("")
        sections.append("#### Summary")
        sections.append("_No summary available; use the transcript below._")

    sections.extend(["", "#### Timestamped Transcript", transcript_text])
    return "\n".join(sections)


def build_generation_prompt(
    transcripts: dict[str, dict[str, Any]],
    summaries: dict[str, str],
    target_count: int,
) -> str:
    """Build the user prompt containing all video sources."""
    video_blocks = [
        build_video_context(video_id, cleaned, summaries.get(video_id))
        for video_id, cleaned in transcripts.items()
    ]

    return f"""\
Generate approximately {target_count} candidate question-answer pairs across the videos below.

Aim for a balanced mix across videos, difficulties, and topics.

{chr(10).join(video_blocks)}
"""


def generate_candidate_questions(
    client: Anthropic,
    model: str,
    max_tokens: int,
    transcripts: dict[str, dict[str, Any]],
    summaries: dict[str, str],
    target_count: int = TARGET_QUESTION_COUNT,
) -> list[CandidateQuestion]:
    """Call Claude to produce structured candidate QA pairs."""
    user_prompt = build_generation_prompt(transcripts, summaries, target_count)

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=CandidateQuestionBatch,
        temperature=0.4,
    )

    parsed = response.parsed_output
    if parsed is None or not parsed.questions:
        raise ValueError("Model returned no candidate questions.")

    return parsed.questions


def save_candidate_questions_csv(questions: list[CandidateQuestion], output_path: Path) -> None:
    """Write candidate questions to CSV with the required columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Question",
        "Answer",
        "Video",
        "Timestamp",
        "Difficulty",
        "Topic",
        "Why this question is useful for evaluating retrieval",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in questions:
            writer.writerow(
                {
                    "Question": item.question.strip(),
                    "Answer": item.answer.strip(),
                    "Video": item.video.strip(),
                    "Timestamp": item.timestamp.strip(),
                    "Difficulty": item.difficulty,
                    "Topic": item.topic.strip(),
                    "Why this question is useful for evaluating retrieval": item.retrieval_usefulness.strip(),
                }
            )


def main() -> int:
    """Generate candidate QA pairs from summaries and transcripts."""
    try:
        transcripts = load_cleaned_transcripts(TRANSCRIPTS_DIR)
        summaries = load_summaries(SUMMARIES_DIR)
        client, model, max_tokens = get_anthropic_client()
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    missing_summaries = [video_id for video_id in transcripts if video_id not in summaries]
    if missing_summaries:
        logger.warning(
            "No summary for %d video(s): %s. Proceeding with transcripts only for those.",
            len(missing_summaries),
            ", ".join(missing_summaries),
        )
    else:
        logger.info("Loaded summaries for all %d video(s).", len(transcripts))

    logger.info(
        "Generating ~%d candidate questions from %d transcript(s) using model '%s'",
        TARGET_QUESTION_COUNT,
        len(transcripts),
        model,
    )

    try:
        questions = generate_candidate_questions(
            client,
            model,
            max_tokens,
            transcripts,
            summaries,
            TARGET_QUESTION_COUNT,
        )
        save_candidate_questions_csv(questions, OUTPUT_CSV)
    except (ValueError, APIError, OSError) as exc:
        logger.error("Failed to generate candidate questions: %s", exc)
        return 1

    logger.info("Saved %d candidate question(s) to %s", len(questions), OUTPUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
