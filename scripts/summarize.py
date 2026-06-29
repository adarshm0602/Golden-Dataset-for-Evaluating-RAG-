#!/usr/bin/env python3
"""
Generate structured Markdown summaries from cleaned video transcripts.

Reads paragraph-level JSON from data/cleaned_transcripts/, calls Claude to extract
an executive summary, topics, concepts, definitions, and examples, then writes
one Markdown file per video to data/summaries/.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIError
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Project paths: script lives in scripts/, repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "cleaned_transcripts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "summaries"

# Transcripts longer than this are summarized in chunks, then merged.
CHUNK_CHAR_LIMIT = 20_000
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Definition(BaseModel):
    """A term and its explanation extracted from the transcript."""

    term: str = Field(description="The concept or term being defined.")
    definition: str = Field(description="A concise definition grounded in the transcript.")


class TranscriptSummary(BaseModel):
    """Structured summary sections used for both chunk and final outputs."""

    executive_summary: str = Field(
        description="A concise 3-5 sentence overview of the video's main message."
    )
    main_topics: list[str] = Field(
        description="High-level topics covered, as short bullet phrases."
    )
    key_concepts: list[str] = Field(
        description="Important ideas, mechanisms, or takeaways from the content."
    )
    important_definitions: list[Definition] = Field(
        description="Explicit or implicit definitions introduced in the transcript."
    )
    important_examples: list[str] = Field(
        description="Concrete examples, analogies, or demonstrations mentioned."
    )


SYSTEM_PROMPT = """\
You are an expert educational content analyst building study materials from video transcripts.

Given a transcript, produce a structured summary with these sections:
- Executive Summary: 3-5 sentences capturing the video's purpose and conclusions.
- Main Topics: ordered list of major themes discussed.
- Key Concepts: core ideas a learner should remember.
- Important Definitions: terms with concise definitions grounded only in the transcript.
- Important Examples: concrete examples, analogies, or demonstrations used.

Rules:
- Base every point strictly on the provided transcript; do not invent facts.
- Prefer clarity and specificity over vague phrasing.
- Write in the same language as the transcript.
- If a section has no relevant content, return an empty list (or a brief note for the summary).
"""


MERGE_SYSTEM_PROMPT = """\
You are consolidating partial summaries of different sections of the same video transcript.

Merge the partial summaries into one cohesive structured summary:
- Remove duplicate topics, concepts, definitions, and examples.
- Preserve the most specific and informative wording.
- Keep the executive summary concise (3-5 sentences) and representative of the full video.
- Write in the same language as the partial summaries.
"""


def load_cleaned_transcripts(input_dir: Path) -> list[Path]:
    """Return all cleaned transcript JSON files sorted by name."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Cleaned transcript directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No cleaned transcript JSON files found in {input_dir}")

    return files


def load_cleaned_transcript(path: Path) -> dict[str, Any]:
    """Load one cleaned transcript document."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def build_transcript_text(cleaned: dict[str, Any]) -> str:
    """
    Flatten timestamped paragraphs into a single text block for the LLM.

    Paragraph headers include timestamps so the model can anchor examples when useful.
    """
    paragraphs = cleaned.get("paragraphs", [])
    lines: list[str] = []

    for index, paragraph in enumerate(paragraphs, start=1):
        start = float(paragraph["start"])
        end = float(paragraph["end"])
        text = str(paragraph.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[Paragraph {index} | {start:.1f}s - {end:.1f}s]\n{text}")

    return "\n\n".join(lines)


def chunk_text(text: str, max_chars: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """
    Split long transcripts on paragraph boundaries to stay within model limits.

    Each chunk stays under max_chars while keeping paragraphs intact.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph) + 2
        if current and current_len + paragraph_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = paragraph_len
        else:
            current.append(paragraph)
            current_len += paragraph_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


PLACEHOLDER_KEYS = {
    "your-api-key-here",
    "your_api_key_here",
    "sk-ant-api03-...your-real-key...",
}


def get_anthropic_client() -> tuple[Anthropic, str, int]:
    """Initialize the Anthropic client and resolve model settings from environment variables."""
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


def summarize_text(
    client: Anthropic,
    model: str,
    max_tokens: int,
    transcript_text: str,
    metadata: dict[str, Any],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    chunk_label: str | None = None,
) -> TranscriptSummary:
    """Call Claude once and parse a validated structured summary response."""
    title = metadata.get("title", "Unknown")
    channel = metadata.get("channel", "Unknown")
    language = metadata.get("language", "Unknown")

    scope = f" (section: {chunk_label})" if chunk_label else ""
    user_prompt = f"""\
Video title: {title}
Channel: {channel}
Language: {language}{scope}

Transcript:
{transcript_text}
"""

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=TranscriptSummary,
        temperature=0.2,
    )

    parsed = response.parsed_output
    if parsed is None:
        raise ValueError(f"Model returned no structured summary for {title}{scope}")

    return parsed


def merge_partial_summaries(
    client: Anthropic,
    model: str,
    max_tokens: int,
    partials: list[TranscriptSummary],
    metadata: dict[str, Any],
) -> TranscriptSummary:
    """Combine chunk-level summaries into one final structured summary."""
    payload = [partial.model_dump() for partial in partials]
    user_prompt = f"""\
Video title: {metadata.get('title', 'Unknown')}
Channel: {metadata.get('channel', 'Unknown')}

Partial summaries (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    return summarize_text(
        client,
        model,
        max_tokens,
        user_prompt,
        metadata,
        system_prompt=MERGE_SYSTEM_PROMPT,
        chunk_label="merged partials",
    )


def summarize_transcript(
    client: Anthropic,
    model: str,
    max_tokens: int,
    cleaned: dict[str, Any],
) -> TranscriptSummary:
    """
    Generate a structured summary for one cleaned transcript.

    Short transcripts are summarized in one pass. Longer transcripts use map-reduce:
    summarize each chunk, then merge the partial summaries.
    """
    transcript_text = build_transcript_text(cleaned)
    if not transcript_text.strip():
        raise ValueError(f"Transcript {cleaned.get('video_id', 'unknown')} has no paragraph text.")

    metadata = {
        "title": cleaned.get("title"),
        "channel": cleaned.get("channel"),
        "language": cleaned.get("language"),
        "language_code": cleaned.get("language_code"),
        "video_id": cleaned.get("video_id"),
        "url": cleaned.get("url"),
    }

    chunks = chunk_text(transcript_text)
    if len(chunks) == 1:
        return summarize_text(client, model, max_tokens, chunks[0], metadata)

    logger.info(
        "Transcript %s is long (%d chars); summarizing in %d chunks",
        cleaned.get("video_id"),
        len(transcript_text),
        len(chunks),
    )

    partials = [
        summarize_text(
            client,
            model,
            max_tokens,
            chunk,
            metadata,
            chunk_label=f"{index}/{len(chunks)}",
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    return merge_partial_summaries(client, model, max_tokens, partials, metadata)


def render_markdown(cleaned: dict[str, Any], summary: TranscriptSummary) -> str:
    """Render the structured summary as a Markdown document."""
    title = cleaned.get("title", "Untitled")
    channel = cleaned.get("channel", "Unknown")
    video_id = cleaned.get("video_id", "unknown")
    url = cleaned.get("url", "")
    language = cleaned.get("language", "Unknown")
    language_code = cleaned.get("language_code", "")

    lines = [
        f"# {title}",
        "",
        f"**Channel:** {channel}  ",
        f"**Video ID:** `{video_id}`  ",
        f"**URL:** {url}  ",
        f"**Language:** {language} ({language_code})  ",
        "",
        "## Executive Summary",
        "",
        summary.executive_summary.strip(),
        "",
        "## Main Topics",
        "",
    ]

    if summary.main_topics:
        lines.extend(f"- {topic.strip()}" for topic in summary.main_topics)
    else:
        lines.append("_No main topics identified._")

    lines.extend(["", "## Key Concepts", ""])
    if summary.key_concepts:
        lines.extend(f"- {concept.strip()}" for concept in summary.key_concepts)
    else:
        lines.append("_No key concepts identified._")

    lines.extend(["", "## Important Definitions", ""])
    if summary.important_definitions:
        for item in summary.important_definitions:
            lines.extend([f"### {item.term.strip()}", "", item.definition.strip(), ""])
    else:
        lines.append("_No important definitions identified._")

    lines.extend(["## Important Examples", ""])
    if summary.important_examples:
        lines.extend(f"- {example.strip()}" for example in summary.important_examples)
    else:
        lines.append("_No important examples identified._")

    return "\n".join(lines).rstrip() + "\n"


def save_summary(cleaned: dict[str, Any], summary: TranscriptSummary, output_dir: Path) -> Path:
    """Write the Markdown summary for one video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = str(cleaned["video_id"])
    output_path = output_dir / f"{video_id}.md"
    output_path.write_text(render_markdown(cleaned, summary), encoding="utf-8")
    return output_path


def main() -> int:
    """Summarize every cleaned transcript and export Markdown files."""
    try:
        cleaned_files = load_cleaned_transcripts(INPUT_DIR)
        client, model, max_tokens = get_anthropic_client()
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    successes = 0
    failures = 0

    logger.info(
        "Summarizing %d cleaned transcript(s) with Claude model '%s' into %s",
        len(cleaned_files),
        model,
        OUTPUT_DIR,
    )

    for index, path in enumerate(cleaned_files, start=1):
        title = path.stem
        try:
            cleaned = load_cleaned_transcript(path)
            title = str(cleaned.get("title", path.stem))
            summary = summarize_transcript(client, model, max_tokens, cleaned)
            output_path = save_summary(cleaned, summary, OUTPUT_DIR)
            successes += 1
            logger.info(
                "[%d/%d] Saved summary for %s -> %s",
                index,
                len(cleaned_files),
                title,
                output_path.name,
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError, APIError) as exc:
            failures += 1
            logger.error("[%d/%d] Failed to summarize %s: %s", index, len(cleaned_files), title, exc)

    logger.info("Finished: %d succeeded, %d failed", successes, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
