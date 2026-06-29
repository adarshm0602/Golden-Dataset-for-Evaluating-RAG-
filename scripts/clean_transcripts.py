#!/usr/bin/env python3
"""
Clean raw YouTube transcripts into coherent, timestamped paragraphs.

Reads JSON files from data/raw_transcripts/, applies preprocessing steps,
and writes cleaned JSON and TXT files to data/cleaned_transcripts/.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Project paths: script lives in scripts/, repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw_transcripts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned_transcripts"

# Paragraph segmentation thresholds.
# Short pauses are normal between subtitle fragments; longer pauses suggest a new thought.
SHORT_PAUSE_SECONDS = 1.5
LONG_PAUSE_SECONDS = 3.0
SOFT_PARAGRAPH_CHARS = 800
HARD_PARAGRAPH_CHARS = 1200

# Sentence-ending punctuation for English and Hindi (Devanagari danda).
SENTENCE_END_PATTERN = re.compile(r"[.!?।…][\"')\]]*$")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm for readable TXT exports."""
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def normalize_whitespace(text: str) -> str:
    """
    Step 1 — Whitespace normalization.

    Strip leading/trailing space and collapse runs of whitespace (including
    non-breaking spaces) into a single ASCII space so fragments can be merged cleanly.
    """
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def snippet_end(snippet: dict[str, Any]) -> float:
    """Return the end time of a subtitle snippet in seconds."""
    return float(snippet["start"]) + float(snippet["duration"])


def ends_sentence(text: str) -> bool:
    """Return True when text appears to end a sentence."""
    return bool(SENTENCE_END_PATTERN.search(text.strip()))


def deduplicate_snippets(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Step 2 — Duplicate fragment removal.

    YouTube captions (especially auto-generated) often repeat lines or emit rolling
    updates where a new snippet fully contains the previous one. This step:
      - drops empty snippets after whitespace cleanup
      - skips exact consecutive duplicates
      - replaces a snippet when the next one is a strict superset (rolling captions)
    """
    deduped: list[dict[str, Any]] = []

    for snippet in snippets:
        text = normalize_whitespace(str(snippet.get("text", "")))
        if not text:
            continue

        if not deduped:
            deduped.append({**snippet, "text": text})
            continue

        previous = deduped[-1]
        previous_text = str(previous["text"])

        # Exact duplicate: keep the first occurrence and ignore the repeat.
        if text == previous_text:
            continue

        # Rolling caption: newer snippet extends the previous one — keep the longer version.
        if previous_text in text and text.startswith(previous_text):
            deduped[-1] = {**snippet, "text": text}
            continue

        # Substring repeat: current snippet adds nothing new.
        if text in previous_text:
            continue

        deduped.append({**snippet, "text": text})

    return deduped


def merge_fragment_text(existing: str, new: str) -> str:
    """
    Step 3a — Fragment text merging.

    Join subtitle fragments with a space. When auto-captions roll forward, the start
    of the new fragment may repeat words from the end of the accumulated text.
    Detect the longest word-level overlap and append only the non-overlapping tail.
    """
    if not existing:
        return new
    if not new:
        return existing
    if existing == new:
        return existing

    existing_words = existing.split()
    new_words = new.split()
    max_overlap = min(len(existing_words), len(new_words))

    overlap_size = 0
    for size in range(max_overlap, 0, -1):
        if existing_words[-size:] == new_words[:size]:
            overlap_size = size
            break

    if overlap_size:
        tail = " ".join(new_words[overlap_size:])
        return f"{existing} {tail}".strip() if tail else existing

    return f"{existing} {new}".strip()


def should_start_new_paragraph(
    current_text: str,
    previous_snippet: dict[str, Any],
    next_snippet: dict[str, Any],
) -> bool:
    """
    Step 3b — Paragraph boundary detection.

    Start a new paragraph when:
      - the speaker pauses for a long time (topic/thought break)
      - the current text ends a sentence and there is a noticeable pause
      - the paragraph is very long (hard limit), or moderately long after a full sentence
    Overlapping timestamps (negative gap) always continue the same paragraph.
    """
    if len(current_text) >= HARD_PARAGRAPH_CHARS:
        return True

    if len(current_text) >= SOFT_PARAGRAPH_CHARS and ends_sentence(current_text):
        return True

    gap = float(next_snippet["start"]) - snippet_end(previous_snippet)

    if gap >= LONG_PAUSE_SECONDS:
        return True

    if ends_sentence(current_text) and gap >= SHORT_PAUSE_SECONDS:
        return True

    return False


def merge_into_paragraphs(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Step 3 — Paragraph assembly.

    Walk deduplicated snippets in time order, merging fragments into paragraphs
    while preserving the start time of the first fragment and the end time of the last.
    """
    if not snippets:
        return []

    paragraphs: list[dict[str, Any]] = []
    paragraph_start = float(snippets[0]["start"])
    paragraph_text = str(snippets[0]["text"])
    paragraph_end = snippet_end(snippets[0])
    source_count = 1

    for index in range(1, len(snippets)):
        previous_snippet = snippets[index - 1]
        snippet = snippets[index]
        snippet_text = str(snippet["text"])

        if should_start_new_paragraph(paragraph_text, previous_snippet, snippet):
            paragraphs.append(
                {
                    "text": paragraph_text,
                    "start": paragraph_start,
                    "end": paragraph_end,
                    "duration": round(paragraph_end - paragraph_start, 3),
                    "source_snippet_count": source_count,
                }
            )
            paragraph_start = float(snippet["start"])
            paragraph_text = snippet_text
            paragraph_end = snippet_end(snippet)
            source_count = 1
            continue

        paragraph_text = merge_fragment_text(paragraph_text, snippet_text)
        paragraph_end = max(paragraph_end, snippet_end(snippet))
        source_count += 1

    paragraphs.append(
        {
            "text": paragraph_text,
            "start": paragraph_start,
            "end": paragraph_end,
            "duration": round(paragraph_end - paragraph_start, 3),
            "source_snippet_count": source_count,
        }
    )

    return paragraphs


def clean_transcript(raw: dict[str, Any]) -> dict[str, Any]:
    """Run the full preprocessing pipeline on one raw transcript document."""
    snippets = raw.get("snippets", [])
    if not isinstance(snippets, list):
        raise ValueError(f"Invalid snippets list for video {raw.get('video_id', 'unknown')}")

    # Pipeline: whitespace cleanup -> deduplication -> paragraph merging.
    normalized = [{**snippet, "text": normalize_whitespace(str(snippet.get("text", "")))} for snippet in snippets]
    normalized = [snippet for snippet in normalized if snippet["text"]]
    deduped = deduplicate_snippets(normalized)
    paragraphs = merge_into_paragraphs(deduped)

    return {
        "video_id": raw.get("video_id"),
        "title": raw.get("title"),
        "channel": raw.get("channel"),
        "url": raw.get("url"),
        "language": raw.get("language"),
        "language_code": raw.get("language_code"),
        "is_generated": raw.get("is_generated"),
        "source_snippet_count": len(snippets),
        "deduped_snippet_count": len(deduped),
        "paragraph_count": len(paragraphs),
        "paragraphs": paragraphs,
    }


def paragraphs_to_txt(payload: dict[str, Any]) -> str:
    """Render cleaned paragraphs as a readable, timestamped plain-text file."""
    header = [
        f"Title: {payload.get('title', 'Unknown')}",
        f"Channel: {payload.get('channel', 'Unknown')}",
        f"Video ID: {payload.get('video_id', 'Unknown')}",
        f"URL: {payload.get('url', '')}",
        f"Language: {payload.get('language')} ({payload.get('language_code')})",
        f"Paragraphs: {payload.get('paragraph_count', 0)}",
        "",
    ]

    body = [
        f"[{format_timestamp(paragraph['start'])} - {format_timestamp(paragraph['end'])}] {paragraph['text']}"
        for paragraph in payload["paragraphs"]
    ]

    return "\n".join(header + body) + "\n"


def save_cleaned_transcript(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write cleaned JSON and TXT outputs for a single video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(payload["video_id"])
    json_path = output_dir / f"{video_id}.json"
    txt_path = output_dir / f"{video_id}.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    txt_path.write_text(paragraphs_to_txt(payload), encoding="utf-8")

    return json_path, txt_path


def load_raw_transcripts(input_dir: Path) -> list[Path]:
    """Return all raw transcript JSON files sorted by name."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Raw transcript directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No raw transcript JSON files found in {input_dir}")

    return files


def process_file(path: Path) -> dict[str, Any]:
    """Load one raw transcript file and return the cleaned payload."""
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)

    return clean_transcript(raw)


def main() -> int:
    """Clean every raw transcript and export results to data/cleaned_transcripts/."""
    try:
        raw_files = load_raw_transcripts(INPUT_DIR)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    successes = 0
    failures = 0

    logger.info("Cleaning %d raw transcript(s) into %s", len(raw_files), OUTPUT_DIR)

    for index, path in enumerate(raw_files, start=1):
        try:
            cleaned = process_file(path)
            json_path, txt_path = save_cleaned_transcript(cleaned, OUTPUT_DIR)
            successes += 1
            logger.info(
                "[%d/%d] %s: %d snippets -> %d paragraphs -> %s, %s",
                index,
                len(raw_files),
                cleaned.get("title", path.stem),
                cleaned["source_snippet_count"],
                cleaned["paragraph_count"],
                json_path.name,
                txt_path.name,
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            failures += 1
            logger.error("[%d/%d] Failed to clean %s: %s", index, len(raw_files), path.name, exc)

    logger.info("Finished: %d succeeded, %d failed", successes, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
