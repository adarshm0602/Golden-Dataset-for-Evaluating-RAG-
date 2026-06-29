#!/usr/bin/env python3
"""
Download YouTube video transcripts and save them as JSON and plain-text files.

Reads video metadata from videos.json, fetches captions via youtube-transcript-api,
and writes timestamped outputs to data/raw_transcripts/.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApiException,
)

# Project paths: script lives in scripts/, repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_JSON = PROJECT_ROOT / "videos.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_transcripts"

# Matches common YouTube URL shapes (watch, embed, shorts, youtu.be).
YOUTUBE_ID_PATTERNS = (
    re.compile(r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"),
    re.compile(r"^[A-Za-z0-9_-]{11}$"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_video_id(url_or_id: str) -> str:
    """
    Extract an 11-character YouTube video ID from a URL or bare ID string.

    Raises:
        ValueError: If no valid video ID can be parsed.
    """
    value = url_or_id.strip()

    for pattern in YOUTUBE_ID_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)

    # Handle watch URLs where v= may appear after other query params.
    parsed = urlparse(value)
    if parsed.netloc.endswith("youtube.com") and parsed.path == "/watch":
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id and len(query_id) == 11:
            return query_id

    raise ValueError(f"Could not extract a YouTube video ID from: {url_or_id!r}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm for human-readable transcript files."""
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def load_videos(config_path: Path) -> list[dict[str, Any]]:
    """Load the video registry from videos.json."""
    if not config_path.exists():
        raise FileNotFoundError(f"Video registry not found: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"No videos found in {config_path}")

    return videos


def build_transcript_payload(
    video: dict[str, Any],
    video_id: str,
    fetched_transcript: Any,
) -> dict[str, Any]:
    """Assemble a JSON-serializable transcript document with metadata and snippets."""
    snippets = [
        {
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration,
        }
        for snippet in fetched_transcript
    ]

    return {
        "video_id": video_id,
        "title": video.get("title"),
        "channel": video.get("channel"),
        "url": video.get("url"),
        "language": fetched_transcript.language,
        "language_code": fetched_transcript.language_code,
        "is_generated": fetched_transcript.is_generated,
        "snippet_count": len(snippets),
        "snippets": snippets,
    }


def transcript_to_txt(payload: dict[str, Any]) -> str:
    """Render a readable plain-text transcript with one timestamped line per snippet."""
    header = [
        f"Title: {payload.get('title', 'Unknown')}",
        f"Channel: {payload.get('channel', 'Unknown')}",
        f"Video ID: {payload['video_id']}",
        f"URL: {payload.get('url', '')}",
        f"Language: {payload.get('language')} ({payload.get('language_code')})",
        "",
    ]

    lines = [
        f"[{format_timestamp(snippet['start'])}] {snippet['text']}"
        for snippet in payload["snippets"]
    ]

    return "\n".join(header + lines) + "\n"


def save_transcript_files(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and TXT transcript files for a single video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = payload["video_id"]
    json_path = output_dir / f"{video_id}.json"
    txt_path = output_dir / f"{video_id}.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    txt_path.write_text(transcript_to_txt(payload), encoding="utf-8")

    return json_path, txt_path


def download_transcript(
    api: YouTubeTranscriptApi,
    video: dict[str, Any],
) -> dict[str, Any]:
    """
    Fetch a single video transcript and return the structured payload.

    Uses language preferences from the video entry when provided; otherwise defaults to English.
    """
    url = video.get("url")
    if not url:
        raise ValueError("Video entry is missing a 'url' field.")

    video_id = extract_video_id(url)
    languages = video.get("languages") or ["en"]

    fetched = api.fetch(video_id, languages=languages)
    return build_transcript_payload(video, video_id, fetched)


def main() -> int:
    """Download transcripts for all videos listed in videos.json."""
    try:
        videos = load_videos(VIDEOS_JSON)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1

    api = YouTubeTranscriptApi()
    successes = 0
    failures = 0

    logger.info("Downloading transcripts for %d video(s) into %s", len(videos), OUTPUT_DIR)

    for index, video in enumerate(videos, start=1):
        title = video.get("title", "Untitled")
        try:
            payload = download_transcript(api, video)
            json_path, txt_path = save_transcript_files(payload, OUTPUT_DIR)
            successes += 1
            logger.info(
                "[%d/%d] Saved %s (%s, %d snippets) -> %s, %s",
                index,
                len(videos),
                title,
                payload["language_code"],
                payload["snippet_count"],
                json_path.name,
                txt_path.name,
            )
        except ValueError as exc:
            failures += 1
            logger.error("[%d/%d] Skipping %s: %s", index, len(videos), title, exc)
        except InvalidVideoId as exc:
            failures += 1
            logger.error("[%d/%d] Invalid video ID for %s: %s", index, len(videos), title, exc)
        except TranscriptsDisabled as exc:
            failures += 1
            logger.error("[%d/%d] Transcripts disabled for %s: %s", index, len(videos), title, exc)
        except NoTranscriptFound as exc:
            failures += 1
            logger.error(
                "[%d/%d] No transcript found for %s (languages=%s): %s",
                index,
                len(videos),
                title,
                video.get("languages", ["en"]),
                exc,
            )
        except VideoUnavailable as exc:
            failures += 1
            logger.error("[%d/%d] Video unavailable for %s: %s", index, len(videos), title, exc)
        except (RequestBlocked, IpBlocked) as exc:
            failures += 1
            logger.error(
                "[%d/%d] Request blocked for %s. Try again later or use a proxy: %s",
                index,
                len(videos),
                title,
                exc,
            )
        except CouldNotRetrieveTranscript as exc:
            failures += 1
            logger.error("[%d/%d] Could not retrieve transcript for %s: %s", index, len(videos), title, exc)
        except YouTubeTranscriptApiException as exc:
            failures += 1
            logger.error("[%d/%d] YouTube API error for %s: %s", index, len(videos), title, exc)
        except OSError as exc:
            failures += 1
            logger.error("[%d/%d] Failed to write files for %s: %s", index, len(videos), title, exc)

    logger.info("Finished: %d succeeded, %d failed", successes, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
