#!/usr/bin/env python3
"""
Select the best golden QA pairs from candidate questions.

Reads candidate_questions.csv, uses Claude to pick the top 5 questions against
explicit quality and diversity criteria, then writes golden_dataset.csv.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from anthropic import Anthropic, APIError
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "candidate_questions.csv"
OUTPUT_CSV = PROJECT_ROOT / "golden_dataset.csv"

GOLDEN_QUESTION_COUNT = 5
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

CSV_COLUMNS = [
    "Question",
    "Answer",
    "Video",
    "Timestamp",
    "Difficulty",
    "Topic",
    "Why this question is useful for evaluating retrieval",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

Difficulty = Literal["Easy", "Medium", "Hard"]


class SelectedQuestion(BaseModel):
    """One question chosen for the golden dataset."""

    candidate_index: int = Field(
        description="0-based row index from the candidate CSV (excluding header)."
    )
    selection_rationale: str = Field(
        description="Brief explanation of why this question meets the golden criteria."
    )


class GoldenSelection(BaseModel):
    """Structured selection of the best candidate questions."""

    selected: list[SelectedQuestion] = Field(
        description=f"Exactly {GOLDEN_QUESTION_COUNT} selected questions."
    )


SYSTEM_PROMPT = f"""\
You are curating a golden evaluation dataset for Retrieval-Augmented Generation (RAG).

From the provided candidate questions, select exactly {GOLDEN_QUESTION_COUNT} for the
final golden dataset.

Selection criteria (all must be satisfied as a set):
1. **Video coverage** — spread selections across different source videos; avoid picking
   multiple questions from the same video unless necessary to reach {GOLDEN_QUESTION_COUNT}.
2. **Concept coverage** — each selected question should test a distinct topic or concept.
3. **Not overly easy** — prefer Medium and Hard questions; include at most one Easy question,
   and only if it still requires precise retrieval of a specific passage.
4. **Not ambiguous** — reject questions with vague wording or answers that could apply to
   multiple videos or generic web content.
5. **Specific retrieval** — favor questions whose answers depend on a particular transcript
   section (timestamp, example, number, or phrasing unique to that video).
6. **High-quality answer** — the answer must be complete, accurate, concise, and clearly
   grounded in the source material.

Return the 0-based candidate_index for each selection and a short selection_rationale.
Do not invent new questions; only choose from the provided candidates.
"""


PLACEHOLDER_KEYS = {
    "your-api-key-here",
    "your_api_key_here",
    "sk-ant-api03-...your-real-key...",
}


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


def load_candidates(input_path: Path) -> pd.DataFrame:
    """Load and validate the candidate questions CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Candidate questions file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [column for column in CSV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {', '.join(missing)}")

    if len(df) < GOLDEN_QUESTION_COUNT:
        raise ValueError(
            f"Need at least {GOLDEN_QUESTION_COUNT} candidates, found {len(df)} in {input_path}"
        )

    return df


def format_candidates_for_prompt(df: pd.DataFrame) -> str:
    """Render candidates as numbered records for the selection prompt."""
    lines: list[str] = []
    for index, row in df.iterrows():
        lines.append(
            "\n".join(
                [
                    f"--- Candidate {index} ---",
                    f"Question: {row['Question']}",
                    f"Answer: {row['Answer']}",
                    f"Video: {row['Video']}",
                    f"Timestamp: {row['Timestamp']}",
                    f"Difficulty: {row['Difficulty']}",
                    f"Topic: {row['Topic']}",
                    f"Retrieval usefulness: {row['Why this question is useful for evaluating retrieval']}",
                ]
            )
        )
    return "\n\n".join(lines)


def select_golden_questions(
    client: Anthropic,
    model: str,
    max_tokens: int,
    df: pd.DataFrame,
) -> GoldenSelection:
    """Ask Claude to pick the best candidate questions."""
    user_prompt = f"""\
Select exactly {GOLDEN_QUESTION_COUNT} questions for the golden dataset.

Candidates:
{format_candidates_for_prompt(df)}
"""

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=GoldenSelection,
        temperature=0.2,
    )

    parsed = response.parsed_output
    if parsed is None or len(parsed.selected) != GOLDEN_QUESTION_COUNT:
        count = 0 if parsed is None else len(parsed.selected)
        raise ValueError(f"Model returned {count} selections; expected {GOLDEN_QUESTION_COUNT}.")

    return parsed


def validate_selection(df: pd.DataFrame, selection: GoldenSelection) -> None:
    """Enforce basic structural and diversity constraints on the selection."""
    indices = [item.candidate_index for item in selection.selected]

    if len(set(indices)) != len(indices):
        raise ValueError("Selection contains duplicate candidate indices.")

    for index in indices:
        if index < 0 or index >= len(df):
            raise ValueError(f"Invalid candidate index {index}; valid range is 0-{len(df) - 1}.")

    selected_df = df.iloc[indices]
    video_count = selected_df["Video"].nunique()
    topic_count = selected_df["Topic"].nunique()
    easy_count = (selected_df["Difficulty"] == "Easy").sum()

    if video_count < min(4, GOLDEN_QUESTION_COUNT):
        logger.warning(
            "Selection spans only %d video(s); ideally cover more distinct sources.",
            video_count,
        )

    if topic_count < GOLDEN_QUESTION_COUNT:
        logger.warning(
            "Selection has only %d distinct topic(s); some concepts may overlap.",
            topic_count,
        )

    if easy_count > 1:
        logger.warning(
            "Selection includes %d Easy question(s); criteria prefer at most one.",
            easy_count,
        )


def build_golden_dataframe(df: pd.DataFrame, selection: GoldenSelection) -> pd.DataFrame:
    """Materialize the golden dataset rows from selected indices."""
    rows: list[dict[str, str]] = []
    for item in selection.selected:
        row = df.iloc[item.candidate_index]
        rows.append(
            {
                **{column: str(row[column]).strip() for column in CSV_COLUMNS},
                "Selection Rationale": item.selection_rationale.strip(),
            }
        )
    return pd.DataFrame(rows)


def save_golden_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Write the golden dataset CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")


def main() -> int:
    """Select golden questions and export the final dataset."""
    try:
        candidates = load_candidates(INPUT_CSV)
        client, model, max_tokens = get_anthropic_client()
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Selecting %d golden question(s) from %d candidate(s) using model '%s'",
        GOLDEN_QUESTION_COUNT,
        len(candidates),
        model,
    )

    try:
        selection = select_golden_questions(client, model, max_tokens, candidates)
        validate_selection(candidates, selection)
        golden_df = build_golden_dataframe(candidates, selection)
        save_golden_dataset(golden_df, OUTPUT_CSV)
    except (ValueError, APIError, OSError) as exc:
        logger.error("Failed to select golden questions: %s", exc)
        return 1

    videos = golden_df["Video"].nunique()
    difficulties = golden_df["Difficulty"].value_counts().to_dict()
    logger.info(
        "Saved %d golden question(s) to %s (%d videos, difficulties: %s)",
        len(golden_df),
        OUTPUT_CSV,
        videos,
        difficulties,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
