# Methodology: Golden Dataset for RAG Evaluation

## 1. Introduction

This report documents the end-to-end methodology used to build a golden evaluation dataset for Retrieval-Augmented Generation (RAG) systems. The dataset is derived from four educational YouTube videos on machine learning and deep learning, processed through a reproducible pipeline of ingestion, cleaning, summarization, candidate generation, expert selection, and retrieval benchmarking.

The final artifacts are:

- `candidate_questions.csv` — 20 grounded QA pairs across three difficulty levels
- `golden_dataset.csv` — five curated question–answer pairs, each tied to a specific transcript passage with a timestamp citation
- `reports/benchmark_results.json` / `.md` — baseline retrieval metrics (Hit@K, MRR) from `scripts/run_retrieval_benchmark.py`

---

## 2. Why a Golden Dataset Is Required for RAG Evaluation

RAG systems combine a retriever (which finds relevant source passages) with a generator (which synthesizes an answer from retrieved context). Evaluating such systems requires more than ad hoc spot checks. A **golden dataset** provides:

| Need | How a golden dataset addresses it |
|------|-----------------------------------|
| **Reproducibility** | Fixed questions and expected answers allow consistent benchmarking across retriever models, chunking strategies, and embedding versions. |
| **Ground truth** | Each QA pair is tied to a known source passage and timestamp, enabling objective measurement of retrieval precision and answer faithfulness. |
| **Regression detection** | Pipeline changes (e.g., new chunk size, different index) can be compared against a stable baseline without re-inventing test cases each time. |
| **Failure diagnosis** | When a system fails, golden pairs distinguish retrieval errors (wrong or missing passage) from generation errors (correct passage, wrong synthesis). |

Without a golden dataset, evaluation tends to rely on subjective judgment or generic benchmarks that do not reflect the specific corpus, terminology, or multilingual content in the knowledge base. This project targets that gap by building evaluation cases directly from the indexed source material.

---

## 3. Source Corpus

Four videos were registered in `videos.json` as the canonical source registry:

| Title | Channel | Language preference |
|-------|---------|---------------------|
| But what is a Neural Network? | 3Blue1Brown | English |
| Transformers, the tech behind LLMs | 3Blue1Brown | English |
| What is Deep Learning? | CampusX | Hindi, English |
| All About ML & Deep Learning | CodeWithHarry | Hindi, English |

The corpus deliberately mixes English and Hindi instructional content, overlapping subject matter (neural networks, deep learning, transformers), and distinct presentation styles. This diversity increases the difficulty of retrieval and makes the evaluation set more representative of real-world knowledge bases built from heterogeneous educational material.

---

## 4. Transcript Collection

Transcripts were collected using `scripts/download_transcripts.py`.

### 4.1 Process

1. **Video registry** — Each entry in `videos.json` supplies title, channel, URL, and preferred caption languages.
2. **API fetch** — The script uses `youtube-transcript-api` to download YouTube's published captions for each video ID.
3. **Language selection** — For each video, the fetcher tries languages in the order listed in the registry (e.g., `["hi", "en"]` for Hindi-first videos). If no preference is set, English is used.
4. **Structured output** — Each transcript is saved to `data/raw_transcripts/` as:
   - A **JSON** file containing metadata (title, channel, URL, language, caption type) and a list of timestamped snippets (`text`, `start`, `duration`).
   - A **plain-text** file with one timestamped line per snippet for human inspection.

### 4.2 Design rationale

YouTube captions provide fine-grained timestamps that are essential for citation and chunk alignment. Storing raw transcripts immutably in `data/raw_transcripts/` preserves lineage: every downstream artifact can be traced back to the original fetched captions.

---

## 5. Transcript Cleaning

Raw captions are noisy: subtitle fragments break mid-sentence, auto-generated captions repeat rolling text, and short pauses do not always indicate topic boundaries. Cleaning is performed by `scripts/clean_transcripts.py`, which transforms snippet-level JSON into paragraph-level documents in `data/cleaned_transcripts/`.

### 5.1 Pipeline stages

**Step 1 — Whitespace normalization**

Leading and trailing whitespace is stripped. Runs of whitespace (including non-breaking spaces) are collapsed to a single ASCII space so fragments can be merged reliably.

**Step 2 — Duplicate fragment removal**

Auto-generated captions frequently emit consecutive duplicates or rolling updates where a new snippet is a strict superset of the previous one. The deduplication step:

- Drops empty snippets.
- Skips exact consecutive duplicates.
- Replaces a snippet when the next one extends it (rolling caption behavior).
- Ignores snippets that add no new information relative to the previous one.

**Step 3 — Paragraph assembly**

Deduplicated snippets are merged into coherent paragraphs using:

- **Fragment merging** — Adjacent snippets are joined with word-level overlap detection to avoid repeated phrases at boundaries.
- **Boundary detection** — A new paragraph starts when:
  - The accumulated text exceeds a hard character limit (1,200 characters), or
  - The text exceeds a soft limit (800 characters) and ends a sentence, or
  - The speaker pauses for ≥ 3.0 seconds (long pause), or
  - The text ends a sentence and the pause is ≥ 1.5 seconds.

Each paragraph retains `start`, `end`, `duration`, and `source_snippet_count` metadata.

### 5.2 Output

Cleaned transcripts are written as JSON and TXT to `data/cleaned_transcripts/`. Paragraph-level chunking is the unit used for retrieval indexing and for grounding QA pairs to specific timestamps.

---

## 6. Summary Generation

Structured summaries were generated by `scripts/summarize.py` and stored in `data/summaries/` as one Markdown file per video.

### 6.1 Method

1. **Input** — Paragraph-level cleaned transcript JSON.
2. **Model** — Claude (`claude-sonnet-4-6` by default) via the Anthropic API.
3. **Structured extraction** — The model returns a validated schema containing:
   - Executive summary (3–5 sentences)
   - Main topics
   - Key concepts
   - Important definitions (term + definition pairs)
   - Important examples
4. **Long transcript handling** — Transcripts exceeding 20,000 characters are summarized in chunks and merged to stay within model context limits.
5. **Constraints** — The system prompt requires all content to be grounded strictly in the transcript; no invented facts.

### 6.2 Role in the pipeline

Summaries serve two purposes: they provide a compact overview for human review, and they give the question-generation model high-level thematic context alongside the full timestamped transcript. This dual input improves coverage of major concepts without replacing the transcript as the authoritative source for answers and citations.

---

## 7. Candidate Question Generation

Approximately 20 candidate question–answer pairs were produced by `scripts/generate_candidate_questions.py` and saved to `candidate_questions.csv`.

### 7.1 Inputs

For each video, the script assembles:

- The Markdown summary from `data/summaries/`
- The full timestamped transcript from `data/cleaned_transcripts/`

### 7.2 Generation method

Claude receives all four videos in a single prompt and is instructed to produce ~20 diverse QA pairs with the following fields:

| Field | Purpose |
|-------|---------|
| Question | Natural-language query a user might ask |
| Answer | Concise, transcript-grounded response |
| Video | Exact source video title |
| Timestamp | Start time (`HH:MM:SS`) of the supporting passage |
| Difficulty | Easy, Medium, or Hard |
| Topic | Short subject label |
| Why this question is useful for evaluating retrieval | Rationale for retrieval benchmarking value |

### 7.3 Design constraints

The generation prompt enforces:

- Strict grounding in source material
- Spread across videos (~5 per video for four videos)
- Difficulty variation (direct facts, conceptual explanations, multi-step synthesis)
- Questions that would fail if retrieval returns the wrong video or a topically adjacent passage
- Language consistency with the source video

The result was 20 candidates with a mix of English and Hindi questions, spanning all four videos and a range of topics from basic definitions to multi-point enumerations.

---

## 8. Golden Question Selection

The final dataset of five questions was curated by `scripts/select_golden_questions.py`, which reads `candidate_questions.csv` and writes `golden_dataset.csv`.

### 8.1 Selection criteria

An LLM-based selector (Claude) evaluated all 20 candidates against explicit criteria:

1. **Video coverage** — Spread selections across different source videos.
2. **Concept coverage** — Each question tests a distinct topic or concept.
3. **Not overly easy** — Prefer Medium and Hard; at most one Easy question, and only if it still requires precise passage retrieval.
4. **Not ambiguous** — Reject vague questions answerable from generic web content or multiple videos.
5. **Specific retrieval** — Favor questions whose answers depend on a particular transcript section (unique example, number, enumeration, or phrasing).
6. **High-quality answer** — Complete, accurate, concise, and clearly grounded in the source.

Post-selection validation checks for duplicate indices, valid row references, and logs warnings if video or topic diversity falls below expectations.

### 8.2 Final golden dataset

The selected set comprises five questions across four videos, with **zero Easy**, **two Medium**, and **three Hard** items:

| # | Video | Topic | Difficulty | Timestamp |
|---|-------|-------|------------|-----------|
| 1 | All About ML & Deep Learning | Reinforcement Learning | Medium | 00:08:10 |
| 2 | What is Deep Learning? | Interpretability (black box) | Hard | 00:28:50 |
| 3 | Transformers, the tech behind LLMs | Word embedding semantics | Hard | 00:15:57 |
| 4 | But what is a Neural Network? | Bias in neural networks | Medium | 00:10:49 |
| 5 | What is Deep Learning? | Deep Learning success factors | Hard | 00:34:56 |

### 8.3 Rationale for these five

Each selection was chosen because it stress-tests retrieval in a distinct way:

- **Reinforcement Learning (Medium)** — Requires disambiguation among three ML paradigms (supervised, unsupervised, reinforcement) covered in the same video. A retriever that returns a generic ML definition or a different learning type fails.

- **Black box interpretability (Hard)** — Combines a conceptual definition with a specific social-media-ban example unique to the CampusX video. Retrieval of only the definition without the example produces an incomplete answer.

- **Germany/Japan/sushi semantic arithmetic (Hard)** — Tests retrieval of a quirky, specific embedding example buried among similar king/queen and man/woman examples in the Transformers video. Wrong-passage retrieval is a common failure mode.

- **Bias in neural networks (Medium)** — Requires disambiguation between bias and weights, which are defined in adjacent passages in the 3Blue1Brown video. Chunk boundaries that split the two concepts cause retrieval failure.

- **Five reasons for Deep Learning's rise (Hard)** — Requires a multi-point enumeration from a long passage (data, hardware, frameworks, architectures, community) with named examples (Jio, TensorFlow, PyTorch). Partial retrieval of a single factor is insufficient.

`golden_dataset.csv` includes a `Selection Rationale` column documenting why each question was chosen.

---

## 9. Retrieval Benchmark

A baseline retrieval evaluation harness is implemented in `scripts/run_retrieval_benchmark.py`. It closes the loop from corpus to measured retrieval accuracy without requiring a vector database or LLM generation step.

### 9.1 Indexing

1. **Corpus** — Load all JSON files from `data/cleaned_transcripts/`. Each cleaned **paragraph** is one retrieval chunk, with metadata: `video_id`, `title`, `start`, `end`, `text`. The current corpus contains **123 chunks** across four videos.
2. **Embedding model** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual, suitable for the mixed Hindi/English corpus). Vectors are L2-normalized; similarity is cosine via dot product.
3. **No external index** — Embeddings are held in memory for this benchmark scale; no Chroma/Pinecone dependency.

### 9.2 Ground-truth matching

For each row in `candidate_questions.csv` (20 rows) and `golden_dataset.csv` (5 rows):

1. Parse `Timestamp` (`HH:MM:SS`) to seconds.
2. A retrieved chunk is **correct** when:
   - `chunk.title` exactly matches the row's `Video` field, **and**
   - `chunk.start <= timestamp_seconds <= chunk.end`

This aligns with how questions were generated: each QA pair cites the paragraph timestamp range where the supporting content appears.

### 9.3 Metrics

| Metric | Definition |
|--------|------------|
| **Hit@1** | Correct chunk is the top-ranked result |
| **Hit@3** | Correct chunk appears in the top 3 results |
| **Hit@5** | Correct chunk appears in the top 5 results |
| **MRR** | Mean reciprocal rank of the first correct chunk (rank 1 → 1.0, miss → 0) |

Results are reported **overall**, **by difficulty** (Easy / Medium / Hard), and **by query type** (factual / medium / multi-hop/synthesis). Difficulty maps to query type as: Easy → factual, Hard → multi-hop/synthesis, Medium → medium.

### 9.4 Baseline results (current run)

| Dataset | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---------|-----------|-------|-------|-------|-----|
| Candidate questions | 20 | 15.0% | 15.0% | 30.0% | 0.185 |
| Golden dataset | 5 | 20.0% | 20.0% | 40.0% | 0.250 |

These scores are intentionally modest: small corpus, small eval set, paragraph-level chunking, and vanilla dense retrieval only (no reranking, hybrid search, or metadata filtering). They establish a **reproducible baseline** for comparing future improvements. Full per-question output is in `reports/benchmark_results.json`.

### 9.5 Reproducing

```bash
python3 scripts/run_retrieval_benchmark.py
```

Requires `data/cleaned_transcripts/` and the CSV files at the project root. No Anthropic API key is needed for this step.

---

## 10. Retrieval Failure Modes This Dataset Can Detect

The golden dataset is designed to surface specific classes of retrieval failure:

### 10.1 Wrong-video retrieval

Several questions target concepts shared across the corpus (neural networks, deep learning, transformers, ML types). If the retriever returns a passage from the wrong video—e.g., the 3Blue1Brown transformers video when the question is about reinforcement learning in the CodeWithHarry video—the answer will be incorrect or ungrounded.

### 10.2 Near-miss / adjacent-topic retrieval

Questions such as bias vs. weights, or the Germany/Japan/sushi example vs. other embedding analogies, test whether the retriever can distinguish closely related content within the same video. Returning a topically similar but incorrect passage is a frequent failure in dense embedding search.

### 10.3 Incomplete passage retrieval

Multi-part questions (five enumerated reasons, black box + example, softmax + temperature) require retrieving a passage that contains the full answer. Chunking strategies that split enumerations or separate definitions from examples will cause partial retrieval and incomplete generation.

### 10.4 Generic vs. source-specific retrieval

Questions grounded in video-specific phrasing, Hindi terminology, named frameworks (TensorFlow 2015, PyTorch 2016), or unique examples (Jio, social media ban) fail when the retriever returns generic web-level content that is semantically related but not corpus-faithful.

### 10.5 Language and cross-lingual retrieval

Two golden questions are in Hindi and three in English. This tests whether the retrieval index handles multilingual queries and returns passages in the correct language and from the correct source video.

### 10.6 Timestamp / citation misalignment

Each golden pair includes a ground-truth timestamp. Evaluation can verify whether retrieved chunks overlap the cited time range, detecting indexing or chunk-alignment errors even when the retrieved text is partially correct.

---

## 11. Pipeline Summary

```
videos.json
    │
    ▼
scripts/download_transcripts.py  →  data/raw_transcripts/
    │
    ▼
scripts/clean_transcripts.py     →  data/cleaned_transcripts/
    │
    ▼
scripts/summarize.py             →  data/summaries/
    │
    ▼
scripts/generate_candidate_questions.py  →  candidate_questions.csv (20 pairs)
    │
    ▼
scripts/select_golden_questions.py       →  golden_dataset.csv (5 pairs)
    │
    ▼
scripts/run_retrieval_benchmark.py       →  reports/benchmark_results.*
```

All scripts are deterministic in their I/O paths. Steps 3–5 require `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, and `ANTHROPIC_MAX_TOKENS` in `.env`. Steps 1–2 and the retrieval benchmark require only `requirements.txt` dependencies and local transcript files. Raw inputs are preserved; each transformation writes to a downstream directory, maintaining full lineage from YouTube caption to golden QA pair and measured retrieval scores.

---

## 12. Limitations and Future Work

- **Sample size** — Five golden questions and 20 candidate questions provide a focused evaluation set, not comprehensive coverage of the corpus. Percentages are noisy at this scale (one hit or miss moves Easy Hit@3 by 20 percentage points).
- **Baseline retrieval only** — The current harness measures dense semantic search with a single multilingual embedder. End-to-end RAG (LLM generation, faithfulness scoring) and vector-database deployment are documented in `reports/future_work.md`.
- **LLM-assisted curation** — Both candidate generation and golden selection use Claude. Human review of selected pairs is recommended before production benchmarking.
- **Caption quality** — Auto-generated captions may contain transcription errors that propagate to QA pairs. Manual spot-checking of cited passages is advised.
- **Static corpus** — The dataset reflects a fixed snapshot of four videos. Corpus updates require re-running the pipeline and re-validating golden pairs.

---

## 13. Conclusion

This project implements a reproducible pipeline from YouTube transcripts to a citation-grounded golden RAG evaluation dataset **with a measured retrieval baseline**. By anchoring each question to a specific video, timestamp, and transcript passage—and by selecting questions that stress-test disambiguation, completeness, and source specificity—the dataset enables systematic measurement of retrieval quality independent of generation quality. The methodology prioritizes traceability, diversity across sources and concepts, explicit documentation of why each evaluation case matters, and reproducible Hit@K / MRR reporting via `scripts/run_retrieval_benchmark.py`.
