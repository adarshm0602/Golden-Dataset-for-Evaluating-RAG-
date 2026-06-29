# Golden Dataset for Evaluating RAG

A reproducible AI engineering project for building a **golden evaluation dataset** to benchmark Retrieval-Augmented Generation (RAG) systems.

The pipeline transforms source video transcripts into structured question–answer pairs with ground-truth citations, enabling consistent measurement of retrieval quality, answer faithfulness, and end-to-end RAG performance.

## Project Goal

Build a curated, version-controlled dataset from educational or technical video content so that RAG pipelines can be evaluated against known questions, expected answers, and supporting source passages.

**Success criteria:**

- Reproducible ingestion and cleaning of raw transcripts
- Traceable lineage from each QA pair back to source material
- Export formats suitable for common RAG evaluation frameworks
- Documented methodology and quality metrics in `reports/`

## Project Structure

```
.
├── data/
│   ├── raw-transcripts/       # Unprocessed transcript files from source videos
│   ├── cleaned-transcripts/   # Normalized, deduplicated, and chunked text
│   ├── summaries/             # Per-video or per-chunk summaries
│   └── datasets/              # Final golden QA datasets (not yet generated)
├── scripts/                   # Reusable pipeline and utility scripts
├── notebooks/                 # Exploratory analysis and prototyping
├── reports/                   # Evaluation reports, metrics, and methodology notes
├── videos.json                # Source video metadata and processing status
├── requirements.txt           # Python dependencies
└── README.md
```

## Pipeline Overview

```
videos.json
    │
    ▼
[data/raw-transcripts/]     ← fetch or import transcripts
    │
    ▼
[data/cleaned-transcripts/] ← clean, normalize, chunk
    │
    ▼
[data/summaries/]           ← generate summaries (optional enrichment)
    │
    ▼
[data/datasets/]            ← generate golden QA pairs with citations
    │
    ▼
[reports/]                  ← quality checks and evaluation results
```

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) Jupyter for notebook workflows

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Edit `videos.json` to add source videos. Each entry tracks metadata and processing status through the pipeline.

## Outputs (Generated Later)

The following artifacts will be produced as the pipeline runs. Placeholders are listed here for reference.

| Artifact | Location | Description |
|----------|----------|-------------|
| Raw transcripts | `data/raw-transcripts/` | <!-- TODO: e.g. 12 transcript files from source videos --> |
| Cleaned transcripts | `data/cleaned-transcripts/` | <!-- TODO: e.g. chunked JSONL with metadata --> |
| Summaries | `data/summaries/` | <!-- TODO: e.g. per-video summary files --> |
| Golden dataset | `data/datasets/` | <!-- TODO: e.g. golden_qa_v1.jsonl with questions, answers, citations --> |
| Quality report | `reports/` | <!-- TODO: e.g. dataset_stats.md, coverage analysis --> |
| Evaluation results | `reports/` | <!-- TODO: e.g. rag_baseline_scores.json --> |

## Reproducibility

- Pin dependencies in `requirements.txt`
- Record video sources and versions in `videos.json`
- Keep raw inputs immutable; write all transformations to downstream folders
- Document any manual steps or LLM prompts in `reports/`

## License

<!-- TODO: Add license (e.g. MIT, Apache 2.0) -->

## Contributing

<!-- TODO: Add contribution guidelines if this becomes a collaborative project -->
