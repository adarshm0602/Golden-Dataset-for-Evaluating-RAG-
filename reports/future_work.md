# Future Work: Using the Golden Dataset in a RAG System

## 1. Introduction

This report describes how the golden evaluation dataset produced by this project (`golden_dataset.csv` and `candidate_questions.csv`) integrates with a Retrieval-Augmented Generation (RAG) application. The dataset provides citation-grounded question–answer pairs spanning four educational videos on machine learning and deep learning. Each pair includes a question, reference answer, source video, timestamp, difficulty level, topic, and retrieval rationale.

**What is already implemented:** `scripts/run_retrieval_benchmark.py` provides a baseline retrieval harness. It indexes paragraph chunks from `data/cleaned_transcripts/`, embeds them with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, and scores Hit@1/3/5 and MRR against both evaluation CSVs. Results are written to `reports/benchmark_results.json` and `reports/benchmark_results.md`. See [`reports/methodology.md`](methodology.md) §9 for the full benchmark methodology.

**What remains:** vector-database deployment, optional reranking, LLM answer generation, and end-to-end metrics (faithfulness, answer correctness, hallucination detection). The sections below cover the full target architecture, evaluation methodology, and a roadmap for scaling from four videos to hundreds.

---

## 2. End-to-End RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE INDEXING                                │
│                                                                         │
│  cleaned_transcripts/  →  Chunking  →  Embeddings  →  Vector DB       │
│                              ▲                                          │
│                    (paragraph chunks — baseline implemented)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ONLINE QUERY                                    │
│                                                                         │
│  User Question  →  Retriever  →  Top-K Chunks  →  LLM  →  Answer       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION                                      │
│                                                                         │
│  candidate_questions.csv + golden_dataset.csv                           │
│       →  Retrieval metrics (Hit@K, MRR)  [implemented]                │
│       →  Generation metrics (faithfulness, correctness)  [future]     │
└─────────────────────────────────────────────────────────────────────────┘
```

The golden and candidate datasets plug into the evaluation layer. They do not participate in indexing or inference, but they define the ground truth against which every pipeline variant is scored.

### 2.1 Current baseline (`run_retrieval_benchmark.py`)

| Component | Current implementation |
|-----------|------------------------|
| Chunking | One chunk per cleaned paragraph (123 chunks, 4 videos) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` via sentence-transformers |
| Retrieval | In-memory cosine similarity, top-K ranking |
| Ground truth | Video title match + timestamp within `[start, end]` |
| Metrics | Hit@1, Hit@3, Hit@5, MRR; stratified by difficulty and query type |
| Datasets | `candidate_questions.csv` (20) + `golden_dataset.csv` (5) |

**Baseline scores (candidate set, 20 questions):** Hit@3 = 15.0%, Hit@5 = 30.0%, MRR = 0.185. Factual (Easy) queries: 20% Hit@3. Multi-hop/synthesis (Hard) queries: 16.7% Hit@3 but 66.7% Hit@5 — the correct passage is often retrieved but ranked below position 3.

**Why scores are modest:** small corpus and eval set, mixed Hindi/English without domain tuning, paragraph boundary effects on timestamp matching, and no reranking or hybrid retrieval. These limitations are features for a baseline: they show where a production RAG system would fail and what to improve next.

---

## 3. Chunking the Transcripts

### 3.1 Starting point

Cleaned transcripts in `data/cleaned_transcripts/` are already segmented into timestamped paragraphs with `start`, `end`, and `text` fields. These paragraphs are the **current benchmark chunking unit** (see `scripts/run_retrieval_benchmark.py`) because they respect sentence boundaries and speaker pauses (see `reports/methodology.md` §5).

### 3.2 Chunking strategies to evaluate

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Paragraph chunks** | One vector per cleaned paragraph | Preserves semantic coherence; may be too long or too short depending on video |
| **Fixed-size windows** | Split paragraphs into ~256–512 token windows with overlap | Uniform chunk sizes; may split definitions from examples |
| **Semantic chunking** | Use embedding similarity to detect topic shifts | Higher quality boundaries; more compute at index time |
| **Hierarchical** | Index both paragraphs and sub-chunk windows | Enables coarse-to-fine retrieval; doubles storage |

### 3.3 Metadata to attach per chunk

Each chunk should carry metadata to support filtering, citation, and evaluation:

```json
{
  "chunk_id": "aircAruvnKk_p07",
  "video_id": "aircAruvnKk",
  "video_title": "But what is a Neural Network?",
  "channel": "3Blue1Brown",
  "start_seconds": 649.0,
  "end_seconds": 692.0,
  "timestamp_start": "00:10:49",
  "language": "en",
  "text": "..."
}
```

The `timestamp_start` field in `golden_dataset.csv` maps directly to these chunk timestamps, enabling automatic labeling of which chunk(s) are relevant for each golden question.

### 3.4 Golden-dataset-driven chunking experiments

Several golden questions are sensitive to chunk boundaries:

- **Bias vs. weights** — Defined in adjacent paragraphs; splitting them into separate chunks tests whether retrieval returns both or only one.
- **Black box + social media example** — Definition and example may span paragraph boundaries; chunk size determines whether both appear in a single retrieved passage.
- **Five enumerated reasons** — A long enumeration may be split across chunks; tests whether the retriever returns all five factors.

Systematically varying chunk size and overlap while holding the golden dataset fixed is one of the highest-value experiments this benchmark enables.

---

## 4. Creating Embeddings

### 4.1 Model selection

Each chunk's `text` field is passed through an embedding model to produce a dense vector.

**Current baseline:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — chosen for multilingual support (Hindi + English corpus) and zero API cost. This is the default in `run_retrieval_benchmark.py` (override with `--model`).

**Models to compare against the baseline:**

| Model family | Strengths | Considerations for this corpus |
|--------------|-----------|-------------------------------|
| `paraphrase-multilingual-MiniLM-L12-v2` | Free, local, multilingual | **Current baseline**; modest Hit@3 on this eval set |
| `text-embedding-3-small/large` (OpenAI) | Strong general performance | API cost; good comparison point |
| `multilingual-e5-large` | Cross-lingual retrieval | Important for Hindi golden questions |
| `BGE-M3` | Multilingual, hybrid dense+sparse | Handles mixed English/Hindi corpus well |
| `voyage-3` | High retrieval accuracy | Commercial API dependency |

Because two golden questions are in Hindi and the corpus mixes languages, a multilingual embedding model is recommended over an English-only model.

### 4.2 Embedding the query

At retrieval time, the user's question is embedded with the same model and compared against chunk vectors. For Hindi questions, the query embedding must land in the same vector space as Hindi transcript chunks—another reason to prefer multilingual models.

### 4.3 Optional enrichments

- **Prefix instructions** — Some models (e.g., E5) expect `"query: "` and `"passage: "` prefixes for asymmetric retrieval.
- **Metadata prepending** — Prepending `Video: {title}\n` to each chunk can help disambiguate overlapping topics across videos.
- **Hybrid embeddings** — Combine dense vectors with BM25 sparse scores to improve retrieval of exact terms (e.g., "TensorFlow", "Jio", "bratwurst").

---

## 5. Storing Vectors in a Vector Database

### 5.1 Database options

| Database | Deployment | Best for |
|----------|------------|----------|
| **Chroma** | Local / embedded | Prototyping, small corpora (this project's scale) |
| **Pinecone** | Managed cloud | Production with minimal ops |
| **Weaviate** | Self-hosted or cloud | Hybrid search, metadata filtering |
| **Qdrant** | Self-hosted or cloud | Performance, payload filtering |
| **pgvector** | PostgreSQL extension | Teams already on Postgres |

For the current four-video corpus (~100 paragraphs), a local Chroma or pgvector instance is sufficient. Scaling to hundreds of videos will require managed infrastructure or sharded self-hosted deployments.

### 5.2 Index schema

Each stored record should contain:

- **Vector** — The embedding (typically 384–3072 dimensions depending on model).
- **Document text** — The chunk text passed to the LLM at generation time.
- **Metadata** — `video_id`, `video_title`, `timestamp_start`, `timestamp_end`, `language`, `chunk_index`.

Metadata filtering is valuable for this corpus. For example, if the retriever can infer the target video from the question ("according to the 3Blue1Brown video"), pre-filtering by `video_title` before vector search can improve precision.

### 5.3 Versioning

Store the embedding model name, chunking parameters, and index build timestamp alongside the index. When re-embedding with a new model, rebuild the index and re-run golden evaluation to measure improvement or regression.

---

## 6. Building a Retriever

### 6.1 Basic retrieval pipeline

```
query → embed(query) → vector_search(top_k=K) → [chunk_1, ..., chunk_K] → rerank (optional) → context
```

1. Embed the user question.
2. Perform approximate nearest-neighbor search against the vector database.
3. Return the top-K chunks by cosine similarity (or dot product, depending on model normalization).
4. Optionally rerank with a cross-encoder for higher precision.

### 6.2 Retrieval parameters to tune

| Parameter | Typical range | Effect |
|-----------|---------------|--------|
| `top_k` | 3–10 | More chunks = more context but more noise |
| `similarity_threshold` | 0.5–0.8 | Filters low-confidence matches |
| `metadata_filter` | video_id, language | Restricts search space |
| `reranker top_n` | 20 → 5 | Cross-encoder rescores initial candidates |

### 6.3 Mapping golden questions to relevant chunks

For evaluation, each question needs a **relevance judgment**: the chunk whose timestamp range contains the cited `Timestamp`. This is implemented in `scripts/run_retrieval_benchmark.py`:

```python
def is_correct_chunk(chunk, row):
    return (
        chunk.title == row.video
        and chunk.start <= row.timestamp_seconds <= chunk.end
    )
```

The benchmark uses **Hit@K** (was any correct chunk in the top K?) rather than Precision@K, because each question has exactly one ground-truth paragraph. MRR measures how highly that paragraph ranks.

Re-running `python3 scripts/run_retrieval_benchmark.py` after changing chunking, embedding model, or retrieval logic produces an updated `reports/benchmark_results.*` for comparison.

---

## 7. Using an LLM for Answer Generation

### 7.1 Prompt structure

Retrieved chunks are assembled into a context block and passed to an LLM:

```
System: You are a helpful assistant. Answer only based on the provided context.
        If the context does not contain enough information, say so.

Context:
[Chunk 1 | Video: ... | Timestamp: 00:08:10]
{chunk text}

[Chunk 2 | ...]
{chunk text}

Question: {user question}

Answer:
```

### 7.2 Generation parameters

- **Temperature** — Use low temperature (0–0.3) for factual QA to reduce hallucination.
- **Max tokens** — Set based on expected answer length; golden answers range from one sentence to multi-point enumerations.
- **Citation** — Instruct the model to cite video title and timestamp, enabling manual verification against golden metadata.

### 7.3 Separating retrieval and generation evaluation

The golden dataset enables a two-stage evaluation:

1. **Retrieval-only** — Did the retriever return the correct chunk(s) before the LLM sees them?
2. **End-to-end** — Given retrieved context, did the LLM produce a correct and faithful answer?

A system can pass retrieval evaluation but fail generation (correct passage, wrong synthesis), or fail retrieval but accidentally produce a plausible answer from wrong context (lucky hallucination). The golden dataset exposes both failure modes.

---

## 8. Evaluating Retrieval Accuracy

### 8.1 Evaluation loop (implemented)

`scripts/run_retrieval_benchmark.py` implements the retrieval-only loop for both `candidate_questions.csv` and `golden_dataset.csv`:

1. Submit the `Question` to the retriever (embed query, rank all paragraph chunks).
2. Record the ranked list of chunk indices.
3. Compare against ground truth: `Video` title match + `Timestamp` within chunk `[start, end]`.
4. Aggregate Hit@1, Hit@3, Hit@5, and MRR overall and by difficulty / query type.

**Not yet implemented:** passing retrieved chunks to an LLM and comparing generated answers against the golden `Answer` (see §7 and §10).

Results are logged per question in `reports/benchmark_results.json` and summarized in `reports/benchmark_results.md`.

### 8.2 Ground-truth relevance labels

| Golden field | Evaluation use |
|--------------|----------------|
| `Question` | Query input to retriever and LLM |
| `Answer` | Reference for answer correctness and faithfulness |
| `Video` | Identifies which video's chunks are relevant |
| `Timestamp` | Locates the specific supporting passage |
| `Difficulty` | Enables stratified reporting (Easy / Medium / Hard) |
| `Topic` | Enables per-topic breakdown of failures |

---

## 9. Retrieval Metrics

### 9.1 Hit@K (implemented)

**Hit@K** (also called Hit Rate@K) is the primary metric in `run_retrieval_benchmark.py`.

**Definition:** For each question, was **any** ground-truth chunk present in the top K retrieved results? Report the fraction of questions where the answer is yes.

| Metric | K values in baseline | Status |
|--------|----------------------|--------|
| **Hit@K** | 1, 3, 5 | **Implemented** |
| **MRR** | — | **Implemented** (see §9.2) |
| Precision@K | 1, 3, 5 | Reference (§9.3) |
| Recall@K | 3, 5, 10 | Reference (§9.4) |

**Interpreting Hit@3 vs Hit@5:** Hit@3 asks whether the right passage is in the top 3. Hit@5 extends the window. A large gap (e.g. 16.7% Hit@3 vs 66.7% Hit@5 on Hard questions) means retrieval finds the passage but ranks it too low — increasing K or adding a reranker may help harder queries.

Report all metrics stratified by `Difficulty` and query type. The baseline already does this; see `reports/benchmark_results.md`.

### 9.2 Mean Reciprocal Rank (MRR) — implemented

**Definition:** The average of the reciprocal rank at which the first relevant chunk appears.

\[
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank of first relevant chunk for } q}
\]

**Example:** If the first relevant chunk for a question appears at rank 3, the reciprocal rank is 1/3 ≈ 0.33. If it appears at rank 1, the reciprocal rank is 1.0. If no relevant chunk appears in the retrieved set, the contribution is 0.

**Interpretation:** MRR rewards retrievers that place the correct passage at the top of the ranked list. It is particularly informative for this dataset because LLM context windows are often filled with the top 3–5 chunks—if the correct passage ranks 8th, the generator may never see it.

### 9.3 Precision@K (reference)

When each question has exactly one ground-truth paragraph, Hit@K and Precision@K convey similar information.

### 9.4 Recall@K (reference)

Recall matters for multi-chunk answers (e.g. definition + example spanning two chunks). The current benchmark uses single-chunk ground truth; multi-chunk relevance is a future extension.

---

## 10. Generation Metrics

### 10.1 Faithfulness

**Definition:** Is the generated answer supported by the retrieved context (and ultimately the source transcript)?

Faithfulness measures whether every claim in the generated answer can be traced to the retrieved passages. It does not require verbatim matching with the golden answer, but the answer must not introduce facts absent from the context.

**Evaluation approaches:**

| Method | Description |
|--------|-------------|
| **LLM-as-judge** | Prompt a separate model: "Given the context and the answer, is every claim in the answer supported by the context? Return supported / unsupported claims." |
| **NLI entailment** | Use a natural language inference model to check whether the context entails each sentence in the answer. |
| **Citation verification** | Require the generator to cite timestamps; verify cited passages contain the claimed facts. |

**Relevance to golden dataset:** The black box question requires both a definition and a specific example. A faithful but incomplete answer might define "black box" without the social media ban example—faithful to what was retrieved, but incomplete relative to the golden answer.

### 10.2 Answer correctness

**Definition:** Does the generated answer match the golden reference answer in substance?

Correctness is stricter than faithfulness: the answer must convey the same information as the golden `Answer`, not merely avoid contradiction.

**Evaluation approaches:**

| Method | Description |
|--------|-------------|
| **Exact match** | Suitable only for short factual answers; too strict for this dataset. |
| **LLM-as-judge** | Compare generated answer to golden answer: "Are these semantically equivalent? Score 1–5." |
| **Rubric-based** | Define required answer elements per question (e.g., all five reasons must be present) and score partial credit. |

**Example rubric for the five-reasons question:**

| Criterion | Points |
|-----------|--------|
| Mentions all five categories (data, hardware, frameworks, architectures, community) | 5 |
| Includes at least one named example (Jio, TensorFlow, PyTorch) | 1 |
| Correctly attributes post-2010 timing | 1 |

### 10.3 Hallucination detection

**Definition:** Does the generated answer contain claims not present in the retrieved context or source material?

Hallucination is the failure mode where the LLM invents facts, misattributes examples, or confuses concepts from different videos.

**Common hallucination patterns this dataset can expose:**

| Pattern | Example |
|---------|---------|
| **Cross-video confusion** | Attributing the Germany/Japan/sushi example to the neural network video |
| **Concept conflation** | Describing bias when the question asks about weights |
| **Invented specifics** | Citing a framework or statistic not mentioned in the transcript |
| **Overconfident generalization** | Answering from parametric knowledge when retrieval returned nothing relevant |

**Detection methods:**

- **Context-answer alignment** — Flag any answer sentence that an NLI model classifies as "not entailed" by the retrieved context.
- **Ablation test** — Run the same question with empty retrieval context; if the answer is identical, the model is relying on parametric knowledge rather than retrieval.
- **Entity grounding** — Extract named entities from the answer and verify each appears in the retrieved chunks.

---

## 11. End-to-End Evaluation Workflow

The **retrieval stage** of this workflow is implemented in `scripts/run_retrieval_benchmark.py`. The **generation stage** remains future work.

A full end-to-end evaluation script would look like:

```
For each row in golden_dataset.csv:
  1. RETRIEVE
     - Run question through retriever → ranked chunks
     - Compute Precision@K, Recall@K, MRR against golden timestamp

  2. GENERATE
     - Pass top-K chunks + question to LLM → generated answer

  3. SCORE
     - Faithfulness: are all claims supported by retrieved context?
     - Correctness: does the answer match golden answer in substance?
     - Hallucination: list any unsupported claims

  4. LOG
     - Per-question results with retrieved chunk IDs, scores, and failure category
     - Aggregate metrics across dataset
```

### 11.1 Failure taxonomy

| Failure type | Symptom | Likely cause |
|--------------|---------|--------------|
| **Retrieval miss** | Correct answer absent; wrong chunks retrieved | Embedding model, chunk size, or K too small |
| **Retrieval noise** | Correct chunk present but ranked low | Reranking needed, or K too small |
| **Faithfulness failure** | Answer adds unsupported claims | LLM temperature too high, or prompt lacks grounding instructions |
| **Incomplete answer** | Correct but missing key details | Chunk split enumeration; K too small |
| **Hallucination** | Confident answer with no retrieval support | Empty or irrelevant retrieval; model falls back to parametric knowledge |

---

## 12. Scaling the Dataset to Hundreds of Videos

The current pipeline processes four videos and produces five golden questions. Scaling to hundreds of videos requires automating each stage while maintaining quality.

### 12.1 Ingestion at scale

| Step | Current approach | Scaled approach |
|------|------------------|-----------------|
| Video registry | Manual `videos.json` entries | Curated playlists, channel feeds, or CSV imports |
| Transcript download | `scripts/download_transcripts.py` per video | Batch job with rate limiting, retry logic, and failure queue |
| Quality gate | Manual inspection | Auto-flag videos with missing captions, low snippet count, or high dedup ratio |

### 12.2 Processing at scale

- **Parallel cleaning** — `clean_transcripts.py` is embarrassingly parallel; run across a worker pool.
- **Batch summarization** — Queue videos for Claude summarization with cost tracking; skip or chunk aggressively for very long videos.
- **Incremental indexing** — Add new videos to the vector database without rebuilding the entire index (supported by most vector DBs).

### 12.3 Candidate generation at scale

| Strategy | Description |
|----------|-------------|
| **Per-video generation** | Generate ~5 candidates per video (current ratio); 200 videos → ~1,000 candidates |
| **Topic-stratified sampling** | Ensure candidates cover all main topics from summaries, not just frequent themes |
| **Difficulty quotas** | Enforce minimum counts of Medium and Hard questions per video |
| **Deduplication** | Embed candidate questions and cluster to remove near-duplicates before selection |

### 12.4 Golden selection at scale

Manual review of five questions does not scale. Options:

1. **LLM selection with human audit** — Current approach (`select_golden_questions.py`), applied per batch with spot-checking.
2. **Active learning** — Prioritize candidates where multiple retriever configurations disagree for human review.
3. **Retrieval-based filtering** — Only keep candidates where a baseline retriever can find the cited passage in top-5; discard questions that are too easy or impossible to retrieve.
4. **Tiered datasets** — Maintain a small "gold" set (human-verified) and a larger "silver" set (LLM-generated, spot-checked).

### 12.5 Evaluation at scale

| Concern | Solution |
|---------|----------|
| Metric stability | Report confidence intervals via bootstrap resampling over questions |
| Stratified reporting | Group by video, topic, difficulty, and language |
| Continuous evaluation | Re-run golden evaluation on every index or model change in CI |
| Synthetic augmentation | Paraphrase golden questions to test retrieval robustness without manual labeling |

### 12.6 Infrastructure for hundreds of videos

```
                    ┌──────────────┐
  videos.json  ───► │  Ingestion   │ ───► raw_transcripts/
                    │  (batch)     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Cleaning    │ ───► cleaned_transcripts/
                    │  (parallel)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌───▼────┐ ┌────▼─────┐
       │ Summarize   │ │ Chunk  │ │ Generate │
       │ (queued)    │ │ + Embed│ │ candidates│
       └──────┬──────┘ └───┬────┘ └────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │  Vector DB   │ ◄─── incremental updates
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Golden QA   │ ───► evaluation in CI
                    │  selection   │
                    └──────────────┘
```

### 12.7 Quality controls at scale

- **Inter-annotator agreement** — When humans review LLM-selected golden questions, measure agreement and resolve conflicts.
- **Temporal drift** — Re-verify golden answers if source transcripts are updated (e.g., creator edits captions).
- **Coverage matrix** — Track which videos, topics, and difficulty levels are represented; flag gaps before expanding the golden set.
- **Cost budgeting** — LLM calls for summarization, candidate generation, and evaluation scale linearly with video count; batch and cache aggressively.

---

## 13. Recommended Next Steps

| Priority | Action | Status | Expected outcome |
|----------|--------|--------|------------------|
| 1 | Chunk `data/cleaned_transcripts/` and embed with multilingual model | **Done** | Paragraph-level baseline in `run_retrieval_benchmark.py` |
| 2 | Implement retrieval evaluation against both CSVs | **Done** | Hit@K, MRR in `reports/benchmark_results.*` |
| 3 | Persist embeddings in a vector database (Chroma, pgvector) | Future | Production-style retrieval API |
| 4 | Add reranking or hybrid BM25 + dense search | Future | Improve baseline Hit@3 |
| 5 | Add LLM generation with retrieved context | Future | End-to-end RAG prototype |
| 6 | Score faithfulness, correctness, and hallucination | Future | Full pipeline benchmark |
| 7 | Experiment with chunk size, embedding model, and K | Future | Identify optimal configuration |
| 8 | Expand `videos.json` and re-run the pipeline | Future | Scale toward a larger golden set |

---

## 14. Conclusion

The golden and candidate datasets are the evaluation anchor for RAG systems built on this project's transcript corpus. They provide fixed questions, reference answers, and timestamp citations that enable rigorous measurement of retrieval accuracy (Hit@K, MRR) and, in future work, generation quality (faithfulness, answer correctness, hallucination detection).

A **baseline retrieval harness is already in place**: `scripts/run_retrieval_benchmark.py` scores paragraph-level dense retrieval across 123 chunks and 25 evaluation questions, with results in `reports/benchmark_results.md`. The modest baseline scores (15% Hit@3 overall on the candidate set) highlight concrete improvement paths—reranking, hybrid search, alternative embedders, and finer chunking—while keeping the evaluation loop reproducible.

The five-question golden set is intentionally small and challenging—spanning four videos, two languages, and a range of retrieval failure modes. As the corpus grows to hundreds of videos, the same pipeline architecture applies: chunk, embed, index, retrieve, generate, and evaluate against an expanding benchmark. The methodology in `reports/methodology.md` and the artifacts in this repository provide a reproducible foundation for that scale-up.
