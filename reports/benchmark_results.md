# Retrieval Benchmark Results

Generated: 2026-06-29T13:46:45.257051+00:00
Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
Corpus chunks: 123

## Overall

| Dataset | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---------|-----------|-------|-------|-------|-----|
| candidate questions | 20 | 15.0% | 15.0% | 30.0% | 0.185 |
| golden dataset | 5 | 20.0% | 20.0% | 40.0% | 0.250 |

## Candidate Questions — By Difficulty

| Difficulty | Count | Hit@1 | Hit@3 | Hit@5 | MRR |
|------------|-------|-------|-------|-------|-----|
| Easy | 5 | 20.0% | 20.0% | 20.0% | 0.200 |
| Hard | 6 | 16.7% | 16.7% | 66.7% | 0.283 |
| Medium | 9 | 11.1% | 11.1% | 11.1% | 0.111 |

## Candidate Questions — By Query Type

| Query Type | Count | Hit@1 | Hit@3 | Hit@5 | MRR |
|------------|-------|-------|-------|-------|-----|
| factual | 5 | 20.0% | 20.0% | 20.0% | 0.200 |
| medium | 9 | 11.1% | 11.1% | 11.1% | 0.111 |
| multi-hop/synthesis | 6 | 16.7% | 16.7% | 66.7% | 0.283 |

## Golden Dataset — By Difficulty

| Difficulty | Count | Hit@1 | Hit@3 | Hit@5 | MRR |
|------------|-------|-------|-------|-------|-----|
| Hard | 3 | 33.3% | 33.3% | 66.7% | 0.417 |
| Medium | 2 | 0.0% | 0.0% | 0.0% | 0.000 |

## Golden Dataset — By Query Type

| Query Type | Count | Hit@1 | Hit@3 | Hit@5 | MRR |
|------------|-------|-------|-------|-------|-----|
| medium | 2 | 0.0% | 0.0% | 0.0% | 0.000 |
| multi-hop/synthesis | 3 | 33.3% | 33.3% | 66.7% | 0.417 |

## Reproduce

```bash
python3 scripts/run_retrieval_benchmark.py
```
