# Evaluation results

24 queries · catalog 180 songs · embedder `BAAI/bge-small-en-v1.5` · LLM off (offline fallbacks)

## Retrieval ablation

| System | P@5 | nDCG@10 | Recall@20 | MRR | p50 latency |
|---|---|---|---|---|---|
| bm25 | 0.675 | 0.685 | 0.627 | 0.735 | 5.9 ms |
| dense | 0.700 | 0.691 | 0.707 | 0.840 | 5.9 ms |
| hybrid | 0.775 | 0.762 | 0.707 | 0.856 | 5.9 ms |

## End-to-end (agent, k=5, neutral profile)

- P@5: **0.767**
- LLM calls per query: **0.00**
- Citation validity (claims kept by the validator): **1.000**
- Recommendations explained by the LLM (vs grounded fallback): **0.000**
- Faithfulness (LLM judge): **n/a (run with --judge)**

| Stage | p50 ms | p95 ms |
|---|---|---|
| plan | 0 | 0 |
| retrieve | 11 | 14 |
| check | 0 | 0 |
| rank | 0 | 0 |
| total | 11 | 15 |

<details><summary>Per query</summary>

| id | tool | P@5 | relevant in catalog | LLM calls | total ms |
|---|---|---|---|---|---|
| study-instrumental | hybrid | 0.80 | 10 | 0 | 10 |
| sleep | semantic_search | 0.00 | 27 | 0 | 6 |
| piano | semantic_search | 1.00 | 10 | 0 | 6 |
| workout | semantic_search | 0.20 | 37 | 0 | 6 |
| metal | semantic_search | 1.00 | 10 | 0 | 6 |
| party | semantic_search | 0.40 | 43 | 0 | 5 |
| sad | semantic_search | 1.00 | 41 | 0 | 6 |
| kpop | semantic_search | 1.00 | 10 | 0 | 6 |
| latin | semantic_search | 1.00 | 10 | 0 | 6 |
| hiphop | semantic_search | 1.00 | 10 | 0 | 9 |
| country-roadtrip | semantic_search | 1.00 | 10 | 0 | 12 |
| folk-acoustic | semantic_search | 0.60 | 5 | 0 | 13 |
| jazz-evening | semantic_search | 1.00 | 8 | 0 | 12 |
| blues | semantic_search | 1.00 | 10 | 0 | 14 |
| funk-groove | semantic_search | 0.80 | 8 | 0 | 11 |
| reggae | semantic_search | 1.00 | 10 | 0 | 11 |
| edm | semantic_search | 0.00 | 14 | 0 | 15 |
| rock-anthem | semantic_search | 0.40 | 17 | 0 | 11 |
| lofi | semantic_search | 1.00 | 10 | 0 | 11 |
| ambient-focus | semantic_search | 0.80 | 28 | 0 | 15 |
| rnb-romantic | semantic_search | 1.00 | 9 | 0 | 11 |
| feel-good | semantic_search | 0.40 | 50 | 0 | 12 |
| introspective | semantic_search | 1.00 | 21 | 0 | 15 |
| intense-latin | semantic_search | 1.00 | 9 | 0 | 11 |

</details>
