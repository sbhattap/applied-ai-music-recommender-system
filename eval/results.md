# Evaluation results

24 queries · catalog 180 songs · embedder `BAAI/bge-small-en-v1.5` · LLM off (offline fallbacks)

## Retrieval ablation

| System | P@5 | nDCG@10 | Recall@20 | MRR | p50 latency |
|---|---|---|---|---|---|
| bm25 | 0.683 | 0.689 | 0.631 | 0.758 | 8.2 ms |
| dense | 0.700 | 0.691 | 0.707 | 0.840 | 8.2 ms |
| hybrid | 0.758 | 0.761 | 0.704 | 0.828 | 8.2 ms |

## End-to-end (agent, k=5, neutral profile)

- P@5: **0.758**
- LLM calls per query: **0.00**
- Citation validity (claims kept by the validator): **1.000**
- Recommendations explained by the LLM (vs grounded fallback): **0.000**
- Faithfulness (LLM judge): **n/a (run with --judge)**

| Stage | p50 ms | p95 ms |
|---|---|---|
| plan | 0 | 0 |
| retrieve | 12 | 19 |
| check | 0 | 0 |
| rank | 0 | 0 |
| total | 13 | 19 |

<details><summary>Per query</summary>

| id | tool | P@5 | relevant in catalog | LLM calls | total ms |
|---|---|---|---|---|---|
| study-instrumental | hybrid | 0.80 | 10 | 0 | 21 |
| sleep | semantic_search | 0.00 | 27 | 0 | 19 |
| piano | semantic_search | 1.00 | 10 | 0 | 16 |
| workout | semantic_search | 0.40 | 37 | 0 | 13 |
| metal | semantic_search | 1.00 | 10 | 0 | 12 |
| party | semantic_search | 0.60 | 43 | 0 | 13 |
| sad | semantic_search | 1.00 | 41 | 0 | 15 |
| kpop | semantic_search | 1.00 | 10 | 0 | 10 |
| latin | semantic_search | 1.00 | 10 | 0 | 9 |
| hiphop | semantic_search | 1.00 | 10 | 0 | 13 |
| country-roadtrip | semantic_search | 1.00 | 10 | 0 | 8 |
| folk-acoustic | semantic_search | 0.60 | 5 | 0 | 19 |
| jazz-evening | semantic_search | 1.00 | 8 | 0 | 12 |
| blues | semantic_search | 1.00 | 10 | 0 | 11 |
| funk-groove | semantic_search | 0.40 | 8 | 0 | 12 |
| reggae | semantic_search | 1.00 | 10 | 0 | 13 |
| edm | semantic_search | 0.20 | 14 | 0 | 11 |
| rock-anthem | semantic_search | 0.40 | 17 | 0 | 11 |
| lofi | semantic_search | 1.00 | 10 | 0 | 14 |
| ambient-focus | semantic_search | 0.80 | 28 | 0 | 12 |
| rnb-romantic | semantic_search | 1.00 | 9 | 0 | 14 |
| feel-good | semantic_search | 0.20 | 50 | 0 | 12 |
| introspective | semantic_search | 0.80 | 21 | 0 | 12 |
| intense-latin | semantic_search | 1.00 | 9 | 0 | 14 |

</details>
