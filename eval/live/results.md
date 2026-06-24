# Evaluation results

24 queries · catalog 180 songs · embedder `BAAI/bge-small-en-v1.5` · LLM on

## Retrieval ablation

| System | P@5 | nDCG@10 | Recall@20 | MRR | p50 latency |
|---|---|---|---|---|---|
| llm (unordered) | 0.767 | 0.779 | 0.660 | 0.734 | 1794.1 ms |
| bm25 | 0.675 | 0.685 | 0.627 | 0.735 | 12.1 ms |
| dense | 0.700 | 0.691 | 0.707 | 0.840 | 12.1 ms |
| hybrid | 0.775 | 0.762 | 0.707 | 0.856 | 12.1 ms |

## End-to-end (agent, k=5, neutral profile)

- P@5: **0.792**
- LLM calls per query: **3.17**
- Citation validity (claims kept by the validator): **1.000**
- Recommendations explained by the LLM (vs grounded fallback): **1.000**
- Faithfulness (LLM judge): **0.851**

| Stage | p50 ms | p95 ms |
|---|---|---|
| plan | 2532 | 3614 |
| retrieve | 20 | 39 |
| check | 2866 | 5972 |
| rank | 0 | 0 |
| explain | 8773 | 11110 |
| total | 14429 | 17640 |

<details><summary>Per query</summary>

| id | tool | P@5 | relevant in catalog | LLM calls | total ms |
|---|---|---|---|---|---|
| study-instrumental | hybrid | 0.80 | 10 | 3 | 17552 |
| sleep | semantic_search | 0.00 | 27 | 3 | 12693 |
| piano | hybrid | 1.00 | 10 | 3 | 13925 |
| workout | hybrid | 0.60 | 37 | 3 | 13398 |
| metal | structured_filter | 1.00 | 10 | 2 | 11581 |
| party | semantic_search | 1.00 | 43 | 3 | 14291 |
| sad | hybrid | 1.00 | 41 | 3 | 14257 |
| kpop | structured_filter | 1.00 | 10 | 2 | 9839 |
| latin | hybrid | 1.00 | 10 | 3 | 14686 |
| hiphop | hybrid | 1.00 | 10 | 3 | 14011 |
| country-roadtrip | hybrid | 1.00 | 10 | 3 | 16759 |
| folk-acoustic | hybrid | 0.60 | 5 | 3 | 13933 |
| jazz-evening | semantic_search | 0.80 | 8 | 4 | 16883 |
| blues | semantic_search | 1.00 | 10 | 4 | 16160 |
| funk-groove | semantic_search | 0.00 | 8 | 3 | 12206 |
| reggae | semantic_search | 1.00 | 10 | 4 | 17640 |
| edm | hybrid | 1.00 | 14 | 3 | 15230 |
| rock-anthem | hybrid | 0.60 | 17 | 3 | 12669 |
| lofi | structured_filter | 1.00 | 10 | 2 | 10251 |
| ambient-focus | semantic_search | 0.80 | 28 | 4 | 16096 |
| rnb-romantic | semantic_search | 1.00 | 9 | 4 | 15963 |
| feel-good | hybrid | 0.60 | 50 | 3 | 14429 |
| introspective | semantic_search | 0.60 | 21 | 4 | 16083 |
| intense-latin | semantic_search | 0.60 | 9 | 4 | 20276 |

</details>
