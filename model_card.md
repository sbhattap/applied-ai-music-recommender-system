# Model Card: Applied AI Music Recommender System

## 1. Model Name

**Applied AI Music Recommender System**

This project is a hybrid system rather than a single standalone model. It combines:

- a deterministic content-based music recommender
- an LLM-driven planning and checking layer
- an LLM-based explanation layer

## 2. Intended Use

This system is designed for educational and portfolio use. It takes a free-text music request plus a listener profile and returns ranked recommendations with an observable agent trace. The main goal is to demonstrate how an AI workflow can sit on top of a traditional recommendation engine without hiding how decisions are made.

It is appropriate for:

- demonstrating agentic workflows
- showing human-readable recommendation reasoning
- exploring retrieval strategy selection
- comparing deterministic ranking with LLM-assisted orchestration

It is not intended for:

- production-grade personalization
- large-scale music recommendation
- fully automated trust without human review
- safety-critical or rights-sensitive decision-making

## 3. System Overview

The system has two major layers.

### Deterministic ranking layer

The original recommender takes a structured listener profile and scores songs using seven signals:

- genre match
- mood match
- energy proximity
- acousticness fit
- danceability
- valence proximity
- popularity fit

After scoring, it applies a diversity cap of at most two songs per genre and returns the top `k` songs.

### LLM orchestration layer

The extended system accepts a free-text request and runs a four-step loop:

1. `PLAN`: interpret the request, extract constraints, choose a retrieval tool, and estimate expected result count
2. `ACT`: run `structured_filter`, `semantic_search`, or `hybrid`
3. `CHECK`: assign a confidence score and flags to the candidate set
4. `RESPOND`: re-rank with the deterministic scorer and generate explanation text

If a candidate set looks weak, the system can refine the plan and retry with broader constraints or a different retrieval strategy.

## 4. Inputs And Outputs

### Inputs

The system uses two kinds of input.

**User query**

- free-text text such as `chill study music, no vocals`


**Listener profile**

| Field | Type | Description |
|---|---|---|
| `genre` | string | preferred genre |
| `mood` | string | target mood |
| `energy` | float 0-1 | preferred energy level |
| `likes_acoustic` | bool | acoustic vs. produced sound preference |
| `target_valence` | float 0-1 | preferred emotional tone |
| `prefer_popular` | bool | mainstream vs. niche preference |

### Outputs

- ranked recommendations
- deterministic score per recommendation
- rule-based explanation breakdown
- LLM-generated or templated narrative per recommendation
- agent trace events
- check confidence and flags

## 5. Data

The current catalog is stored in `data/songs.csv`.

Current dataset snapshot:

- `180` songs
- `18` genres
- `5` primary moods

Genres currently represented:

- `ambient`, `blues`, `classical`, `country`, `edm`, `folk`, `funk`, `hip-hop`, `indie`, `jazz`, `k-pop`, `latin`, `lofi`, `metal`, `pop`, `r&b`, `reggae`, `rock`

Primary moods currently represented:

- `chill`, `happy`, `intense`, `melancholic`, `relaxed`

Each song record includes:

| Field | Used In Ranking? | Used In Retrieval? |
|---|---|---|
| `genre` | Yes | Yes |
| `mood` | Yes | Yes |
| `mood_tags` | Yes | Yes |
| `energy` | Yes | Yes |
| `tempo_bpm` | No | Yes |
| `instrumentalness` | No | Yes |
| `valence` | Yes | Exposed to semantic retrieval |
| `danceability` | Yes | Exposed to semantic retrieval |
| `acousticness` | Yes | Exposed to semantic retrieval |
| `popularity` | Yes | Not directly filtered |
| `liveness` | No | No |
| `duration_sec` | No | No |
| `explicit` | No | No |
| `release_decade` | No | No |

This is a small catalog, which makes the system easier to inspect but also limits coverage for niche combinations of mood, genre, and constraints.

## 6. How Ranking Works

The deterministic ranker uses these default weights:

| Signal | Weight |
|---|---|
| Energy proximity | 25% |
| Genre match | 20% |
| Mood match | 15% |
| Valence proximity | 12% |
| Acousticness fit | 12% |
| Danceability | 8% |
| Popularity fit | 8% |

The score is a weighted combination of those signals. A song can still rank well without matching every field if it is strong on the most heavily weighted features.

Mood matching has a soft-match path:

- exact mood match gets full credit
- mood tag match gets partial credit
- no mood match gets zero credit

The system also supports alternative weight presets:

- `genre_first`
- `mood_first`
- `energy_similarity`

These presets make it easier to compare how different recommendation philosophies change the final output.

## 7. Retrieval And Agent Behavior

The LLM layer does not directly score songs. Instead, it controls which candidate songs are passed to the deterministic ranker.

### Retrieval tools

- `structured_filter`: pure logic filter on genre, mood, BPM, energy, and instrumentalness
- `semantic_search`: LLM selects matching song ids from the catalog
- `hybrid`: deterministic narrowing first, then semantic selection inside the narrowed set

### Check output

The check step returns:

- a confidence score from `0.0` to `1.0`
- zero or more flags

Supported flags:

- `looks_good`
- `low_result_count`
- `constraint_violated`
- `intent_mismatch`

These signals are used for observability and for deciding whether the system should refine and retry.

## 8. Strengths

- The system handles free-text requests that the original structured-only recommender could not accept.
- The deterministic scorer keeps the final ranking reproducible and testable.
- The check step makes uncertainty visible instead of silently hiding weak retrieval results.
- The fallback explainer keeps the system functional even if the LLM explanation call fails.
- The separation between retrieval and ranking makes the architecture easier to debug than a fully end-to-end LLM recommender.

## 9. Limitations And Biases

- The catalog is still relatively small, so some requests fail because the data is sparse rather than because the logic is wrong.
- Genre matching in the ranker is binary, so related genres do not receive partial credit.
- The system can still struggle with contradictory requests such as wanting a very high-energy but strongly melancholic acoustic track.
- `tempo_bpm` and `instrumentalness` affect retrieval but not final ranking, so some constraints shape the candidate set more than the final score.
- `prefer_popular=True` by default can tilt the system toward mainstream music unless changed.
- LLM planning and semantic retrieval are dependent on prompt quality and API availability.
- The recommendations are not personalized from listening history, so they should not be interpreted as behaviorally learned taste modeling.

## 10. Reliability And Safety Features

The project includes several guardrails and evaluation-oriented features:

- structured Pydantic schemas for plan, semantic picks, and check outputs
- confidence scoring and issue flags in the check step
- retry/refine loop capped at 2 iterations
- templated fallback when the LLM explainer fails
- structured logging for LLM calls in `logs/llm_calls.jsonl`
- automated tests covering agent flow, tools, explainer behavior, and deterministic ranking

Known testing status:

- the suite passes with `PYTHONPATH=. pytest -p no:cacheprovider`
- plain `pytest` currently has an import-path issue because tests import `src.recommender`

## 11. Evaluation Summary

The system has been evaluated primarily through automated tests and representative manual runs.

What has been validated:

- deterministic ranking remains stable
- tool dispatch routes correctly
- structured filtering respects hard constraints
- semantic retrieval prompt wiring is correct under mocks
- the agent emits trace events in the correct order
- the check step is skipped when no candidates are returned and replaced with a synthesized low-confidence verdict
- the explainer falls back safely when the API call fails
- refine behavior switches tools or relaxes constraints in the tested scenarios

This evaluation is good for correctness and reliability of the workflow, but it is not yet a full recommendation-quality benchmark.

## 12. Misuse Considerations

Possible misuses include:

- overstating the system as a production-quality personalized recommender
- treating LLM narratives as objective truth rather than user-facing explanations
- assuming missing songs imply bad taste matches instead of sparse catalog coverage

Ways the project reduces that risk:

- exposing confidence and flags
- keeping the ranking logic inspectable
- logging intermediate decisions
- documenting limitations directly in the README and model card

## 13. Future Improvements

- fix the import-path issue so plain `pytest` works
- build a formal evaluation harness with saved benchmark queries
- incorporate more song features into final ranking
- expand the catalog for better genre and mood coverage
- add softer similarity between related genres
- compare templated vs. LLM explanation quality with a repeatable rubric

## 14. Reflection

This project reinforced that an AI system is rarely just "the model." The real behavior comes from how planning, retrieval, scoring, fallback logic, and error handling interact. The most valuable design choice here was keeping the LLM responsible for interpretation and explanation while leaving final ranking to deterministic logic. That split made the system easier to reason about, easier to test, and more honest about uncertainty.
