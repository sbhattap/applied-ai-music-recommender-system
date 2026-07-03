# Model Card: Applied AI Music Recommender System

## 1. Model Name

**Applied AI Music Recommender System**

This project is a hybrid system rather than a single standalone model. It combines:

- a deterministic content-based scorer (the original 7 profile signals)
- a hybrid retriever (BM25 + local `BAAI/bge-small-en-v1.5` embeddings) over a small knowledge store
- an LLM agent (Claude, via the `anthropic` SDK) that plans, checks, and explains
- optional Spotify personalization from the listener's own history

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

The system has four layers.

### Deterministic scoring layer

The original recommender scores songs with seven profile signals:

- genre match
- mood match
- energy proximity
- acousticness fit
- danceability
- valence proximity
- popularity fit

### Retrieval layer (RAG)

- **Documents:** `song:` cards (attributes rendered as words), plus `artist:` and `genre:` summaries from Wikipedia, plus optional Last.fm `tags:`. A Spotify listener adds `hist:` and `similar:` documents.
- **Search:** BM25 and dense search, fused with reciprocal rank fusion. Every hit records its BM25 rank, its dense rank, and the query terms it matched, along with the evidence documents that support it.

### LLM orchestration layer

1. `PLAN`: interpret the request, extract hard filters, choose a retrieval tool, and estimate the expected result count
2. `ACT`: run `structured_filter`, `semantic_search` (the hybrid retriever), or `hybrid` (filter, then retrieve)
3. `CHECK`: assign a confidence score and flags. Clear-cut cases are decided in code without an LLM call.
4. `REFINE`: if the check is weak, relax filters or switch tools (at most once)
5. `RANK`: the additive reranker, `score = 0.5·query_match + 0.15·taste_match + 0.05·novelty + Σ wᵢ·profile_signalᵢ`
6. `EXPLAIN`: one batched call writes short claims per song. Each claim must cite evidence ids, and a deterministic validator drops claims that don't.

### Personalization layer (Spotify, optional)

Listening history is matched to the catalog, and unmatched tracks are labelled with catalog-style genre, mood, and energy by one cached LLM call. The result is a derived profile, a taste vector (which adds a `taste_match` signal), a recently-played set (no novelty bonus for those songs), and citable `hist:` / `similar:` evidence.

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

**Spotify profile (optional)**

When the listener logs in, the profile above is derived from their history instead of picked from a preset. It also carries `taste_vector`, `recently_played`, and `history_docs`.

### Outputs

- ranked recommendations
- additive score per recommendation, broken down by signal
- retrieved evidence per recommendation, with source and link
- cited claims (LLM-written, or template claims built from the same evidence when offline)
- claims dropped by the citation validator
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

Knowledge store (`data/rag/`): 187 Wikipedia artist summaries and 18 genre summaries. Last.fm tags are added only when `LASTFM_API_KEY` is set.

This is a small catalog, which makes the system easier to inspect but also limits coverage for niche combinations of mood, genre, and constraints.

## 6. How Ranking Works

The final score is additive: `0.5·query_match + 0.15·taste_match + 0.05·novelty` plus the profile signals below. A profile signal only counts when the listener stated that preference, and the UI's per-signal bars sum exactly to the score.

The profile signals use these default weights:

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

The LLM does not score songs. It controls which candidates reach the reranker and writes the explanations.

### Retrieval tools

- `structured_filter`: pure logic filter on genre, mood, BPM, energy, and instrumentalness
- `semantic_search`: the hybrid retriever over the knowledge store (~12 ms). The legacy backend, where an LLM reads the whole catalog, is kept behind `RECOMMENDER_SEMANTIC_BACKEND=llm` for ablations.
- `hybrid`: hard filters first, then the hybrid retriever inside the filtered set

### Check output

The check step returns:

- a confidence score from `0.0` to `1.0`
- zero or more flags

Supported flags:

- `looks_good`
- `low_result_count`
- `constraint_violated`
- `intent_mismatch`

These signals drive the single refine step and are shown in the UI.

### Explanations

Each claim must cite evidence ids: a score signal (`sig:energy`), a song card, an artist or genre summary, or a listening-history doc. Claims with no ids, or with ids that weren't offered for that song, are dropped and recorded in the trace.

## 8. Strengths

- The system handles free-text requests that the original structured-only recommender could not accept.
- The deterministic scorer keeps the final ranking reproducible and testable.
- The check step makes uncertainty visible instead of silently hiding weak retrieval results.
- The fallback explainer keeps the system functional even if the LLM explanation call fails.
- Every explanation claim is tied to a retrieved document or a score signal, and unsupported claims are removed before the user sees them.
- Hybrid retrieval beats the original LLM catalog search on P@5 at about 1/150th of the latency.
- The separation between retrieval and ranking makes the architecture easier to debug than a fully end-to-end LLM recommender.

## 9. Limitations And Biases

- The catalog is still relatively small, so some requests fail because the data is sparse rather than because the logic is wrong.
- Genre matching in the ranker is binary, so related genres do not receive partial credit.
- The system can still struggle with contradictory requests such as wanting a very high-energy but strongly melancholic acoustic track.
- `tempo_bpm` and `instrumentalness` affect retrieval but not final ranking, so some constraints shape the candidate set more than the final score.
- `prefer_popular=True` by default can tilt the system toward mainstream music unless changed.
- LLM planning and semantic retrieval are dependent on prompt quality and API availability.
- Without a Spotify login, recommendations are not personalized from listening history.
- With a Spotify login, only a small share of a real history is in the 180-song catalog (1 of 132 tracks in testing). Most personalization therefore comes from labels and similarity, not real plays.
- Spotify no longer gives new apps audio features, and gives genres for few artists. History labels are LLM estimates of genre, mood, and energy, not measurements.
- `similar:` links compare artist, genre, and mood text with embeddings. They are restricted to the same genre to avoid title-word false matches, but they are still approximate.
- Wikipedia summaries can be stale, and the explainer may only repeat what they say.

## 10. Reliability And Safety Features

- Pydantic-validated structured outputs for the plan, check, explanations, and Spotify labels
- confidence scoring and issue flags in the check step; deterministic checks for clear-cut cases
- the refine loop is capped at one retry
- citation validation for every explanation claim
- offline fallbacks for every LLM step (heuristic plan, deterministic check, template explanation, embedding-neighbour profile)
- every LLM call is logged to `logs/llm_calls.jsonl` (model, latency, tokens; no prompts)
- 63 offline tests (`uv run pytest`) covering scoring, tools, retrieval, citations, the agent loop, the explainer, and Spotify auth, client, profile, and labelling

## 11. Evaluation Summary

`scripts/run_eval.py` runs 24 labeled queries. These are live results with Claude (`eval/live/results.md`):

| Retrieval | P@5 | nDCG@10 | Recall@20 | MRR | p50 latency |
|---|---|---|---|---|---|
| LLM reads whole catalog | 0.767 | 0.779 | 0.660 | 0.734 | 1794 ms |
| BM25 | 0.675 | 0.685 | 0.627 | 0.735 | 12 ms |
| Dense | 0.700 | 0.691 | 0.707 | 0.840 | 12 ms |
| Hybrid | 0.775 | 0.762 | 0.707 | 0.856 | 12 ms |

End to end (k=5): P@5 0.792, 3.17 LLM calls per query, citation validity 1.000, LLM-judge faithfulness 0.851, p50 latency 14.4 s (explain 8.8 s of that).

Spotify personalization was tested live: login, profile derivation, three personalized queries, and the Streamlit connect flow. There is no labeled benchmark for personalized quality yet.

## 12. Misuse Considerations

Possible misuses include:

- overstating the system as a production-quality personalized recommender
- treating LLM narratives as objective truth rather than user-facing explanations
- assuming missing songs imply bad taste matches instead of sparse catalog coverage
- reading "similar to a song you play" as a measured audio match rather than an estimate

Ways the project reduces that risk:

- exposing confidence and flags
- keeping the ranking logic inspectable
- logging intermediate decisions
- documenting limitations directly in the README and model card

## 13. Future Improvements

- import a listener's top tracks into the catalog, with estimated features clearly labeled
- an "exclude recently played" toggle and a counterfactual view (remove a history track or zero out a signal, then re-rank)
- Last.fm crowd tags in the knowledge store
- softer similarity between related genres
- a labeled benchmark for personalized recommendations
- bring p50 latency under 5 s

## 14. Privacy

Spotify personalization handles personal listening data, so it is kept local by default:

- **Tokens:** `.spotify_tokens.json`, file mode 600, gitignored. They're only sent to Spotify. Scopes are read-only: `user-read-recently-played`, `user-top-read`, `user-library-read`.
- **History:** fetched live and kept in memory. Streamlit caches the derived profile for 30 minutes. Raw history is never written to disk.
- **Labels:** titles and artists of history tracks missing from the catalog are sent to Claude once for labelling. The labels are cached in `.cache/spotify_track_labels.json`.
- **Logs:** `logs/llm_calls.jsonl` holds model, latency, and token counts, never prompts or history.
- **Deletion:** **Disconnect Spotify** deletes the tokens. Delete `.cache/spotify_track_labels.json` to remove the labels.
- **Access:** the Spotify app runs in Development Mode, so only users allowlisted in the dashboard (max 5) can log in.

## 15. Reflection

This project reinforced that an AI system is rarely just "the model." The real behavior comes from how planning, retrieval, scoring, fallback logic, and error handling interact. The most valuable design choice here was keeping the LLM responsible for interpretation and explanation while leaving final ranking to deterministic logic. That split made the system easier to reason about, easier to test, and more honest about uncertainty.
