# Music Recommender - Data Flow

## Original System (v1)

The first version: a fixed profile scored against every song in the catalog.

```mermaid
flowchart TD
    A["Input: User Preferences
    -------------------
    favorite_genre
    favorite_mood
    target_energy
    likes_acoustic"]

    B["Song Catalog (CSV)
    -------------------
    new_songs.csv
    id, title, artist, genre
    mood, energy, tempo_bpm
    valence, danceability, acousticness"]

    C{"The Loop
    For every song in catalog..."}

    D["Scoring Formula
    -------------------
    genre_match    x 0.25
    mood_match     x 0.20
    energy_proximity x 0.30
    acousticness_fit x 0.15
    danceability   x 0.10
    -------------------
    total score (0.0 - 1.0)"]

    E["Diversity Filter
    -------------------
    Max 2 songs per genre
    (keeps results varied)"]

    F["Output: Top-K Recommendations
    -------------------
    Ranked list of K songs
    + score + explanation
    per song"]

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
```

## Current System: Agentic RAG + Spotify

The scoring idea from v1 survives as the 7 profile signals inside an additive reranker. Around it, an agent plans and checks retrieval over a small knowledge store, and an optional Spotify login turns listening history into a profile, a taste vector, and citable evidence.

```mermaid
flowchart LR
    subgraph SOURCES["Sources"]
        CSV["data/songs.csv<br/>180 songs, 18 genres"]
        WIKI["Wikipedia<br/>artist + genre summaries"]
        LFM["Last.fm tags<br/>optional"]
        SPAPI["Spotify Web API<br/>recent, top tracks, top artists, saved"]
    end

    subgraph BUILD["Built once (scripts/build_knowledge.py)"]
        DOCS["data/rag/*.jsonl<br/>artist:, genre:, tags: docs"]
    end

    subgraph INDEX["Index (.cache/)"]
        CARDS["song: cards"]
        VEC["Embedding cache<br/>bge-small, keyed by text hash"]
        BM["BM25 index"]
    end

    subgraph USER["Per listener"]
        TOK[".spotify_tokens.json<br/>mode 600, gitignored"]
        LAB[".cache/spotify_track_labels.json<br/>LLM labels per track"]
        PROF["Profile<br/>derived prefs, taste vector,<br/>recently played, hist: + similar: docs"]
    end

    CSV --> CARDS
    WIKI --> DOCS
    LFM --> DOCS
    DOCS --> VEC
    CARDS --> VEC
    CARDS --> BM
    DOCS --> BM
    TOK --> SPAPI
    SPAPI --> PROF
    LAB <--> PROF

    Q["Query"] --> AGENT["Agent<br/>plan, act, check, refine"]
    PROF --> AGENT
    VEC --> AGENT
    BM --> AGENT
    AGENT --> RANK["Additive reranker"]
    RANK --> EXP["Explainer + citation validator"]
    EXP --> OUT["UI / CLI"]
    AGENT -.-> LOG["logs/llm_calls.jsonl<br/>model, latency, tokens"]
    EXP -.-> LOG
```

### Pipeline per query

```mermaid
flowchart TD
    A["Query + profile"] --> B["PLAN<br/>intent, filters, expected count, tool"]
    B --> C{"Tool"}
    C --> D["structured_filter<br/>hard filters in code"]
    C --> E["semantic_search<br/>hybrid retriever"]
    C --> F["hybrid<br/>filter, then retrieve"]
    D --> G["Candidates + evidence docs per song"]
    E --> G
    F --> G
    G --> H["CHECK<br/>confidence + flags"]
    H -- "weak, max 1 refine" --> B
    H --> I["Rerank<br/>query_match, taste_match, novelty<br/>+ 7 profile signals"]
    I --> J["EXPLAIN<br/>one batched call, claims cite ids"]
    J --> K["Validate citations<br/>drop uncited claims"]
    K --> L["Ranked cards, score bars,<br/>cited evidence, trace"]
```

### What data goes where

| Data | Where it lives | Leaves the machine? |
|---|---|---|
| Song catalog, knowledge docs | `data/` (in the repo) | No |
| Embeddings, BM25 index | `.cache/` | No (embeddings run locally) |
| Spotify tokens | `.spotify_tokens.json`, mode 600, gitignored | Only to Spotify |
| Listening history | in memory; Streamlit caches it for 30 min | No |
| History track labels (title, artist → genre, mood, energy) | `.cache/spotify_track_labels.json` | Titles and artists of unmatched tracks are sent to Claude once, for labelling |
| Query, catalog snippets, evidence | in memory | Sent to Claude for plan, check, and explain |
| LLM call log | `logs/llm_calls.jsonl` | No. Holds model, latency, and token counts, not prompts |

To remove your Spotify data: click **Disconnect Spotify** (deletes the tokens), then delete `.cache/spotify_track_labels.json`.
