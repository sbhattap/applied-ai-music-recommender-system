# Music Recommender - Data Flow

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

## Extended System: Agentic Workflow Data Flow

The original deterministic data flow above is still part of the project. The extended system adds an LLM-driven orchestration layer in front of that scorer so the app can accept free-text requests, choose retrieval behavior, evaluate candidate quality, and explain the final recommendations.

```mermaid
flowchart TD
    A["User Input
    free-text query
    + selected listener profile"]

    B["Agent: PLAN
    parse intent
    extract constraints
    choose retrieval tool
    set expected result count"]

    C{"Chosen Tool"}

    D["structured_filter
    deterministic filtering on
    genre, mood, BPM, energy,
    and instrumentalness"]

    E["semantic_search
    LLM reviews the catalog
    and returns matching song ids"]

    F["hybrid
    structured filter first,
    then semantic search inside
    the narrowed candidate set"]

    G["Candidate Songs"]

    H["Agent: CHECK
    assign confidence
    emit flags:
    looks_good
    low_result_count
    constraint_violated
    intent_mismatch"]

    I{"Refine?"}

    J["Refined Plan
    relax filters
    widen numeric ranges
    or switch tools
    max 2 retries"]

    K["Deterministic Re-ranker
    7-signal scoring
    + diversity cap"]

    L["Agent: EXPLAIN
    generate recommendation
    narrative from score breakdown
    and check verdict
    fallback to templated text
    on API failure"]

    M["Outputs
    Streamlit recommendations
    CLI recommendations
    trace events
    logs/llm_calls.jsonl"]

    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    D --> G
    E --> G
    F --> G
    G --> H
    H --> I
    I -- Yes --> J
    J --> B
    I -- No --> K
    K --> L
    L --> M
```

### Notes On The New Flow

- The original rule-based recommender is still the final ranking authority.
- The new AI layer adds a `plan -> act -> check -> respond` loop.
- Retrieval is dynamic: the agent can choose `structured_filter`, `semantic_search`, or `hybrid`.
- The system now exposes confidence, flags, trace events, and structured LLM logs as reliability signals.
- If the first pass looks weak, the refine loop can broaden or switch retrieval strategies before ranking.
