# Applied AI Music Recommender System

This project extends my rule-based deterministic music recommender into an agentic music recommender that accepts free-text requests, retrieves candidate songs, checks whether the results make sense, and then explains the final recommendations. It matters because it turns a transparent but rigid rule-based recommender into a more realistic AI workflow: the user can ask for a vibe in natural language, while the system still keeps its reasoning observable and its ranking reproducible.

The original project was a deterministic content-based recommender built around `Song`, `UserProfile`, and a weighted scoring function. Given a structured profile, it ranked songs by genre, mood, energy, acousticness, danceability, valence, and popularity. This extension keeps that ranking backbone, but adds an LLM-driven agent so the system can plan from free text, choose a retrieval strategy, evaluate candidate quality, and generate user-facing explanations.

## New Additions

- It adds a substantial new AI feature: an agentic workflow with `plan -> act -> check -> respond`.
- It keeps the deterministic scorer from the original project as the final ranking authority.
- It exposes intermediate reasoning through trace logs, confidence scores, flags, and the Streamlit UI.
- It includes reliability mechanisms instead of treating the model output as automatically correct.

## Architecture Overview

The system has two entry points: a CLI and a Streamlit app. Both send a free-text query and a listener profile into the same backend pipeline.

```text
User query + listener profile
            |
            v
      [Agent: PLAN]
      Extract intent, filters,
      expected result shape,
      and retrieval tool
            |
            v
   +--------+---------+----------------+
   |                  |                |
   v                  v                v
[structured]      [semantic]       [hybrid]
[filter]          [search]         [search]
   \                  |                /
    \                 |               /
     +----------------+--------------+
                      |
                      v
               Candidate songs
                      |
                      v
              [Agent: CHECK]
        Confidence score + flags:
        looks_good / low_result_count /
        constraint_violated / intent_mismatch
                      |
                      v
         [Deterministic re-ranker]
      7-signal recommendation scoring
                      |
                      v
             [Agent: EXPLAIN]
                      |
                      v
     CLI output / Streamlit cards / trace panel
```

In plain language: the LLM is responsible for understanding the request, deciding how to search, judging whether the retrieved songs fit the request, and narrating the results. The final ordering still comes from the rule-based recommender, which makes the ranking easier to debug and test.

## Design Decisions

- I kept the original scorer as the source of truth. This trades some flexibility for reproducibility and easier testing.
- I used three retrieval modes: `structured_filter`, `semantic_search`, and `hybrid`. This lets the agent handle both clear constraints and fuzzier vibe-based requests.
- I added a check step with confidence and flags instead of trusting retrieval blindly. This improves observability and gives the UI a way to surface uncertainty.
- I capped the refine loop so the agent can broaden or switch strategies without turning one user request into an unbounded chain of retries.
- I did not add a vector database or multi-source RAG because the song catalog is small enough that simpler retrieval strategies are easier to explain and maintain.

The main trade-off is that the system is more capable on free-text requests, but it also depends on API availability and prompt quality. The deterministic baseline is simpler and cheaper; the agentic version is more expressive and more realistic.

## Setup Instructions

### Prerequisites

- Python 3.12+
- A Gemini API key stored as `GEMINI_API_KEY`

### Installation

1. Clone the repository and enter the project folder.
2. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -e .
```

4. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

5. Optional configuration:

```env
RECOMMENDER_MODEL=gemini-3-flash-preview
RECOMMENDER_EFFORT=medium
```

### Run The App

Start the Streamlit UI:

```powershell
streamlit run streamlit_app.py
```

Run the CLI version:

```powershell
python -m src.recommender.main --query "chill study music, no vocals" --profile chill_lofi
```

Useful CLI options:

- `--k 5` to control how many recommendations are returned
- `--no-llm-explainer` to use the templated fallback explanation
- `--songs data/songs.csv` to point at a different catalog file

### Run Tests

The current test suite passes with:

```powershell
$env:PYTHONPATH='.'
pytest -p no:cacheprovider
```

This runs the offline test suite with mocked LLM calls and avoids the current `src.recommender` import-path issue that appears with plain `pytest`.

## Sample Interactions

These examples show the current system behavior on the repository's song catalog. They illustrate both successful matches and one guardrail case.

### 1. Workout Pop

**Input**

- Query: `upbeat pop for a workout`
- Profile: `High-Energy Pop`

**Output**

- `I Ain't Worried` by OneRepublic
  Recommended because it's pop, matches the happy mood, and its energy is close to the target.
- `Calm Down (with Selena Gomez)` by Rema, Selena Gomez
  Recommended because it's pop, matches the happy mood, and stays close to the target energy.
- Agent check: `looks_good`
- Confidence: `0.84`

### 2. Intense Night Listening

**Input**

- Query: `intense late-night rock`
- Profile: `Deep Intense Rock`

**Output**

- `Sex, Drugs, Etc.` by Beach Weather
  Strong match on genre, mood, energy, and non-acoustic sound.
- `Hold Me Closer` by Elton John, Britney Spears
  Strong match on genre, mood, and overall production feel.
- `The Next Episode` by Dr. Dre, Snoop Dogg
  Lower-ranked fallback that still matches the intense, high-energy request.
- Agent check: `looks_good`
- Confidence: `0.79`

### 3. Guardrail Example

**Input**

- Query: `songs from a fake genre`
- Profile: `Chill Lofi`

**Output**

- No recommendations returned
- Agent check: `low_result_count`
- Confidence: `0.00`
- System behavior: the pipeline synthesizes a low-confidence verdict instead of pretending the result is valid

## Reliability And Testing Summary

This project includes several reliability mechanisms required by the rubric:

- Automated tests for the deterministic recommender, retrieval tools, explainer behavior, and agent loop
- Confidence scoring and issue flags from the `check` step
- Structured logging for LLM calls in `logs/llm_calls.jsonl`
- Error handling and fallback explanations when the LLM is unavailable

Current test result:

- `24/24` tests passed with `$env:PYTHONPATH='.'; pytest -p no:cacheprovider`

What worked:

- The mocked end-to-end agent tests passed, including plan, act, check, respond, and refine behavior.
- The deterministic scorer still produces stable rankings after the AI layer was added.
- The explainer fallback keeps the app usable even if the LLM call fails.

What did not work cleanly:

- Plain `pytest` currently fails during test collection because the tests import `src.recommender`.
- The live agent path depends on a valid Gemini API key, so full online runs are not completely offline-reproducible.
- Very narrow or contradictory constraints can still produce sparse candidate sets.

What I learned:

- Reliability features are not optional extras in AI systems; they are part of the product.
- A small deterministic core paired with an LLM orchestration layer is easier to evaluate than an all-LLM recommender.
- Surfacing uncertainty helps users trust the system more than hiding failure states.

## Reflection

This project taught me that building with AI is less about asking a model for an answer and more about designing a workflow around uncertainty. The most important shift from the earlier modules was not just "adding an LLM," but deciding where the model should have freedom and where the system should stay deterministic. Letting the agent interpret free-text intent was valuable; letting it control the final ranking would have made the system much harder to debug and justify.

It also highlighted the limitations and biases of small recommendation systems. The catalog is limited, so some genres or moods are underrepresented. That means the agent can understand a request correctly and still fail to satisfy it because the data is weak. The system could also be misused if someone assumed the recommendations were personalized at Spotify-scale; to reduce that risk, I expose the trace, the confidence score, and the recommendation rationale instead of presenting the output as magically authoritative.

My collaboration with AI during development was useful but not flawless. One helpful suggestion was separating the workflow into plan, check, and explain stages while keeping the original rule-based scorer in charge of ranking; that made the system much easier to test. One flawed suggestion was an earlier configuration path that pointed at the wrong provider-style setup, which would have sent the project toward mismatched environment-variable and model assumptions. Correcting that reinforced an important lesson: AI can accelerate design and debugging, but it still needs human review, especially around architecture and integration details.

## Future Improvements

- Fix the test import path so plain `pytest` works without setting `PYTHONPATH`
- Add a formal evaluation harness with saved benchmark queries and quality metrics
- Add a portfolio demo video link and a cleaner architecture image for recruiters
- Expand the song catalog so the agent has better coverage for niche tastes

## Portfolio Notes

- GitHub repository: https://github.com/sunitab55/applied-ai-music-recommender-system
- Loom walkthrough: add your Loom link here when the demo video is ready
- What this project says about me as an AI engineer: I care about building AI systems that are inspectable, testable, and honest about uncertainty rather than just impressive at first glance.
