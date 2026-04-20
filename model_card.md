# Model Card: Music Recommender Simulation

## 1. Model Name

**MoodMusical 1.5**

---

## 2. Intended Use

MoodMusical 1.5 generates personalized song recommendations based on a listener's genre preference, mood, desired energy level, acoustic preference, emotional tone (valence), and mainstream vs. niche taste. It is designed for classroom exploration of how content-based filtering works — not for production deployment. The system assumes each user can be described by a fixed preference vector and that those preferences are stable across a session.

---

## 3. How the Model Works

Each song in the catalog has seven scored attributes: genre, mood, energy (intensity, 0–1), danceability (groove-friendliness, 0–1), acousticness (how "unplugged" it sounds, 0–1), valence (musical positiveness, 0–1), and popularity (chart presence, 0–100 normalized to 0–1).

**User profile fields:**

| Field | Type | Description |
|---|---|---|
| `favorite_genre` | string | The genre the user most wants to hear |
| `favorite_mood` | string | The mood or vibe the user is looking for |
| `target_energy` | float 0–1 | Preferred intensity level |
| `likes_acoustic` | bool | Acoustic vs. produced/electronic preference |
| `target_valence` | float 0–1 | Emotional tone: 0 = darker/sadder, 1 = upbeat/positive (default 0.5) |
| `prefer_popular` | bool | Mainstream taste vs. underground/niche (default True) |

**Default scoring weights:**

| Signal | Weight | How it's calculated |
|---|---|---|
| Energy proximity | 25% | `1 − │song.energy − target_energy│` |
| Genre match | 20% | 1.0 if exact match, else 0.0 |
| Mood match | 15% | 1.0 primary match · 0.5 tag match · 0.0 no match |
| Valence proximity | 12% | `1 − │song.valence − target_valence│` |
| Acousticness fit | 12% | `song.acousticness` if acoustic preferred, else `1 − song.acousticness` |
| Danceability | 8% | Raw danceability value |
| Popularity fit | 8% | Normalized popularity if mainstream preferred, inverted if niche preferred |

**Soft mood matching:** Each song carries a `mood_tags` field (e.g., `"happy|uplifting|summer"`). If a song's primary mood doesn't match but the user's mood appears in its tags, it receives 0.5 credit instead of 0. This allows partial matches across mood boundaries.

Songs are ranked by their total score. A **diversity cap** limits results to at most two songs from any single genre.

**Named ranking strategies** replace the earlier single weight-shift experiment. Setting `ACTIVE_STRATEGY` in `main.py` switches the right-column output to one of three presets, each giving one signal a dominant share:

| Strategy | Dominant signal | Intent |
|---|---|---|
| `genre_first` | genre 50% | Genre loyalty — stay within the user's category above all else |
| `mood_first` | mood 50% | Vibe match — emotional state matters more than genre label |
| `energy_similarity` | energy 55% | Audio twin — find the closest feel regardless of genre or mood |

Each profile run prints two side-by-side tables — default weights vs. active strategy — along with per-song bullet-point explanations of which factors drove the score.

---

## 4. Data

The catalog is `data/songs.csv`, which contains **30 songs** across **27 genres** (ambient, blues, bossa nova, classical, country, edm, folk, funk, hip-hop, indie pop, indie rock, jazz, jazz fusion, k-pop, latin, lofi, metal, new wave, pop, post-rock, r&b, reggae, rock, soul, synthwave, techno, trap) and **9 moods** (chill, confident, focused, happy, intense, melancholic, moody, relaxed, romantic). The dataset is entirely synthetic.

Each song carries the following fields — those in **bold** are used in scoring:

| Field | Type | Scored? |
|---|---|---|
| **genre** | string | Yes |
| **mood** | string | Yes (primary) |
| **mood_tags** | pipe-separated string | Yes (soft match) |
| **energy** | float 0–1 | Yes |
| **valence** | float 0–1 | Yes |
| **danceability** | float 0–1 | Yes |
| **acousticness** | float 0–1 | Yes |
| **popularity** | int 0–100 | Yes |
| tempo_bpm | float | No |
| liveness | float 0–1 | No |
| duration_sec | int | No |
| explicit | 0 or 1 | No |
| release_decade | string | No |

Notable gaps: most genres still have only one or two representative songs, which means the diversity cap rarely constrains niche genres. There are no songs for some mood×genre combinations (e.g., no jazz song is tagged "happy").

---

## 5. Strengths

The system works well for users whose preferences are internally consistent — a high-energy pop fan or a chill lofi listener gets sensible, coherent results because genre, mood, and energy all point toward the same songs. The diversity cap meaningfully improves results for users whose top genre has limited catalog coverage, forcing the recommender to surface interesting alternatives.

**Soft mood matching** reduces the harsh cliff between "exact match" and "no match": a song tagged `"happy|uplifting|summer"` still earns partial credit for a user who wants happy music even if its primary mood label differs.

**Valence and popularity scoring** add two dimensions that were previously ignored. Valence lets the model distinguish between bright/upbeat songs and darker ones even within the same genre. The popularity dimension can be inverted, so users who prefer underground or niche music are actively rewarded rather than penalized by default mainstream bias.

**Named ranking strategies** (`genre_first`, `mood_first`, `energy_similarity`) make it easy to see how a single design decision — which signal should dominate — changes the entire character of the output. The explanation feature generates per-song bullet-point reasons, making it easy to audit exactly which factors drove each result.

---

## 6. Limitations and Bias

- **Energy still dominates.** At 25%, energy remains the single largest factor. A user who wants melancholic classical music but sets a high energy target will rarely see low-energy melancholic songs because the energy gap overwhelms mood and genre bonuses.
- **Genre matching is still binary.** There is no partial credit for related genres — "lofi" and "ambient" are treated as completely unrelated. A user who runs out of lofi songs in the catalog gets arbitrary genre fallbacks rather than acoustically similar alternatives.
- **Sparse genre coverage.** Most genres still have only one or two songs, so the diversity cap rarely constrains results and some genre/mood combinations simply don't exist (e.g., no jazz song is tagged "happy").
- **Popularity bias.** `prefer_popular` defaults to `True`, which means by default every scoring run gives a small advantage to chart-popular songs. Users who don't set this explicitly will subtly be steered toward mainstream tracks.
- **Tempo and several new fields are unused.** `tempo_bpm`, `liveness`, `duration_sec`, `explicit`, and `release_decade` are loaded from the CSV but carry zero weight in scoring. Two songs with very different tempos or live vs. studio feels are still treated identically.
- **No personalization over time.** The model has no memory of past listens, so it cannot learn or adapt.

---

## 7. Evaluation

Six profiles were run side-by-side under the default weights vs. each named strategy:

| Profile | What was checked |
|---|---|
| High-Energy Pop | Expected pop songs near energy 0.88 to dominate |
| Chill Lofi | Expected lofi songs + diversity cap to kick in after 2 |
| Deep Intense Rock | Only one rock song exists — tested fallback behavior |
| Conflicting Energy + Mood | High energy vs. melancholic mood pulling in opposite directions |
| Acoustic EDM Contradiction | Genre bonus for EDM vs. acousticness penalty |
| No Exact Match in Catalogue | Jazz + happy = no song qualifies; tested graceful degradation |

**Key findings:**

- Under **default weights**, "Copper Cathedral" (classical, melancholic, energy=0.30) appears at rank 1 for the Conflicting Energy + Mood profile because the balanced weights give enough combined credit to its genre, mood, and acousticness to overcome its energy penalty.
- Under **`genre_first`**, "Coffee Shop Stories" (the only jazz track) jumps to rank 1 for the No Exact Match profile even though its energy (0.37) is nearly opposite to the user's target (0.9). The 50% genre bonus overrides every other signal — demonstrating how strategy choice can completely invert a ranking.
- Under **`energy_similarity`**, genre and mood labels become almost irrelevant. The top 5 for a "jazz/happy/energy=0.9" user are entirely non-jazz songs ranked by how close their energy is to 0.9.
- The **soft mood tag matching** had a measurable effect in several profiles where a song's primary mood differed from the user's preference but its tags included a partial match — those songs ranked noticeably higher than they would have under binary matching.

---

## 8. Future Work

- **Soft genre similarity** — cluster related genres (lofi, ambient, acoustic folk) so fallback recommendations feel intentional rather than arbitrary.
- **Use the remaining unused fields** — `tempo_bpm`, `liveness`, `duration_sec`, `explicit`, and `release_decade` are loaded but not scored. Tempo would let the model distinguish fast-happy from slow-happy; liveness could serve users who prefer live recordings; release decade could support nostalgia or recency preferences.
- **Learned weights and learned strategies** — instead of hand-tuning, infer both the weight distribution and the appropriate strategy from implicit feedback (skips, replays, queue additions).
- **Popularity calibration** — the current popularity signal treats a score of 70 the same regardless of genre. Normalizing within genre (a popularity of 70 in jazz means something very different than in pop) would reduce cross-genre bias.
- **Contextual profiles** — support time-of-day or activity context (working, exercising, winding down) rather than a single static preference vector. This would also give the strategy system a natural trigger: automatically switch to `energy_similarity` for a workout context.

---

## 9. Personal Reflection

Building this made the energy-dominance problem immediately visible in a way that reading about recommender bias never did — watching "Copper Cathedral" disappear from results despite being a perfect mood match was a concrete, memorable illustration of how feature weighting shapes what users never see. It also changed how I think about music apps: the songs surfaced at the top of a playlist are not "the best matches" in some objective sense; they are the winners of a weighted arithmetic race whose rules were set by engineers, and small changes to those weights produce meaningfully different listening experiences.