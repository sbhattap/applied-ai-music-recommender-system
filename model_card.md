# Model Card: Music Recommender Simulation

## 1. Model Name

**MoodMusical 1.0**

---

## 2. Intended Use

MoodMusical 1.0 generates personalized song recommendations based on a listener's genre preference, mood, desired energy level, and whether they prefer acoustic sounds. It is designed for classroom exploration of how content-based filtering works — not for production deployment. The system assumes each user can be described by a single genre, mood, energy target, and acoustic preference, and that those preferences are stable across a session.

---

## 3. How the Model Works

Each song in the catalog has five measurable attributes: genre, mood, energy (how intense it feels, 0–1), danceability (how groove-friendly it is, 0–1), and acousticness (how "unplugged" it sounds, 0–1).

When a user profile is supplied, the model compares those attributes against what the user prefers and awards partial credit for each match:

- **30%** goes to how close the song's energy is to what the user wants.
- **25%** goes to whether the genre is an exact match.
- **20%** goes to whether the mood is an exact match.
- **15%** rewards songs that are acoustic (or non-acoustic) based on the user's preference.
- **10%** is a flat bonus for danceability — higher-danceability songs always get a small edge.

Songs are ranked by their total score. A **diversity cap** limits results to at most two songs from any single genre, so the top five recommendations span different styles rather than repeating the same genre.

A second weight configuration (`SHIFTED_WEIGHTS`) doubles the energy influence and halves genre influence, letting you compare how sensitive the rankings are to weight choices.

The main addition over the starter logic is the configurable weights system and the per-recommendation explanation that tells you exactly which factors drove the match.

---

## 4. Data

The active catalog is `songs.csv`, which contains **10 songs** across **8 genres**: pop, lofi, rock, ambient, jazz, synthwave, indie pop, and indie pop. Moods represented include happy, chill, intense, relaxed, focused, and moody. A second file (`new_songs.csv`) exists in the repo with 8 additional songs covering country, EDM, k-pop, classical, hip-hop, latin, metal, and folk, but it is never loaded by the application. The dataset is entirely synthetic. Notable gaps: no R&B, no reggae, no blues, no melancholic songs, and most genres have only one representative song, which means the diversity cap is rarely a constraint for niche genres.

---

## 5. Strengths

The system works well for users whose preferences are internally consistent — a high-energy pop fan or a chill lofi listener gets sensible, coherent results because genre, mood, and energy all point toward the same songs. The diversity cap meaningfully improves results for users whose top genre has limited catalog coverage, forcing the recommender to surface interesting alternatives. The explanation feature clearly communicates which factors drove each suggestion, making it easy to audit and learn from.

---

## 6. Limitations and Bias

- **Energy dominates.** At 30%, energy is the single largest factor. A user who wants melancholic classical music but accidentally sets a high energy target will almost never see the one melancholic classical song in the catalog because its low energy score overwhelms the mood and genre bonuses.
- **Binary genre/mood matching.** There is no partial credit for related genres (e.g., "lofi" and "ambient" are treated as completely unrelated). A user who likes lofi but the catalog runs out of lofi songs gets unrelated genres rather than acoustic/chill alternatives.
- **Thin catalog.** With only 1–2 songs per genre, the diversity cap rarely has work to do, and some genre/mood combinations simply do not exist (e.g., no jazz song is tagged "happy").
- **Tempo and valence are unused.** Both attributes are stored in the data but never scored, so two songs with identical genre/mood/energy but very different feels (bright vs. dark, slow vs. fast) are treated identically.
- **No personalization over time.** The model has no memory of past listens, so it cannot learn or adapt.

---

## 7. Evaluation

Six profiles were run side-by-side under both weight configurations:

| Profile | What was checked |
|---|---|
| High-Energy Pop | Expected pop songs near energy 0.88 to dominate |
| Chill Lofi | Expected lofi songs + diversity cap to kick in after 2 |
| Deep Intense Rock | Only one rock song exists — tested fallback behavior |
| Conflicting Energy + Mood | High energy vs. melancholic mood pulling in opposite directions |
| Acoustic EDM Contradiction | Genre bonus for EDM vs. acousticness penalty |
| No Exact Match in Catalogue | Jazz + happy = no song qualifies; tested graceful degradation |

The most surprising finding: in the `CONFLICTING_ENERGY_MOOD` adversarial case the one truly melancholic song ("Copper Cathedral", energy 0.30) never appeared in the top 5 under either weight set — energy proximity to 0.9 crushed its score entirely. This shows how a single dominant feature can silence a perfect mood match.

The weight shift (doubled energy, halved genre) produced visible rank changes in 4 of 6 profiles, confirming the system is sensitive to weight choices in meaningful ways.

---

## 8. Future Work

- **Soft genre similarity** — cluster related genres (lofi, ambient, acoustic folk) so fallback recommendations feel intentional rather than arbitrary.
- **Use tempo and valence** — incorporating these would let the model distinguish fast-happy from slow-happy songs.
- **Learned weights** — instead of hand-tuning, infer weights from implicit feedback (skips, replays).
- **Richer catalog** — more songs per genre would make the diversity cap useful and reduce the "no exact match" problem.
- **Contextual profiles** — support time-of-day or activity context (working, exercising, winding down) rather than a single static preference vector.

---

## 9. Personal Reflection

Building this made the energy-dominance problem immediately visible in a way that reading about recommender bias never did — watching "Copper Cathedral" disappear from results despite being a perfect mood match was a concrete, memorable illustration of how feature weighting shapes what users never see. It also changed how I think about music apps: the songs surfaced at the top of a playlist are not "the best matches" in some objective sense; they are the winners of a weighted arithmetic race whose rules were set by engineers, and small changes to those weights produce meaningfully different listening experiences.