"""Retriever, reranker and citation tests (offline hashing embedder via conftest)."""

import pytest

from src.recommender.core.rerank import active_weights, rerank
from src.recommender.rag import citations
from src.recommender.rag.bm25 import BM25, tokenize
from src.recommender.rag.documents import Document, build_documents, describe, slug
from src.recommender.rag.retriever import Retriever, get_retriever


def _song(id_, title, genre, mood, energy, **kw):
    base = {"id": id_, "title": title, "artist": kw.pop("artist", f"Artist {id_}"),
            "genre": genre, "mood": mood, "mood_tags": kw.pop("mood_tags", mood),
            "energy": energy, "tempo_bpm": kw.pop("tempo_bpm", 100.0),
            "valence": 0.5, "danceability": 0.5, "acousticness": 0.5,
            "instrumentalness": kw.pop("instrumentalness", 0.0), "popularity": 50,
            "release_decade": "", "liveness": 0.1, "duration_sec": 200, "explicit": 0}
    base.update(kw)
    return base


@pytest.fixture
def songs():
    return [
        _song(1, "Rainy Library", "lofi", "chill", 0.25, mood_tags="chill|study|rain",
              tempo_bpm=72, instrumentalness=0.9, acousticness=0.8),
        _song(2, "Stadium Anthem", "rock", "intense", 0.95, mood_tags="intense|loud", tempo_bpm=150),
        _song(3, "Sunday Piano", "classical", "relaxed", 0.15, mood_tags="relaxed|calm|sleep",
              tempo_bpm=60, instrumentalness=0.98, acousticness=0.95),
        _song(4, "Club Night", "edm", "happy", 0.9, mood_tags="happy|party|dance", tempo_bpm=128,
              danceability=0.9, acousticness=0.05),
    ]


def test_tokenize_normalizes_genre_spellings():
    assert tokenize("Hip Hop and lo-fi, R&B, no vocals") == ["hip-hop", "lofi", "r&b", "vocals"]


def test_describe_turns_features_into_words(songs):
    words = " ".join(describe(songs[0]))
    assert "low energy" in words and "slow tempo" in words and "no vocals" in words


def test_slug_strips_accents():
    assert slug("Frédéric Chopin") == "frederic-chopin"


def test_bm25_reports_matched_terms():
    bm = BM25(["chill lofi study beats", "loud rock anthem"])
    scores = bm.scores("lofi for studying and chill")
    assert scores[0] > scores[1]
    assert bm.matched_terms("lofi for chill", 0) == ["lofi", "chill"]


def test_retriever_ranks_relevant_song_first_with_provenance(songs):
    r = Retriever(songs, build_documents(songs))
    hits = r.search("rain study lofi", top_n=2)
    assert hits[0].song_id == 1
    assert hits[0].query_match == pytest.approx(1.0)
    assert "lofi" in hits[0].matched_terms
    assert hits[0].found_by in {"both", "bm25", "dense"}
    assert hits[0].evidence[0].doc_id == "song:1" or any(e.doc_id == "song:1" for e in hits[0].evidence)
    assert len(hits) == 2


def test_retriever_restricts_to_candidates(songs):
    r = get_retriever(songs)
    hits = r.search("lofi", candidates=[songs[1], songs[3]])
    assert {h.song_id for h in hits} == {2, 4}


def test_retriever_attaches_context_docs_and_taste(songs):
    extra = [Document(doc_id="hist:3", kind="hist", source="spotify", title="Sunday Piano",
                      text="You played Sunday Piano 12 times.", song_ids=[3])]
    r = get_retriever(songs, extra)
    taste = r.song_vector(3)
    hits = {h.song_id: h for h in r.search("calm piano", taste_vector=taste)}
    assert any(e.doc_id == "hist:3" for e in hits[3].evidence)
    assert hits[3].taste_sim == pytest.approx(1.0, abs=1e-3)
    assert hits[3].taste_sim > hits[2].taste_sim


def test_active_weights_sum_to_one():
    for flags in [(True, False, False), (True, True, True), (False, False, False)]:
        w = active_weights(has_query=flags[0], has_taste=flags[1], has_history=flags[2])
        assert sum(w.values()) == pytest.approx(1.0)


def test_rerank_is_additive_and_explains_query_match(songs):
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.3, "likes_acoustic": True}
    hits = {h.song_id: h.to_dict() for h in Retriever(songs, build_documents(songs)).search("rain study lofi")}
    ranked = rerank(prefs, songs, hits, k=2)
    song, score, signals, _ = ranked[0]
    assert song["id"] == 1
    assert score == pytest.approx(sum(s.contribution for s in signals), abs=1e-3)
    qm = next(s for s in signals if s.name == "query_match")
    assert "matched: " in qm.detail and "lofi" in qm.detail


def test_rerank_novelty_penalizes_recent_plays(songs):
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.3}
    ranked = rerank(prefs, songs, {}, k=4, recently_played={1})
    novelty = {s[0]["id"]: next(x for x in s[2] if x.name == "novelty").value for s in ranked}
    assert novelty[1] == 0.0 and novelty[2] == 1.0


def test_validate_drops_uncited_and_foreign_citations():
    allowed = {"sig:energy", "song:1"}
    kept, dropped = citations.validate([
        {"text": "Energy fits.", "evidence_ids": ["sig:energy"]},
        {"text": "No source.", "evidence_ids": []},
        {"text": "Wrong song.", "evidence_ids": ["song:2"]},
        {"text": "Mixed.", "evidence_ids": ["song:1", "artist:x"]},
    ], allowed)
    assert [c["text"] for c in kept] == ["Energy fits."]
    assert [d["reason"] for d in dropped][0] == "no citation"
    assert len(dropped) == 3


def test_grounded_claims_cite_only_offered_evidence():
    signals = [
        {"name": "query_match", "label": "Query match", "contribution": 0.3, "detail": "retrieval match 1.00"},
        {"name": "energy", "label": "Energy", "contribution": 0.2, "detail": "energy 0.30 vs 0.35"},
        {"name": "genre", "label": "Genre", "contribution": 0.0, "detail": "x"},
    ]
    docs = [{"doc_id": "song:1", "kind": "song", "text": "card"},
            {"doc_id": "artist:a", "kind": "artist", "text": "A is a band. More text."}]
    claims = citations.grounded_claims(signals, docs, low_confidence=True)
    allowed = citations.evidence_ids(signals, docs)
    kept, dropped = citations.validate(claims, allowed)
    assert len(kept) == 3 and not dropped
    assert claims[0]["text"].startswith("Closest match available")
    assert claims[2] == {"text": "A is a band.", "evidence_ids": ["artist:a"]}
