"""Spotify auth, API client and taste-profile tests. No network: urlopen is faked."""

import io
import json
import urllib.error

import pytest

from src.recommender.llm import agent
from src.recommender.spotify import SpotifyClient, Tokens, auth, build_profile
from src.recommender.spotify import labels
from src.recommender.spotify.taste import aggregate, track_key


@pytest.fixture(autouse=True)
def no_llm_labels(monkeypatch):
    """build_profile must not call the LLM or touch the real label cache in tests."""
    monkeypatch.setattr("src.recommender.spotify.taste.label_tracks", lambda tracks: {})


def _song(id_, title, artist, genre, mood, energy, **kw):
    base = {"id": id_, "title": title, "artist": artist, "genre": genre, "mood": mood,
            "mood_tags": mood, "energy": energy, "tempo_bpm": 120.0, "valence": 0.4,
            "danceability": 0.5, "acousticness": 0.1, "instrumentalness": 0.0,
            "popularity": 70, "release_decade": "", "liveness": 0.1, "duration_sec": 200,
            "explicit": 0}
    base.update(kw)
    return base


@pytest.fixture
def songs():
    return [
        _song(1, "Enter Sandman", "Metallica", "metal", "intense", 0.9),
        _song(2, "Numb", "Linkin Park", "metal", "intense", 0.85),
        _song(3, "Clair de Lune", "Claude Debussy", "classical", "relaxed", 0.1,
              acousticness=0.99, instrumentalness=0.99, valence=0.2),
        _song(4, "Blinding Lights", "The Weeknd", "pop", "happy", 0.8, valence=0.8),
    ]


def _t(name, artist):
    return {"id": name, "name": name, "artists": [{"id": artist, "name": artist}], "url": ""}


@pytest.fixture
def history():
    return {
        "user": {"id": "u1", "display_name": "Sam"},
        "recently_played": [_t("Enter Sandman", "Metallica"), _t("Enter Sandman", "Metallica"),
                            _t("Unlisted Riff", "Sleep Token")],
        "top_tracks": {"short_term": [_t("Numb", "Linkin Park")], "medium_term": [], "long_term": []},
        "top_artists": [{"id": "st", "name": "Sleep Token", "genres": ["progressive metal"]}],
        "saved_tracks": [_t("Numb (Live)", "Linkin Park")],
    }


def test_pkce_challenge_matches_rfc7636_vector():
    assert auth.challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == \
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_authorize_url_uses_loopback_ip_and_pkce(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.delenv("SPOTIFY_REDIRECT_URI", raising=False)
    url = auth.authorize_url("st8", "verifier")
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8501" in url
    assert "code_challenge_method=S256" in url and "state=st8" in url
    assert "user-read-recently-played" in url


def test_token_store_roundtrip(tmp_path):
    store = auth.TokenStore(tmp_path / "t.json")
    store.save(Tokens("a", "r", 9e12))
    assert store.valid().access_token == "a"
    store.clear()
    assert store.valid() is None


def test_track_key_strips_feat_and_remaster():
    assert track_key("Numb (Live)", "Linkin Park") == track_key("Numb", "Linkin Park")
    assert track_key("Song - Remastered 2011", "X") == track_key("Song", "X")


def test_aggregate_weights_plays_tops_and_saves(history):
    tracks = aggregate(history)
    sandman = tracks[track_key("Enter Sandman", "Metallica")]
    numb = tracks[track_key("Numb", "Linkin Park")]
    assert sandman.plays == 2 and sandman.weight == pytest.approx(2.0)
    assert numb.saved and numb.top_ranges == ["short_term"]
    assert tracks[track_key("Unlisted Riff", "Sleep Token")].genres == ["progressive metal"]


def test_build_profile_derives_prefs_and_evidence(history, songs):
    p = build_profile(history, songs)
    assert p["genre"] == "metal" and p["mood"] == "intense"
    assert p["energy"] > 0.6 and p["likes_acoustic"] is False
    assert p["recently_played"] == [1]
    assert len(p["taste_vector"]) > 0
    hist_ids = {d["doc_id"] for d in p["history_docs"] if d["kind"] == "hist"}
    assert hist_ids == {"hist:1", "hist:2"}
    sandman = next(d for d in p["history_docs"] if d["doc_id"] == "hist:1")
    assert "played 2 times" in sandman["text"]
    assert p["history_summary"]["matched_to_catalog"] == 2


def test_build_profile_uses_labels_for_unmatched_tracks(history, songs):
    pop = {"genre": "pop", "mood": "happy", "energy": 0.7, "valence": 0.9, "acousticness": 0.2}
    seen = []

    def labeler(tracks):
        seen.extend(t[1] for t in tracks)
        return {t[0]: pop for t in tracks}

    history["recently_played"] += [_t("Unlisted Riff", "Sleep Token")] * 10
    p = build_profile(history, songs, labeler=labeler)
    assert seen == ["Unlisted Riff"]               # only tracks missing from the catalog
    assert p["genre"] == "pop" and p["mood"] == "happy"
    assert p["history_summary"]["labelled"] == 1


def test_similar_docs_ignore_title_overlap_across_genres(history, songs):
    """A pop track titled like a classical piece must not yield a "similar to" doc."""
    pop = {"genre": "pop", "mood": "happy", "energy": 0.7, "valence": 0.9, "acousticness": 0.2}
    history["recently_played"].append(_t("Clair de Lune Remix", "DJ Pop"))
    p = build_profile(history, songs, labeler=lambda tracks: {t[0]: pop for t in tracks})
    similar = [d for d in p["history_docs"] if d["kind"] == "similar"]
    assert all("Clair de Lune" not in d["text"] or d["song_ids"] != [3] for d in similar)
    assert all('"' in d["text"] for d in similar)   # history title is quoted


def test_label_tracks_caches_and_skips_llm_when_cached(monkeypatch, tmp_path):
    calls = []

    def fake_call(**kw):
        calls.append(kw["user"])
        return labels.TrackLabels(labels=[labels.TrackLabel(
            i=0, genre="pop", mood="happy", energy=0.7, valence=0.8, acousticness=0.1)])

    monkeypatch.setattr(labels.client, "is_configured", lambda: True)
    monkeypatch.setattr(labels.client, "call_messages_parsed", fake_call)
    path = tmp_path / "labels.json"
    tracks = [(("blue moon", "zara larsson"), "Blue Moon", ["Zara Larsson"], [])]
    first = labels.label_tracks(tracks, cache_path=path)
    second = labels.label_tracks(tracks, cache_path=path)
    assert first == second and first[("blue moon", "zara larsson")]["genre"] == "pop"
    assert len(calls) == 1 and "Blue Moon — Zara Larsson" in calls[0]


def test_label_tracks_offline_returns_cache_only(monkeypatch, tmp_path):
    monkeypatch.setattr(labels.client, "is_configured", lambda: False)
    assert labels.label_tracks([(("a", "b"), "A", ["B"], [])], cache_path=tmp_path / "x.json") == {}


def test_agent_uses_spotify_profile_signals_and_history_evidence(monkeypatch, history, songs):
    monkeypatch.setattr("src.recommender.llm.agent.client.is_configured", lambda: False)
    p = build_profile(history, songs)
    result = agent.run("loud heavy riffs", p, songs, k=3)
    rec = result.recommendations[0]
    names = {s["name"] for s in rec.signals}
    assert {"query_match", "taste_match", "novelty"} <= names
    played = next(r for r in result.recommendations if r.song["id"] == 1) if any(
        r.song["id"] == 1 for r in result.recommendations) else None
    if played:
        assert next(s for s in played.signals if s["name"] == "novelty")["value"] == 0.0
    all_evidence = {e["doc_id"] for r in result.recommendations for e in r.evidence}
    assert all_evidence & {"hist:1", "hist:2"}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_refreshes_once_on_401(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(req.headers["Authorization"])
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 401, "expired", {}, io.BytesIO(b"{}"))
        return _Resp(json.dumps({"id": "u1", "display_name": "Sam"}).encode())

    monkeypatch.setattr("src.recommender.spotify.api.urllib.request.urlopen", fake_urlopen)
    client = SpotifyClient(Tokens("old", "r", 0), on_refresh=lambda t: Tokens("new", "r", 9e12))
    assert client.me() == {"id": "u1", "display_name": "Sam"}
    assert calls == ["Bearer old", "Bearer new"]


def test_client_surfaces_quota_exceeded(monkeypatch):
    def fake_urlopen(req, timeout=0):
        body = io.BytesIO(b'{"error": {"status": 429, "reason": "QUOTA_EXCEEDED"}}')
        raise urllib.error.HTTPError(req.full_url, 429, "quota", {}, body)

    monkeypatch.setattr("src.recommender.spotify.api.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(Exception, match="quota"):
        SpotifyClient(Tokens("a", "r", 9e12)).me()
