"""Hybrid retriever: dense + BM25, fused with reciprocal rank fusion.

Every hit keeps its provenance — both ranks, the matched BM25 terms, and the top
evidence documents behind its dense score — so the UI can show *how* a song was found
and the explainer can cite *what* it was found from.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import config
from .bm25 import BM25
from .documents import Document, build_documents
from .embedder import get_embedder, get_vector_cache


RRF_K = 60
CARD_WEIGHT = 0.7       # dense score = 0.7 * card similarity + 0.3 * best context doc
EVIDENCE_PER_SONG = 3


@dataclass
class Evidence:
    doc_id: str
    kind: str
    source: str
    title: str
    text: str
    url: str
    score: float


@dataclass
class SongHit:
    song_id: int
    dense: float
    bm25: float
    dense_rank: int
    bm25_rank: Optional[int]          # None when no query term matched
    rrf: float
    query_match: float                # rrf min-max scaled over this result set: 1 best, 0 weakest
    found_by: str                     # "dense" | "bm25" | "both"
    matched_terms: List[str] = field(default_factory=list)
    taste_sim: Optional[float] = None
    evidence: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Retriever:
    def __init__(self, songs: Sequence[Dict[str, Any]], docs: List[Document]) -> None:
        self.songs = list(songs)
        self.song_ids = [s["id"] for s in self.songs]
        self._pos = {sid: i for i, sid in enumerate(self.song_ids)}
        self.docs = docs
        self._doc_vecs = get_vector_cache().get_many([d.text for d in docs])

        self._card_idx = {d.song_ids[0]: i for i, d in enumerate(docs) if d.kind == "song"}
        self._context_idx: Dict[int, List[int]] = {sid: [] for sid in self.song_ids}
        for i, d in enumerate(docs):
            if d.kind != "song":
                for sid in d.song_ids:
                    if sid in self._context_idx:
                        self._context_idx[sid].append(i)

        # BM25 over a per-song text: card + short track-level evidence (not long bios,
        # which would let an artist's biography outvote the track itself).
        per_song = []
        for sid in self.song_ids:
            parts = [docs[self._card_idx[sid]].text]
            parts += [docs[i].text for i in self._context_idx[sid] if docs[i].kind in ("tags", "hist")]
            per_song.append(" ".join(parts))
        self._bm25 = BM25(per_song)

    def song_vector(self, song_id: int) -> np.ndarray:
        return self._doc_vecs[self._card_idx[song_id]]

    def search(
        self,
        query: str,
        candidates: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        top_n: Optional[int] = None,
        taste_vector: Optional[Sequence[float]] = None,
    ) -> List[SongHit]:
        """Rank ``candidates`` (default: all songs) for ``query``."""
        ids = [s["id"] for s in candidates] if candidates is not None else self.song_ids
        ids = [i for i in ids if i in self._pos]
        if not ids:
            return []

        q = get_embedder().embed_query(query)
        doc_sims = self._doc_vecs @ q

        dense = np.empty(len(ids), dtype=np.float32)
        evidence: List[List[Evidence]] = []
        for j, sid in enumerate(ids):
            card = self._card_idx[sid]
            ctx = self._context_idx[sid]
            best_ctx = max((doc_sims[i] for i in ctx), default=doc_sims[card])
            dense[j] = CARD_WEIGHT * doc_sims[card] + (1 - CARD_WEIGHT) * best_ctx
            ranked = sorted([card] + ctx, key=lambda i: -doc_sims[i])[:EVIDENCE_PER_SONG]
            if card not in ranked:          # the card is always citable
                ranked = [card] + ranked[:-1]
            evidence.append([self._evidence(i, float(doc_sims[i])) for i in ranked])

        all_bm25 = self._bm25.scores(query)
        bm25 = np.array([all_bm25[self._pos[sid]] for sid in ids], dtype=np.float32)

        dense_order = np.argsort(-dense)
        dense_rank = np.empty(len(ids), dtype=int)
        dense_rank[dense_order] = np.arange(1, len(ids) + 1)
        bm25_order = [j for j in np.argsort(-bm25) if bm25[j] > 0]
        bm25_rank: Dict[int, int] = {j: r for r, j in enumerate(bm25_order, 1)}

        rrf = np.array([
            1 / (RRF_K + dense_rank[j]) + (1 / (RRF_K + bm25_rank[j]) if j in bm25_rank else 0.0)
            for j in range(len(ids))
        ])
        # RRF is flat near the top (rank 1 vs 12 differ by ~15%), so scale within the
        # result set; otherwise query relevance barely moves the downstream rerank.
        lo, hi = float(rrf.min()), float(rrf.max())
        span = hi - lo

        taste = None
        if taste_vector is not None:
            t = np.asarray(taste_vector, dtype=np.float32)
            t = t / (np.linalg.norm(t) or 1.0)
            taste = [float(self.song_vector(sid) @ t) for sid in ids]

        cutoff = top_n or len(ids)
        hits = []
        for j in np.argsort(-rrf)[:cutoff]:
            in_dense = dense_rank[j] <= cutoff
            in_bm25 = j in bm25_rank and bm25_rank[j] <= cutoff
            hits.append(SongHit(
                song_id=ids[j],
                dense=round(float(dense[j]), 4),
                bm25=round(float(bm25[j]), 4),
                dense_rank=int(dense_rank[j]),
                bm25_rank=bm25_rank.get(j),
                rrf=round(float(rrf[j]), 6),
                query_match=round((float(rrf[j]) - lo) / span, 4) if span > 0 else 1.0,
                found_by="both" if in_dense and in_bm25 else "bm25" if in_bm25 and not in_dense else "dense",
                matched_terms=self._bm25.matched_terms(query, self._pos[ids[j]]),
                taste_sim=round(taste[j], 4) if taste is not None else None,
                evidence=evidence[j],
            ))
        return hits

    def _evidence(self, i: int, score: float) -> Evidence:
        d = self.docs[i]
        return Evidence(d.doc_id, d.kind, d.source, d.title, d.text, d.url, round(score, 4))


_retrievers: Dict[str, Retriever] = {}


def _key(songs: Sequence[Dict[str, Any]], extra: Sequence[Document]) -> str:
    h = hashlib.sha1()
    h.update(get_embedder().name.encode())
    for s in songs:
        h.update(f"{s['id']}|{s['title']}|{s['artist']}|{s['genre']}|{s['mood']}|{s.get('mood_tags', '')}".encode())
    for d in extra:
        h.update(d.doc_id.encode() + d.text.encode())
    return h.hexdigest()


def get_retriever(
    songs: Sequence[Dict[str, Any]],
    extra_docs: Sequence[Document] = (),
) -> Retriever:
    """Memoized per (songs, extra docs, embedder). Vectors come from the disk cache, so
    a new subset only costs building BM25 over it."""
    key = _key(songs, extra_docs)
    if key not in _retrievers:
        if len(_retrievers) > 64:
            _retrievers.clear()
        docs = build_documents(songs, config.RAG_DATA_DIR) + list(extra_docs)
        _retrievers[key] = Retriever(songs, docs)
    return _retrievers[key]
