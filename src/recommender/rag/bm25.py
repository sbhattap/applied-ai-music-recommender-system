"""Small Okapi BM25. Hand-rolled so it can report *which* query terms matched each
document, which the UI shows as "matched on: lofi, study"."""

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


_STOPWORDS = frozenset("""
a an and are as at be but by for from i in into is it its me my no of on or so some
something songs song music track tracks that the this to up want with like play give get
""".split())

# Collapse spelling variants so "lo-fi" matches "lofi" etc.
_SYNONYMS = {
    "lo-fi": "lofi", "hiphop": "hip-hop", "rnb": "r&b", "kpop": "k-pop",
    "vocal": "vocals", "singing": "sung",
}


# Multi-word spellings of single catalog tokens, applied before splitting.
_PHRASES = (
    (re.compile(r"\bhip[\s-]?hop\b"), "hip-hop"),
    (re.compile(r"\blo[\s-]fi\b"), "lofi"),
    (re.compile(r"\bk[\s-]pop\b"), "k-pop"),
    (re.compile(r"\br\s?(?:and|n|&)\s?b\b"), "r&b"),
)


def tokenize(text: str) -> List[str]:
    # "no" is a stopword: catalog cards say "no vocals" vs "has singing", so the
    # signal lives in "vocals"; keeping "no" matched titles like "No Role Modelz".
    text = text.lower()
    for pattern, replacement in _PHRASES:
        text = pattern.sub(replacement, text)
    tokens = re.findall(r"[a-z0-9&]+(?:-[a-z0-9]+)*", text)
    return [_SYNONYMS.get(t, t) for t in tokens if t not in _STOPWORDS]


class BM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.docs) else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf: Dict[str, float] = {
            t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        q = set(tokenize(query))
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not self.docs:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_len, 1e-9))
        for term in q:
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def matched_terms(self, query: str, index: int) -> List[str]:
        doc = self.docs[index]
        seen: List[str] = []
        for t in tokenize(query):
            if t in doc and t not in seen:
                seen.append(t)
        return seen
