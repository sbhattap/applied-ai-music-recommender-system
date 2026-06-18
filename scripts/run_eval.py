"""Evaluate retrieval quality, end-to-end recommendations, citations and latency.

  uv run python scripts/run_eval.py                 # retrieval ablation + end-to-end
  uv run python scripts/run_eval.py --judge         # + LLM faithfulness judge (needs key)
  uv run python scripts/run_eval.py --llm-search    # + legacy LLM catalog search (needs key)
  uv run python scripts/run_eval.py --gate eval/baseline.json   # fail on regressions

Writes eval/results.md and eval/results.json. Relevance labels live in
eval/queries.json (attribute-based silver labels; see its ``_about``).
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from src.recommender import config  # noqa: E402
from src.recommender.core.io import load_songs  # noqa: E402
from src.recommender.llm import agent, client  # noqa: E402
from src.recommender.llm.types import Plan  # noqa: E402
from src.recommender.rag import get_retriever  # noqa: E402
from src.recommender.tools import semantic_search  # noqa: E402


EVAL_DIR = ROOT / "eval"
NEUTRAL_PROFILE = {"name": "Neutral", "genre": None, "mood": None, "energy": 0.5,
                   "likes_acoustic": False, "target_valence": 0.5, "prefer_popular": True}

# Metrics the regression gate checks, with how much they may drop.
GATED = {"hybrid.ndcg@10": 0.03, "hybrid.p@5": 0.05, "e2e.p@5": 0.05, "e2e.citation_valid_rate": 0.05}


# ── relevance ────────────────────────────────────────────────────────────────

def is_relevant(song: Dict[str, Any], rel: Dict[str, Any]) -> bool:
    if "genres" in rel and song["genre"] not in rel["genres"]:
        return False
    if "moods" in rel:
        tags = {song["mood"], *song.get("mood_tags", "").split("|")}
        if not tags & set(rel["moods"]):
            return False
    bounds = {
        "energy": song["energy"], "valence": song["valence"],
        "danceability": song["danceability"], "acoustic": song["acousticness"],
        "instrumental": song.get("instrumentalness", 0.0),
    }
    for name, value in bounds.items():
        if f"{name}_min" in rel and value < rel[f"{name}_min"]:
            return False
        if f"{name}_max" in rel and value > rel[f"{name}_max"]:
            return False
    return True


def precision_at(ranked: Sequence[int], relevant: set, k: int) -> float:
    top = list(ranked)[:k]
    return sum(i in relevant for i in top) / k if top else 0.0


def recall_at(ranked: Sequence[int], relevant: set, k: int) -> float:
    return sum(i in relevant for i in list(ranked)[:k]) / len(relevant) if relevant else 0.0


def ndcg_at(ranked: Sequence[int], relevant: set, k: int) -> float:
    dcg = sum(1 / math.log2(r + 2) for r, i in enumerate(list(ranked)[:k]) if i in relevant)
    ideal = sum(1 / math.log2(r + 2) for r in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def mrr(ranked: Sequence[int], relevant: set) -> float:
    for r, i in enumerate(ranked, 1):
        if i in relevant:
            return 1 / r
    return 0.0


def _mean(xs: List[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1)


# ── retrieval ablation ───────────────────────────────────────────────────────

def retrieval_rankings(songs, query: str) -> Dict[str, List[int]]:
    hits = get_retriever(songs).search(query)
    by_dense = sorted(hits, key=lambda h: h.dense_rank)
    by_bm25 = sorted(hits, key=lambda h: (h.bm25_rank is None, h.bm25_rank or 0, h.song_id))
    return {
        "bm25": [h.song_id for h in by_bm25],
        "dense": [h.song_id for h in by_dense],
        "hybrid": [h.song_id for h in hits],
    }


def eval_retrieval(songs, queries, *, llm_search: bool) -> Dict[str, Dict[str, float]]:
    systems: Dict[str, Dict[str, List[float]]] = {}
    for q in queries:
        relevant = {s["id"] for s in songs if is_relevant(s, q["relevant"])}
        started = time.monotonic()
        rankings = retrieval_rankings(songs, q["query"])
        retrieval_ms = (time.monotonic() - started) * 1000
        if llm_search:
            started = time.monotonic()
            plan = Plan(intent=q["query"], chosen_tool="semantic_search", expected_count_min=5,
                        expected_count_max=12, semantic_query=q["query"])
            picked = [s["id"] for s in semantic_search.run_llm(plan, songs)]
            rankings["llm (unordered)"] = picked
            systems.setdefault("llm (unordered)", {}).setdefault("ms", []).append((time.monotonic() - started) * 1000)
        for name, ranked in rankings.items():
            m = systems.setdefault(name, {})
            m.setdefault("p@5", []).append(precision_at(ranked, relevant, 5))
            m.setdefault("ndcg@10", []).append(ndcg_at(ranked, relevant, 10))
            m.setdefault("recall@20", []).append(recall_at(ranked, relevant, 20))
            m.setdefault("mrr", []).append(mrr(ranked, relevant))
            if name != "llm (unordered)":
                m.setdefault("ms", []).append(retrieval_ms)
    return {
        name: {**{k: _mean(v) for k, v in m.items() if k != "ms"}, "p50_ms": _pct(m["ms"], 0.5)}
        for name, m in systems.items()
    }


# ── end-to-end ───────────────────────────────────────────────────────────────

class ClaimVerdict(BaseModel):
    index: int
    supported: bool


class JudgeOut(BaseModel):
    verdicts: List[ClaimVerdict] = Field(default_factory=list)


JUDGE_SYSTEM = """You check whether each claim is supported by the evidence it cites.
A claim is supported only if the cited evidence states or directly implies it. Claims that
add facts not present in the evidence are NOT supported. Return one verdict per claim index."""


def judge(result) -> List[bool]:
    lines, n = [], 0
    for rec in result.recommendations:
        texts = {f"sig:{s['name']}": f"{s['label']}: {s['detail']}" for s in rec.signals}
        texts.update({d["doc_id"]: d["text"][:400] for d in rec.evidence})
        for claim in rec.claims:
            cited = "\n".join(f"    [{i}] {texts.get(i, '(missing)')}" for i in claim["evidence_ids"])
            lines.append(f"Claim {n}: {claim['text']}\n  Evidence:\n{cited}")
            n += 1
    if not n:
        return []
    out = client.call_messages_parsed(
        system=JUDGE_SYSTEM, user="\n\n".join(lines), label="judge", schema=JudgeOut,
        max_tokens=8000, effort="low", model=config.FAST_MODEL,
    )
    return [v.supported for v in out.verdicts]


def _count_llm_calls() -> Callable[[], Dict[str, int]]:
    counts: Dict[str, int] = {}
    original = client._generate

    def counting(**kwargs):
        counts[kwargs["label"]] = counts.get(kwargs["label"], 0) + 1
        return original(**kwargs)

    client._generate = counting
    return lambda: dict(counts)


def eval_end_to_end(songs, queries, *, use_judge: bool) -> Dict[str, Any]:
    llm = client.is_configured()
    read_counts = _count_llm_calls()
    per_query, stage_ms = [], {}
    kept = dropped = llm_recs = recs = 0
    judged: List[bool] = []
    for q in queries:
        before = sum(read_counts().values())
        result = agent.run(q["query"], NEUTRAL_PROFILE, songs, k=5)
        calls = sum(read_counts().values()) - before
        relevant = {s["id"] for s in songs if is_relevant(s, q["relevant"])}
        ranked = [r.song["id"] for r in result.recommendations]
        cites = next((e for e in result.trace_events if e["kind"] == "citations"), None)
        if cites:
            kept += cites["kept"]
            dropped += cites["dropped"]
        fallback = {i for e in result.trace_events if e["kind"] == "explainer_fallback"
                    for i in e.get("song_ids", ranked)}
        llm_recs += sum(i not in fallback for i in ranked) if llm else 0
        recs += len(ranked)
        if use_judge and llm and result.narratives_from_llm:
            judged += judge(result)
        for stage, ms in result.timings_ms.items():
            stage_ms.setdefault(stage, []).append(ms)
        per_query.append({
            "id": q["id"], "query": q["query"], "tool": result.plan.chosen_tool,
            "p@5": round(precision_at(ranked, relevant, 5), 3), "n_relevant": len(relevant),
            "confidence": result.verdict.confidence, "llm_calls": calls,
            "total_ms": result.timings_ms.get("total", 0),
        })

    total_claims = kept + dropped
    return {
        "llm_configured": llm,
        "p@5": _mean([r["p@5"] for r in per_query]),
        "llm_calls_per_query": _mean([r["llm_calls"] for r in per_query]),
        "citation_valid_rate": round(kept / total_claims, 4) if total_claims else (1.0 if not llm else 0.0),
        "llm_explained_rate": round(llm_recs / recs, 4) if recs and llm else 0.0,
        "faithfulness": round(sum(judged) / len(judged), 4) if judged else None,
        "latency_ms": {s: {"p50": _pct(v, 0.5), "p95": _pct(v, 0.95)} for s, v in stage_ms.items()},
        "per_query": per_query,
    }


# ── report ───────────────────────────────────────────────────────────────────

def to_markdown(report: Dict[str, Any]) -> str:
    r, e = report["retrieval"], report["end_to_end"]
    lines = [
        "# Evaluation results", "",
        f"{report['n_queries']} queries · catalog {report['n_songs']} songs · "
        f"embedder `{report['embedder']}` · LLM {'on' if e['llm_configured'] else 'off (offline fallbacks)'}",
        "", "## Retrieval ablation", "",
        "| System | P@5 | nDCG@10 | Recall@20 | MRR | p50 latency |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in r.items():
        lines.append(f"| {name} | {m['p@5']:.3f} | {m['ndcg@10']:.3f} | {m['recall@20']:.3f} | "
                     f"{m['mrr']:.3f} | {m['p50_ms']:.1f} ms |")
    faith = f"{e['faithfulness']:.3f}" if e["faithfulness"] is not None else "n/a (run with --judge)"
    lines += [
        "", "## End-to-end (agent, k=5, neutral profile)", "",
        f"- P@5: **{e['p@5']:.3f}**",
        f"- LLM calls per query: **{e['llm_calls_per_query']:.2f}**",
        f"- Citation validity (claims kept by the validator): **{e['citation_valid_rate']:.3f}**",
        f"- Recommendations explained by the LLM (vs grounded fallback): **{e['llm_explained_rate']:.3f}**",
        f"- Faithfulness (LLM judge): **{faith}**",
        "", "| Stage | p50 ms | p95 ms |", "|---|---|---|",
    ]
    for stage, v in e["latency_ms"].items():
        lines.append(f"| {stage} | {v['p50']:.0f} | {v['p95']:.0f} |")
    lines += ["", "<details><summary>Per query</summary>", "",
              "| id | tool | P@5 | relevant in catalog | LLM calls | total ms |", "|---|---|---|---|---|---|"]
    for q in e["per_query"]:
        lines.append(f"| {q['id']} | {q['tool']} | {q['p@5']:.2f} | {q['n_relevant']} | "
                     f"{q['llm_calls']} | {q['total_ms']} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def flatten(report: Dict[str, Any]) -> Dict[str, float]:
    flat = {f"{name}.{k}": v for name, m in report["retrieval"].items() for k, v in m.items()}
    flat.update({f"e2e.{k}": v for k, v in report["end_to_end"].items() if isinstance(v, (int, float))})
    return flat


def gate(report: Dict[str, Any], baseline_path: Path) -> List[str]:
    baseline = flatten(json.loads(baseline_path.read_text()))
    current = flatten(report)
    return [
        f"{metric}: {current.get(metric, 0):.3f} < baseline {baseline[metric]:.3f} - {tol}"
        for metric, tol in GATED.items()
        if metric in baseline and current.get(metric, 0) < baseline[metric] - tol
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default=str(EVAL_DIR / "queries.json"))
    parser.add_argument("--judge", action="store_true", help="LLM faithfulness judge")
    parser.add_argument("--llm-search", action="store_true", help="include legacy LLM search")
    parser.add_argument("--gate", type=Path, help="baseline results.json to compare against")
    parser.add_argument("--out", type=Path, default=EVAL_DIR)
    args = parser.parse_args(argv)

    songs = load_songs(str(config.SONGS_CSV))
    queries = json.loads(Path(args.queries).read_text())["queries"]
    get_retriever(songs).search("warm up")  # model load shouldn't count as query latency

    report = {
        "n_queries": len(queries),
        "n_songs": len(songs),
        "embedder": config.EMBEDDING_MODEL,
        "retrieval": eval_retrieval(songs, queries, llm_search=args.llm_search and client.is_configured()),
        "end_to_end": eval_end_to_end(songs, queries, use_judge=args.judge),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    (args.out / "results.md").write_text(to_markdown(report))
    print(to_markdown(report))

    if args.gate:
        failures = gate(report, args.gate)
        for f in failures:
            print(f"REGRESSION {f}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
