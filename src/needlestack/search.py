import httpx
import numpy as np

from .embedder import Embedder
from .store import Store

# Weight blend: captions carry the detail, CLIP catches visual similarity
FTS_WEIGHT = 0.6
CLIP_WEIGHT = 0.4

EXPAND_PROMPT = (
    "You are a search synonym expander for a photo archive. "
    "Given a search phrase, return ONLY direct synonyms and alternate names for the exact same thing. "
    "Do NOT include related or associated items — only other names for the identical subject. "
    "Examples: 'caboose' → 'cabin car, waycar, hack, crummy, van'; "
    "'tank car' → 'tanker, cistern car, pressure car'; "
    "'steam locomotive' → 'steam engine, steamer'. "
    "Return ONLY a comma-separated list, no explanation, no punctuation other than commas. "
    "If there are no meaningful synonyms, return only the original term."
    "\n\nSearch phrase: {query}"
)

MIN_SCORE = 0.38


def _expand_query(query: str, ollama_url: str = "http://localhost:11434", model: str = "llava:13b") -> list[str]:
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": EXPAND_PROMPT.format(query=query), "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
        seen = {query.lower()}
        unique = [query] + [t for t in terms if t not in seen and not seen.add(t)]
        return unique[:13]
    except Exception:
        return [query]


def _fts_query(terms: list[str]) -> str:
    # FTS5 OR query across all expanded terms
    escaped = [f'"{t}"' for t in terms]
    return " OR ".join(escaped)


def search(
    query: str,
    store: Store,
    embedder: Embedder,
    limit: int = 40,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llava:13b",
    preexpanded_terms: list[str] | None = None,
) -> list[dict]:
    terms = preexpanded_terms if preexpanded_terms else _expand_query(query, ollama_url=ollama_url, model=ollama_model)
    fts_q = _fts_query(terms)

    # CLIP similarity over all indexed embeddings
    query_vec = embedder.embed_text(query)
    ids, _paths, matrix = store.all_embeddings()

    clip_scores: dict[int, float] = {}
    if len(ids) > 0:
        raw = (matrix @ query_vec).astype(float)
        mn, mx = raw.min(), raw.max()
        norm = (raw - mn) / (mx - mn) if mx > mn else raw - mn
        clip_scores = dict(zip(ids, norm.tolist()))

    # FTS5 over captions using expanded query
    fts_rows = store.fts_search(fts_q, limit=limit * 3)
    fts_scores: dict[int, float] = {}
    if fts_rows:
        # rank is negative BM25: more negative = better match
        ranks = np.array([r[2] for r in fts_rows], dtype=float)
        ranks = ranks - ranks.min()
        mx = ranks.max()
        if mx > 0:
            ranks = ranks / mx
        # invert so 1.0 = best
        for (image_id, _path, _rank), norm in zip(fts_rows, (1.0 - ranks).tolist()):
            fts_scores[image_id] = norm

    # Merge: union of both result sets
    all_ids = set(clip_scores) | set(fts_scores)
    combined = {
        iid: CLIP_WEIGHT * clip_scores.get(iid, 0.0) + FTS_WEIGHT * fts_scores.get(iid, 0.0)
        for iid in all_ids
    }

    top_ids = sorted(
        (k for k, v in combined.items() if v >= MIN_SCORE),
        key=lambda k: combined[k], reverse=True
    )[:limit]
    rows = store.get_by_ids(top_ids)
    by_id = {r["id"]: r for r in rows}

    results = []
    for iid in top_ids:
        if iid in by_id:
            r = by_id[iid]
            r["score"] = round(combined[iid], 4)
            results.append(r)
    return results
