import logging

import httpx
import numpy as np

_log = logging.getLogger(__name__)

from needlestack_core import taxonomy
from needlestack_core.taxonomy import Domain
from needlestack_core.constants import DEFAULT_MODEL, OLLAMA_URL
from needlestack_core.embedder import Embedder
from .store import Store

# Weight blend: captions carry the detail, CLIP catches visual similarity
FTS_WEIGHT = 0.6
CLIP_WEIGHT = 0.4

MIN_SCORE = 0.38


def _make_expand_prompt(domains: list[Domain]) -> str:
    names = " and ".join(d.name for d in domains)
    all_vocab = "; ".join(d.subject_types_prompt() for d in domains)
    vocab = (f"The archive covers {names} photography and uses this vocabulary: "
             f"{all_vocab}.")
    return (
        "You are a search synonym expander for a photo archive. "
        "Given a search phrase, return ONLY direct synonyms and alternate names for the exact same thing. "
        "Do NOT include related or associated items — only other names for the identical subject. "
        f"{vocab} "
        "Return ONLY a comma-separated list, no explanation, no punctuation other than commas. "
        "If there are no meaningful synonyms, return only the original term."
        "\n\nSearch phrase: {query}"
    )


def _expand_query(query: str, ollama_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL,
                  domain: Domain | None = None,
                  domains: list[Domain] | None = None) -> list[str]:
    # Deterministic domain synonyms are always included, so known terms expand even
    # when the LLM is unavailable or flubs; the LLM widens coverage beyond the taxonomy.
    # `domains` takes precedence over `domain`; both are accepted for backward compat.
    all_domains = domains if domains else ([domain] if domain is not None else [taxonomy.RAILROAD])
    seen_local: set[str] = set()
    local: list[str] = []
    for d in all_domains:
        for syn in d.synonyms_for(query):
            if syn.lower() not in seen_local:
                seen_local.add(syn.lower())
                local.append(syn)
    prompt = _make_expand_prompt(all_domains)
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt.format(query=query), "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("done_reason") == "length":
            _log.warning("Query expansion truncated at token limit (model=%s)", model)
        raw = data["response"].strip()
        terms = [t.strip() for t in raw.split(",") if t.strip()]
    except Exception as e:
        _log.warning("Query expansion failed, using bare query + taxonomy: %s", e)
        terms = []

    # Union: original first, then local synonyms, then LLM terms; dedup case-insensitively.
    seen = {query.lower()}
    unique = [query]
    for t in [*local, *terms]:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t.lower())
    if len(unique) > 13:
        _log.debug("Expansion terms capped at 13 (had %d)", len(unique))
    return unique[:13]


def _normalize_clip(raw: np.ndarray) -> np.ndarray:
    """Min-max normalize CLIP cosine scores to [0, 1].

    Tied case (all equal, including single-image index) maps to 1.0 so a lone
    result clears MIN_SCORE (CLIP_WEIGHT × 1.0 = 0.40 > 0.38).
    """
    mn, mx = raw.min(), raw.max()
    if mx > mn:
        return (raw - mn) / (mx - mn)
    return np.full_like(raw, 1.0)


def _fts_query(terms: list[str]) -> str:
    # FTS5 OR query; embed each term as a phrase, doubling any internal " per FTS5 spec.
    escaped = ['"' + t.replace('"', '""') + '"' for t in terms]
    return " OR ".join(escaped)


def search(
    query: str,
    store: Store,
    embedder: Embedder,
    limit: int = 40,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = DEFAULT_MODEL,
    preexpanded_terms: list[str] | None = None,
    domain: Domain | None = None,
    domains: list[Domain] | None = None,
) -> list[dict]:
    terms = preexpanded_terms if preexpanded_terms is not None else _expand_query(
        query, ollama_url=ollama_url, model=ollama_model, domain=domain, domains=domains
    )
    fts_q = _fts_query(terms)

    # CLIP similarity over all indexed embeddings
    query_vec = embedder.embed_text(query)
    ids, _paths, matrix = store.all_embeddings()

    clip_scores: dict[int, float] = {}
    if len(ids) > 0:
        raw = (matrix @ query_vec).astype(float)
        norm = _normalize_clip(raw)
        clip_scores = dict(zip(ids, norm.tolist()))

    # FTS5 over captions using expanded query
    fts_rows = store.fts_search(fts_q, limit=limit * 3)
    fts_scores: dict[int, float] = {}
    if fts_rows:
        # rank is negative BM25: more negative = better match
        ranks = np.array([r[2] for r in fts_rows], dtype=float)
        ranks = ranks - ranks.min()
        mx = ranks.max()
        # When all ranks are equal (single result or all tied), assign 0 so the
        # inversion (1.0 − 0) = 1.0, letting a lone FTS match clear MIN_SCORE.
        if mx > 0:
            ranks = ranks / mx
        else:
            ranks = np.zeros_like(ranks)
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
