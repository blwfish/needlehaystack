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

MAX_EXPANSION_TERMS = 13


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


def _expand_query_raw(query: str, ollama_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL,
                      domain: Domain | None = None,
                      domains: list[Domain] | None = None) -> list[str]:
    """Uncapped expansion union — shared by _expand_query and
    expand_query_with_truncation so the cap/truncation logic has one source of truth
    (see _cap_terms) instead of being duplicated at both call sites."""
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
    except (httpx.HTTPError, KeyError, ValueError) as e:
        # HTTPError: network/connect/status failure. KeyError: "response" missing
        # from an otherwise-valid JSON body. ValueError (json.JSONDecodeError is a
        # subclass): malformed JSON. Narrowed from a bare `except Exception` so a
        # genuinely unexpected error (e.g. a bug in this function) isn't silently
        # swallowed into the same "Ollama must be down" degraded path.
        _log.warning("Query expansion failed, using bare query + taxonomy: %s", e)
        terms = []

    # Union: original first, then local synonyms, then LLM terms; dedup case-insensitively.
    seen = {query.lower()}
    unique = [query]
    for t in [*local, *terms]:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t.lower())
    return unique


def _cap_terms(unique: list[str]) -> tuple[list[str], bool]:
    truncated = len(unique) > MAX_EXPANSION_TERMS
    if truncated:
        _log.info("Expansion terms capped at %d (had %d)", MAX_EXPANSION_TERMS, len(unique))
    return unique[:MAX_EXPANSION_TERMS], truncated


def _expand_query(query: str, ollama_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL,
                  domain: Domain | None = None,
                  domains: list[Domain] | None = None) -> list[str]:
    """search() calls this, not expand_query_with_truncation -- the truncation
    flag is intentionally not propagated through search()'s return value (a list
    of result rows, with no natural place for a query-level flag without an
    invasive API change across its callers). It's still visible via _cap_terms'
    log line above. A caller that genuinely needs the flag (server.py's /expand
    endpoint) uses expand_query_with_truncation directly instead."""
    unique = _expand_query_raw(query, ollama_url=ollama_url, model=model,
                               domain=domain, domains=domains)
    capped, _truncated = _cap_terms(unique)
    return capped


def expand_query_with_truncation(
    query: str, ollama_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL,
    domain: Domain | None = None, domains: list[Domain] | None = None,
) -> tuple[list[str], bool]:
    """Like _expand_query, but also reports whether the term list was capped — for
    API consumers (server.py's /expand) that need to surface this to the UI instead
    of silently discarding the distinction between 'exactly N terms' and 'overflow'."""
    unique = _expand_query_raw(query, ollama_url=ollama_url, model=model,
                               domain=domain, domains=domains)
    return _cap_terms(unique)


def _normalize_clip(raw: np.ndarray) -> np.ndarray:
    """Clip raw CLIP cosine similarities to [0, 1] so MIN_SCORE reflects genuine
    match quality rather than a per-query, corpus-relative rank.

    Tied case (all scores equal, including a single-image index) is the one
    exception: there's no discriminating signal to rank by, so it maps to 1.0 —
    a lone candidate clears MIN_SCORE (CLIP_WEIGHT × 1.0 = 0.40 > 0.38) instead
    of being arbitrarily excluded for lack of a runner-up to compare against.

    Previously this used full min-max normalization ((raw - min) / (max - min))
    for every query, not just the tied case — which meant the single
    best-scoring candidate was ALWAYS rescaled to exactly 1.0, on every search,
    regardless of how weak its actual cosine similarity was. Combined with
    CLIP_WEIGHT × 1.0 = 0.40 > MIN_SCORE = 0.38, that meant the best-of-a-bad-lot
    image always cleared the "no good match" floor — for any query, including
    ones with zero genuinely relevant results.
    """
    mn, mx = raw.min(), raw.max()
    if mx > mn:
        return np.clip(raw, 0.0, 1.0)
    return np.full_like(raw, 1.0)


def _normalize_fts_ranks(ranks: np.ndarray) -> np.ndarray:
    """Min-max normalize BM25 ranks (already negated so higher = better) to
    [0, 1], with a tied case (single result or all tied) mapping to 1.0.

    Deliberately NOT unified with _normalize_clip despite the superficial
    resemblance (both min-max normalize, both special-case ties to 1.0):
    FTS5's MATCH already pre-filters fts_rows to rows that genuinely matched
    the text query, so "best rank among only genuine matches" is a meaningful
    relative signal. CLIP scores, by contrast, are computed against EVERY
    indexed embedding regardless of relevance -- relative top-1 rank there was
    exactly the bug _normalize_clip's docstring describes fixing. Same-looking
    math, different correctness properties; kept as two named functions rather
    than one shared one so a future change to either doesn't silently change
    the other's behavior.
    """
    shifted = ranks - ranks.min()
    mx = shifted.max()
    if mx > 0:
        return 1.0 - (shifted / mx)
    return np.full_like(ranks, 1.0)


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
        norm = _normalize_fts_ranks(ranks)
        for (image_id, _path, _rank), n in zip(fts_rows, norm.tolist()):
            fts_scores[image_id] = n

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
    missing = len(top_ids) - len(results)
    if missing:
        # An id scored by CLIP/FTS but absent from get_by_ids' result (a row
        # deleted between scoring and lookup) silently shrank the result count
        # with no signal why -- log it so a user-visible "fewer results than
        # expected" has a trail to follow.
        _log.warning("%d scored result(s) not found in store.get_by_ids lookup", missing)
    return results
