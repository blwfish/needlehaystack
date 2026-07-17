import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from needlestack.search import _fts_query, _expand_query, search, MIN_SCORE, _make_expand_prompt
from needlestack_core import taxonomy
from needlestack.store import Store


# --- _make_expand_prompt ---

def test_make_expand_prompt_single_domain_includes_vocab():
    prompt = _make_expand_prompt([taxonomy.RAILROAD])
    assert taxonomy.RAILROAD.subject_types_prompt() in prompt


def test_make_expand_prompt_multi_domain_includes_vocab_for_all():
    prompt = _make_expand_prompt([taxonomy.RAILROAD, taxonomy.BIRDS])
    assert taxonomy.RAILROAD.subject_types_prompt() in prompt
    assert taxonomy.BIRDS.subject_types_prompt() in prompt


# --- _fts_query ---

def test_fts_query_single_term():
    assert _fts_query(["caboose"]) == '"caboose"'


def test_fts_query_multiple_terms():
    result = _fts_query(["caboose", "waycar", "hack"])
    assert result == '"caboose" OR "waycar" OR "hack"'


def test_fts_query_phrases():
    result = _fts_query(["tank car", "tanker"])
    assert '"tank car"' in result
    assert '"tanker"' in result


# --- _expand_query ---

def test_expand_query_returns_original_plus_taxonomy_on_failure():
    """On LLM failure, the original term plus deterministic taxonomy synonyms still
    come back — known railroad terms expand even with Ollama down."""
    with patch("needlestack.search.httpx.post", side_effect=Exception("timeout")):
        result = _expand_query("caboose")
    assert result[0] == "caboose"
    assert "waycar" in result and "crummy" in result


def test_expand_query_unknown_term_on_failure_is_bare():
    """A term not in the taxonomy and with the LLM down expands to just itself."""
    with patch("needlestack.search.httpx.post", side_effect=Exception("timeout")):
        result = _expand_query("automobile")
    assert result == ["automobile"]


def test_expand_query_unions_taxonomy_on_success():
    """Taxonomy synonyms are unioned with LLM output even when the LLM succeeds."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "some-llm-synonym"}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        result = _expand_query("caboose")
    assert "waycar" in result          # from taxonomy
    assert "some-llm-synonym" in result  # from LLM


def test_expand_query_deduplicates():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "caboose, waycar, caboose, hack"}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        result = _expand_query("caboose")
    assert result.count("caboose") == 1


def test_expand_query_prepends_original():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "waycar, hack, crummy"}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        result = _expand_query("caboose")
    assert result[0] == "caboose"


def test_expand_query_caps_length():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": ", ".join(f"term{i}" for i in range(50))}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        result = _expand_query("caboose")
    assert len(result) == 13  # cap is exactly 13, not "at most 13"


def test_expand_query_strips_whitespace():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "  waycar ,  hack  , crummy "}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        result = _expand_query("caboose")
    assert "waycar" in result
    assert "hack" in result


# --- search() ---

def fake_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def populated_store(tmp_path):
    from needlestack.store import Store
    s = Store(tmp_path / "test.db")
    s.upsert("/caboose.jpg", "h1", "a yellow caboose at the end of a freight train", fake_embedding(1), b"t")
    s.upsert("/loco.jpg",    "h2", "a steam locomotive pulling a long consist",       fake_embedding(2), b"t")
    s.upsert("/tank.jpg",    "h3", "a tank car loaded with petroleum products",       fake_embedding(3), b"t")
    yield s
    s.close()


def mock_embedder(query_seed: int = 99):
    embedder = MagicMock()
    embedder.embed_text.return_value = fake_embedding(query_seed)
    return embedder


def test_search_returns_list(populated_store):
    with patch("needlestack.search._expand_query", return_value=["caboose", "waycar"]):
        results = search("caboose", populated_store, mock_embedder())
    assert isinstance(results, list)


def test_search_result_has_required_fields(populated_store):
    # mock_embedder(1) matches the caboose doc embedding → deterministic, non-empty.
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        results = search("caboose", populated_store, mock_embedder(1))
    assert results, "expected at least one result for a matching caption"
    r = results[0]
    assert "id" in r
    assert "path" in r
    assert "caption" in r
    assert "score" in r


def test_search_fts_match_scores_high(populated_store):
    # "caboose" is in only one caption — that image should appear and score well
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        results = search("caboose", populated_store, mock_embedder(1))
    assert results
    paths = [r["path"] for r in results]
    assert "/caboose.jpg" in paths


def test_min_score_threshold_is_load_bearing(populated_store):
    """Not 'all returned scores >= MIN_SCORE' (which is tautological — the function
    filters on exactly that). Instead prove the threshold actually excludes: a high
    threshold returns strictly fewer results than a permissive one."""
    emb = mock_embedder(1)
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        with patch("needlestack.search.MIN_SCORE", 0.0):
            permissive = search("caboose", populated_store, emb)
        with patch("needlestack.search.MIN_SCORE", 0.99):
            strict = search("caboose", populated_store, emb)
    assert len(permissive) > len(strict)   # threshold is doing real work
    assert len(strict) < 3                  # not everything clears a 0.99 bar


def test_search_empty_index(tmp_path):
    from needlestack.store import Store
    store = Store(tmp_path / "empty.db")
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        results = search("caboose", store, mock_embedder())
    assert results == []
    store.close()


def test_search_results_ordered_by_score(populated_store):
    # "caboose" FTS-matches caboose.jpg; "locomotive" FTS-matches loco.jpg.
    # mock_embedder(1) aligns with the caboose embedding — caboose.jpg scores highest
    # via both FTS and CLIP, loco.jpg scores via FTS alone. Both clear MIN_SCORE.
    with patch("needlestack.search._expand_query", return_value=["caboose", "locomotive"]):
        results = search("caboose", populated_store, mock_embedder(1))
    assert len(results) >= 2, "need ≥2 results to verify ordering"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# --- MIN_SCORE boundary: >= not > ---

def test_min_score_boundary_is_gte_not_gt(tmp_path):
    """Score exactly equal to MIN_SCORE must be included (>= not >)."""
    from needlestack.store import Store
    s = Store(tmp_path / "b.db")
    vec = fake_embedding(1)
    s.upsert("/only.jpg", "h1", "locomotive", vec, b"t")

    embedder = MagicMock()
    embedder.embed_text.return_value = vec  # perfect match → CLIP normalizes to 1.0

    # Single doc: CLIP tie → 1.0, FTS tie → 1.0. Combined = CLIP_WEIGHT*1 + FTS_WEIGHT*1 = 1.0.
    # At MIN_SCORE=1.0 the result is exactly on the boundary — must be included.
    with patch("needlestack.search.MIN_SCORE", 1.0):
        at_boundary = search("locomotive", s, embedder, preexpanded_terms=["locomotive"])
    # At MIN_SCORE=1.001 the result is just above — must be excluded.
    with patch("needlestack.search.MIN_SCORE", 1.001):
        above_boundary = search("locomotive", s, embedder, preexpanded_terms=["locomotive"])

    assert len(at_boundary) == 1    # >= means at-boundary is included
    assert len(above_boundary) == 0  # strictly above is excluded
    s.close()


# --- _fts_query empty list ---

def test_fts_query_empty_list_returns_empty_string():
    assert _fts_query([]) == ""


def test_search_with_empty_preexpanded_terms(tmp_path):
    """preexpanded_terms=[] must not raise; FTS produces empty query → handled gracefully."""
    from needlestack.store import Store
    s = Store(tmp_path / "e.db")
    s.upsert("/a.jpg", "h1", "a caboose", fake_embedding(1), b"t")
    results = search("anything", s, mock_embedder(), preexpanded_terms=[])
    assert isinstance(results, list)  # no exception
    s.close()


# --- _expand_query(domains=[]) empty list is same fallback as None ---

def test_expand_query_domains_empty_list_falls_back_to_railroad():
    """Empty list is falsy → same RAILROAD fallback as domains=None. Pinned explicitly."""
    with patch("needlestack.search.httpx.post", side_effect=Exception("timeout")):
        result = _expand_query("caboose", domains=[])
    assert "waycar" in result  # RAILROAD synonyms included via fallback


def test_expand_query_domains_none_falls_back_to_railroad():
    with patch("needlestack.search.httpx.post", side_effect=Exception("timeout")):
        result = _expand_query("caboose", domains=None)
    assert "waycar" in result


# --- single-image edge cases (H4 / M6 normalization fixes) ---

def test_single_image_index_returns_result(tmp_path):
    """H4: single-image index must not zero out CLIP scores and drop the only result."""
    from needlestack.store import Store
    s = Store(tmp_path / "single.db")
    vec = fake_embedding(1)
    s.upsert("/only.jpg", "h1", "a steam locomotive", vec, b"t")

    embedder = MagicMock()
    embedder.embed_text.return_value = vec  # perfect match → mn == mx

    results = search("locomotive", s, embedder, preexpanded_terms=["locomotive"])
    assert len(results) == 1
    s.close()


def test_single_doc_returns_and_scores_correctly(tmp_path):
    """M6: a lone document that matches on both CLIP and FTS must appear and score 1.0."""
    from needlestack.store import Store
    s = Store(tmp_path / "fts.db")
    vec = fake_embedding(1)
    s.upsert("/only.jpg", "h1", "a steam locomotive on the mainline", vec, b"t")

    embedder = MagicMock()
    embedder.embed_text.return_value = vec  # perfect CLIP match

    results = search("locomotive", s, embedder, preexpanded_terms=["locomotive", "steam engine"])
    assert results, "single matching doc should still return a result"
    # Single doc: CLIP tie-break → 1.0, FTS tie-break → 1.0. Combined = 0.4*1 + 0.6*1 = 1.0
    assert results[0]["score"] == 1.0
    s.close()


# --- _expand_query logging ---

def test_expand_query_with_truncation_reports_true_when_capped():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": ", ".join(f"term{i}" for i in range(50))}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        from needlestack.search import expand_query_with_truncation
        terms, truncated = expand_query_with_truncation("caboose")
    assert len(terms) == 13
    assert truncated is True


def test_expand_query_with_truncation_reports_false_when_not_capped():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "waycar, hack"}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        from needlestack.search import expand_query_with_truncation
        terms, truncated = expand_query_with_truncation("caboose")
    assert truncated is False


def test_expand_query_multi_domain_unions_synonyms():
    """BIRDS synonyms for 'raptor' appear when BIRDS domain is included."""
    with patch("needlestack.search.httpx.post", side_effect=Exception("timeout")):
        railroad_only = _expand_query("raptor", domains=[taxonomy.RAILROAD])
        both_domains  = _expand_query("raptor", domains=[taxonomy.RAILROAD, taxonomy.BIRDS])
    assert "hawk" not in railroad_only   # RAILROAD has no raptor synonyms
    assert "hawk" in both_domains        # union includes BIRDS synonyms


# --- _expand_query logging ---

def test_expand_query_logs_warning_on_failure(caplog):
    import logging
    with patch("needlestack.search.httpx.post", side_effect=Exception("conn refused")):
        with caplog.at_level(logging.WARNING, logger="needlestack.search"):
            result = _expand_query("caboose")
    assert result[0] == "caboose"
    assert any("expansion failed" in r.message.lower() for r in caplog.records)


def test_expand_query_logs_warning_on_done_reason_length(caplog):
    import logging
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "waycar, hack", "done_reason": "length"}
    with patch("needlestack.search.httpx.post", return_value=mock_resp):
        with caplog.at_level(logging.WARNING, logger="needlestack.search"):
            _expand_query("caboose")
    assert any("truncated" in r.message.lower() or "token limit" in r.message.lower()
               for r in caplog.records)


# --- _fts_query: quote escaping (a broken escaper degrades to silent zero results,
# since store.fts_search swallows sqlite3.OperationalError on malformed FTS5 syntax) ---

def test_fts_query_escapes_internal_quotes():
    result = _fts_query(['36" gauge'])
    assert result == '"36"" gauge"'


def test_search_handles_term_with_literal_quote(tmp_path):
    """Integration-level: a broken escaper produces malformed FTS5 syntax, which
    store.fts_search silently converts to an empty result list — this test would go
    from green to a false '(no results)' rather than an obvious crash."""
    from needlestack.store import Store
    s = Store(tmp_path / "q.db")
    s.upsert("/g.jpg", "h1", 'a 36" gauge model railroad layout', fake_embedding(1), b"t")
    with patch("needlestack.search._expand_query", return_value=['36" gauge']):
        results = search('36" gauge', s, mock_embedder(1))
    assert results
    s.close()


# --- score-merge weights: CLIP-only and FTS-only arms, and FTS_WEIGHT/CLIP_WEIGHT values ---
# (Critical: dict.get(id, 0.0) defaults on both sides of the merge were completely
# unpinned — a mutant default of 1.0, or a FTS_WEIGHT/CLIP_WEIGHT swap, survived the
# full suite. This test pins both at once via two docs, each missing from one side.)

def test_score_merge_pins_weights_and_missing_side_defaults(tmp_path):
    from needlestack.store import Store
    from needlestack.search import FTS_WEIGHT, CLIP_WEIGHT
    s = Store(tmp_path / "merge.db")

    query_vec = fake_embedding(1)
    # clip_only.jpg: caption never matches "caboose" (FTS-absent) but its embedding
    # is the query vector itself — the best-aligned of the two, so it normalizes to 1.0.
    s.upsert("/clip_only.jpg", "h1", "nothing relevant in this caption", query_vec, b"t")
    # fts_only.jpg: caption matches "caboose" (sole FTS hit -> tie-break norm 1.0), but
    # its embedding is set to NULL after insert, so it's excluded from clip_scores
    # entirely (all_embeddings() only selects WHERE embedding IS NOT NULL) — this is
    # the real-world "captioned but embedding failed" row doctor.py already tracks.
    off_axis = fake_embedding(2)
    s.upsert("/fts_only.jpg", "h2", "a caboose at the yard", off_axis, b"t")
    s.conn.execute("UPDATE images SET embedding = NULL WHERE path = '/fts_only.jpg'")
    s.conn.commit()
    s._embedding_cache = None

    embedder = MagicMock()
    embedder.embed_text.return_value = query_vec

    with patch("needlestack.search.MIN_SCORE", 0.0):  # don't filter anything out
        results = search("caboose", s, embedder, preexpanded_terms=["caboose"])

    by_path = {r["path"]: r["score"] for r in results}
    # clip_only: CLIP norm 1.0 (best-aligned, the only embedded doc) + FTS-absent default.
    assert by_path["/clip_only.jpg"] == round(CLIP_WEIGHT * 1.0 + FTS_WEIGHT * 0.0, 4)
    # fts_only: FTS norm 1.0 (sole match, tie-break) + CLIP-absent default (excluded row).
    assert by_path["/fts_only.jpg"] == round(CLIP_WEIGHT * 0.0 + FTS_WEIGHT * 1.0, 4)
    s.close()
