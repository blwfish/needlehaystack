import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from needlestack.search import _fts_query, _expand_query, search, MIN_SCORE
from needlestack.store import Store


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

def test_expand_query_returns_original_on_failure():
    with patch("needlestack.search.httpx.post", side_effect=Exception("timeout")):
        result = _expand_query("caboose")
    assert result == ["caboose"]


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
    assert len(result) <= 13


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
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        results = search("caboose", populated_store, mock_embedder())
    if results:
        r = results[0]
        assert "id" in r
        assert "path" in r
        assert "caption" in r
        assert "score" in r


def test_search_fts_match_scores_high(populated_store):
    # "caboose" is in only one caption — that image should appear and score well
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        results = search("caboose", populated_store, mock_embedder())
    paths = [r["path"] for r in results]
    assert "/caboose.jpg" in paths


def test_search_min_score_filters_results(populated_store):
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        results = search("caboose", populated_store, mock_embedder())
    assert all(r["score"] >= MIN_SCORE for r in results)


def test_search_empty_index(tmp_path):
    from needlestack.store import Store
    store = Store(tmp_path / "empty.db")
    with patch("needlestack.search._expand_query", return_value=["caboose"]):
        results = search("caboose", store, mock_embedder())
    assert results == []
    store.close()


def test_search_results_ordered_by_score(populated_store):
    with patch("needlestack.search._expand_query", return_value=["caboose", "waycar"]):
        results = search("caboose", populated_store, mock_embedder())
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


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


def test_single_fts_result_score_not_inflated(tmp_path):
    """M6: single FTS result must not be inflated to score 1.0."""
    from needlestack.store import Store
    s = Store(tmp_path / "fts.db")
    vec = fake_embedding(1)
    s.upsert("/only.jpg", "h1", "a steam locomotive on the mainline", vec, b"t")

    embedder = MagicMock()
    embedder.embed_text.return_value = vec

    results = search("locomotive", s, embedder, preexpanded_terms=["locomotive", "steam engine"])
    if results:
        # Score with neutral FTS (0.5) + CLIP (0.5) = 0.6*0.5 + 0.4*0.5 = 0.5, not 0.8
        assert results[0]["score"] < 0.75
    s.close()


# --- _expand_query logging ---

def test_expand_query_logs_warning_on_failure(caplog):
    import logging
    with patch("needlestack.search.httpx.post", side_effect=Exception("conn refused")):
        with caplog.at_level(logging.WARNING, logger="needlestack.search"):
            result = _expand_query("caboose")
    assert result == ["caboose"]
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
