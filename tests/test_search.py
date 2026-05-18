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
