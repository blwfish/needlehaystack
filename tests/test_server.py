import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import needlestack.server as srv
from needlestack.server import app
from needlestack_core.constants import DEFAULT_MODEL, OLLAMA_URL


def _reset_server_globals():
    """Restore module globals so server tests don't leak state into one another."""
    srv.init(store=None, embedder=None, ui_path=None, db_path=None,
             ollama_url=OLLAMA_URL, ollama_model=DEFAULT_MODEL, setup_mode=False)
    srv._index_state = srv._IndexState()


@pytest.fixture
def client_no_store(tmp_path):
    """Server in setup mode with no index."""
    (tmp_path / "setup.html").write_text("<html>setup</html>")
    (tmp_path / "index.html").write_text("<html>index</html>")
    srv._index_state = srv._IndexState()
    srv.init(store=None, embedder=None, ui_path=tmp_path,
             db_path=tmp_path / "test.db",
             ollama_url=OLLAMA_URL,
             ollama_model=DEFAULT_MODEL,
             setup_mode=True)
    from fastapi.testclient import TestClient
    yield TestClient(app, raise_server_exceptions=False)
    _reset_server_globals()


@pytest.fixture
def client_with_store(tmp_path):
    """Server with a real (empty) store."""
    from needlestack.store import Store
    (tmp_path / "setup.html").write_text("<html>setup</html>")
    (tmp_path / "index.html").write_text("<html>index</html>")
    store = Store(tmp_path / "index.db")
    embedder = MagicMock()
    embedder.embed_text.return_value = np.zeros(512, dtype=np.float32)
    srv._index_state = srv._IndexState()
    srv._index_state.done = True  # skip setup redirect
    srv.init(store=store, embedder=embedder, ui_path=tmp_path,
             db_path=tmp_path / "index.db",
             ollama_url=OLLAMA_URL,
             ollama_model=DEFAULT_MODEL,
             setup_mode=False)
    from fastapi.testclient import TestClient
    yield TestClient(app, raise_server_exceptions=False), store
    store.close()
    _reset_server_globals()


# --- H1: None guards ---

def test_search_returns_503_when_store_none(client_no_store):
    resp = client_no_store.post("/search", json={"query": "locomotive"})
    assert resp.status_code == 503


def test_thumbnail_returns_503_when_store_none(client_no_store):
    resp = client_no_store.get("/thumbnail/1")
    assert resp.status_code == 503


def test_open_returns_503_when_store_none(client_no_store):
    resp = client_no_store.post("/open/1")
    assert resp.status_code == 503


def test_reveal_returns_503_when_store_none(client_no_store):
    resp = client_no_store.post("/reveal/1")
    assert resp.status_code == 503


# --- H2: folder validation ---

def test_start_indexing_rejects_missing_folder(client_no_store):
    resp = client_no_store.post("/api/setup/start", json={"folder": "/no/such/path"})
    assert resp.status_code == 400


def test_start_indexing_rejects_file_not_directory(tmp_path, client_no_store):
    f = tmp_path / "file.jpg"
    f.write_bytes(b"x")
    resp = client_no_store.post("/api/setup/start", json={"folder": str(f)})
    assert resp.status_code == 400


def test_start_indexing_rejects_empty_folder_string(client_no_store):
    resp = client_no_store.post("/api/setup/start", json={"folder": ""})
    assert resp.status_code == 400


# --- progress endpoint ---

def test_progress_endpoint_returns_expected_fields(client_no_store):
    resp = client_no_store.get("/api/setup/progress")
    assert resp.status_code == 200
    data = resp.json()
    for field in ("running", "done", "error", "total", "indexed", "skipped", "failed", "current"):
        assert field in data


# --- search with store ---

def test_search_returns_empty_list_for_blank_query(client_with_store):
    client, _ = client_with_store
    resp = client.post("/search", json={"query": "   "})
    assert resp.status_code == 200
    assert resp.json() == []


def test_thumbnail_returns_404_for_missing_id(client_with_store):
    client, _ = client_with_store
    resp = client.get("/thumbnail/99999")
    assert resp.status_code == 404


def test_open_returns_404_for_missing_id(client_with_store):
    client, _ = client_with_store
    resp = client.post("/open/99999")
    assert resp.status_code == 404


# --- sync-status (new/stale caption reporting) ---

# --- _all_domains / _resolve_domains ---

def test_all_domains_returns_distinct_in_order(tmp_path):
    from needlestack.store import Store
    store = Store(tmp_path / "test.db")
    store.add_root("/photos/rr", "railroad")
    store.add_root("/photos/birds", "birds")
    srv._store = store
    try:
        result = srv._all_domains()
        assert [d.name for d in result] == ["railroad", "birds"]
    finally:
        store.close()
        srv._store = None


def test_all_domains_deduplicates_same_domain(tmp_path):
    from needlestack.store import Store
    store = Store(tmp_path / "test.db")
    store.add_root("/photos/a", "railroad")
    store.add_root("/photos/b", "railroad")
    srv._store = store
    try:
        result = srv._all_domains()
        assert len(result) == 1
        assert result[0].name == "railroad"
    finally:
        store.close()
        srv._store = None


def test_resolve_domains_false_returns_primary(tmp_path):
    from needlestack.store import Store
    store = Store(tmp_path / "test.db")
    store.add_root("/photos", "birds")
    srv._store = store
    try:
        req = srv.SearchRequest(query="test", all_domains=False)
        result = srv._resolve_domains(req)
        assert len(result) == 1
        assert result[0].name == "birds"
    finally:
        store.close()
        srv._store = None


def test_resolve_domains_true_returns_all(tmp_path):
    from needlestack.store import Store
    store = Store(tmp_path / "test.db")
    store.add_root("/photos/rr", "railroad")
    store.add_root("/photos/birds", "birds")
    srv._store = store
    try:
        req = srv.SearchRequest(query="test", all_domains=True)
        result = srv._resolve_domains(req)
        assert {d.name for d in result} == {"railroad", "birds"}
    finally:
        store.close()
        srv._store = None


def test_expand_endpoint_returns_terms(client_with_store):
    from unittest.mock import patch
    client, store = client_with_store
    store.add_root("/photos", "railroad")
    with patch("needlestack.search._expand_query", return_value=["caboose", "waycar"]):
        resp = client.post("/expand", json={"query": "caboose"})
    assert resp.status_code == 200
    assert resp.json()["terms"] == ["caboose", "waycar"]


def test_expand_endpoint_all_domains_passes_all_domain_objects(client_with_store):
    from unittest.mock import patch
    client, store = client_with_store
    store.add_root("/photos/rr", "railroad")
    store.add_root("/photos/birds", "birds")
    with patch("needlestack.search._expand_query", return_value=["hawk"]) as mock_expand:
        resp = client.post("/expand", json={"query": "hawk", "all_domains": True})
    assert resp.status_code == 200
    domains_arg = mock_expand.call_args.kwargs.get("domains", [])
    assert {d.name for d in domains_arg} == {"railroad", "birds"}


# --- sync-status ---

def test_sync_status_none_store(client_no_store):
    resp = client_no_store.get("/api/sync-status")
    assert resp.status_code == 200
    assert resp.json()["new"] == 0


def test_sync_status_reports_stale_captions(client_with_store, tmp_path):
    client, store = client_with_store
    root = tmp_path / "photos"
    root.mkdir()
    store.set_config("indexed_root", str(root))
    # A row whose caption_version predates the current model → counts as stale.
    store.upsert(
        str(root / "a.jpg"), "h", "an old caption", np.zeros(512, dtype=np.float32),
        b"t", caption_version="ancient:v0",
    )
    resp = client.get("/api/sync-status")
    assert resp.status_code == 200
    d = resp.json()
    assert d["stale"] == 1
    assert d["new"] == 0   # no actual file on disk → nothing "new"


def test_sync_status_no_stale_when_current(client_with_store, tmp_path):
    client, store = client_with_store
    root = tmp_path / "photos"
    root.mkdir()
    store.set_config("indexed_root", str(root))
    from needlestack_core.constants import caption_version
    store.upsert(
        str(root / "a.jpg"), "h", "a fresh caption", np.zeros(512, dtype=np.float32),
        b"t", caption_version=caption_version(srv._ollama_model),
    )
    resp = client.get("/api/sync-status")
    assert resp.json()["stale"] == 0
