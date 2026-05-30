import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import needlestack.server as srv
from needlestack.server import app


@pytest.fixture
def client_no_store(tmp_path):
    """Server in setup mode with no index."""
    (tmp_path / "setup.html").write_text("<html>setup</html>")
    (tmp_path / "index.html").write_text("<html>index</html>")
    srv._index_state = srv._IndexState()
    srv.init(store=None, embedder=None, ui_path=tmp_path,
             db_path=tmp_path / "test.db",
             ollama_url="http://localhost:11434",
             ollama_model="llava:13b",
             setup_mode=True)
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


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
             ollama_url="http://localhost:11434",
             ollama_model="llava:13b",
             setup_mode=False)
    from fastapi.testclient import TestClient
    yield TestClient(app, raise_server_exceptions=False), store
    store.close()


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
    from needlestack.constants import caption_version
    store.upsert(
        str(root / "a.jpg"), "h", "a fresh caption", np.zeros(512, dtype=np.float32),
        b"t", caption_version=caption_version(srv._ollama_model),
    )
    resp = client.get("/api/sync-status")
    assert resp.json()["stale"] == 0
