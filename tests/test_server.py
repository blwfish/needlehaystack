import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import needlestack.server as srv
from needlestack.server import app
from needlestack_core import taxonomy
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


# --- setup.html domain options: generated from taxonomy.DOMAINS, not hand-typed ---

def test_setup_html_options_generated_from_all_domains(tmp_path):
    """Regression test for the missing-motorsports bug: the <select> is now
    generated from taxonomy.DOMAINS server-side, so it can't silently omit a
    domain the way the old hardcoded HTML list did."""
    from needlestack.cli import UI_PATH
    from fastapi.testclient import TestClient
    srv._index_state = srv._IndexState()
    srv.init(store=None, embedder=None, ui_path=UI_PATH, db_path=tmp_path / "test.db",
             ollama_url=OLLAMA_URL, ollama_model=DEFAULT_MODEL, setup_mode=True)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/setup")
        assert resp.status_code == 200
        for name in taxonomy.DOMAINS:
            assert f'value="{name}"' in resp.text
        assert "<!-- DOMAIN_OPTIONS -->" not in resp.text  # marker replaced, not leaked
    finally:
        client.close()
        _reset_server_globals()


def test_root_setup_redirect_also_gets_domain_options(tmp_path):
    """The `/` root route's own setup.html redirect must go through the same
    rendering path as GET /setup — not a second, unrendered read of the raw file."""
    from needlestack.cli import UI_PATH
    from fastapi.testclient import TestClient
    srv._index_state = srv._IndexState()
    srv.init(store=None, embedder=None, ui_path=UI_PATH, db_path=tmp_path / "test.db",
             ollama_url=OLLAMA_URL, ollama_model=DEFAULT_MODEL, setup_mode=True)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/")
        assert 'value="motorsports"' in resp.text
    finally:
        client.close()
        _reset_server_globals()


# --- sync-status (new/stale caption reporting) ---

# --- _all_domains / _resolve_domains ---

def test_root_returns_setup_page_when_store_empty_and_indexing_not_done(tmp_path):
    """root() must redirect to setup.html when store is present but empty and done=False."""
    from needlestack.store import Store
    from fastapi.testclient import TestClient
    (tmp_path / "setup.html").write_text("<html>setup</html>")
    (tmp_path / "index.html").write_text("<html>index</html>")
    store = Store(tmp_path / "index.db")  # empty, count() == 0
    srv._index_state = srv._IndexState()  # done=False by default
    srv.init(store=store, embedder=MagicMock(), ui_path=tmp_path,
             db_path=tmp_path / "index.db",
             ollama_url=OLLAMA_URL, ollama_model=DEFAULT_MODEL, setup_mode=False)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "setup" in resp.text.lower()
    finally:
        store.close()
        _reset_server_globals()


def test_all_domains_unknown_domain_falls_back_to_railroad(tmp_path, caplog):
    """An unrecognized domain name in the index logs a warning and falls back to railroad."""
    import logging
    from needlestack.store import Store
    store = Store(tmp_path / "test.db")
    store.add_root("/photos", "unknown_domain_xyz")
    srv._store = store
    try:
        with caplog.at_level(logging.WARNING, logger="needlestack.server"):
            result = srv._all_domains()
        assert result[0].name == "railroad"
        assert any("unknown_domain_xyz" in r.message for r in caplog.records)
    finally:
        store.close()
        srv._store = None


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
    with patch("needlestack.search.expand_query_with_truncation",
               return_value=(["caboose", "waycar"], False)):
        resp = client.post("/expand", json={"query": "caboose"})
    assert resp.status_code == 200
    assert resp.json()["terms"] == ["caboose", "waycar"]
    assert resp.json()["truncated"] is False


def test_expand_endpoint_reports_truncated_flag(client_with_store):
    from unittest.mock import patch
    client, store = client_with_store
    store.add_root("/photos", "railroad")
    with patch("needlestack.search.expand_query_with_truncation",
               return_value=(["caboose"] * 13, True)):
        resp = client.post("/expand", json={"query": "caboose"})
    assert resp.json()["truncated"] is True


def test_expand_endpoint_all_domains_passes_all_domain_objects(client_with_store):
    from unittest.mock import patch
    client, store = client_with_store
    store.add_root("/photos/rr", "railroad")
    store.add_root("/photos/birds", "birds")
    with patch("needlestack.search.expand_query_with_truncation",
               return_value=(["hawk"], False)) as mock_expand:
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


def test_sync_status_store_present_zero_roots(client_with_store):
    """Store exists but nothing has been registered as an indexed root yet — a real
    reachable state (index.db created but add_root never called), distinct from
    both the None-store case and the happy path."""
    client, store = client_with_store
    resp = client.get("/api/sync-status")
    assert resp.json() == {"new": 0, "removed": 0, "roots": []}


# --- resolve_image_path: deleted-file-with-existing-row (distinct from missing-row 404) ---

def test_resolve_image_path_404_when_backing_file_deleted(client_with_store, tmp_path):
    client, store = client_with_store
    img = tmp_path / "gone.jpg"
    img.write_bytes(b"x")
    store.upsert(str(img), "h", "a caption", np.zeros(512, dtype=np.float32), b"t")
    row = store.get_by_ids([1])
    assert row  # sanity: the DB row exists
    img.unlink()  # backing file removed after indexing, before the request
    resp = client.post(f"/open/{row[0]['id']}")
    assert resp.status_code == 404


# --- browse_folder: cancel vs error branches (macOS non-zero exit == user cancelled) ---

def test_browse_folder_nonzero_exit_is_cancelled_not_error(monkeypatch):
    import sys as _sys
    monkeypatch.setattr(_sys, "platform", "darwin")
    fake_result = MagicMock(stdout="", returncode=1, stderr="User canceled.")
    monkeypatch.setattr(srv.subprocess, "run", lambda *a, **k: fake_result)
    import asyncio
    result = asyncio.run(srv.browse_folder())
    assert result == {"cancelled": True}


def test_browse_folder_zero_exit_empty_path_is_error(monkeypatch):
    import sys as _sys
    monkeypatch.setattr(_sys, "platform", "darwin")
    fake_result = MagicMock(stdout="", returncode=0, stderr="")
    monkeypatch.setattr(srv.subprocess, "run", lambda *a, **k: fake_result)
    import asyncio
    result = asyncio.run(srv.browse_folder())
    assert "error" in result


def test_browse_folder_returns_path_on_success(monkeypatch):
    import sys as _sys
    monkeypatch.setattr(_sys, "platform", "darwin")
    fake_result = MagicMock(stdout="/Users/me/Photos\n", returncode=0, stderr="")
    monkeypatch.setattr(srv.subprocess, "run", lambda *a, **k: fake_result)
    import asyncio
    result = asyncio.run(srv.browse_folder())
    assert result == {"path": "/Users/me/Photos"}


def test_browse_folder_unsupported_platform(monkeypatch):
    import sys as _sys
    monkeypatch.setattr(_sys, "platform", "linux")
    import asyncio
    result = asyncio.run(srv.browse_folder())
    assert "error" in result


# --- /api/reindex-all ---

def _wait_for_index_done_or_error(timeout=5.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with srv._index_lock:
            if srv._index_state.done or srv._index_state.error:
                return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for background indexing to finish")


def test_reindex_all_no_roots(client_with_store):
    client, store = client_with_store
    resp = client.post("/api/reindex-all")
    assert resp.json() == {"status": "no_roots"}


def test_reindex_all_returns_503_when_store_none(client_no_store):
    resp = client_no_store.post("/api/reindex-all")
    assert resp.status_code == 503


def test_reindex_all_already_running(client_with_store):
    client, store = client_with_store
    store.add_root("/photos", "railroad")
    srv._index_state.running = True
    try:
        resp = client.post("/api/reindex-all")
        assert resp.json() == {"status": "already_running"}
    finally:
        srv._index_state.running = False


def test_reindex_all_success_uses_own_connection_and_invalidates_cache(
    client_with_store, tmp_path, monkeypatch
):
    """The writer must be a Store separate from the live `_store` (connection
    safety fix), and on success `_store`'s embedding cache must be invalidated so a
    subsequent search re-reads what the writer just committed."""
    client, store = client_with_store
    store.add_root(str(tmp_path), "railroad")

    calls = []

    def fake_loop(roots, s, url, model, embedder):
        calls.append((roots, s))

    monkeypatch.setattr(srv, "_run_indexing_loop", fake_loop)
    monkeypatch.setattr(srv, "Embedder", MagicMock)
    store._embedding_cache = (["sentinel"], ["sentinel"], np.zeros((1, 512)))

    resp = client.post("/api/reindex-all")
    assert resp.json() == {"status": "started"}
    _wait_for_index_done_or_error()

    assert srv._index_state.done is True
    assert srv._index_state.error == ""
    assert len(calls) == 1
    roots_arg, writer_store_arg = calls[0]
    assert roots_arg[0]["path"] == str(tmp_path)
    assert writer_store_arg is not store  # separate connection, not the live _store
    assert store._embedding_cache is None  # invalidated after the writer's commit
    assert store.get_config("last_indexed_model") == DEFAULT_MODEL
    writer_store_arg.close()


def test_reindex_all_failure_sets_error_and_closes_writer(
    client_with_store, tmp_path, monkeypatch
):
    client, store = client_with_store
    store.add_root(str(tmp_path), "railroad")

    def failing_loop(roots, s, url, model, embedder):
        raise RuntimeError("model not found")

    monkeypatch.setattr(srv, "_run_indexing_loop", failing_loop)
    monkeypatch.setattr(srv, "Embedder", MagicMock)

    resp = client.post("/api/reindex-all")
    assert resp.json() == {"status": "started"}
    _wait_for_index_done_or_error()

    assert srv._index_state.done is False
    assert "model not found" in srv._index_state.error


# --- start_indexing / reindex_all parity: both funnel through _run_indexing_loop ---

def test_start_indexing_and_reindex_all_both_use_shared_loop(
    client_with_store, tmp_path, monkeypatch
):
    """Regression guard for the duplicated-implementation finding: both endpoints
    must call the one shared _run_indexing_loop rather than each inlining their own
    copy that can silently diverge."""
    client, store = client_with_store
    calls = []

    def fake_loop(roots, s, url, model, embedder):
        calls.append("shared_loop")

    monkeypatch.setattr(srv, "_run_indexing_loop", fake_loop)
    monkeypatch.setattr(srv, "Embedder", MagicMock)

    # start_indexing path (setup wizard)
    resp = client.post("/api/setup/start", json={"folder": str(tmp_path), "domain": "railroad"})
    assert resp.json()["status"] == "started"
    _wait_for_index_done_or_error()
    assert calls == ["shared_loop"]

    # reindex_all path
    srv._index_state = srv._IndexState()
    store.add_root(str(tmp_path), "railroad")
    resp = client.post("/api/reindex-all")
    assert resp.json()["status"] == "started"
    _wait_for_index_done_or_error()
    assert calls == ["shared_loop", "shared_loop"]
