"""Tests for the CLI entry points via click.testing.CliRunner.

Each command's heavy dependencies (Captioner, Store, Embedder, uvicorn) are mocked so
tests are fast and hermetic. Port-logic tests mock socket to control which ports appear
in use.
"""

import socket
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from click.testing import CliRunner

from needlestack.cli import main
from needlestack.store import Store


def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------

def _mock_captioner_ok():
    m = MagicMock()
    m.check.return_value = (True, "ok")
    return m


def _mock_captioner_fail(msg="Ollama not reachable"):
    m = MagicMock()
    m.check.return_value = (False, msg)
    return m


def _index_patches(captioner=None, indexed=5, skipped=2, failed=0):
    """Context that mocks all index-command dependencies."""
    cap = captioner or _mock_captioner_ok()
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    return (
        patch("needlestack_core.captioner.Captioner", return_value=cap),
        patch("needlestack.store.Store", return_value=mock_store),
        patch("needlestack_core.embedder.Embedder", return_value=MagicMock()),
        patch("needlestack.indexer.index_directory", return_value=(indexed, skipped, failed)),
    )


def test_index_model_and_preset_mutual_exclusion(tmp_path):
    r = runner()
    result = r.invoke(main, ["index", str(tmp_path), "--model", "foo", "--preset", "fast"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_index_captioner_check_fails_exits_1(tmp_path):
    r = runner()
    with patch("needlestack_core.captioner.Captioner", return_value=_mock_captioner_fail()):
        result = r.invoke(main, ["index", str(tmp_path), "--db", str(tmp_path / "i.db")])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_index_happy_path_reports_counts(tmp_path):
    r = runner()
    patches = _index_patches(indexed=7, skipped=3, failed=1)
    with patches[0], patches[1], patches[2], patches[3]:
        result = r.invoke(main, ["index", str(tmp_path), "--db", str(tmp_path / "i.db")])
    assert result.exit_code == 0
    assert "indexed: 7" in result.output
    assert "skipped: 3" in result.output
    assert "failed: 1" in result.output


def test_index_persists_domain_to_config(tmp_path):
    r = runner()
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    with (
        patch("needlestack_core.captioner.Captioner", return_value=_mock_captioner_ok()),
        patch("needlestack.store.Store", return_value=mock_store),
        patch("needlestack_core.embedder.Embedder", return_value=MagicMock()),
        patch("needlestack.indexer.index_directory", return_value=(0, 0, 0)),
    ):
        result = r.invoke(
            main, ["index", str(tmp_path), "--db", str(tmp_path / "i.db"), "--domain", "naval"]
        )
    assert result.exit_code == 0
    mock_store.add_root.assert_called_once_with(str(tmp_path.resolve()), "naval")


def test_index_records_last_indexed_model(tmp_path):
    r = runner()
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    with (
        patch("needlestack_core.captioner.Captioner", return_value=_mock_captioner_ok()),
        patch("needlestack.store.Store", return_value=mock_store),
        patch("needlestack_core.embedder.Embedder", return_value=MagicMock()),
        patch("needlestack.indexer.index_directory", return_value=(0, 0, 0)),
    ):
        result = r.invoke(
            main, ["index", str(tmp_path), "--db", str(tmp_path / "i.db"), "--preset", "quality"]
        )
    assert result.exit_code == 0
    from needlestack_core.constants import MODEL_PRESETS
    mock_store.set_config.assert_any_call("last_indexed_model", MODEL_PRESETS["quality"])


def test_index_domain_naval_passes_naval_domain_to_captioner(tmp_path):
    r = runner()
    captured = []

    def captioner_factory(*args, **kwargs):
        captured.append(kwargs.get("domain"))
        return _mock_captioner_ok()

    mock_store = MagicMock()
    mock_store.count.return_value = 0
    with (
        patch("needlestack_core.captioner.Captioner", side_effect=captioner_factory),
        patch("needlestack.store.Store", return_value=mock_store),
        patch("needlestack_core.embedder.Embedder", return_value=MagicMock()),
        patch("needlestack.indexer.index_directory", return_value=(0, 0, 0)),
    ):
        r.invoke(
            main, ["index", str(tmp_path), "--db", str(tmp_path / "i.db"), "--domain", "naval"]
        )
    assert captured and captured[0].name == "naval"


def test_index_default_domain_is_railroad(tmp_path):
    r = runner()
    captured = []

    def captioner_factory(*args, **kwargs):
        captured.append(kwargs.get("domain"))
        return _mock_captioner_ok()

    mock_store = MagicMock()
    mock_store.count.return_value = 0
    with (
        patch("needlestack_core.captioner.Captioner", side_effect=captioner_factory),
        patch("needlestack.store.Store", return_value=mock_store),
        patch("needlestack_core.embedder.Embedder", return_value=MagicMock()),
        patch("needlestack.indexer.index_directory", return_value=(0, 0, 0)),
    ):
        r.invoke(main, ["index", str(tmp_path), "--db", str(tmp_path / "i.db")])
    assert captured and captured[0].name == "railroad"


def test_index_preset_resolves_model(tmp_path):
    """--preset fast should resolve to the fast preset model, not the default."""
    from needlestack_core.constants import MODEL_PRESETS
    r = runner()
    captured_models = []

    def captioner_factory(*args, **kwargs):
        captured_models.append(kwargs.get("model"))
        return _mock_captioner_ok()

    mock_store = MagicMock()
    mock_store.count.return_value = 0
    with (
        patch("needlestack_core.captioner.Captioner", side_effect=captioner_factory),
        patch("needlestack.store.Store", return_value=mock_store),
        patch("needlestack_core.embedder.Embedder", return_value=MagicMock()),
        patch("needlestack.indexer.index_directory", return_value=(0, 0, 0)),
    ):
        r.invoke(main, ["index", str(tmp_path), "--db", str(tmp_path / "i.db"), "--preset", "fast"])
    assert captured_models and captured_models[0] == MODEL_PRESETS["fast"]


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------

def test_doctor_calls_run_and_prints(tmp_path):
    r = runner()
    with patch("needlestack.doctor.run", return_value="FAKE REPORT") as mock_run:
        result = r.invoke(main, ["doctor", "--db", str(tmp_path / "missing.db")])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    assert "FAKE REPORT" in result.output


def test_doctor_saves_output_to_file(tmp_path):
    r = runner()
    out_file = tmp_path / "report.txt"
    with patch("needlestack.doctor.run", return_value="DIAGNOSIS OUTPUT"):
        result = r.invoke(
            main, ["doctor", "--db", str(tmp_path / "missing.db"), "--out", str(out_file)]
        )
    assert result.exit_code == 0
    assert out_file.read_text() == "DIAGNOSIS OUTPUT"


# ---------------------------------------------------------------------------
# serve command — port logic
# ---------------------------------------------------------------------------

def _mock_socket_factory(in_use_ports: set):
    """Mock socket.socket where specified ports appear in use (connect_ex returns 0)."""
    def factory(*args, **kwargs):
        m = MagicMock()
        m.__enter__ = lambda self: m
        m.__exit__ = MagicMock(return_value=False)
        def connect_ex(addr):
            _, port = addr
            return 0 if port in in_use_ports else 1
        m.connect_ex = connect_ex
        return m
    return factory


def _serve_patches(tmp_path, in_use_ports=None, httpx_resp=None):
    """Context stack for serve tests. Creates a real empty DB so setup_mode=False."""
    db_path = tmp_path / "index.db"
    Store(db_path).close()   # create real schema
    sock_factory = _mock_socket_factory(in_use_ports or set())
    return db_path, (
        patch("socket.socket", side_effect=sock_factory),
        patch("needlestack_core.embedder.Embedder", return_value=MagicMock()),
        patch("needlestack.server.init"),
        patch("uvicorn.run"),
        patch("webbrowser.open"),
        patch("httpx.get", return_value=httpx_resp) if httpx_resp else patch("httpx.get", side_effect=Exception("no")),
    )


def test_serve_port_free_starts_uvicorn(tmp_path):
    db_path, patches = _serve_patches(tmp_path, in_use_ports=set())
    r = runner()
    with patches[0], patches[1], patches[2], patches[3] as mock_uvicorn, patches[4], patches[5]:
        result = r.invoke(main, ["serve", "--db", str(db_path), "--no-browser"])
    assert result.exit_code == 0
    mock_uvicorn.assert_called_once()


def test_serve_port_in_use_needlestack_running_opens_browser(tmp_path):
    """Port in use + it's already needlestack → open browser, no uvicorn."""
    mock_resp = MagicMock()
    mock_resp.text = "needlestack photo search"   # contains "needlestack"
    db_path, patches = _serve_patches(tmp_path, in_use_ports={8484}, httpx_resp=mock_resp)
    r = runner()
    with patches[0], patches[1], patches[2], patches[3] as mock_uvicorn, patches[4] as mock_browser, patches[5]:
        result = r.invoke(main, ["serve", "--db", str(db_path)])
    assert result.exit_code == 0
    mock_uvicorn.assert_not_called()
    mock_browser.assert_called_once()


def test_serve_port_in_use_different_process_finds_next_free(tmp_path):
    """Port in use + not needlestack → find next free port, start there."""
    db_path, patches = _serve_patches(tmp_path, in_use_ports={8484})
    r = runner()
    with patches[0], patches[1], patches[2], patches[3] as mock_uvicorn, patches[4], patches[5]:
        result = r.invoke(main, ["serve", "--db", str(db_path), "--no-browser"])
    assert result.exit_code == 0
    # uvicorn called with a port other than 8484
    assert mock_uvicorn.call_args.kwargs["port"] != 8484


def test_serve_no_free_port_exits_1(tmp_path):
    """All 20 candidate ports in use → exit 1."""
    in_use = set(range(8484, 8484 + 25))   # more than the 20-port scan range (port+1..port+20)
    db_path, patches = _serve_patches(tmp_path, in_use_ports=in_use)
    r = runner()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        result = r.invoke(main, ["serve", "--db", str(db_path), "--no-browser"])
    assert result.exit_code == 1


def test_serve_scan_width_upper_edge_exactly_20_busy_fails(tmp_path):
    """Base port + exactly the 20-candidate range (port+1..port+20) busy, port+21
    free. Pins the upper edge: a widened scan (e.g. to port+25) would reach the
    free port+21 and succeed instead of exiting 1."""
    in_use = {8484} | set(range(8485, 8505))   # base + port+1..port+20 (20 candidates)
    db_path, patches = _serve_patches(tmp_path, in_use_ports=in_use)
    r = runner()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        result = r.invoke(main, ["serve", "--db", str(db_path), "--no-browser"])
    assert result.exit_code == 1


def test_serve_scan_width_lower_edge_finds_port_plus_20(tmp_path):
    """Base port + port+1..port+19 busy, port+20 free. Pins the lower edge: a
    narrowed scan (e.g. only port+1..port+15) would never reach port+20 and would
    exit 1 instead of succeeding there."""
    in_use = {8484} | set(range(8485, 8504))   # base + port+1..port+19 (19 candidates)
    db_path, patches = _serve_patches(tmp_path, in_use_ports=in_use)
    r = runner()
    with patches[0], patches[1], patches[2], patches[3] as mock_uvicorn, patches[4], patches[5]:
        result = r.invoke(main, ["serve", "--db", str(db_path), "--no-browser"])
    assert result.exit_code == 0
    assert mock_uvicorn.call_args.kwargs["port"] == 8504  # port+20


# --- serve: --model/--preset mutual exclusion (same contract as `index`) ---

def test_serve_model_and_preset_mutual_exclusion(tmp_path):
    db_path = tmp_path / "index.db"  # need not exist — check runs before any I/O
    r = runner()
    result = r.invoke(main, ["serve", "--db", str(db_path), "--model", "x", "--preset", "fast"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


# --- serve: default model consults last_indexed_model (caption-model tracking) ---

def test_serve_defaults_to_last_indexed_model_when_no_model_or_preset(tmp_path):
    db_path = tmp_path / "index.db"
    s = Store(db_path)
    s.set_config("last_indexed_model", "qwen2.5vl:7b")
    s.close()
    sock_factory = _mock_socket_factory(set())
    r = runner()
    with patch("socket.socket", side_effect=sock_factory), \
         patch("needlestack_core.embedder.Embedder", return_value=MagicMock()), \
         patch("needlestack.server.init") as mock_init, \
         patch("uvicorn.run"), patch("webbrowser.open"), \
         patch("httpx.get", side_effect=Exception("no")):
        result = r.invoke(main, ["serve", "--db", str(db_path), "--no-browser"])
    assert result.exit_code == 0
    assert mock_init.call_args.kwargs["ollama_model"] == "qwen2.5vl:7b"


def test_serve_falls_back_to_default_model_when_no_last_indexed_model(tmp_path):
    db_path = tmp_path / "index.db"
    Store(db_path).close()
    sock_factory = _mock_socket_factory(set())
    r = runner()
    with patch("socket.socket", side_effect=sock_factory), \
         patch("needlestack_core.embedder.Embedder", return_value=MagicMock()), \
         patch("needlestack.server.init") as mock_init, \
         patch("uvicorn.run"), patch("webbrowser.open"), \
         patch("httpx.get", side_effect=Exception("no")):
        result = r.invoke(main, ["serve", "--db", str(db_path), "--no-browser"])
    from needlestack_core.constants import DEFAULT_MODEL
    assert mock_init.call_args.kwargs["ollama_model"] == DEFAULT_MODEL


def test_serve_explicit_model_overrides_last_indexed_model(tmp_path):
    db_path = tmp_path / "index.db"
    s = Store(db_path)
    s.set_config("last_indexed_model", "qwen2.5vl:7b")
    s.close()
    sock_factory = _mock_socket_factory(set())
    r = runner()
    with patch("socket.socket", side_effect=sock_factory), \
         patch("needlestack_core.embedder.Embedder", return_value=MagicMock()), \
         patch("needlestack.server.init") as mock_init, \
         patch("uvicorn.run"), patch("webbrowser.open"), \
         patch("httpx.get", side_effect=Exception("no")):
        result = r.invoke(
            main, ["serve", "--db", str(db_path), "--no-browser", "--model", "other:model"]
        )
    assert mock_init.call_args.kwargs["ollama_model"] == "other:model"
