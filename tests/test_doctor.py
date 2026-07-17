"""Tests for doctor.run() — the diagnostic report function.

All httpx calls are mocked so no real Ollama is needed. Store calls use a real
temp DB so migrations and SQL are exercised. Embedder is mocked for query-trace
tests (it would load CLIP, which is slow).
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from needlestack.doctor import run
from needlestack.store import Store


# --- helpers ---

def _tags_resp(model_names):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"models": [{"name": n} for n in model_names]}
    return m


def _gen_resp(text):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"response": text}
    return m


def _ollama_ok(models, reply="OK"):
    """Return (tags_mock, generate_mock) for a working Ollama with given models."""
    return _tags_resp(models), _gen_resp(reply)


def _make_populated_store(db_path: Path) -> None:
    """Insert one image row so the 'count > 0' branch runs."""
    s = Store(db_path)
    s.upsert(
        "/photos/train.jpg", "hash1",
        "A steam locomotive at a depot.",
        np.zeros(512, dtype=np.float32), b"thumb",
        reporting_marks="ATSF", equipment="steam locomotive",
        is_railroad=1, caption_version="v1",
    )
    s.close()


# --- DB presence ---

def test_db_not_found_shows_not_found(tmp_path):
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=tmp_path / "missing.db")
    assert "NOT FOUND" in report


def test_db_not_found_still_runs(tmp_path):
    """A missing DB should not crash doctor — it should report and continue."""
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=tmp_path / "missing.db")
    assert isinstance(report, str) and len(report) > 0


def test_db_empty_shows_zero_count(tmp_path):
    db_path = tmp_path / "index.db"
    Store(db_path).close()   # creates schema, no rows
    with (
        patch("httpx.get", side_effect=Exception("down")),
        patch("httpx.post", side_effect=Exception("down")),
    ):
        report = run(db_path=db_path)
    assert "Images indexed" in report
    assert "0" in report


def test_db_populated_shows_count_and_dates(tmp_path):
    db_path = tmp_path / "index.db"
    _make_populated_store(db_path)
    with (
        patch("httpx.get", side_effect=Exception("down")),
        patch("httpx.post", side_effect=Exception("down")),
    ):
        report = run(db_path=db_path)
    assert "1" in report
    assert "First indexed" in report
    assert "Last indexed" in report


# --- Ollama status ---

def test_ollama_unreachable(tmp_path):
    with patch("httpx.get", side_effect=Exception("connection refused")):
        report = run(db_path=tmp_path / "missing.db")
    assert "NOT REACHABLE" in report


def test_ollama_running_model_present(tmp_path):
    tags, gen = _ollama_ok(["qwen2.5vl:7b"])
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", return_value=gen),
    ):
        report = run(db_path=tmp_path / "missing.db", ollama_model="qwen2.5vl:7b")
    assert "running" in report
    assert "present" in report


def test_ollama_model_not_found(tmp_path):
    tags, gen = _ollama_ok(["qwen2.5vl:3b"])
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", return_value=gen),
    ):
        report = run(db_path=tmp_path / "missing.db", ollama_model="qwen2.5vl:7b")
    assert "NOT FOUND" in report   # model not found message


def test_model_present_prefix_match_different_tag_same_base(tmp_path):
    """A different tag/size of the SAME base model (e.g. :3b present when :7b was
    requested) satisfies the prefix-match branch and is reported present."""
    db_path = tmp_path / "index.db"
    _make_populated_store(db_path)
    tags, gen = _ollama_ok(["qwen2.5vl:3b"])
    with patch("httpx.get", return_value=tags), patch("httpx.post", return_value=gen):
        report = run(db_path=db_path, ollama_model="qwen2.5vl:7b")
    model_line = next(l for l in report.splitlines() if "Model qwen2.5vl:7b" in l)
    assert "present" in model_line


def test_model_present_false_when_different_base_model(tmp_path):
    """A different base model entirely does not satisfy the prefix match."""
    db_path = tmp_path / "index.db"
    _make_populated_store(db_path)
    tags, gen = _ollama_ok(["llava:7b"])
    with patch("httpx.get", return_value=tags), patch("httpx.post", return_value=gen):
        report = run(db_path=db_path, ollama_model="qwen2.5vl:7b")
    model_line = next(l for l in report.splitlines() if "Model qwen2.5vl:7b" in l)
    assert "NOT FOUND" in model_line


def test_quality_tier_note_shown(tmp_path):
    """The quality-tier warning must appear when the quality model is loaded."""
    tags, gen = _ollama_ok(["qwen3-vl:32b"])
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", return_value=gen),
    ):
        report = run(db_path=tmp_path / "missing.db", ollama_model="qwen3-vl:32b")
    assert "quality" in report.lower()
    assert "slow" in report.lower()


def test_custom_model_note_shown(tmp_path):
    tags, gen = _ollama_ok(["my-custom-model:latest"])
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", return_value=gen),
    ):
        report = run(db_path=tmp_path / "missing.db", ollama_model="my-custom-model:latest")
    assert "custom" in report.lower()


def test_test_inference_ok(tmp_path):
    tags, gen = _ollama_ok(["qwen2.5vl:7b"], reply="OK I will help you")
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", return_value=gen),
    ):
        report = run(db_path=tmp_path / "missing.db", ollama_model="qwen2.5vl:7b")
    assert "Test inference" in report
    assert "FAILED" not in report.split("Test inference")[1][:50]


def test_test_inference_failure(tmp_path):
    tags = _tags_resp(["qwen2.5vl:7b"])
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", side_effect=Exception("generate timeout")),
    ):
        report = run(db_path=tmp_path / "missing.db", ollama_model="qwen2.5vl:7b")
    assert "FAILED" in report


# --- query trace ---

def test_query_trace_expansion_failure(tmp_path):
    """doctor.py's own try/except around _expand_query must catch a failure and
    report FAILED, without crashing the rest of the trace.

    Exercised by patching needlestack.doctor._expand_query directly (not by mocking
    httpx) — _expand_query already swallows every httpx exception internally and
    never raises, so mocking httpx down could never actually reach this except
    branch. The previous version of this test did exactly that: its assertion
    `"FAILED" in report or "Query expansion" in report` was unconditionally true
    because doctor.py writes the "Query expansion:" header before the try block
    even runs, regardless of what happens inside it — the test always passed
    without ever proving the FAILED branch worked.
    """
    db_path = tmp_path / "index.db"
    Store(db_path).close()

    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = np.zeros(512, dtype=np.float32)

    with (
        patch("needlestack.doctor._expand_query", side_effect=RuntimeError("boom")),
        patch("needlestack_core.embedder.Embedder", return_value=mock_embedder),
    ):
        report = run(db_path=db_path, query="steam locomotive")
    assert "Search trace" in report
    assert "Query expansion:" in report
    assert "FAILED: boom" in report


def test_query_trace_clip_failure_is_graceful(tmp_path):
    """If Embedder.embed_text raises, the CLIP section shows FAILED, no crash."""
    db_path = tmp_path / "index.db"
    Store(db_path).close()

    mock_embedder = MagicMock()
    mock_embedder.embed_text.side_effect = RuntimeError("CLIP model not loaded")

    tags, gen = _ollama_ok(["qwen2.5vl:7b"])
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", return_value=gen),
        patch("needlestack_core.embedder.Embedder", return_value=mock_embedder),
    ):
        report = run(db_path=db_path, query="caboose", ollama_model="qwen2.5vl:7b")
    assert "CLIP" in report
    assert "FAILED" in report


def test_query_trace_empty_index_no_crash(tmp_path):
    """Trace with zero images in the index should not crash."""
    db_path = tmp_path / "index.db"
    Store(db_path).close()

    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = np.zeros(512, dtype=np.float32)

    tags, gen = _ollama_ok(["qwen2.5vl:7b"])
    with (
        patch("httpx.get", return_value=tags),
        patch("httpx.post", return_value=gen),
        patch("needlestack_core.embedder.Embedder", return_value=mock_embedder),
    ):
        report = run(db_path=db_path, query="steam locomotive", ollama_model="qwen2.5vl:7b")
    assert "Search trace" in report
    assert "index empty" in report.lower() or "none" in report.lower()


# --- report structure ---

def test_report_contains_version_and_platform(tmp_path):
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=tmp_path / "missing.db")
    assert "needlestack diagnostic report" in report
    assert "Version:" in report
    assert "Platform" in report


# --- domain awareness (doctor previously hardcoded RAILROAD regardless of what
# was actually indexed) ---

def test_domain_row_reflects_indexed_domain(tmp_path):
    db_path = tmp_path / "index.db"
    s = Store(db_path)
    s.add_root("/photos", "aviation")
    s.close()
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=db_path)
    assert "Domain(s)" in report
    assert "aviation" in report


def test_vocabulary_section_uses_indexed_domain_not_railroad(tmp_path):
    db_path = tmp_path / "index.db"
    s = Store(db_path)
    s.upsert("/p/bird.jpg", "h", "a red-tailed hawk soaring", np.zeros(512, dtype=np.float32), b"t")
    s.add_root("/p", "birds")
    s.close()
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=db_path)
    assert "Birds / bird photography vocabulary in captions" in report
    assert "Railroad / railway vocabulary in captions" not in report


def test_vocabulary_section_multi_domain_shows_each(tmp_path):
    db_path = tmp_path / "index.db"
    s = Store(db_path)
    s.upsert("/p/a.jpg", "h1", "a caboose", np.zeros(512, dtype=np.float32), b"t")
    s.add_root("/p/rr", "railroad")
    s.add_root("/p/naval", "naval")
    s.close()
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=db_path)
    assert "Railroad / railway vocabulary in captions" in report
    assert "Naval / warships vocabulary in captions" in report


def test_query_expansion_trace_passes_indexed_domain(tmp_path):
    """The query-trace's expansion call must use the store's actual domain(s), not
    a hardcoded railroad default."""
    db_path = tmp_path / "index.db"
    s = Store(db_path)
    s.add_root("/photos", "naval")
    s.close()

    with (
        patch("needlestack.doctor._expand_query", return_value=["destroyer"]) as mock_expand,
        patch("httpx.get", side_effect=Exception("down")),
    ):
        run(db_path=db_path, query="destroyer")

    domains_arg = mock_expand.call_args.kwargs.get("domains")
    assert domains_arg is not None
    assert [d.name for d in domains_arg] == ["naval"]


# --- corrupt embeddings (neither NULL nor decodable — invisible to every other check) ---

def test_corrupt_embeddings_reported(tmp_path):
    db_path = tmp_path / "index.db"
    s = Store(db_path)
    s.upsert("/p/a.jpg", "h1", "a caboose", np.zeros(512, dtype=np.float32), b"t")
    s.conn.execute("UPDATE images SET embedding = ? WHERE path = '/p/a.jpg'", (b"garbage",))
    s.conn.commit()
    s.close()
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=db_path)
    assert "Corrupt embeddings" in report
    assert "1" in [l.split()[-1] for l in report.splitlines() if "Corrupt embeddings" in l][0]


def test_corrupt_embeddings_row_absent_when_none(tmp_path):
    db_path = tmp_path / "index.db"
    _make_populated_store(db_path)
    with patch("httpx.get", side_effect=Exception("down")):
        report = run(db_path=db_path)
    assert "Corrupt embeddings" not in report
