import logging
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from needlestack.captioner import Captioner


def make_image():
    return Image.new("RGB", (100, 100), color=(200, 100, 50))


def mock_tags_response(model_names):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"models": [{"name": n} for n in model_names]}
    return resp


def mock_generate_response(text, done_reason="stop"):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": text, "done_reason": done_reason, "done": True}
    return resp


# --- caption() ---

def test_caption_returns_response_text():
    c = Captioner()
    with patch.object(c._client, "post", return_value=mock_generate_response("a steam locomotive")):
        result = c.caption(make_image())
    assert result == "a steam locomotive"
    c.close()


def test_caption_strips_whitespace():
    c = Captioner()
    with patch.object(c._client, "post", return_value=mock_generate_response("  a boxcar  ")):
        result = c.caption(make_image())
    assert result == "a boxcar"
    c.close()


def test_caption_logs_warning_on_truncation(caplog):
    c = Captioner()
    with patch.object(c._client, "post", return_value=mock_generate_response("truncated mid", done_reason="length")):
        with caplog.at_level(logging.WARNING, logger="needlestack.captioner"):
            c.caption(make_image())
    assert any("truncated" in r.message.lower() or "token limit" in r.message.lower()
               for r in caplog.records)
    c.close()


def test_caption_no_warning_on_normal_stop(caplog):
    c = Captioner()
    with patch.object(c._client, "post", return_value=mock_generate_response("a caboose", done_reason="stop")):
        with caplog.at_level(logging.WARNING, logger="needlestack.captioner"):
            c.caption(make_image())
    assert not any("truncated" in r.message.lower() or "token limit" in r.message.lower()
                   for r in caplog.records)
    c.close()


# --- check() ---

def test_check_returns_false_when_ollama_unreachable():
    c = Captioner()
    with patch.object(c._client, "get", side_effect=Exception("connection refused")):
        ok, msg = c.check()
    assert not ok
    assert "not reachable" in msg.lower()
    c.close()


def test_check_requires_exact_model_match():
    """M5: llava:7b must NOT satisfy a requirement for llava:13b."""
    c = Captioner(model="llava:13b")
    with patch.object(c._client, "get", return_value=mock_tags_response(["llava:7b"])):
        ok, msg = c.check()
    assert not ok
    assert "llava:13b" in msg
    c.close()


def test_check_returns_ok_for_exact_match():
    c = Captioner(model="llava:13b")
    with patch.object(c._client, "get", return_value=mock_tags_response(["llava:13b", "llava:7b"])):
        ok, msg = c.check()
    assert ok
    c.close()


def test_check_returns_false_when_model_absent():
    c = Captioner(model="llava:13b")
    with patch.object(c._client, "get", return_value=mock_tags_response([])):
        ok, msg = c.check()
    assert not ok
    assert "none" in msg.lower() or "available" in msg.lower()
    c.close()


def test_connect_timeout_is_separate():
    """M8: connect timeout must be shorter than read timeout."""
    import httpx
    c = Captioner()
    t = c._client.timeout
    assert isinstance(t, httpx.Timeout)
    assert t.connect < t.read
    c.close()
