import json
import logging
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from needlestack.captioner import Captioner, CaptionResult


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


def mock_json_generate(payload, done_reason="stop"):
    return mock_generate_response(json.dumps(payload), done_reason)


# --- caption(): structured output ---

def test_caption_returns_structured_result():
    c = Captioner()
    payload = {
        "is_railroad": True,
        "description": "A steam locomotive at a depot.",
        "setting": "depot",
        "era": "steam era",
        "equipment": [
            {"type": "steam locomotive", "road_name": "Santa Fe",
             "reporting_marks": "ATSF", "road_number": "3751", "details": "4-8-4"}
        ],
        "visible_text": ["ATSF", "3751", "SANTA FE"],
    }
    with patch.object(c._client, "post", return_value=mock_json_generate(payload)):
        result = c.caption(make_image())
    assert isinstance(result, CaptionResult)
    assert result.is_railroad is True
    assert "steam locomotive" in result.caption
    # High-value identifiers land in the weighted reporting_marks field.
    assert "ATSF" in result.reporting_marks and "3751" in result.reporting_marks
    # Equipment type + road name land in the equipment field.
    assert "steam locomotive" in result.equipment and "Santa Fe" in result.equipment
    # Raw JSON retained so nothing is silently dropped.
    assert json.loads(result.structured_json)["era"] == "steam era"
    c.close()


def test_caption_description_strips_and_populates_caption():
    c = Captioner()
    payload = {"is_railroad": True, "description": "  a boxcar  "}
    with patch.object(c._client, "post", return_value=mock_json_generate(payload)):
        result = c.caption(make_image())
    assert result.description == "a boxcar"
    assert result.caption.startswith("a boxcar")
    c.close()


def test_caption_missing_keys_safe_defaults():
    """Only the required `description` present — everything else defaults, no crash."""
    c = Captioner()
    payload = {"is_railroad": False, "description": "a generic photo"}
    with patch.object(c._client, "post", return_value=mock_json_generate(payload)):
        result = c.caption(make_image())
    assert result.is_railroad is False
    assert result.reporting_marks == ""
    assert result.equipment == ""
    assert result.caption == "a generic photo"
    c.close()


def test_caption_non_railroad_routing():
    """is_railroad False with no equipment → caption is just the description."""
    c = Captioner()
    payload = {"is_railroad": False, "description": "a dog in a field",
               "equipment": [], "visible_text": []}
    with patch.object(c._client, "post", return_value=mock_json_generate(payload)):
        result = c.caption(make_image())
    assert result.is_railroad is False
    assert result.caption == "a dog in a field"
    c.close()


def test_caption_unknown_equipment_type_kept_and_logged(caplog):
    """Ambiguous case: a categorical the taxonomy doesn't know must be KEPT (not
    dropped or coerced) and surfaced in the log."""
    c = Captioner()
    payload = {
        "is_railroad": True,
        "description": "an odd car",
        "equipment": [{"type": "schnabel car", "road_name": "", "reporting_marks": "",
                       "road_number": "", "details": ""}],
    }
    with patch.object(c._client, "post", return_value=mock_json_generate(payload)):
        with caplog.at_level(logging.INFO, logger="needlestack.captioner"):
            result = c.caption(make_image())
    assert "schnabel car" in result.equipment          # kept
    assert any("schnabel car" in r.message for r in caplog.records)  # logged
    c.close()


def test_caption_malformed_json_falls_back_to_plain():
    """The ambiguous failure case: non-JSON model output must fall back to a plain
    free-text caption rather than crashing or losing the image."""
    c = Captioner()
    responses = [
        mock_generate_response("this is not json at all"),   # structured call
        mock_generate_response("a plain caption of a train"),  # fallback call
    ]
    with patch.object(c._client, "post", side_effect=responses):
        result = c.caption(make_image())
    assert isinstance(result, CaptionResult)
    assert result.caption == "a plain caption of a train"
    c.close()


def test_caption_thorough_merges_ocr_pass():
    c = Captioner()
    payload = {"is_railroad": True, "description": "a tank car",
               "equipment": [], "visible_text": ["UTLX"]}
    responses = [
        mock_json_generate(payload),                        # structured call
        mock_generate_response("UTLX\n640123\nSHELL"),       # dedicated OCR pass
    ]
    with patch.object(c._client, "post", side_effect=responses):
        result = c.caption(make_image(), thorough=True)
    # OCR-only tokens not in the structured pass are merged into the marks field.
    assert "640123" in result.reporting_marks
    assert "SHELL" in result.reporting_marks
    c.close()


def test_caption_marks_without_visible_text_still_in_caption():
    """Equipment carries reporting marks but the model returned no visible_text list.
    The marks must still surface in the caption text (via the Equipment phrase) and in
    the weighted reporting_marks column — without a redundant second 'Reporting marks:'
    line duplicating them."""
    c = Captioner()
    payload = {
        "is_railroad": True, "description": "a boxcar",
        "equipment": [{"type": "boxcar", "road_name": "", "reporting_marks": "ATSF",
                       "road_number": "1234", "details": ""}],
        "visible_text": [],
    }
    with patch.object(c._client, "post", return_value=mock_json_generate(payload)):
        result = c.caption(make_image())
    assert "Equipment:" in result.caption
    assert "ATSF" in result.caption and "1234" in result.caption
    assert "ATSF" in result.reporting_marks and "1234" in result.reporting_marks
    # No duplicate marks line — marks appear once (in the Equipment phrase).
    assert result.caption.count("ATSF") == 1
    c.close()


def test_caption_thorough_ocr_failure_is_safe():
    """If the dedicated OCR pass errors, the structured result is preserved, no crash."""
    import httpx
    c = Captioner()
    payload = {"is_railroad": True, "description": "a tank car",
               "equipment": [], "visible_text": ["UTLX"]}
    responses = [mock_json_generate(payload), httpx.ConnectError("ollama down")]
    with patch.object(c._client, "post", side_effect=responses):
        result = c.caption(make_image(), thorough=True)
    assert "UTLX" in result.reporting_marks   # structured pass survived
    c.close()


def test_caption_total_failure_returns_empty_not_crash():
    """Structured call returns junk AND the plain-text fallback also errors → an empty
    caption is returned (never lose the row, never raise)."""
    import httpx
    c = Captioner()
    responses = [mock_generate_response("not json"), httpx.ConnectError("ollama down")]
    with patch.object(c._client, "post", side_effect=responses):
        result = c.caption(make_image())
    assert isinstance(result, CaptionResult)
    assert result.caption == ""
    c.close()


def test_caption_logs_warning_on_truncation(caplog):
    c = Captioner()
    payload = {"is_railroad": True, "description": "truncated mid"}
    with patch.object(c._client, "post", return_value=mock_json_generate(payload, done_reason="length")):
        with caplog.at_level(logging.WARNING, logger="needlestack.captioner"):
            c.caption(make_image())
    assert any("truncated" in r.message.lower() or "token limit" in r.message.lower()
               for r in caplog.records)
    c.close()


def test_caption_no_warning_on_normal_stop(caplog):
    c = Captioner()
    payload = {"is_railroad": True, "description": "a caboose"}
    with patch.object(c._client, "post", return_value=mock_json_generate(payload, done_reason="stop")):
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
    """M5: a different tag must NOT satisfy a requirement for an exact model."""
    c = Captioner(model="qwen2.5vl:7b")
    with patch.object(c._client, "get", return_value=mock_tags_response(["qwen2.5vl:3b"])):
        ok, msg = c.check()
    assert not ok
    assert "qwen2.5vl:7b" in msg
    c.close()


def test_check_returns_ok_for_exact_match():
    c = Captioner(model="qwen2.5vl:7b")
    with patch.object(c._client, "get", return_value=mock_tags_response(["qwen2.5vl:7b", "qwen2.5vl:3b"])):
        ok, msg = c.check()
    assert ok
    c.close()


def test_check_returns_false_when_model_absent():
    c = Captioner(model="qwen2.5vl:7b")
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
