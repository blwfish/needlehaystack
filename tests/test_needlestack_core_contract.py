"""Contract tests against the REAL needlestack_core classes -- no mocking.

Every other test in this suite mocks Captioner/Embedder out entirely (they need
a live Ollama server / a loaded CLIP model respectively), which means no
CI-run test ever verified that the real classes still expose the shape this
package actually depends on. A renamed method, a removed CaptionResult field,
or a changed Captioner constructor signature in needlestack-core would pass
every other test in this suite unnoticed, and only break in production on the
next real index/serve run.

These tests construct the real Captioner class (cheap: __init__ only builds an
httpx.Client, no network call) and introspect it -- they never call .caption()
or .check(), which need a live Ollama server (that's what the separate
@pytest.mark.integration suite in test_birds_integration.py is for). This is
deliberately narrower than a full integration test: it catches "the shape
changed" (a rename, a removed field, a signature change), not "the behavior is
wrong" -- but that's exactly the class of drift that was previously invisible
to every CI run.
"""
import inspect

from needlestack_core.captioner import Captioner, CaptionResult
from needlestack_core.embedder import Embedder
from needlestack_core.taxonomy import RAILROAD


# -- Captioner: constructor + methods this package actually calls ------------

def test_captioner_constructor_accepts_model_base_url_domain():
    Captioner(model="test-model", base_url="http://localhost:11434", domain=RAILROAD)


def test_captioner_has_model_attribute():
    """indexer.py's current_caption_version reads captioner.model directly."""
    c = Captioner(model="test-model")
    assert c.model == "test-model"


def test_captioner_has_domain_property():
    """indexer.py's current_caption_version reads captioner.domain.name."""
    c = Captioner(domain=RAILROAD)
    assert c.domain is RAILROAD


def test_captioner_check_signature():
    """cli.py/server.py call captioner.check() with no arguments and unpack a
    2-tuple (ok, msg)."""
    sig = inspect.signature(Captioner.check)
    assert list(sig.parameters) == ["self"]


def test_captioner_caption_signature():
    """indexer.py calls captioner.caption(image, thorough=thorough)."""
    sig = inspect.signature(Captioner.caption)
    params = sig.parameters
    assert "image" in params
    assert "thorough" in params
    assert params["thorough"].default is False


def test_captioner_has_close_method():
    """cli.py/server.py call captioner.close() with no arguments."""
    sig = inspect.signature(Captioner.close)
    assert list(sig.parameters) == ["self"]


def test_captioner_has_stats_with_expected_fields():
    """cli.py/indexer.py read captioner.stats.calls / .avg_seconds_per_call /
    .tokens_per_second."""
    c = Captioner()
    assert hasattr(c.stats, "calls")
    assert hasattr(c.stats, "avg_seconds_per_call")
    assert hasattr(c.stats, "tokens_per_second")


# -- CaptionResult: every field indexer.py reads off Captioner.caption()'s
# return value, verified against the real dataclass rather than a MagicMock ---

def test_caption_result_has_every_field_indexer_consumes():
    fields = {f.name for f in __import__("dataclasses").fields(CaptionResult)}
    consumed_by_indexer = {
        "caption", "reporting_marks", "equipment", "structured_json",
        "is_railroad", "view",
    }
    missing = consumed_by_indexer - fields
    assert not missing, f"indexer.py reads CaptionResult fields that don't exist: {missing}"


# -- Embedder: methods search.py/indexer.py/doctor.py call -------------------

def test_embedder_embed_image_signature():
    sig = inspect.signature(Embedder.embed_image)
    assert "image" in sig.parameters


def test_embedder_embed_text_signature():
    sig = inspect.signature(Embedder.embed_text)
    assert "text" in sig.parameters


def test_embedder_has_dim_class_attribute():
    """store.py's _embedding_dim() reads Embedder.dim without instantiating the
    class (to avoid loading the CLIP model just to get a shape)."""
    assert isinstance(Embedder.dim, int)
    assert Embedder.dim > 0
