"""Lightweight, dependency-free constants.

Kept separate from captioner.py so the CLI can import the default model name without
pulling in PIL/httpx at startup (the CLI lazy-imports heavy modules inside commands).
Single source of truth for these values — every other module imports from here.
"""

DEFAULT_MODEL = "qwen2.5vl:7b"
OLLAMA_URL = "http://localhost:11434"

# Bump whenever the caption PROMPT, the JSON schema, or caption synthesis changes in a
# way that should invalidate existing captions. Combined with the model name into the
# per-image caption_version so an upgrade auto-re-captions.
PROMPT_SCHEMA_VERSION = "v2"


def caption_version(model: str) -> str:
    """Canonical per-image caption-version string. Single source of truth for the
    format so the indexer (which writes it) and the server (which counts staleness)
    never disagree on how model+schema map to a version."""
    return f"{model}:{PROMPT_SCHEMA_VERSION}"
