"""Lightweight, dependency-free constants.

Kept separate from captioner.py so the CLI can import the default model name without
pulling in PIL/httpx at startup (the CLI lazy-imports heavy modules inside commands).
Single source of truth for these values — every other module imports from here.
"""

DEFAULT_MODEL = "qwen2.5vl:7b"
OLLAMA_URL = "http://localhost:11434"

# Named model presets — convenience aliases over --model.
# fast:     low-RAM / CPU-only machines; good for type + identifier, approximate components
# balanced: default; good GPU/Apple-Silicon fit; ~4s/photo
# quality:  high-VRAM machines; best component/armament detail; ~90-120s/photo — warn users
MODEL_PRESETS: dict[str, str] = {
    "fast":     "minicpm-v:latest",
    "balanced": "qwen2.5vl:7b",
    "quality":  "qwen3-vl:32b",
}

# Reverse map: model name → tier label (for doctor display)
MODEL_TIERS: dict[str, str] = {v: k for k, v in MODEL_PRESETS.items()}

# Bump whenever the caption PROMPT, the JSON schema, or caption synthesis changes in a
# way that should invalidate existing captions. Combined with the model name into the
# per-image caption_version so an upgrade auto-re-captions.
PROMPT_SCHEMA_VERSION = "v2"


def caption_version(model: str) -> str:
    """Canonical per-image caption-version string. Single source of truth for the
    format so the indexer (which writes it) and the server (which counts staleness)
    never disagree on how model+schema map to a version."""
    return f"{model}:{PROMPT_SCHEMA_VERSION}"
