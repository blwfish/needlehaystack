import base64
import io
import json
import logging
from dataclasses import dataclass, field

import httpx
from PIL import Image

from . import taxonomy
from .constants import DEFAULT_MODEL, OLLAMA_URL, PROMPT_SCHEMA_VERSION

_log = logging.getLogger(__name__)

# Re-exported from .constants (single source) for backward compatibility.
__all__ = ["Captioner", "CaptionResult", "DEFAULT_MODEL", "OLLAMA_URL", "PROMPT_SCHEMA_VERSION"]

# JSON schema handed to Ollama's `format` so the model returns parseable fields rather
# than free prose. Every field here is enumerated and given a disposition in store/
# synthesis — none is captured without a home (data-capture backward-chaining rule).
CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_railroad": {"type": "boolean"},
        "description": {"type": "string"},
        "setting": {"type": "string"},
        "era": {"type": "string"},
        "equipment": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "road_name": {"type": "string"},
                    "reporting_marks": {"type": "string"},
                    "road_number": {"type": "string"},
                    "details": {"type": "string"},
                },
            },
        },
        "visible_text": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_railroad", "description"],
}


def _prompt() -> str:
    return (
        "This is a railroad or railway photograph (or might be). Analyze it for a "
        "searchable photo index and return JSON matching the schema.\n"
        "- is_railroad: true only if the photo actually shows railroad subject matter.\n"
        "- description: plain-sentence description with railroad specificity. If it is "
        "NOT a railroad photo, describe it normally with equal specificity.\n"
        "- equipment: one entry per distinct piece of visible rolling stock. Use exact "
        f"terminology for `type` from this list when it applies: {taxonomy.equipment_terms_prompt()}. "
        "For steam, name the wheel arrangement if legible, e.g. "
        f"{taxonomy.wheel_arrangements_prompt()}. Fill road_name (railroad), "
        "reporting_marks (e.g. ATSF, UP), and road_number ONLY from text you can "
        "actually read on the equipment; leave blank if not legible.\n"
        f"- setting: one of, or near: {taxonomy.settings_prompt()}.\n"
        "- era: approximate period if inferable (e.g. 'steam era', '1950s', 'modern').\n"
        "- visible_text: EVERY piece of text you can read anywhere in the image — "
        "reporting marks, road numbers, heralds, builder's plates, station signs, "
        "lettering. Transcribe exactly; do not guess."
    )

# Free-text prompt used as the fallback when structured JSON can't be parsed, and as the
# base of the dedicated OCR pass.
_FALLBACK_PROMPT = (
    "This is a railroad or railway photograph. Describe it for a searchable photo index "
    "using exact railroad terminology for any rolling stock, including railroad names, "
    "reporting marks, and road numbers if legible. Describe the setting and era. "
    "If this is not a railroad photo, describe it normally with equal specificity. "
    "Write in plain sentences, no preamble."
)

_OCR_PROMPT = (
    "List every piece of text legible in this image — reporting marks, road numbers, "
    "railroad names, heralds, builder's plates, station signs, any lettering. "
    "One item per line. Transcribe exactly what you can read; do not guess or invent. "
    "If no text is legible, reply with nothing."
)


@dataclass
class CaptionResult:
    """Structured caption output. `caption` is the synthesized FTS text; the other
    fields map to dedicated store columns (see Store.upsert)."""
    caption: str
    description: str = ""
    is_railroad: bool = False
    reporting_marks: str = ""   # flattened marks/road numbers/visible text (FTS-weighted high)
    equipment: str = ""         # flattened equipment types + road names (FTS-weighted mid)
    structured_json: str = ""   # raw model JSON, so nothing is ever silently dropped


class Captioner:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))

    # -- public API ---------------------------------------------------------------

    def caption(self, image: Image.Image, thorough: bool = False) -> CaptionResult:
        """Caption an image, returning structured fields.

        Default is a single JSON-schema-constrained call. With `thorough=True`, a
        second dedicated OCR pass is merged in to maximize reporting-mark recall.
        """
        b64 = self._encode(image)
        try:
            data = self._generate(_prompt(), b64, schema=CAPTION_SCHEMA)
            parsed = json.loads(data["response"])
            if not isinstance(parsed, dict):
                raise ValueError("model returned non-object JSON")
        except (json.JSONDecodeError, KeyError, ValueError, httpx.HTTPError) as e:
            _log.warning("Structured caption failed (%s); falling back to plain text", e)
            return self._plain_caption(b64)

        if thorough:
            self._merge_ocr_pass(parsed, b64)

        return self._build_result(parsed)

    def check(self) -> tuple[bool, str]:
        """Return (ok, message). Checks Ollama is running and model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except Exception:
            return False, f"Ollama not reachable at {self.base_url}"

        models = [m["name"] for m in resp.json().get("models", [])]
        if self.model not in models:
            available = ", ".join(models) or "none"
            return False, (
                f"Model '{self.model}' not found in Ollama. "
                f"Available: {available}. "
                f"Run: ollama pull {self.model}"
            )
        return True, "ok"

    def close(self) -> None:
        self._client.close()

    # -- internals ----------------------------------------------------------------

    def _encode(self, image: Image.Image) -> str:
        img = image.copy()
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _generate(self, prompt: str, b64: str, schema: dict | None = None) -> dict:
        body = {"model": self.model, "prompt": prompt, "images": [b64], "stream": False}
        if schema is not None:
            body["format"] = schema
        resp = self._client.post(f"{self.base_url}/api/generate", json=body)
        resp.raise_for_status()
        data = resp.json()
        if data.get("done_reason") == "length":
            _log.warning("Caption truncated at token limit (model=%s)", self.model)
        return data

    def _plain_caption(self, b64: str) -> CaptionResult:
        """Old single-call behavior, used when structured parsing fails."""
        try:
            data = self._generate(_FALLBACK_PROMPT, b64)
            text = data["response"].strip()
        except (KeyError, httpx.HTTPError) as e:
            _log.warning("Plain caption also failed: %s", e)
            text = ""
        return CaptionResult(caption=text, description=text)

    def _merge_ocr_pass(self, parsed: dict, b64: str) -> None:
        """Add a dedicated OCR pass's lines into parsed['visible_text'] (deduped)."""
        try:
            data = self._generate(_OCR_PROMPT, b64)
            lines = [ln.strip(" -•\t") for ln in data["response"].splitlines()]
        except (KeyError, httpx.HTTPError) as e:
            _log.warning("OCR pass failed: %s", e)
            return
        existing = parsed.get("visible_text") or []
        if not isinstance(existing, list):
            existing = []
        seen = {str(t).lower() for t in existing}
        for ln in lines:
            if ln and ln.lower() not in seen:
                existing.append(ln)
                seen.add(ln.lower())
        parsed["visible_text"] = existing

    def _build_result(self, parsed: dict) -> CaptionResult:
        description = str(parsed.get("description") or "").strip()
        setting = str(parsed.get("setting") or "").strip()
        era = str(parsed.get("era") or "").strip()
        is_railroad = bool(parsed.get("is_railroad"))

        equipment = parsed.get("equipment")
        equipment = equipment if isinstance(equipment, list) else []
        visible_text = parsed.get("visible_text")
        visible_text = [str(t).strip() for t in visible_text if str(t).strip()] \
            if isinstance(visible_text, list) else []

        valid_types = taxonomy.valid_equipment_types()
        mark_tokens: list[str] = []      # high-value identifiers (marks, road numbers)
        equip_tokens: list[str] = []     # equipment types + road names
        equip_phrases: list[str] = []    # human-readable per-item phrases for the caption

        for item in equipment:
            if not isinstance(item, dict):
                continue
            etype = str(item.get("type") or "").strip()
            road = str(item.get("road_name") or "").strip()
            marks = str(item.get("reporting_marks") or "").strip()
            number = str(item.get("road_number") or "").strip()
            details = str(item.get("details") or "").strip()

            if etype and etype.lower() not in valid_types:
                # Categorical from an external source: keep it, but surface the unknown
                # rather than silently dropping or coercing it.
                _log.info("Unknown equipment type from model (kept): %r", etype)
            if etype:
                equip_tokens.append(etype)
            if road:
                equip_tokens.append(road)
            if marks:
                mark_tokens.append(marks)
            if number:
                mark_tokens.append(number)
            phrase = " ".join(p for p in [road, marks, number, etype, details] if p)
            if phrase:
                equip_phrases.append(phrase)

        # visible_text is the OCR catch-all — every legible token, weighted as a mark.
        mark_tokens.extend(visible_text)

        caption = self._synthesize(description, equip_phrases, setting, era,
                                   visible_text, mark_tokens)
        return CaptionResult(
            caption=caption,
            description=description,
            is_railroad=is_railroad,
            reporting_marks=" ".join(dict.fromkeys(mark_tokens)),
            equipment=" ".join(dict.fromkeys(equip_tokens)),
            structured_json=json.dumps(parsed, ensure_ascii=False),
        )

    @staticmethod
    def _synthesize(description, equip_phrases, setting, era, visible_text, mark_tokens) -> str:
        parts: list[str] = []
        if description:
            parts.append(description)
        if equip_phrases:
            parts.append("Equipment: " + "; ".join(equip_phrases) + ".")
        if setting:
            parts.append(f"Setting: {setting}.")
        if era:
            parts.append(f"Era: {era}.")
        if visible_text:
            parts.append("Visible text: " + ", ".join(visible_text) + ".")
        elif mark_tokens:
            parts.append("Reporting marks: " + ", ".join(mark_tokens) + ".")
        return "\n".join(parts).strip()
