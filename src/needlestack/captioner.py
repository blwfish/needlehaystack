import base64
import io
import json
import logging
from dataclasses import dataclass, field

import httpx
from PIL import Image

from . import taxonomy
from .taxonomy import Domain
from .constants import DEFAULT_MODEL, OLLAMA_URL

_log = logging.getLogger(__name__)

_OCR_PROMPT = (
    "List every piece of text legible in this image — identifiers, names, numbers, "
    "heralds, plates, signs, any lettering. "
    "One item per line. Transcribe exactly what you can read; do not guess or invent. "
    "If no text is legible, reply with nothing."
)


def _make_schema(domain: Domain) -> dict:
    """Build the JSON schema Ollama enforces for this domain's caption output."""
    item_props = {f: {"type": "string"} for f, _ in domain.item_fields}
    return {
        "type": "object",
        "properties": {
            domain.subject_field: {"type": "boolean"},
            "description": {"type": "string"},
            "setting": {"type": "string"},
            "era": {"type": "string"},
            "view": {"type": "string"},
            domain.items_field: {
                "type": "array",
                "items": {"type": "object", "properties": item_props},
            },
            "visible_text": {"type": "array", "items": {"type": "string"}},
        },
        "required": [domain.subject_field, "description"],
    }


def _make_prompt(domain: Domain) -> str:
    """Build the caption prompt for this domain."""
    frags = domain.prompt_fragments
    preamble = frags["preamble"]
    subject_qualifier = frags["subject_qualifier"]
    item_singular = frags["item_singular"]
    id_instruction = frags["id_instruction"]
    era_examples = frags["era_examples"]
    view_instruction = frags["view_instruction"]
    type_note = frags.get("type_note", "")

    type_instruction = (
        f"Use exact terminology for `type` from this list when it applies: "
        f"{domain.subject_types_prompt()}."
    )
    if type_note:
        type_instruction = f"{type_instruction} {type_note}"

    return (
        f"{preamble} Analyze it for a searchable photo index and return JSON matching "
        "the schema.\n"
        f"- {domain.subject_field}: {subject_qualifier}.\n"
        "- description: plain-sentence description with specific detail. Name only what "
        "you can visually confirm; use 'appears to be' when uncertain. If this is not a "
        "matching photo, describe what it actually shows.\n"
        f"- {domain.items_field}: one entry per distinct {item_singular} visible. "
        f"{type_instruction} {id_instruction}\n"
        f"- setting: one of, or similar to: {domain.settings_prompt()}.\n"
        f"- era: approximate period if inferable (e.g. {era_examples}).\n"
        f"- view: camera perspective — one of: {view_instruction}.\n"
        "- visible_text: EVERY piece of text you can read anywhere in the image. "
        "Transcribe exactly what you can read; do not guess or invent."
    )


@dataclass
class CaptionResult:
    """Structured caption output. `caption` is the synthesized FTS text; the other
    fields map to dedicated store columns (see Store.upsert)."""
    caption: str
    description: str = ""
    is_railroad: bool = False   # semantically: "is_subject" — true if on-topic for the domain
    reporting_marks: str = ""   # flattened high-priority identifiers (FTS-weighted high)
    equipment: str = ""         # flattened subject types + class/road names (FTS-weighted mid)
    structured_json: str = ""   # raw model JSON, so nothing is ever silently dropped
    view: str = ""              # camera perspective (broadside, bow quarter, etc.)


class Captioner:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_URL,
        domain: Domain = taxonomy.RAILROAD,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._domain = domain
        self._client = httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))

    # -- public API ---------------------------------------------------------------

    def caption(self, image: Image.Image, thorough: bool = False) -> CaptionResult:
        """Caption an image, returning structured fields.

        Default is a single JSON-schema-constrained call. With `thorough=True`, a
        second dedicated OCR pass is merged in to maximize identifier recall.
        """
        b64 = self._encode(image)
        schema = _make_schema(self._domain)
        prompt = _make_prompt(self._domain)
        try:
            data = self._generate(prompt, b64, schema=schema)
            parsed = json.loads(data["response"])
            if not isinstance(parsed, dict):
                raise ValueError("model returned non-object JSON")
        except (json.JSONDecodeError, KeyError, ValueError, httpx.HTTPError) as e:
            _log.warning("Structured caption failed (%s); falling back to plain text", e)
            return self._plain_caption(b64)

        if thorough:
            self._merge_ocr_pass(parsed, b64)

        return self._build_result(parsed, self._domain)

    def check(self) -> tuple[bool, str]:
        """Return (ok, message). Checks Ollama is running and model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except Exception:
            return False, f"Ollama not reachable at {self.base_url}"

        models = [m["name"] for m in resp.json().get("models", [])]
        base = self.model.split(":")[0]
        # Accept exact match or the untagged pull (stored as :latest by Ollama).
        model_found = self.model in models or f"{base}:latest" in models
        if not model_found:
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
        fallback_prompt = self._domain.prompt_fragments.get("fallback_preamble", "")
        try:
            data = self._generate(fallback_prompt, b64)
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

    def _build_result(self, parsed: dict, domain: Domain) -> CaptionResult:
        description = str(parsed.get("description") or "").strip()
        setting = str(parsed.get("setting") or "").strip()
        era = str(parsed.get("era") or "").strip()
        view = str(parsed.get("view") or "").strip()
        is_subject = bool(parsed.get(domain.subject_field))

        items = parsed.get(domain.items_field)
        items = items if isinstance(items, list) else []
        visible_text = parsed.get("visible_text")
        visible_text = [str(t).strip() for t in visible_text if str(t).strip()] \
            if isinstance(visible_text, list) else []

        mark_tokens: list[str] = []      # high-value identifiers (hull numbers, marks)
        equip_tokens: list[str] = []     # subject types + class/road names
        equip_phrases: list[str] = []    # human-readable per-item phrases for the caption

        for item in items:
            if not isinstance(item, dict):
                continue
            etype = str(item.get("type") or "").strip()
            if etype and etype.lower() not in domain.valid_subject_types:
                _log.info("Unknown %s type from model (kept): %r", domain.name, etype)

            phrase_parts: list[str] = []
            for field_name, fts_weight in domain.item_fields:
                value = str(item.get(field_name) or "").strip()
                if not value:
                    continue
                if fts_weight == "high":
                    mark_tokens.append(value)
                elif fts_weight == "mid":
                    equip_tokens.append(value)
                phrase_parts.append(value)

            phrase = " ".join(phrase_parts)
            if phrase:
                equip_phrases.append(phrase)

        # visible_text is the OCR catch-all — every legible token, weighted as a mark.
        mark_tokens.extend(visible_text)

        caption = self._synthesize(description, equip_phrases, setting, era, view, visible_text)
        return CaptionResult(
            caption=caption,
            description=description,
            is_railroad=is_subject,
            reporting_marks=" ".join(dict.fromkeys(mark_tokens)),
            equipment=" ".join(dict.fromkeys(equip_tokens)),
            structured_json=json.dumps(parsed, ensure_ascii=False),
            view=view,
        )

    @staticmethod
    def _synthesize(
        description: str,
        equip_phrases: list[str],
        setting: str,
        era: str,
        view: str,
        visible_text: list[str],
    ) -> str:
        parts: list[str] = []
        if description:
            parts.append(description)
        if equip_phrases:
            parts.append("Equipment: " + "; ".join(equip_phrases) + ".")
        if setting:
            parts.append(f"Setting: {setting}.")
        if era:
            parts.append(f"Era: {era}.")
        if view:
            parts.append(f"View: {view}.")
        if visible_text:
            parts.append("Visible text: " + ", ".join(visible_text) + ".")
        return "\n".join(parts).strip()
