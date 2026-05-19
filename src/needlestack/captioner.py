import base64
import io
import logging

import httpx
from PIL import Image

_log = logging.getLogger(__name__)

PROMPT = (
    "This is a railroad or railway photograph. Describe it for a searchable photo index. "
    "For all visible rolling stock, use exact railroad terminology: "
    "steam locomotive, diesel locomotive, electric locomotive; "
    "boxcar, flatcar, gondola, hopper car, tank car, refrigerator car, stock car, "
    "passenger coach, baggage car, caboose, etc. "
    "Include railroad name or reporting marks if visible. "
    "Describe the setting (yard, depot, mainline, bridge, tunnel, etc.), era, "
    "architectural features, and any infrastructure. "
    "If this is not a railroad photo, describe it normally with equal specificity. "
    "Write in plain sentences, no preamble."
)

DEFAULT_MODEL = "llava:13b"
OLLAMA_URL = "http://localhost:11434"


class Captioner:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    def caption(self, image: Image.Image) -> str:
        img = image.copy()
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = self._client.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": PROMPT, "images": [b64], "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("done_reason") == "length":
            _log.warning("Caption truncated at token limit (model=%s)", self.model)
        return data["response"].strip()

    def check(self) -> tuple[bool, str]:
        """Return (ok, message). Checks Ollama is running and model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except Exception:
            return False, f"Ollama not reachable at {self.base_url}"

        models = [m["name"] for m in resp.json().get("models", [])]
        base = self.model.split(":")[0]
        if not any(m == self.model or m.startswith(base + ":") for m in models):
            available = ", ".join(models) or "none"
            return False, (
                f"Model '{self.model}' not found in Ollama. "
                f"Available: {available}. "
                f"Run: ollama pull {self.model}"
            )
        return True, "ok"

    def close(self) -> None:
        self._client.close()
