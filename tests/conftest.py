"""Shared pytest fixtures."""
import io
import json
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from needlestack_core.constants import OLLAMA_URL

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "birds"


_HEADERS = {"User-Agent": "needlestack-tests/1.0 (integration test; bird domain)"}


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _wikipedia_image_url(article_title: str) -> str | None:
    """Return the original image URL for a Wikipedia article's lead photo."""
    api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article_title}"
    try:
        data = json.loads(_get(api))
        src = data.get("originalimage", {}).get("source")
        if not src:
            src = data.get("thumbnail", {}).get("source")
        return src
    except Exception:
        return None


def _download_photo(article_title: str, cache_path: Path) -> Path | None:
    """Download a Wikipedia lead photo, resize to ≤1024px, cache as JPEG."""
    if cache_path.exists():
        return cache_path
    url = _wikipedia_image_url(article_title)
    if not url:
        return None
    try:
        img = Image.open(io.BytesIO(_get(url, timeout=30))).convert("RGB")
    except Exception:
        return None
    img.thumbnail((1024, 1024))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache_path, "JPEG", quality=85)
    return cache_path


@pytest.fixture(scope="session")
def ollama_running():
    """Skip the test if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
            pass
    except Exception:
        pytest.skip("Ollama not reachable — start Ollama to run integration tests")


@pytest.fixture(scope="session")
def bird_photos(ollama_running):
    """Download and cache one photo per bird family group. Returns dict of group→Path."""
    sources = {
        "waterfowl":   "Mallard",
        "wading bird": "Great_blue_heron",
        "raptor":      "Red-tailed_hawk",
        "songbird":    "American_robin",
    }
    photos = {}
    for group, title in sources.items():
        path = _download_photo(title, FIXTURE_DIR / f"{group.replace(' ', '_')}.jpg")
        if path:
            photos[group] = path
    if not photos:
        pytest.skip("Could not download any test photos — check network access")
    return photos


@pytest.fixture(scope="session")
def non_bird_photo(ollama_running):
    """Download and cache a landscape photo (no birds)."""
    path = _download_photo("Grand_Canyon", FIXTURE_DIR / "non_bird.jpg")
    if not path:
        pytest.skip("Could not download non-bird test photo")
    return path
