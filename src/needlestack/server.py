import base64
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import search as search_module
from .constants import DEFAULT_MODEL, OLLAMA_URL
from .embedder import Embedder
from .store import Store

app = FastAPI(title="needlestack")

_store: Store | None = None
_embedder: Embedder | None = None
_ui_path: Path | None = None
_db_path: Path | None = None
_ollama_url: str = OLLAMA_URL
_ollama_model: str = DEFAULT_MODEL
_setup_mode: bool = False


@dataclass
class _IndexState:
    running: bool = False
    done: bool = False
    error: str = ""
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    current: str = ""


_index_state = _IndexState()
_index_lock = threading.Lock()


def init(
    store: Store | None,
    embedder: Embedder | None,
    ui_path: Path,
    db_path: Path,
    ollama_url: str,
    ollama_model: str,
    setup_mode: bool = False,
) -> None:
    global _store, _embedder, _ui_path, _db_path, _ollama_url, _ollama_model, _setup_mode
    _store = store
    _embedder = embedder
    _ui_path = ui_path
    _db_path = db_path
    _ollama_url = ollama_url
    _ollama_model = ollama_model
    _setup_mode = setup_mode


class SearchRequest(BaseModel):
    query: str
    limit: int = 40
    terms: list[str] | None = None  # pre-expanded terms, skips expansion if provided


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    try:
        if _setup_mode or _index_state.running or (_store is not None and _store.count() == 0 and not _index_state.done):
            return (_ui_path / "setup.html").read_text()
        return (_ui_path / "index.html").read_text()
    except FileNotFoundError as e:
        raise HTTPException(500, f"UI file missing: {e}")


# --- setup wizard endpoints ---

@app.get("/api/setup/browse")
async def browse_folder() -> dict:
    """Open a native folder picker and return the selected path."""
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["osascript", "-e",
                 'POSIX path of (choose folder with prompt "Select your photos folder:")'],
                capture_output=True, text=True, timeout=60,
            )
            path = result.stdout.strip()
        elif sys.platform == "win32":
            script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
                "$d.Description = 'Select your photos folder';"
                "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=60,
            )
            path = result.stdout.strip()
        else:
            return {"error": "Folder picker not supported on this platform"}

        if not path:
            return {"cancelled": True}
        return {"path": path}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/setup/start")
async def start_indexing(req: dict) -> dict:
    """Kick off indexing in a background thread."""
    global _store, _embedder, _setup_mode

    folder = req.get("folder", "").strip()
    if not folder:
        raise HTTPException(400, "No folder provided")

    root = Path(folder)
    if not root.exists():
        raise HTTPException(400, f"Folder not found: {folder}")
    if not root.is_dir():
        raise HTTPException(400, f"Not a directory: {folder}")

    with _index_lock:
        if _index_state.running:
            return {"status": "already_running"}
        _index_state.running = True
        _index_state.done = False
        _index_state.error = ""
        _index_state.total = 0
        _index_state.indexed = 0
        _index_state.skipped = 0
        _index_state.failed = 0
        _index_state.current = ""

    # Capture globals now so the thread doesn't race with a future init() call.
    _captured_db_path = _db_path
    _captured_ollama_url = _ollama_url
    _captured_ollama_model = _ollama_model

    def _run():
        global _store, _embedder, _setup_mode
        try:
            from .captioner import Captioner
            from .indexer import index_directory

            _captured_db_path.parent.mkdir(parents=True, exist_ok=True)
            store = Store(_captured_db_path)
            captioner = Captioner(model=_captured_ollama_model, base_url=_captured_ollama_url)
            embedder = Embedder()

            def _progress(total, indexed, skipped, failed, current):
                with _index_lock:
                    _index_state.total = total
                    _index_state.indexed = indexed
                    _index_state.skipped = skipped
                    _index_state.failed = failed
                    _index_state.current = current

            index_directory(root, store, captioner, embedder, on_progress=_progress)
            store.set_config("indexed_root", str(root.resolve()))
            captioner.close()


            with _index_lock:
                _index_state.running = False
                _index_state.done = True

            # Switch server to search mode
            _store = store
            _embedder = embedder
            _setup_mode = False

        except Exception as e:
            with _index_lock:
                _index_state.running = False
                _index_state.error = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/sync-status")
async def sync_status() -> dict:
    """Check for new or missing files since last index."""
    if _store is None:
        return {"new": 0, "removed": 0, "root": None}
    root_str = _store.get_config("indexed_root")
    if not root_str:
        return {"new": 0, "removed": 0, "root": None}
    root = Path(root_str)
    from .constants import caption_version
    stale = _store.count_stale_captions(caption_version(_ollama_model))
    if not root.exists():
        return {"new": 0, "removed": 0, "stale": stale, "root": root_str, "root_missing": True}
    new_count = _store.count_unindexed(root)
    return {
        "new": new_count, "removed": 0, "stale": stale,
        "root": root_str, "count": _store.count(),
    }


@app.get("/api/setup/progress")
async def index_progress() -> dict:
    with _index_lock:
        return {
            "running": _index_state.running,
            "done": _index_state.done,
            "error": _index_state.error,
            "total": _index_state.total,
            "indexed": _index_state.indexed,
            "skipped": _index_state.skipped,
            "failed": _index_state.failed,
            "current": _index_state.current,
        }


@app.post("/expand")
async def expand(req: SearchRequest) -> dict:
    from .search import _expand_query
    terms = _expand_query(req.query, ollama_url=_ollama_url, model=_ollama_model)
    return {"terms": terms}


@app.post("/search")
async def search(req: SearchRequest) -> list[dict]:
    if _store is None or _embedder is None:
        raise HTTPException(503, "Index not ready")
    if not req.query.strip():
        return []
    results = search_module.search(
        req.query, _store, _embedder, limit=req.limit,
        ollama_url=_ollama_url, ollama_model=_ollama_model,
        preexpanded_terms=req.terms,
    )
    return [
        {
            "id": r["id"],
            "path": r["path"],
            "caption": r["caption"],
            "score": r["score"],
            "thumbnail": (
                base64.b64encode(r["thumbnail"]).decode() if r["thumbnail"] else None
            ),
        }
        for r in results
    ]


@app.get("/thumbnail/{image_id}")
async def thumbnail(image_id: int) -> Response:
    if _store is None:
        raise HTTPException(503, "Index not ready")
    data = _store.get_thumbnail(image_id)
    if not data:
        raise HTTPException(404)
    return Response(content=data, media_type="image/jpeg")


def _resolve_image_path(image_id: int) -> str:
    """Look up a stored path and validate it still exists and is an image."""
    from .indexer import IMAGE_EXTENSIONS
    if _store is None:
        raise HTTPException(503, "Index not ready")
    rows = _store.get_by_ids([image_id])
    if not rows:
        raise HTTPException(404)
    path = rows[0]["path"]
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404)
    return path


@app.post("/open/{image_id}")
async def open_image(image_id: int) -> dict:
    path = _resolve_image_path(image_id)
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform == "win32":
        subprocess.Popen(["cmd", "/c", "start", "", path], shell=False)
    return {"status": "ok"}


@app.post("/reveal/{image_id}")
async def reveal_image(image_id: int) -> dict:
    path = _resolve_image_path(image_id)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", path])
    return {"status": "ok"}
