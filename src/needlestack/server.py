import asyncio
import base64
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import search as search_module
from needlestack_core import taxonomy
from needlestack_core.constants import DEFAULT_MODEL, OLLAMA_URL
from needlestack_core.embedder import Embedder
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


def close() -> None:
    """Release the active store. Called by the CLI after uvicorn exits.

    Handles both the normal path (store opened by the CLI, passed via init)
    and the setup-wizard path (store created in the background thread and
    assigned to _store after indexing completes).
    """
    global _store
    if _store is not None:
        _store.close()
        _store = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 40
    terms: list[str] | None = None  # pre-expanded terms, skips expansion if provided
    all_domains: bool = False        # expand synonyms across all indexed domains


@app.get("/setup", response_class=HTMLResponse)
async def setup_page() -> str:
    try:
        return (_ui_path / "setup.html").read_text()
    except FileNotFoundError as e:
        raise HTTPException(500, f"UI file missing: {e}")


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

    domain_name = req.get("domain", "railroad").strip()
    if domain_name not in taxonomy.DOMAINS:
        raise HTTPException(
            400,
            f"Unknown domain {domain_name!r}. Available: {', '.join(taxonomy.DOMAINS)}",
        )

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
    _captured_domain_name = domain_name

    def _run():
        global _store, _embedder, _setup_mode
        store = None
        captioner = None
        try:
            from needlestack_core.captioner import Captioner
            from .indexer import index_directory

            _captured_db_path.parent.mkdir(parents=True, exist_ok=True)
            store = Store(_captured_db_path)
            selected_domain = taxonomy.DOMAINS.get(_captured_domain_name, taxonomy.RAILROAD)
            captioner = Captioner(
                model=_captured_ollama_model,
                base_url=_captured_ollama_url,
                domain=selected_domain,
            )

            # Fail fast and visibly if Ollama/the model isn't ready — otherwise
            # index_directory swallows the per-image errors and the run looks "done"
            # with everything failed and no explanation (the CLI path check()s too).
            ok, msg = captioner.check()
            if not ok:
                captioner.close()
                store.close()
                with _index_lock:
                    _index_state.running = False
                    _index_state.error = msg
                return

            embedder = Embedder()

            def _progress(total, indexed, skipped, failed, current):
                with _index_lock:
                    _index_state.total = total
                    _index_state.indexed = indexed
                    _index_state.skipped = skipped
                    _index_state.failed = failed
                    _index_state.current = current

            index_directory(root, store, captioner, embedder, on_progress=_progress)
            store.add_root(str(root.resolve()), _captured_domain_name)
            captioner.close()
            captioner = None

            # Publish globals before signaling done so any request that observes
            # done=True also sees a ready _store and _embedder.
            _store = store
            _embedder = embedder
            _setup_mode = False
            store = None  # handed off to _store; don't close in the except path

            with _index_lock:
                _index_state.running = False
                _index_state.done = True

        except Exception as e:
            if captioner is not None:
                try:
                    captioner.close()
                except Exception:
                    pass
            if store is not None:
                try:
                    store.close()
                except Exception:
                    pass
            with _index_lock:
                _index_state.running = False
                _index_state.error = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/sync-status")
async def sync_status() -> dict:
    """Check for new or missing files across all indexed roots."""
    if _store is None:
        return {"new": 0, "removed": 0, "roots": []}
    store = _store
    roots = store.get_roots()
    if not roots:
        return {"new": 0, "removed": 0, "roots": []}
    from needlestack_core.constants import caption_version
    stale = store.count_stale_captions(caption_version(_ollama_model))
    existing_roots = [Path(r["path"]) for r in roots if Path(r["path"]).exists()]

    def _count_new() -> int:
        return sum(store.count_unindexed(p) for p in existing_roots)

    # Both calls do blocking filesystem I/O (rglob + stat); offload to a thread
    # so the async event loop isn't blocked during directory scans.
    new_count, removed_count = await asyncio.gather(
        asyncio.to_thread(_count_new),
        asyncio.to_thread(store.count_missing),
    )
    return {
        "new": new_count, "removed": removed_count, "stale": stale,
        "roots": roots, "count": store.count(),
    }


@app.post("/api/reindex-all")
async def reindex_all() -> dict:
    """Re-index all stored roots in a background thread."""
    if _store is None:
        raise HTTPException(503, "Index not ready")
    roots = _store.get_roots()
    if not roots:
        return {"status": "no_roots"}

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

    _captured_roots = list(roots)
    _captured_store = _store
    _captured_ollama_url_r = _ollama_url
    _captured_ollama_model_r = _ollama_model

    def _run_all():
        global _embedder
        captioner = None
        current_domain_name = None
        try:
            from needlestack_core.captioner import Captioner
            from .indexer import index_directory

            embedder = Embedder()

            for root_info in _captured_roots:
                root = Path(root_info["path"])
                domain_name = root_info["domain"]
                if not root.exists():
                    continue

                selected_domain = taxonomy.DOMAINS.get(domain_name, taxonomy.RAILROAD)
                if domain_name != current_domain_name:
                    if captioner is not None:
                        captioner.close()
                    captioner = Captioner(
                        model=_captured_ollama_model_r,
                        base_url=_captured_ollama_url_r,
                        domain=selected_domain,
                    )
                    ok, msg = captioner.check()
                    if not ok:
                        captioner.close()
                        with _index_lock:
                            _index_state.running = False
                            _index_state.error = msg
                        return
                    current_domain_name = domain_name

                def _progress(total, indexed, skipped, failed, current):
                    with _index_lock:
                        _index_state.total = total
                        _index_state.indexed = indexed
                        _index_state.skipped = skipped
                        _index_state.failed = failed
                        _index_state.current = current

                index_directory(root, _captured_store, captioner, embedder,
                                on_progress=_progress)

            if captioner is not None:
                captioner.close()
            _embedder = embedder
            with _index_lock:
                _index_state.running = False
                _index_state.done = True

        except Exception as e:
            if captioner is not None:
                try:
                    captioner.close()
                except Exception:
                    pass
            with _index_lock:
                _index_state.running = False
                _index_state.error = str(e)

    threading.Thread(target=_run_all, daemon=True).start()
    return {"status": "started"}


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


def _primary_domain() -> taxonomy.Domain:
    """Domain for query expansion: first stored root's domain, or railroad fallback."""
    if _store is None:
        return taxonomy.RAILROAD
    roots = _store.get_roots()
    name = roots[0]["domain"] if roots else (_store.get_config("indexed_domain") or "railroad")
    return taxonomy.DOMAINS.get(name, taxonomy.RAILROAD)


def _all_domains() -> list[taxonomy.Domain]:
    """All distinct domains across indexed roots, preserving first-seen order."""
    if _store is None:
        return [taxonomy.RAILROAD]
    roots = _store.get_roots()
    seen: set[str] = set()
    result: list[taxonomy.Domain] = []
    for r in roots:
        name = r["domain"]
        if name not in seen:
            seen.add(name)
            if name not in taxonomy.DOMAINS:
                _log.warning("Unknown domain %r in index — falling back to railroad", name)
            result.append(taxonomy.DOMAINS.get(name, taxonomy.RAILROAD))
    return result or [taxonomy.RAILROAD]


def _resolve_domains(req: SearchRequest) -> list[taxonomy.Domain]:
    return _all_domains() if req.all_domains else [_primary_domain()]


@app.post("/expand")
async def expand(req: SearchRequest) -> dict:
    from .search import _expand_query
    domains = _resolve_domains(req)
    terms = _expand_query(req.query, ollama_url=_ollama_url, model=_ollama_model,
                          domains=domains)
    return {"terms": terms}


@app.post("/search")
async def search(req: SearchRequest) -> list[dict]:
    if _store is None or _embedder is None:
        raise HTTPException(503, "Index not ready")
    if not req.query.strip():
        return []
    domains = _resolve_domains(req)
    results = search_module.search(
        req.query, _store, _embedder, limit=req.limit,
        ollama_url=_ollama_url, ollama_model=_ollama_model,
        preexpanded_terms=req.terms,
        domains=domains,
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
    else:
        return {"status": "unsupported", "message": "Open not supported on this platform"}
    return {"status": "ok"}


@app.post("/reveal/{image_id}")
async def reveal_image(image_id: int) -> dict:
    path = _resolve_image_path(image_id)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", path])
    else:
        return {"status": "unsupported", "message": "Reveal not supported on this platform"}
    return {"status": "ok"}
