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


class StartIndexRequest(BaseModel):
    folder: str
    domain: str = "railroad"


def _run_indexing_loop(
    roots: list[dict], store: Store, ollama_url: str, ollama_model: str, embedder: Embedder,
) -> None:
    """Caption+embed every root in `roots`, switching Captioner when the domain
    changes. Shared by start_indexing and reindex_all so the setup-wizard flow and
    the reindex flow can't silently diverge the way they previously did (two
    separate copies of this loop, only one of which fast-failed consistently).

    Raises on a captioner.check() failure or any exception from index_directory —
    callers wrap this in their own try/except to set _index_state.error.
    """
    from needlestack_core.captioner import Captioner
    from .indexer import index_directory

    captioner: Captioner | None = None
    current_domain_name: str | None = None
    try:
        for root_info in roots:
            root = Path(root_info["path"])
            domain_name = root_info["domain"]
            if not root.exists():
                continue

            if domain_name != current_domain_name:
                if captioner is not None:
                    captioner.close()
                selected_domain = taxonomy.resolve_domain(domain_name)
                captioner = Captioner(model=ollama_model, base_url=ollama_url, domain=selected_domain)
                # Fail fast and visibly if Ollama/the model isn't ready — otherwise
                # index_directory swallows the per-image errors and the run looks
                # "done" with everything failed and no explanation.
                ok, msg = captioner.check()
                if not ok:
                    raise RuntimeError(msg)
                current_domain_name = domain_name

            def _progress(total, indexed, skipped, failed, current):
                with _index_lock:
                    _index_state.total = total
                    _index_state.indexed = indexed
                    _index_state.skipped = skipped
                    _index_state.failed = failed
                    _index_state.current = current

            index_directory(root, store, captioner, embedder, on_progress=_progress)
    finally:
        if captioner is not None:
            captioner.close()


def _render_domain_options() -> str:
    """Generate the setup wizard's <option> list from taxonomy.DOMAINS — the single
    source of truth — instead of a hand-typed HTML list that can drift out of sync
    with the registry (this is exactly how "motorsports" ended up missing from the
    picker after being added to DOMAINS)."""
    import html
    return "\n".join(
        f'        <option value="{html.escape(name)}">{html.escape(domain.display_label)}</option>'
        for name, domain in taxonomy.DOMAINS.items()
    )


def _read_setup_html() -> str:
    try:
        text = (_ui_path / "setup.html").read_text()
    except FileNotFoundError as e:
        raise HTTPException(500, f"UI file missing: {e}")
    return text.replace("<!-- DOMAIN_OPTIONS -->", _render_domain_options())


@app.get("/setup", response_class=HTMLResponse)
async def setup_page() -> str:
    return _read_setup_html()


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    if _setup_mode or _index_state.running or (_store is not None and _store.count() == 0 and not _index_state.done):
        return _read_setup_html()
    try:
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
            # On macOS/Windows, clicking Cancel makes the picker exit non-zero (e.g.
            # osascript's "User canceled." AppleScript error -128) — that is the
            # NORMAL cancellation path, not a failure. Empty output with a zero exit
            # is the unexpected case (the dialog claimed success but gave no path).
            if result.returncode != 0:
                return {"cancelled": True}
            return {"error": "Folder picker returned no path unexpectedly (exit 0)"}
        return {"path": path}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/setup/start")
async def start_indexing(req: StartIndexRequest) -> dict:
    """Kick off indexing in a background thread."""
    global _store, _embedder, _setup_mode

    folder = req.folder.strip()
    if not folder:
        raise HTTPException(400, "No folder provided")

    root = Path(folder)
    if not root.exists():
        raise HTTPException(400, f"Folder not found: {folder}")
    if not root.is_dir():
        raise HTTPException(400, f"Not a directory: {folder}")

    domain_name = req.domain.strip()
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
        try:
            _captured_db_path.parent.mkdir(parents=True, exist_ok=True)
            store = Store(_captured_db_path)
            embedder = Embedder()

            _run_indexing_loop(
                [{"path": str(root), "domain": _captured_domain_name}],
                store, _captured_ollama_url, _captured_ollama_model, embedder,
            )
            store.add_root(str(root.resolve()), _captured_domain_name)
            # Record which model produced these captions — cli.py serve's default
            # model then consults this instead of a hardcoded default, so a bare
            # `needlestack serve` can't silently disagree with what was indexed.
            store.set_config("last_indexed_model", _captured_ollama_model)

            # Publish globals and signal done atomically under the lock so any thread
            # that observes done=True is guaranteed to also see _store and _embedder set.
            with _index_lock:
                _store = store
                _embedder = embedder
                _setup_mode = False
                store = None  # handed off to _store; don't close in the except path
                _index_state.running = False
                _index_state.done = True

        except Exception as e:
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
    """Re-index all stored roots in a background thread.

    Writes go through a brand-new Store (its own sqlite connection) pointed at the
    same db file, rather than the live `_store` — `_store`'s connection is
    concurrently serving /search, /thumbnail, and /api/sync-status on other
    threads/tasks, and Python's sqlite3 module does not guarantee safe concurrent
    use of one Connection object across threads (only WAL's multiple-connections
    model is; check_same_thread=False on `_store` is there only for the
    setup-wizard's single-handoff case, not for concurrent read+write). WAL makes
    the writer's commits visible to `_store` on its next read, but `_store`'s
    in-process embedding cache needs an explicit invalidation, which happens below.
    """
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
    _captured_db_path = _db_path
    _captured_ollama_url_r = _ollama_url
    _captured_ollama_model_r = _ollama_model

    def _run_all():
        global _embedder
        writer_store = None
        try:
            writer_store = Store(_captured_db_path)
            embedder = Embedder()

            _run_indexing_loop(
                _captured_roots, writer_store, _captured_ollama_url_r,
                _captured_ollama_model_r, embedder,
            )
            writer_store.set_config("last_indexed_model", _captured_ollama_model_r)
            writer_store.close()
            writer_store = None

            with _index_lock:
                _embedder = embedder
                if _store is not None:
                    _store.invalidate_embedding_cache()
                _index_state.running = False
                _index_state.done = True

        except Exception as e:
            if writer_store is not None:
                try:
                    writer_store.close()
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
    """Domain for query expansion: first stored root's domain, or railroad fallback.

    Delegates to Store.primary_domain() — the single implementation of "unknown
    domain name -> fall back to RAILROAD, logged" (see store.py's domains()) — so
    this and _all_domains can't independently drift on that fallback the way the
    three separate copies in this file previously could.
    """
    if _store is None:
        return taxonomy.RAILROAD
    return _store.primary_domain()


def _all_domains() -> list[taxonomy.Domain]:
    """All distinct domains across indexed roots, preserving first-seen order."""
    if _store is None:
        return [taxonomy.RAILROAD]
    return _store.domains()


def _resolve_domains(req: SearchRequest) -> list[taxonomy.Domain]:
    return _all_domains() if req.all_domains else [_primary_domain()]


@app.post("/expand")
async def expand(req: SearchRequest) -> dict:
    from .search import expand_query_with_truncation
    domains = _resolve_domains(req)
    terms, truncated = expand_query_with_truncation(
        req.query, ollama_url=_ollama_url, model=_ollama_model, domains=domains
    )
    return {"terms": terms, "truncated": truncated}


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
