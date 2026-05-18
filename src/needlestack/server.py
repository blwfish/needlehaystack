import base64
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import search as search_module
from .embedder import Embedder
from .store import Store

app = FastAPI(title="needlestack")

_store: Store | None = None
_embedder: Embedder | None = None
_ui_path: Path | None = None
_ollama_url: str = "http://localhost:11434"
_ollama_model: str = "llava:13b"


def init(store: Store, embedder: Embedder, ui_path: Path, ollama_url: str, ollama_model: str) -> None:
    global _store, _embedder, _ui_path, _ollama_url, _ollama_model
    _store = store
    _embedder = embedder
    _ui_path = ui_path
    _ollama_url = ollama_url
    _ollama_model = ollama_model


class SearchRequest(BaseModel):
    query: str
    limit: int = 40
    terms: list[str] | None = None  # pre-expanded terms, skips expansion if provided


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return (_ui_path / "index.html").read_text()


@app.post("/expand")
async def expand(req: SearchRequest) -> dict:
    from .search import _expand_query
    terms = _expand_query(req.query, ollama_url=_ollama_url, model=_ollama_model)
    return {"terms": terms}


@app.post("/search")
async def search(req: SearchRequest) -> list[dict]:
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
    data = _store.get_thumbnail(image_id)
    if not data:
        raise HTTPException(404)
    return Response(content=data, media_type="image/jpeg")


@app.post("/open/{image_id}")
async def open_image(image_id: int) -> dict:
    rows = _store.get_by_ids([image_id])
    if not rows:
        raise HTTPException(404)
    subprocess.Popen(["open", "-R", rows[0]["path"]])  # reveal in Finder
    return {"status": "ok"}
