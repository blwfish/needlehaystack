"""Diagnostic report for remote debugging."""

import io
import sys
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

from .search import MIN_SCORE, _expand_query, _fts_query


def _section(out: io.StringIO, title: str) -> None:
    out.write(f"\n{'═' * 60}\n{title}\n{'═' * 60}\n")


def _row(out: io.StringIO, label: str, value: str) -> None:
    out.write(f"  {label:<28} {value}\n")


def run(
    db_path: Path,
    query: str | None = None,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llava:13b",
) -> str:
    out = io.StringIO()

    from . import __version__
    out.write(f"needlestack diagnostic report\n")
    out.write(f"Version:   {__version__}\n")
    out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- system ---
    _section(out, "System")
    import platform
    _row(out, "Platform", platform.platform())
    _row(out, "Python", sys.version.split()[0])

    try:
        import torch
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        _row(out, "PyTorch", torch.__version__)
        _row(out, "Compute device", device)
    except ImportError:
        _row(out, "PyTorch", "NOT FOUND")

    # --- ollama ---
    _section(out, "Ollama")
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        _row(out, "Ollama URL", ollama_url)
        _row(out, "Status", "running")
        _row(out, "Models available", ", ".join(models) or "none")
        model_present = any(
            m == ollama_model or m.startswith(ollama_model.split(":")[0] + ":")
            for m in models
        )
        _row(out, f"Model {ollama_model}", "present" if model_present else "NOT FOUND")
    except Exception as e:
        _row(out, "Status", f"NOT REACHABLE — {e}")

    # Test inference
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": ollama_model, "prompt": "Reply with just the word OK.", "stream": False},
            timeout=30.0,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "").strip()
        _row(out, "Test inference", f"OK — model replied: {response_text[:40]!r}")
    except Exception as e:
        _row(out, "Test inference", f"FAILED — {e}")

    # --- index ---
    _section(out, "Index")
    _row(out, "Database", str(db_path))

    if not db_path.exists():
        _row(out, "Status", "NOT FOUND — run: needlestack index <directory>")
    else:
        from .store import Store
        store = Store(db_path)

        count = store.count()
        _row(out, "Images indexed", str(count))

        if count > 0:
            row = store.conn.execute(
                "SELECT MIN(indexed_at), MAX(indexed_at) FROM images"
            ).fetchone()
            _row(out, "First indexed", row[0] or "unknown")
            _row(out, "Last indexed", row[1] or "unknown")

            no_caption = store.conn.execute(
                "SELECT COUNT(*) FROM images WHERE caption IS NULL OR caption = ''"
            ).fetchone()[0]
            _row(out, "Missing captions", str(no_caption))

            no_embedding = store.conn.execute(
                "SELECT COUNT(*) FROM images WHERE embedding IS NULL"
            ).fetchone()[0]
            _row(out, "Missing embeddings", str(no_embedding))

            db_size_mb = db_path.stat().st_size / 1024 / 1024
            _row(out, "Database size", f"{db_size_mb:.1f} MB")

            # Sample captions
            _section(out, "Caption samples (5 random)")
            rows = store.conn.execute(
                "SELECT path, caption FROM images WHERE caption IS NOT NULL ORDER BY RANDOM() LIMIT 5"
            ).fetchall()
            for path, caption in rows:
                out.write(f"\n  {Path(path).name}\n")
                out.write(f"  {(caption or '').strip()[:300]}\n")

            # Railroad vocabulary check
            _section(out, "Railroad vocabulary in captions")
            terms = [
                "locomotive", "steam", "diesel", "boxcar", "box car",
                "tank car", "tanker", "caboose", "flatcar", "hopper",
                "gondola", "freight", "passenger", "depot", "yard",
            ]
            for term in terms:
                n = store.conn.execute(
                    "SELECT COUNT(*) FROM images WHERE caption LIKE ?", (f"%{term}%",)
                ).fetchone()[0]
                bar = "█" * min(n, 40)
                out.write(f"  {term:<16} {n:>4}  {bar}\n")

        # --- query trace ---
        if query:
            _section(out, f"Search trace: {query!r}")

            # Expansion
            out.write("\n  Query expansion:\n")
            try:
                terms = _expand_query(query, ollama_url=ollama_url, model=ollama_model)
                for t in terms:
                    out.write(f"    • {t}\n")
                fts_q = _fts_query(terms)
            except Exception as e:
                out.write(f"    FAILED: {e}\n")
                terms = [query]
                fts_q = f'"{query}"'

            # FTS results (before score threshold)
            out.write("\n  FTS5 matches (before score threshold):\n")
            fts_rows = store.fts_search(fts_q, limit=20)
            if fts_rows:
                for image_id, path, rank in fts_rows[:10]:
                    rows2 = store.get_by_ids([image_id])
                    caption = rows2[0]["caption"][:120] if rows2 else ""
                    out.write(f"    [{rank:+.2f}] {Path(path).name}\n")
                    out.write(f"           {caption}\n")
            else:
                out.write("    (none)\n")

            # CLIP scores
            out.write("\n  CLIP top-5 matches:\n")
            try:
                from .embedder import Embedder
                embedder = Embedder()
                query_vec = embedder.embed_text(query)
                ids, paths, matrix = store.all_embeddings()
                if len(ids) > 0:
                    raw = (matrix @ query_vec).astype(float)
                    top_idx = np.argsort(raw)[::-1][:5]
                    mn, mx = raw.min(), raw.max()
                    for i in top_idx:
                        norm = (raw[i] - mn) / (mx - mn) if mx > mn else 0.0
                        rows2 = store.get_by_ids([ids[i]])
                        caption = rows2[0]["caption"][:100] if rows2 else ""
                        out.write(f"    [{norm:.3f}] {Path(paths[i]).name}\n")
                        out.write(f"           {caption}\n")
                else:
                    out.write("    (index empty)\n")
            except Exception as e:
                out.write(f"    FAILED: {e}\n")

            # Final merged results
            out.write(f"\n  Final results (MIN_SCORE={MIN_SCORE}):\n")
            try:
                from .search import search
                results = search(
                    query, store, embedder,
                    limit=10, ollama_url=ollama_url, ollama_model=ollama_model,
                    preexpanded_terms=terms,
                )
                if results:
                    for r in results:
                        out.write(f"    [{r['score']:.3f}] {Path(r['path']).name}\n")
                        out.write(f"           {(r['caption'] or '')[:100]}\n")
                else:
                    out.write(f"    (no results above threshold {MIN_SCORE})\n")
            except Exception as e:
                out.write(f"    FAILED: {e}\n")

        store.close()

    out.write("\n")
    return out.getvalue()
