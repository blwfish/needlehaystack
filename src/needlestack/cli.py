import sys
import threading
import time
import webbrowser
from pathlib import Path

import click
import uvicorn
from rich.console import Console

from .constants import DEFAULT_MODEL, OLLAMA_URL as DEFAULT_OLLAMA, MODEL_PRESETS
from .taxonomy import DOMAINS

DEFAULT_DB = Path.home() / ".needlestack" / "index.db"
DEFAULT_PORT = 8484
UI_PATH = Path(__file__).parent / "ui"

console = Console()


@click.group()
@click.version_option(package_name="needlestack")
def main() -> None:
    """needlestack — find photos by describing what's in them."""


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="Index database path")
@click.option("--model", default=None, help="Ollama vision model (overrides --preset)")
@click.option("--preset", default=None,
              type=click.Choice(list(MODEL_PRESETS), case_sensitive=False),
              help="Model tier: fast / balanced (default) / quality")
@click.option("--ollama", default=DEFAULT_OLLAMA, show_default=True, help="Ollama base URL")
@click.option("--force", is_flag=True, help="Re-index already-indexed files")
@click.option("--thorough", is_flag=True, help="Add a dedicated OCR pass for reporting marks (~2x slower)")
@click.option("--domain", default="railroad",
              type=click.Choice(list(DOMAINS), case_sensitive=False),
              help="Photo collection domain (default: railroad)")
def index(directory: Path, db: str, model: str | None, preset: str | None,
          ollama: str, force: bool, thorough: bool, domain: str) -> None:
    """Index a directory of images."""
    from .captioner import Captioner
    from .embedder import Embedder
    from .indexer import index_directory
    from .store import Store
    from .taxonomy import get_domain

    if model and preset:
        console.print("[red]Error:[/red] --model and --preset are mutually exclusive.")
        sys.exit(1)
    resolved_model = model or MODEL_PRESETS.get(preset or "", DEFAULT_MODEL)
    selected_domain = get_domain(domain)
    captioner = Captioner(model=resolved_model, base_url=ollama, domain=selected_domain)
    ok, msg = captioner.check()
    if not ok:
        console.print(f"[red]Error:[/red] {msg}")
        sys.exit(1)

    db_path = Path(db)
    store = Store(db_path)
    embedder = Embedder()

    console.print(f"[green]Indexing[/green] {directory}")
    console.print(
        f"  model: [cyan]{resolved_model}[/cyan]  "
        f"domain: [cyan]{domain}[/cyan]  "
        f"db: [cyan]{db_path}[/cyan]"
    )
    if store.count() > 0 and not force:
        console.print(f"  resuming — [dim]{store.count()} already indexed[/dim]")

    indexed, skipped, failed = index_directory(
        directory, store, captioner, embedder, force=force, thorough=thorough
    )
    store.set_config("indexed_root", str(directory.resolve()))
    store.set_config("indexed_domain", domain)

    console.print(
        f"\n[green]Done.[/green]  "
        f"indexed: {indexed}  skipped: {skipped}  failed: {failed}"
    )
    store.close()
    captioner.close()


@main.command()
@click.argument("query", required=False)
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="Index database path")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model")
@click.option("--ollama", default=DEFAULT_OLLAMA, show_default=True, help="Ollama base URL")
@click.option("--out", "output", default=None, help="Save report to file")
def doctor(query: str | None, db: str, model: str, ollama: str, output: str | None) -> None:
    """Run diagnostics. Optionally trace a specific query."""
    from .doctor import run

    report = run(
        db_path=Path(db),
        query=query,
        ollama_url=ollama,
        ollama_model=model,
    )

    console.print(report)

    if output:
        Path(output).write_text(report)
        console.print(f"[green]Report saved to[/green] {output}")
    else:
        console.print(
            "[dim]Tip: add --out report.txt to save a file you can share for support.[/dim]"
        )


@main.command()
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="Index database path")
@click.option("--port", default=DEFAULT_PORT, show_default=True, help="Server port")
@click.option("--model", default=None, help="Ollama model for query expansion (overrides --preset)")
@click.option("--preset", default=None,
              type=click.Choice(list(MODEL_PRESETS), case_sensitive=False),
              help="Model tier: fast / balanced (default) / quality")
@click.option("--ollama", default=DEFAULT_OLLAMA, show_default=True, help="Ollama base URL")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def serve(db: str, port: int, model: str | None, preset: str | None,
          ollama: str, no_browser: bool) -> None:
    """Start the search server and open the browser."""
    from .embedder import Embedder
    from .server import app, init
    from .store import Store

    db_path = Path(db)
    setup_mode = not db_path.exists()

    if setup_mode:
        console.print("[green]needlestack[/green]  no index yet — opening setup wizard")
        store = None
        embedder = None
    else:
        store = Store(db_path)

        # Silently remove index entries for deleted files
        removed = store.remove_missing()
        if removed:
            console.print(f"[dim]Removed {removed} deleted files from index.[/dim]")

        n = store.count()
        console.print(f"[green]needlestack[/green]  {n} photos indexed")
        embedder = Embedder()

    resolved_model = model or MODEL_PRESETS.get(preset or "", DEFAULT_MODEL)
    init(store, embedder, UI_PATH, db_path=db_path,
         ollama_url=ollama, ollama_model=resolved_model, setup_mode=setup_mode)

    # If port is in use, check if it's already us — if so, just open the browser
    import socket
    import httpx as _httpx

    def _port_free(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) != 0

    if not _port_free(port):
        try:
            resp = _httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0)
            if "needlestack" in resp.text.lower():
                console.print(f"[dim]needlestack already running on port {port} — opening browser.[/dim]")
                webbrowser.open(f"http://localhost:{port}")
                return
        except Exception:
            pass
        # Something else owns the port — find the next free one
        original = port
        for candidate in range(port + 1, port + 20):
            if _port_free(candidate):
                port = candidate
                break
        else:
            console.print(f"[red]Port {original} is in use and no free port found nearby.[/red]")
            sys.exit(1)
        console.print(f"[yellow]Port {original} in use — using {port} instead.[/yellow]")

    url = f"http://localhost:{port}"
    if not no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    console.print(f"Listening on [cyan]{url}[/cyan]  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    if store:
        store.close()
