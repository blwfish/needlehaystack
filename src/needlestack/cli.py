import sys
import threading
import time
import webbrowser
from pathlib import Path

import click
import uvicorn
from rich.console import Console

DEFAULT_DB = Path.home() / ".needlestack" / "index.db"
DEFAULT_MODEL = "llava:13b"
DEFAULT_OLLAMA = "http://localhost:11434"
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
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Ollama vision model")
@click.option("--ollama", default=DEFAULT_OLLAMA, show_default=True, help="Ollama base URL")
@click.option("--force", is_flag=True, help="Re-index already-indexed files")
def index(directory: Path, db: str, model: str, ollama: str, force: bool) -> None:
    """Index a directory of images."""
    from .captioner import Captioner
    from .embedder import Embedder
    from .indexer import index_directory
    from .store import Store

    captioner = Captioner(model=model, base_url=ollama)
    ok, msg = captioner.check()
    if not ok:
        console.print(f"[red]Error:[/red] {msg}")
        sys.exit(1)

    db_path = Path(db)
    store = Store(db_path)
    embedder = Embedder()

    console.print(f"[green]Indexing[/green] {directory}")
    console.print(f"  model: [cyan]{model}[/cyan]  db: [cyan]{db_path}[/cyan]")
    if store.count() > 0 and not force:
        console.print(f"  resuming — [dim]{store.count()} already indexed[/dim]")

    indexed, skipped, failed = index_directory(directory, store, captioner, embedder, force=force)
    store.set_config("indexed_root", str(directory.resolve()))

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
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model for query expansion")
@click.option("--ollama", default=DEFAULT_OLLAMA, show_default=True, help="Ollama base URL")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def serve(db: str, port: int, model: str, ollama: str, no_browser: bool) -> None:
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

    init(store, embedder, UI_PATH, db_path=db_path,
         ollama_url=ollama, ollama_model=model, setup_mode=setup_mode)

    url = f"http://localhost:{port}"
    if not no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    console.print(f"Listening on [cyan]{url}[/cyan]  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    store.close()
