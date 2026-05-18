import io
from pathlib import Path

import rawpy
from PIL import Image
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .captioner import Captioner
from .embedder import Embedder
from .store import Store

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp",
    ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf", ".dng", ".pef",
}

THUMBNAIL_SIZE = (300, 300)


def _file_hash(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime:.0f}-{stat.st_size}"


def _load_image(path: Path) -> Image.Image | None:
    try:
        if path.suffix.lower() in {
            ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf", ".dng", ".pef"
        }:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            return Image.fromarray(rgb)
        else:
            img = Image.open(path)
            img.load()
            return img.convert("RGB")
    except Exception:
        return None


def _thumbnail(image: Image.Image) -> bytes:
    thumb = image.copy()
    thumb.thumbnail(THUMBNAIL_SIZE)
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def find_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def index_directory(
    root: Path,
    store: Store,
    captioner: Captioner,
    embedder: Embedder,
    force: bool = False,
) -> tuple[int, int, int]:
    """Index a directory. Returns (indexed, skipped, failed)."""
    images = find_images(root)
    indexed = skipped = failed = 0

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Indexing", total=len(images))

        for path in images:
            progress.update(task, description=f"[cyan]{path.name[:40]}[/cyan]")
            hash_ = _file_hash(path)

            if not force and store.get_hash(str(path)) == hash_:
                skipped += 1
                progress.advance(task)
                continue

            image = _load_image(path)
            if image is None:
                failed += 1
                progress.advance(task)
                continue

            try:
                caption = captioner.caption(image)
                embedding = embedder.embed_image(image)
                thumb = _thumbnail(image)
                store.upsert(str(path), hash_, caption, embedding, thumb)
                indexed += 1
            except Exception as e:
                progress.log(f"[red]Failed[/red] {path.name}: {e}")
                failed += 1

            progress.advance(task)

    return indexed, skipped, failed
