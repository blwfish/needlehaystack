import io
from collections.abc import Callable
from pathlib import Path

import rawpy
from PIL import Image, ImageOps
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

RAW_EXTENSIONS = {
    ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf", ".dng", ".pef",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp",
} | RAW_EXTENSIONS

THUMBNAIL_SIZE = (300, 300)
MAX_PIXELS = 25_000_000  # 25MP cap; downstream uses max 1024px so higher res is wasted


def _file_hash(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime:.0f}-{stat.st_size}"


def _cap_image(img: Image.Image) -> Image.Image:
    """Downsample in-memory if pixel count exceeds MAX_PIXELS. Original file untouched."""
    w, h = img.size
    if w * h <= MAX_PIXELS:
        return img
    scale = (MAX_PIXELS / (w * h)) ** 0.5
    img.thumbnail((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            # Apply half_size only when the raw sensor exceeds the cap — avoids
            # unnecessarily halving small RAWs while still protecting memory on
            # high-megapixel bodies.
            sizes = raw.sizes
            half = sizes.raw_width * sizes.raw_height > MAX_PIXELS
            rgb = raw.postprocess(use_camera_wb=True, half_size=half)
        return _cap_image(Image.fromarray(rgb))
    else:
        img = Image.open(path)
        w, h = img.size
        if w * h > MAX_PIXELS:
            scale = (MAX_PIXELS / (w * h)) ** 0.5
            # For JPEG, draft() decodes natively at lower resolution (no full load)
            img.draft("RGB", (int(w * scale), int(h * scale)))
        img.load()
        img = ImageOps.exif_transpose(img)
        return _cap_image(img.convert("RGB"))


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
    on_progress: Callable[[int, int, int, int, str], None] | None = None,
) -> tuple[int, int, int]:
    """Index a directory. Returns (indexed, skipped, failed).

    on_progress(total, indexed, skipped, failed, current_filename) is called
    after each image when provided — used by the setup wizard for browser polling.
    """
    images = find_images(root)
    indexed = skipped = failed = 0
    silent = on_progress is not None

    def _notify(current: str = ""):
        if on_progress:
            on_progress(len(images), indexed, skipped, failed, current)

    _notify()

    progress_ctx = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) if not silent else None

    def _run():
        nonlocal indexed, skipped, failed

        task = progress_ctx.add_task("Indexing", total=len(images)) if progress_ctx else None

        for path in images:
            if progress_ctx:
                progress_ctx.update(task, description=f"[cyan]{path.name[:40]}[/cyan]")

            hash_ = _file_hash(path)

            if not force and store.get_hash(str(path)) == hash_:
                skipped += 1
                if progress_ctx:
                    progress_ctx.advance(task)
                _notify(path.name)
                continue

            try:
                image = _load_image(path)
                caption = captioner.caption(image)
                embedding = embedder.embed_image(image)
                thumb = _thumbnail(image)
                store.upsert(str(path), hash_, caption, embedding, thumb)
                indexed += 1
            except Exception as e:
                if progress_ctx:
                    progress_ctx.log(f"[red]Failed[/red] {path.name}: {e}")
                failed += 1

            if progress_ctx:
                progress_ctx.advance(task)
            _notify(path.name)

    if progress_ctx:
        with progress_ctx:
            _run()
    else:
        _run()

    return indexed, skipped, failed
