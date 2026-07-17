import io
import json as _json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import rawpy
from PIL import ExifTags, Image, ImageOps

_log = logging.getLogger(__name__)
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from needlestack_core.captioner import Captioner
from needlestack_core.constants import caption_version
from needlestack_core.embedder import Embedder
from .store import Store


def current_caption_version(captioner: Captioner) -> str:
    """Identity of the captioning pipeline. Changing the model or the prompt schema
    changes this string, which auto-invalidates older captions on the next index."""
    return caption_version(captioner.model)

RAW_EXTENSIONS = {
    ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf", ".dng", ".pef",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp",
} | RAW_EXTENSIONS

THUMBNAIL_SIZE = (300, 300)
MAX_PIXELS = 25_000_000  # 25MP cap; downstream uses max 1024px so higher res is wasted


def _file_hash(path: Path) -> str:
    # Uses mtime+size, not content hash — fast, but a same-second same-size in-place
    # edit won't be detected. Acceptable trade-off for indexing throughput.
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


_GPS_IFD_TAG = 0x8825  # EXIF tag id of the "GPSInfo" sub-IFD pointer


def _json_safe(value):
    """Recursively coerce an EXIF value into something json.dumps can serialize,
    without dropping it: PIL's IFDRational -> float, bytes -> text (or hex if not
    decodable), tuples -> lists, dict keys -> str. Everything else is passed
    through, or str()'d as a last resort — no EXIF value is silently discarded."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "numerator") and hasattr(value, "denominator"):  # IFDRational
        try:
            return value.numerator / value.denominator
        except (ZeroDivisionError, TypeError, ValueError):
            return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("ascii").rstrip("\x00")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _dms_to_decimal(dms, ref: str) -> float | None:
    """Convert an EXIF GPS (degrees, minutes, seconds) tuple + hemisphere ref
    ('N'/'S'/'E'/'W') to signed decimal degrees."""
    try:
        degrees, minutes, seconds = (float(v) for v in dms)
    except (TypeError, ValueError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    return -decimal if ref in ("S", "W") else decimal


def _extract_exif(path: Path) -> str:
    """Extract EXIF metadata as a JSON string, or "" if there's none / the format
    isn't supported.

    Disposition (Data-Capture Backward-Chaining Rule): every tag Pillow exposes is
    preserved under "raw" (raw-only) or promoted to a named top-level key
    (extracted: date_taken, gps_lat/lon, make, model, iso, f_number, exposure_time,
    focal_length) — nothing is silently discarded. RAW camera formats are
    dropped-with-reason: Pillow can't open most of them, so extraction is skipped
    and logged rather than silently producing an empty result.
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return ""
            raw = {ExifTags.TAGS.get(k, str(k)): _json_safe(v) for k, v in exif.items()}

            gps_raw: dict = {}
            try:
                gps_ifd = exif.get_ifd(_GPS_IFD_TAG)
                gps_raw = {ExifTags.GPSTAGS.get(k, str(k)): v for k, v in gps_ifd.items()}
            except (KeyError, AttributeError):
                pass
            if gps_raw:
                raw["GPS"] = _json_safe(gps_raw)

            result: dict = {"raw": raw}
            for src_key, out_key in (
                ("DateTimeOriginal", "date_taken"), ("DateTime", "date_taken"),
                ("Make", "make"), ("Model", "model"),
                ("ISOSpeedRatings", "iso"), ("FNumber", "f_number"),
                ("ExposureTime", "exposure_time"), ("FocalLength", "focal_length"),
            ):
                if out_key not in result and src_key in raw:
                    result[out_key] = raw[src_key]

            if gps_raw:
                lat = _dms_to_decimal(gps_raw.get("GPSLatitude"), gps_raw.get("GPSLatitudeRef", ""))
                lon = _dms_to_decimal(gps_raw.get("GPSLongitude"), gps_raw.get("GPSLongitudeRef", ""))
                if lat is not None:
                    result["gps_lat"] = lat
                if lon is not None:
                    result["gps_lon"] = lon

            return _json.dumps(result, ensure_ascii=False)
    except Exception as e:
        _log.debug("EXIF extraction skipped for %s: %s", path.name, e)
        return ""


def find_images(root: Path) -> list[Path]:
    # os.walk with onerror logs-and-continues on a per-directory failure (permission
    # denied, disappeared mid-walk, unreadable NAS mount point) instead of letting
    # root.rglob("*") raise and abort the entire indexing run on one bad directory.
    def _on_walk_error(err: OSError) -> None:
        _log.warning("Skipping unreadable directory during scan: %s", err)

    results: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, onerror=_on_walk_error):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                results.append(p)
    return sorted(results)


def index_directory(
    root: Path,
    store: Store,
    captioner: Captioner,
    embedder: Embedder,
    force: bool = False,
    thorough: bool = False,
    on_progress: Callable[[int, int, int, int, str], None] | None = None,
) -> tuple[int, int, int]:
    """Index a directory. Returns (indexed, skipped, failed).

    on_progress(total, indexed, skipped, failed, current_filename) is called
    after each image when provided — used by the setup wizard for browser polling.
    """
    images = find_images(root)
    indexed = skipped = failed = 0
    silent = on_progress is not None
    version = current_caption_version(captioner)

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

            # Skip only when the file is unchanged AND its caption came from the
            # current model/prompt — a model upgrade or prompt bump re-captions.
            stored_hash, stored_version = store.get_hash_and_version(str(path))
            if not force and stored_hash == hash_ and stored_version == version:
                skipped += 1
                if progress_ctx:
                    progress_ctx.advance(task)
                _notify(path.name)
                continue

            try:
                image = _load_image(path)
                exif_json = _extract_exif(path)
                result = captioner.caption(image, thorough=thorough)
                # An empty caption means captioning failed (e.g. Ollama was unreachable
                # and even the plain-text fallback returned nothing). Treat it as a
                # failure so the row is NOT stored with the current caption_version —
                # otherwise the skip check above would suppress retries forever.
                if not result.caption.strip():
                    raise RuntimeError("captioner returned an empty caption")
                embedding = embedder.embed_image(image)
                thumb = _thumbnail(image)
                store.upsert(
                    str(path), hash_, result.caption, embedding, thumb,
                    # result.description is intentionally omitted: it's already synthesized
                    # into result.caption, so a separate column would be redundant.
                    reporting_marks=result.reporting_marks,
                    equipment=result.equipment,
                    structured_json=result.structured_json,
                    is_railroad=int(result.is_railroad),
                    caption_version=version,
                    view=result.view,
                    exif_json=exif_json,
                )
                indexed += 1
            except Exception as e:
                # Broad catch is intentional: no single image should kill the whole run.
                # Always log via the module logger so failures are visible even on the
                # browser-polling (on_progress) path where progress_ctx is None.
                _log.warning("Failed to index %s: %s", path.name, e)
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

    # captioner.stats is real measured throughput (Ollama's own per-request
    # telemetry), replacing the hardcoded tier-speed guesses in constants.py —
    # only log it if any captioning actually happened this run.
    if captioner.stats.calls:
        _log.info(
            "Captioning: %d calls, %.1fs/call avg, %.1f tok/s",
            captioner.stats.calls, captioner.stats.avg_seconds_per_call,
            captioner.stats.tokens_per_second,
        )

    return indexed, skipped, failed
