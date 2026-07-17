import io
import json as _json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from needlestack.indexer import find_images, _file_hash, _thumbnail, _cap_image, _load_image, index_directory, IMAGE_EXTENSIONS, MAX_PIXELS


# --- _cap_image ---

def test_cap_image_small_image_unchanged():
    img = Image.new("RGB", (100, 100))
    result = _cap_image(img)
    assert result.size == (100, 100)


def test_cap_image_at_exact_boundary_not_downsampled():
    # 5000x5000 == MAX_PIXELS exactly — must not downsample (boundary is <=)
    assert 5000 * 5000 == MAX_PIXELS
    img = Image.new("RGB", (5000, 5000))
    result = _cap_image(img)
    assert result.size == (5000, 5000)


def test_cap_image_at_boundary_takes_early_return_branch():
    """The at-boundary image must take the early-return path, not thumbnail().

    Size alone doesn't prove this: Pillow's thumbnail() at target=(5000,5000) is a
    no-op, so both branches produce the same output. Use a mock to detect the branch.
    """
    img = Image.new("RGB", (5000, 5000))
    with patch.object(img, "thumbnail", wraps=img.thumbnail) as mock_thumb:
        _cap_image(img)
    mock_thumb.assert_not_called()  # AT boundary takes the early-return, not thumbnail


def test_cap_image_just_below_boundary_not_downsampled():
    # 4999x5000 = 24_995_000 < MAX_PIXELS
    img = Image.new("RGB", (4999, 5000))
    result = _cap_image(img)
    assert result.size == (4999, 5000)


def test_cap_image_just_over_boundary_downsampled():
    # 5001x5000 = 25_005_000 > MAX_PIXELS
    img = Image.new("RGB", (5001, 5000))
    result = _cap_image(img)
    w, h = result.size
    assert w * h <= MAX_PIXELS


def test_cap_image_large_image_downsampled():
    # 10000x10000 = 100MP — simulate LoC digitization image
    img = Image.new("RGB", (10000, 10000))
    result = _cap_image(img)
    w, h = result.size
    assert w * h <= MAX_PIXELS


def test_cap_image_preserves_aspect_ratio():
    img = Image.new("RGB", (10000, 5000))  # 2:1, 50MP
    result = _cap_image(img)
    w, h = result.size
    assert abs(w / h - 2.0) < 0.02


# --- _load_image ---

def make_jpeg_file(path, width, height):
    Image.new("RGB", (width, height), color=(200, 100, 50)).save(str(path), format="JPEG")


def make_png_file(path, width, height):
    Image.new("RGB", (width, height), color=(50, 100, 200)).save(str(path), format="PNG")


def test_load_image_normal_jpeg_returns_rgb(tmp_path):
    p = tmp_path / "img.jpg"
    make_jpeg_file(p, 800, 600)
    result = _load_image(p)
    assert result is not None
    assert result.mode == "RGB"


def test_load_image_normal_jpeg_size_unchanged(tmp_path):
    p = tmp_path / "img.jpg"
    make_jpeg_file(p, 800, 600)
    result = _load_image(p)
    assert result.width <= 800
    assert result.height <= 600


def test_load_image_oversized_jpeg_capped(tmp_path, monkeypatch):
    # Monkeypatch so we can use a tiny test image (100x100 = 10000 px)
    monkeypatch.setattr("needlestack.indexer.MAX_PIXELS", 5000)
    p = tmp_path / "big.jpg"
    make_jpeg_file(p, 100, 100)
    result = _load_image(p)
    assert result is not None
    w, h = result.size
    assert w * h <= 5000


def test_load_image_oversized_png_capped(tmp_path, monkeypatch):
    # PNG has no draft() path — load then cap via _cap_image
    monkeypatch.setattr("needlestack.indexer.MAX_PIXELS", 5000)
    p = tmp_path / "big.png"
    make_png_file(p, 100, 100)
    result = _load_image(p)
    assert result is not None
    w, h = result.size
    assert w * h <= 5000


def test_load_image_corrupt_raises(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(Exception):
        _load_image(p)


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        _load_image(tmp_path / "nonexistent.jpg")


# --- _load_image RAW branch: half_size threshold (MAX_PIXELS), previously 0% covered ---

def _mock_rawpy_context(raw_width: int, raw_height: int):
    """A mock rawpy.imread(...) context manager reporting the given sensor
    dimensions, whose postprocess() returns a tiny solid RGB array — exercises the
    half_size threshold decision without needing a real RAW file or decoding."""
    mock_raw = MagicMock()
    mock_raw.sizes.raw_width = raw_width
    mock_raw.sizes.raw_height = raw_height
    mock_raw.postprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_raw
    mock_ctx.__exit__.return_value = False
    return mock_ctx, mock_raw


def test_load_image_raw_below_max_pixels_no_half_size(tmp_path):
    p = tmp_path / "small.cr2"
    p.write_bytes(b"fake raw")
    w, h = 5000, 4999
    assert w * h < MAX_PIXELS
    mock_ctx, mock_raw = _mock_rawpy_context(w, h)
    with patch("needlestack.indexer.rawpy.imread", return_value=mock_ctx):
        _load_image(p)
    mock_raw.postprocess.assert_called_once_with(use_camera_wb=True, half_size=False)


def test_load_image_raw_at_exact_max_pixels_no_half_size(tmp_path):
    """Exactly at MAX_PIXELS must NOT trigger half_size — the comparison is strict
    '>', not '>='."""
    p = tmp_path / "exact.cr2"
    p.write_bytes(b"fake raw")
    w, h = 5000, 5000
    assert w * h == MAX_PIXELS
    mock_ctx, mock_raw = _mock_rawpy_context(w, h)
    with patch("needlestack.indexer.rawpy.imread", return_value=mock_ctx):
        _load_image(p)
    mock_raw.postprocess.assert_called_once_with(use_camera_wb=True, half_size=False)


def test_load_image_raw_above_max_pixels_uses_half_size(tmp_path):
    p = tmp_path / "big.cr2"
    p.write_bytes(b"fake raw")
    w, h = 5001, 5000
    assert w * h > MAX_PIXELS
    mock_ctx, mock_raw = _mock_rawpy_context(w, h)
    with patch("needlestack.indexer.rawpy.imread", return_value=mock_ctx):
        _load_image(p)
    mock_raw.postprocess.assert_called_once_with(use_camera_wb=True, half_size=True)


def test_load_image_applies_exif_orientation(tmp_path):
    import io as _io
    p = tmp_path / "rotated.jpg"
    # Landscape image (200x100) with EXIF orientation=6 (90° CW).
    # After exif_transpose the result should be portrait (100x200).
    img = Image.new("RGB", (200, 100), color=(200, 100, 50))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation: 90° CW
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    p.write_bytes(buf.getvalue())
    result = _load_image(p)
    assert result.width < result.height  # was landscape, now portrait


# --- index_directory on_progress ---

def test_index_calls_on_progress_callback(tmp_path):
    from needlestack.store import Store
    img_path = tmp_path / "img" / "test.jpg"
    img_path.parent.mkdir()
    make_test_image(img_path)

    store = Store(tmp_path / "index.db")
    calls = []

    def on_progress(total, indexed, skipped, failed, current):
        calls.append((total, indexed, skipped, failed, current))

    index_directory(img_path.parent, store, make_captioner(), make_embedder(),
                    on_progress=on_progress)

    assert len(calls) >= 1
    totals  = [c[0] for c in calls]
    indexed = [c[1] for c in calls]
    assert all(t == 1 for t in totals)
    assert max(indexed) == 1
    store.close()


# --- find_images ---

def test_find_images_finds_jpegs(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.jpeg").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    result = find_images(tmp_path)
    names = {p.name for p in result}
    assert "a.jpg" in names
    assert "b.jpeg" in names
    assert "c.txt" not in names


def test_find_images_recurses(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "x.png").write_bytes(b"")
    result = find_images(tmp_path)
    assert any(p.name == "x.png" for p in result)


def test_find_images_all_extensions(tmp_path):
    for ext in IMAGE_EXTENSIONS:
        (tmp_path / f"img{ext}").write_bytes(b"")
    result = find_images(tmp_path)
    assert len(result) == len(IMAGE_EXTENSIONS)


def test_find_images_empty_dir(tmp_path):
    assert find_images(tmp_path) == []


def test_find_images_skips_unreadable_subdirectory_without_aborting(tmp_path):
    """A previously fatal case: root.rglob("*") raises PermissionError on the first
    unreadable subdirectory and aborts the whole scan. os.walk's onerror callback
    must log and continue, still finding images in siblings."""
    import os as _os
    good = tmp_path / "good"
    good.mkdir()
    (good / "a.jpg").write_bytes(b"")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "hidden.jpg").write_bytes(b"")
    _os.chmod(blocked, 0o000)
    try:
        result = find_images(tmp_path)
        names = {p.name for p in result}
        assert "a.jpg" in names
    finally:
        _os.chmod(blocked, 0o755)  # restore so pytest can clean up tmp_path


def test_find_images_ignores_broken_symlink(tmp_path):
    target = tmp_path / "does_not_exist.jpg"
    link = tmp_path / "broken_link.jpg"
    link.symlink_to(target)
    assert find_images(tmp_path) == []


# --- _file_hash ---

def test_file_hash_is_string(tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"data")
    assert isinstance(_file_hash(f), str)


def test_file_hash_same_file_is_stable(tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"data")
    assert _file_hash(f) == _file_hash(f)


def test_file_hash_differs_after_content_change(tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"original")
    h1 = _file_hash(f)
    # Write more bytes so file size changes — no sleep needed, size is always different
    f.write_bytes(b"changed_with_extra_bytes")
    h2 = _file_hash(f)
    assert h1 != h2


# --- _thumbnail ---

def make_rgb_image(w=800, h=600) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 64, 32))


def test_thumbnail_returns_bytes():
    img = make_rgb_image()
    result = _thumbnail(img)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_thumbnail_is_valid_jpeg():
    img = make_rgb_image()
    data = _thumbnail(img)
    # JPEG magic bytes
    assert data[:2] == b"\xff\xd8"


def test_thumbnail_respects_max_size():
    img = make_rgb_image(2000, 2000)
    data = _thumbnail(img)
    recovered = Image.open(io.BytesIO(data))
    assert recovered.width <= 300
    assert recovered.height <= 300


# --- index_directory skip logic ---

def make_captioner():
    from needlestack_core.captioner import CaptionResult
    c = MagicMock()
    c.model = "test-model"
    c.caption.return_value = CaptionResult(
        caption="a steam locomotive on the mainline",
        description="a steam locomotive on the mainline",
        is_railroad=True,
        reporting_marks="ATSF 3751",
        equipment="steam locomotive Santa Fe",
        structured_json="{}",
    )
    return c


def make_embedder():
    e = MagicMock()
    e.embed_image.return_value = np.zeros(512, dtype=np.float32)
    return e


def make_test_image(path: Path):
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    img.save(str(path), format="JPEG")


def test_index_skips_already_indexed(tmp_path):
    from needlestack.store import Store
    img_path = tmp_path / "img" / "test.jpg"
    img_path.parent.mkdir()
    make_test_image(img_path)

    store = Store(tmp_path / "index.db")
    captioner = make_captioner()
    embedder = make_embedder()

    # first run
    indexed, skipped, failed = index_directory(img_path.parent, store, captioner, embedder)
    assert indexed == 1
    assert skipped == 0

    # second run — same hash, should skip
    indexed2, skipped2, failed2 = index_directory(img_path.parent, store, captioner, embedder)
    assert indexed2 == 0
    assert skipped2 == 1

    store.close()


def test_index_force_reindexes(tmp_path):
    from needlestack.store import Store
    img_path = tmp_path / "img" / "test.jpg"
    img_path.parent.mkdir()
    make_test_image(img_path)

    store = Store(tmp_path / "index.db")
    captioner = make_captioner()
    embedder = make_embedder()

    index_directory(img_path.parent, store, captioner, embedder)
    indexed2, skipped2, _ = index_directory(img_path.parent, store, captioner, embedder, force=True)
    assert indexed2 == 1
    assert skipped2 == 0

    store.close()


def test_index_recaptions_when_version_stale(tmp_path):
    """The headline upgrade behavior: an unchanged file whose caption came from an
    OLDER model/prompt must be re-captioned, NOT skipped."""
    from needlestack.store import Store
    img_path = tmp_path / "img" / "test.jpg"
    img_path.parent.mkdir()
    make_test_image(img_path)

    store = Store(tmp_path / "index.db")
    embedder = make_embedder()

    # First index with an "old" model.
    old_cap = make_captioner()
    old_cap.model = "old-model"
    indexed1, _, _ = index_directory(img_path.parent, store, old_cap, embedder)
    assert indexed1 == 1

    # Re-index, same file (same hash), but a NEW model → version differs → re-caption.
    new_cap = make_captioner()
    new_cap.model = "new-model"
    indexed2, skipped2, _ = index_directory(img_path.parent, store, new_cap, embedder)
    assert indexed2 == 1   # re-captioned despite unchanged file
    assert skipped2 == 0
    new_cap.caption.assert_called()  # the new captioner actually ran

    store.close()


def test_index_empty_caption_not_stored_and_retried(tmp_path):
    """A captioner that returns an empty caption (e.g. Ollama was down and even the
    fallback returned nothing) must be treated as a FAILURE: the row is not stored with
    the current caption_version, so the next run retries instead of skipping forever."""
    from needlestack.store import Store
    from needlestack_core.captioner import CaptionResult
    img_path = tmp_path / "img" / "test.jpg"
    img_path.parent.mkdir()
    make_test_image(img_path)

    store = Store(tmp_path / "index.db")
    embedder = make_embedder()

    # First run: captioner returns empty → counted failed, NOT stored.
    failing = MagicMock()
    failing.model = "m"
    failing.caption.return_value = CaptionResult(caption="", description="")
    indexed1, skipped1, failed1 = index_directory(img_path.parent, store, failing, embedder)
    assert (indexed1, skipped1, failed1) == (0, 0, 1)
    assert store.get_caption_version(str(img_path)) is None  # nothing persisted

    # Second run with a working captioner: the image is retried, not skipped.
    indexed2, skipped2, _ = index_directory(img_path.parent, store, make_captioner(), embedder)
    assert indexed2 == 1
    assert skipped2 == 0

    store.close()


def test_index_handles_unreadable_file(tmp_path):
    from needlestack.store import Store
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    (img_dir / "bad.jpg").write_bytes(b"not an image")

    store = Store(tmp_path / "index.db")
    captioner = make_captioner()
    embedder = make_embedder()

    indexed, skipped, failed = index_directory(img_dir, store, captioner, embedder)
    assert failed == 1
    assert indexed == 0

    store.close()


# --- _extract_exif / _json_safe / _dms_to_decimal ---

from needlestack.indexer import _extract_exif, _json_safe, _dms_to_decimal  # noqa: E402


def _make_jpeg_with_exif(path: Path, exif_tags: dict, gps_tags: dict | None = None) -> None:
    img = Image.new("RGB", (50, 50), color=(10, 20, 30))
    exif = img.getexif()
    for tag_id, value in exif_tags.items():
        exif[tag_id] = value
    if gps_tags:
        gps_ifd = exif.get_ifd(0x8825)
        for tag_id, value in gps_tags.items():
            gps_ifd[tag_id] = value
        exif[0x8825] = gps_ifd  # Pillow only serializes a sub-IFD once reassigned
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    path.write_bytes(buf.getvalue())


def test_extract_exif_no_exif_returns_empty_string(tmp_path):
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (10, 10)).save(p, format="JPEG")
    assert _extract_exif(p) == ""


def test_extract_exif_unreadable_file_returns_empty_and_does_not_raise(tmp_path):
    p = tmp_path / "not_an_image.cr2"
    p.write_bytes(b"this is not a real RAW file")
    assert _extract_exif(p) == ""  # dropped-with-reason (logged at debug), not a crash


def test_extract_exif_promotes_date_and_camera_fields(tmp_path):
    p = tmp_path / "dated.jpg"
    _make_jpeg_with_exif(p, {
        0x9003: "2020:05:17 14:30:00",  # DateTimeOriginal
        0x010F: "Canon",                # Make
        0x0110: "EOS 90D",              # Model
    })
    data = _json.loads(_extract_exif(p))
    assert data["date_taken"] == "2020:05:17 14:30:00"
    assert data["make"] == "Canon"
    assert data["model"] == "EOS 90D"
    assert "raw" in data  # nothing dropped — full tag set preserved too


def test_extract_exif_unmapped_tag_preserved_under_raw(tmp_path):
    p = tmp_path / "misc.jpg"
    _make_jpeg_with_exif(p, {0x927C: "some maker note data"})  # MakerNote — not promoted
    data = _json.loads(_extract_exif(p))
    assert "MakerNote" in data["raw"]


def test_extract_exif_gps_converted_to_decimal(tmp_path):
    p = tmp_path / "geo.jpg"
    _make_jpeg_with_exif(
        p, {0x9003: "2020:01:01 00:00:00"},
        gps_tags={
            1: "N", 2: (40.0, 26.0, 46.0),   # GPSLatitudeRef, GPSLatitude
            3: "W", 4: (79.0, 58.0, 56.0),   # GPSLongitudeRef, GPSLongitude
        },
    )
    data = _json.loads(_extract_exif(p))
    assert data["gps_lat"] == pytest.approx(40.446111, abs=1e-4)
    assert data["gps_lon"] == pytest.approx(-79.982222, abs=1e-4)


def test_dms_to_decimal_north_east_positive():
    assert _dms_to_decimal((10, 0, 0), "N") == pytest.approx(10.0)
    assert _dms_to_decimal((10, 0, 0), "E") == pytest.approx(10.0)


def test_dms_to_decimal_south_west_negative():
    assert _dms_to_decimal((10, 0, 0), "S") == pytest.approx(-10.0)
    assert _dms_to_decimal((10, 0, 0), "W") == pytest.approx(-10.0)


def test_dms_to_decimal_invalid_input_returns_none():
    assert _dms_to_decimal(("not", "a", "number"), "N") is None
    assert _dms_to_decimal(None, "N") is None


def test_json_safe_bytes_ascii_decodes():
    assert _json_safe(b"ASCII\x00") == "ASCII"


def test_json_safe_bytes_non_ascii_falls_back_to_hex():
    assert _json_safe(b"\xff\xfe") == "fffe"


def test_json_safe_ifdrational_like_becomes_float():
    class _FakeRational:
        numerator = 1
        denominator = 2
    assert _json_safe(_FakeRational()) == 0.5


def test_json_safe_nested_tuple_becomes_list():
    assert _json_safe((1, (2, 3), b"x")) == [1, [2, 3], "x"]


def test_index_directory_stores_exif_json(tmp_path):
    from needlestack.store import Store
    img_path = tmp_path / "img" / "dated.jpg"
    img_path.parent.mkdir()
    _make_jpeg_with_exif(img_path, {0x9003: "2021:06:15 09:00:00"})

    store = Store(tmp_path / "index.db")
    index_directory(img_path.parent, store, make_captioner(), make_embedder())
    exif = store.get_exif(str(img_path))
    assert exif
    assert _json.loads(exif)["date_taken"] == "2021:06:15 09:00:00"
    store.close()
