import io
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


def test_load_image_corrupt_returns_none(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    assert _load_image(p) is None


def test_load_image_missing_file_returns_none(tmp_path):
    assert _load_image(tmp_path / "nonexistent.jpg") is None


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
    import time; time.sleep(0.01)
    f.write_bytes(b"changed")
    # mtime or size changed
    h2 = _file_hash(f)
    # size changed, so hash must differ
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
    c = MagicMock()
    c.caption.return_value = "a steam locomotive on the mainline"
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
