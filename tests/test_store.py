import io
import numpy as np
import pytest
from pathlib import Path
from needlestack.store import Store, _enc, _dec


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def fake_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


# --- embedding codec ---

def test_embedding_roundtrip():
    arr = fake_embedding()
    assert np.allclose(arr, _dec(_enc(arr)), atol=1e-6)


def test_embedding_roundtrip_preserves_norm():
    arr = fake_embedding(42)
    recovered = _dec(_enc(arr))
    assert abs(np.linalg.norm(recovered) - 1.0) < 1e-5


# --- upsert / get_hash ---

def test_upsert_and_get_hash(store):
    store.upsert("/a/b.jpg", "hash1", "a red boxcar", fake_embedding(), b"thumb")
    assert store.get_hash("/a/b.jpg") == "hash1"


def test_get_hash_missing(store):
    assert store.get_hash("/nonexistent.jpg") is None


def test_upsert_updates_existing(store):
    emb = fake_embedding()
    store.upsert("/a/b.jpg", "hash1", "old caption", emb, b"thumb")
    store.upsert("/a/b.jpg", "hash2", "new caption", emb, b"thumb2")
    assert store.get_hash("/a/b.jpg") == "hash2"
    rows = store.get_by_ids(
        [store.conn.execute("SELECT id FROM images WHERE path='/a/b.jpg'").fetchone()[0]]
    )
    assert rows[0]["caption"] == "new caption"


# --- count ---

def test_count_empty(store):
    assert store.count() == 0


def test_count_after_inserts(store):
    for i in range(5):
        store.upsert(f"/img/{i}.jpg", f"h{i}", f"caption {i}", fake_embedding(i), b"t")
    assert store.count() == 5


# --- all_embeddings ---

def test_all_embeddings_empty(store):
    ids, paths, matrix = store.all_embeddings()
    assert ids == []
    assert paths == []
    assert matrix.shape == (0, 512)


def test_all_embeddings_returns_correct_shape(store):
    for i in range(3):
        store.upsert(f"/img/{i}.jpg", f"h{i}", f"caption {i}", fake_embedding(i), b"t")
    ids, paths, matrix = store.all_embeddings()
    assert len(ids) == 3
    assert len(paths) == 3
    assert matrix.shape == (3, 512)


def test_all_embeddings_preserves_values(store):
    emb = fake_embedding(7)
    store.upsert("/x.jpg", "hx", "caption", emb, b"t")
    ids, paths, matrix = store.all_embeddings()
    assert np.allclose(matrix[0], emb, atol=1e-6)


# --- fts_search ---

def test_fts_search_finds_match(store):
    store.upsert("/a.jpg", "h1", "a yellow caboose on the main line", fake_embedding(), b"t")
    store.upsert("/b.jpg", "h2", "a steam locomotive pulling freight", fake_embedding(1), b"t")
    results = store.fts_search("caboose")
    paths = [r[1] for r in results]
    assert "/a.jpg" in paths
    assert "/b.jpg" not in paths


def test_fts_search_or_query(store):
    store.upsert("/a.jpg", "h1", "a caboose at the end of the train", fake_embedding(), b"t")
    store.upsert("/b.jpg", "h2", "a waycar coupled behind the locomotive", fake_embedding(1), b"t")
    store.upsert("/c.jpg", "h3", "a diesel locomotive pulling hoppers", fake_embedding(2), b"t")
    results = store.fts_search('"caboose" OR "waycar"')
    paths = [r[1] for r in results]
    assert "/a.jpg" in paths
    assert "/b.jpg" in paths
    assert "/c.jpg" not in paths


def test_fts_search_no_results(store):
    store.upsert("/a.jpg", "h1", "a steam locomotive", fake_embedding(), b"t")
    assert store.fts_search("tank car") == []


def test_fts_search_malformed_query(store):
    store.upsert("/a.jpg", "h1", "a steam locomotive", fake_embedding(), b"t")
    # malformed FTS5 syntax should return empty, not raise
    results = store.fts_search("AND OR")
    assert isinstance(results, list)


# --- get_by_ids ---

def test_get_by_ids_returns_correct_fields(store):
    store.upsert("/a.jpg", "h1", "a red boxcar", fake_embedding(), b"\xff\xd8")
    row_id = store.conn.execute("SELECT id FROM images WHERE path='/a.jpg'").fetchone()[0]
    rows = store.get_by_ids([row_id])
    assert len(rows) == 1
    assert rows[0]["path"] == "/a.jpg"
    assert rows[0]["caption"] == "a red boxcar"
    assert rows[0]["thumbnail"] == b"\xff\xd8"


def test_get_by_ids_empty_list(store):
    assert store.get_by_ids([]) == []


def test_get_by_ids_missing_id(store):
    assert store.get_by_ids([99999]) == []


# --- thumbnail ---

def test_get_thumbnail(store):
    store.upsert("/a.jpg", "h1", "caption", fake_embedding(), b"thumbdata")
    row_id = store.conn.execute("SELECT id FROM images WHERE path='/a.jpg'").fetchone()[0]
    assert store.get_thumbnail(row_id) == b"thumbdata"


def test_get_thumbnail_missing(store):
    assert store.get_thumbnail(99999) is None
