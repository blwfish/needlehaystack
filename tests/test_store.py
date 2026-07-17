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


# --- config ---

def test_get_config_missing_key_returns_none(store):
    assert store.get_config("nonexistent") is None


def test_get_config_missing_key_returns_explicit_default(store):
    """Non-None default is returned on a missing key — distinct call form from None default."""
    assert store.get_config("nonexistent", "railroad") == "railroad"


def test_get_roots_domain_falls_back_to_railroad_when_not_set(store):
    """Legacy single-root path: if indexed_domain is absent, the domain defaults to 'railroad'."""
    store.set_config("indexed_root", "/photos")
    # indexed_domain NOT set
    roots = store.get_roots()
    assert roots == [{"path": "/photos", "domain": "railroad"}]


def test_get_roots_returns_empty_on_malformed_json(store):
    """Malformed indexed_roots JSON logs a warning and returns [] rather than raising."""
    store.set_config("indexed_roots", "not valid json {{")
    roots = store.get_roots()
    assert roots == []


def test_set_and_get_config(store):
    store.set_config("indexed_root", "/photos")
    assert store.get_config("indexed_root") == "/photos"


# --- domains() / primary_domain() ---

def test_domains_empty_store_falls_back_to_railroad(store):
    from needlestack_core import taxonomy
    assert store.domains() == [taxonomy.RAILROAD]
    assert store.primary_domain() == taxonomy.RAILROAD


def test_domains_returns_distinct_domains_in_first_seen_order(store):
    from needlestack_core import taxonomy
    store.add_root("/a", "naval")
    store.add_root("/b", "railroad")
    store.add_root("/c", "naval")  # duplicate domain, different root
    assert store.domains() == [taxonomy.NAVAL, taxonomy.RAILROAD]
    assert store.primary_domain() == taxonomy.NAVAL


def test_domains_unknown_name_falls_back_to_railroad_and_logs(store, caplog):
    import logging
    store.add_root("/a", "steampunk")  # not a real domain
    from needlestack_core import taxonomy
    # Logged by taxonomy.resolve_domain (the single source of truth for this
    # fallback), not by needlestack.store — domains() delegates rather than
    # re-implementing the check.
    with caplog.at_level(logging.WARNING, logger="needlestack_core.taxonomy"):
        result = store.domains()
    assert result == [taxonomy.RAILROAD]
    assert any("steampunk" in r.message for r in caplog.records)


# --- count_corrupt_embeddings() ---

def test_count_corrupt_embeddings_none_corrupt(store):
    store.upsert("/a.jpg", "h1", "a caption", fake_embedding(1), b"t")
    assert store.count_corrupt_embeddings() == 0


def test_count_corrupt_embeddings_detects_corrupt_blob(store):
    store.upsert("/a.jpg", "h1", "a caption", fake_embedding(1), b"t")
    store.upsert("/b.jpg", "h2", "a caption", fake_embedding(2), b"t")
    store.conn.execute("UPDATE images SET embedding = ? WHERE path = ?", (b"not-a-valid-npy-blob", "/a.jpg"))
    store.conn.commit()
    assert store.count_corrupt_embeddings() == 1


def test_count_corrupt_embeddings_excludes_null(store):
    """A NULL embedding (never indexed) is not 'corrupt' — it's simply absent."""
    store.conn.execute(
        "INSERT INTO images (path, hash, caption) VALUES ('/no-embed.jpg', 'h', 'c')"
    )
    store.conn.commit()
    assert store.count_corrupt_embeddings() == 0


def test_set_config_overwrites(store):
    store.set_config("key", "first")
    store.set_config("key", "second")
    assert store.get_config("key") == "second"


# --- remove_missing ---

def test_remove_missing_deletes_nonexistent_paths(store, tmp_path):
    real = tmp_path / "real.jpg"
    real.write_bytes(b"x")
    store.upsert(str(real), "h1", "caption", fake_embedding(), b"t")
    store.upsert("/nonexistent/ghost.jpg", "h2", "caption", fake_embedding(1), b"t")
    assert store.count() == 2

    removed = store.remove_missing()
    assert removed == 1
    assert store.count() == 1


def test_remove_missing_keeps_existing_files(store, tmp_path):
    real = tmp_path / "real.jpg"
    real.write_bytes(b"x")
    store.upsert(str(real), "h1", "caption", fake_embedding(), b"t")

    assert store.remove_missing() == 0
    assert store.count() == 1


def test_remove_missing_empty_index(store):
    assert store.remove_missing() == 0


# --- count_unindexed ---

def test_count_unindexed_finds_new_files(store, tmp_path):
    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"x")
    (img_dir / "b.jpg").write_bytes(b"x")

    assert store.count_unindexed(img_dir) == 2
    store.upsert(str(img_dir / "a.jpg"), "h1", "caption", fake_embedding(), b"t")
    assert store.count_unindexed(img_dir) == 1


def test_count_unindexed_ignores_non_images(store, tmp_path):
    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    (img_dir / "readme.txt").write_bytes(b"x")
    assert store.count_unindexed(img_dir) == 0


# --- embedding cache ---

def test_embedding_cache_populated_on_first_call(store):
    store.upsert("/img/a.jpg", "h1", "caption", fake_embedding(), b"t")
    store._embedding_cache = None  # ensure cold

    ids, paths, matrix = store.all_embeddings()

    assert store._embedding_cache is not None
    assert len(ids) == 1


def test_embedding_cache_returns_same_object_on_second_call(store):
    store.upsert("/img/a.jpg", "h1", "caption", fake_embedding(), b"t")

    r1 = store.all_embeddings()
    r2 = store.all_embeddings()
    assert r1 is r2


def test_embedding_cache_invalidated_on_upsert(store):
    store.upsert("/img/a.jpg", "h1", "caption", fake_embedding(), b"t")
    store.all_embeddings()  # populate
    assert store._embedding_cache is not None

    store.upsert("/img/b.jpg", "h2", "caption2", fake_embedding(1), b"t")
    assert store._embedding_cache is None


def test_embedding_cache_repopulates_after_invalidation(store):
    store.upsert("/img/a.jpg", "h1", "caption", fake_embedding(), b"t")
    store.all_embeddings()
    store.upsert("/img/b.jpg", "h2", "caption2", fake_embedding(1), b"t")

    ids, _paths, matrix = store.all_embeddings()
    assert len(ids) == 2
    assert matrix.shape == (2, 512)


# --- structured columns + caption_version ---

def test_upsert_roundtrips_structured_fields(store):
    store.upsert(
        "/a.jpg", "h", "cap", fake_embedding(), b"t",
        reporting_marks="ATSF 3751", equipment="steam locomotive Santa Fe",
        structured_json='{"era":"steam"}', is_railroad=1, caption_version="m:v2",
    )
    row = store.conn.execute(
        "SELECT reporting_marks, equipment, structured_json, is_railroad, caption_version "
        "FROM images WHERE path='/a.jpg'"
    ).fetchone()
    assert row == ("ATSF 3751", "steam locomotive Santa Fe", '{"era":"steam"}', 1, "m:v2")
    assert store.get_caption_version("/a.jpg") == "m:v2"


def test_upsert_roundtrips_exif_json(store):
    store.upsert("/a.jpg", "h", "cap", fake_embedding(), b"t",
                 exif_json='{"date_taken": "2020-01-01T00:00:00"}')
    assert store.get_exif("/a.jpg") == '{"date_taken": "2020-01-01T00:00:00"}'


def test_get_exif_missing(store):
    assert store.get_exif("/nope.jpg") is None


def test_get_caption_version_missing(store):
    assert store.get_caption_version("/nope.jpg") is None


def test_get_hash_and_version_combined(store):
    store.upsert("/a.jpg", "h9", "c", fake_embedding(), b"t", caption_version="m:v2")
    assert store.get_hash_and_version("/a.jpg") == ("h9", "m:v2")


def test_get_hash_and_version_missing(store):
    assert store.get_hash_and_version("/nope.jpg") == (None, None)


def test_count_stale_captions(store):
    store.upsert("/cur.jpg", "h", "c", fake_embedding(), b"t", caption_version="m:v2")
    store.upsert("/old.jpg", "h", "c", fake_embedding(1), b"t", caption_version="m:v1")
    assert store.count_stale_captions("m:v2") == 1   # only /old.jpg is stale
    assert store.count_stale_captions("m:v3") == 2    # both stale vs a newer version


# --- weighted multi-column FTS ---

def test_fts_bm25_weights_marks_above_prose(store):
    # img A: the mark appears buried in generic prose (low-weight caption column)
    store.upsert(
        "/prose.jpg", "h1", "a train rolls past lots of generic scenery and ATSF appears "
        "amid many unrelated descriptive filler words here there everywhere",
        fake_embedding(), b"t",
    )
    # img B: the mark is in the high-weight reporting_marks column
    store.upsert(
        "/marked.jpg", "h2", "a train", fake_embedding(1), b"t",
        reporting_marks="ATSF",
    )
    results = store.fts_search('"ATSF"')
    paths = [r[1] for r in results]
    assert paths[0] == "/marked.jpg"   # reporting-mark hit outranks buried-prose hit
    assert "/prose.jpg" in paths


# --- migration from the original single-column schema ---

_OLD_SCHEMA = """
CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL, hash TEXT NOT NULL, caption TEXT,
    embedding BLOB, thumbnail BLOB, indexed_at TEXT DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE captions_fts USING fts5(caption, content=images, content_rowid=id);
CREATE TRIGGER images_ai AFTER INSERT ON images BEGIN
    INSERT INTO captions_fts(rowid, caption) VALUES (new.id, new.caption);
END;
"""


def test_migrate_upgrades_old_db(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO images (path, hash, caption, embedding, thumbnail) VALUES (?,?,?,?,?)",
        ("/old.jpg", "h", "an old vintage caboose photo", _enc(fake_embedding()), b"t"),
    )
    conn.commit()
    conn.close()

    # Opening with the current Store must migrate in place.
    s = Store(db)
    cols = {row[1] for row in s.conn.execute("PRAGMA table_info(images)")}
    assert {"reporting_marks", "equipment", "structured_json",
            "is_railroad", "caption_version", "view", "exif_json"} <= cols

    fts_cols = {row[1] for row in s.conn.execute("PRAGMA table_info(captions_fts)")}
    assert "reporting_marks" in fts_cols      # FTS widened to multi-column

    # Old row preserved and still searchable after the FTS rebuild.
    assert s.count() == 1
    assert s.fts_search('"caboose"')[0][1] == "/old.jpg"
    # Old row has no caption_version → counts as stale, triggering re-caption later.
    assert s.get_caption_version("/old.jpg") is None
    assert s.count_stale_captions("m:v2") == 1
    s.close()


def test_migrate_is_idempotent_on_fresh_db(store):
    """A freshly created Store is already current — re-running _migrate is a no-op."""
    store.upsert("/a.jpg", "h", "a caboose", fake_embedding(), b"t",
                 reporting_marks="ATSF", caption_version="m:v2")
    store._migrate()  # should not raise or lose data
    assert store.count() == 1
    assert store.fts_search('"caboose"')


def test_migrate_logs_schema_version_transition(tmp_path, caplog):
    import logging
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO config (key, value) VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()

    from needlestack.store import SCHEMA_VERSION
    with caplog.at_level(logging.INFO, logger="needlestack.store"):
        s = Store(db)
    assert any("v1" in r.message and f"v{SCHEMA_VERSION}" in r.message
               for r in caplog.records)
    s.close()


def test_migrate_does_not_log_on_matching_version(store, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="needlestack.store"):
        store._migrate()  # fresh store is already at SCHEMA_VERSION
    assert not any("Migrating index schema" in r.message for r in caplog.records)


# --- FTS_COLUMN_WEIGHTS: caption vs equipment ordering (only marks-vs-caption was pinned) ---

def test_fts_bm25_weights_equipment_above_caption(store):
    """equipment (weight 4.0) must rank above plain caption prose (weight 1.0) when
    the equipment column is the only match — pins the middle tier, not just the
    marks-vs-caption edge (which the existing test above already covers)."""
    # img A: the term appears buried in generic prose (low-weight caption column)
    store.upsert(
        "/prose.jpg", "h1", "a train rolls past lots of generic scenery and boxcar appears "
        "amid many unrelated descriptive filler words here there everywhere",
        fake_embedding(1), b"t",
    )
    # img B: the term is in the mid-weight equipment column
    store.upsert("/equip.jpg", "h2", "a train", fake_embedding(2), b"t", equipment="boxcar")
    rows = store.fts_search('"boxcar"')
    # rank is negative BM25 (more negative = better match, per fts_search's own convention).
    ranked_paths = [path for _id, path, _rank in sorted(rows, key=lambda r: r[2])]
    assert ranked_paths[0] == "/equip.jpg"
