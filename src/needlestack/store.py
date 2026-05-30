import io
import sqlite3
from pathlib import Path

import numpy as np

# bm25 column weights for the multi-column FTS index: reporting marks (road numbers,
# heralds, builder's plates) are the highest-value railfan search tokens, so they
# outrank generic prose; equipment type/road name sits in between.
FTS_COLUMN_WEIGHTS = (1.0, 8.0, 4.0)  # (caption, reporting_marks, equipment)

# Bump when the on-disk schema changes; stamped into config by _migrate().
SCHEMA_VERSION = 3

# Columns added to `images` beyond the original set. Single source of truth: the
# CREATE TABLE in SCHEMA and the ALTER TABLE migration loop are both generated from
# this dict, so they cannot drift.
_EXTRA_COLUMNS = {
    "reporting_marks": "TEXT",
    "equipment": "TEXT",
    "structured_json": "TEXT",
    "is_railroad": "INTEGER",
    "caption_version": "TEXT",
    "view": "TEXT",
}
_EXTRA_DDL = "".join(f",\n    {name} {decl}" for name, decl in _EXTRA_COLUMNS.items())

_FTS_CREATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(
    caption,
    reporting_marks,
    equipment,
    content=images,
    content_rowid=id
);
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO captions_fts(rowid, caption, reporting_marks, equipment)
    VALUES (new.id, new.caption, new.reporting_marks, new.equipment);
END;

CREATE TRIGGER IF NOT EXISTS images_au
AFTER UPDATE OF caption, reporting_marks, equipment ON images BEGIN
    INSERT INTO captions_fts(captions_fts, rowid, caption, reporting_marks, equipment)
    VALUES ('delete', old.id, old.caption, old.reporting_marks, old.equipment);
    INSERT INTO captions_fts(rowid, caption, reporting_marks, equipment)
    VALUES (new.id, new.caption, new.reporting_marks, new.equipment);
END;

CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO captions_fts(captions_fts, rowid, caption, reporting_marks, equipment)
    VALUES ('delete', old.id, old.caption, old.reporting_marks, old.equipment);
END;
"""

# config + images only. The FTS table and its triggers are created in _migrate(), NOT
# here, so that if a previous run crashed after dropping captions_fts, the next open
# rebuilds and repopulates it — a `CREATE ... IF NOT EXISTS` here would instead recreate
# it empty and hide the loss.
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT UNIQUE NOT NULL,
    hash            TEXT NOT NULL,
    caption         TEXT,
    embedding       BLOB,
    thumbnail       BLOB,
    indexed_at      TEXT DEFAULT (datetime('now')){_EXTRA_DDL}
);
"""


def _enc(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def _dec(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: WAL mode is safe for concurrent reads from
        # multiple threads; the background indexing thread hands the Store to
        # the uvicorn thread after indexing completes.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()
        self._embedding_cache: tuple[list[int], list[str], "np.ndarray"] | None = None

    def _migrate(self) -> None:
        """Bring the DB to the current schema, and create/repair the FTS index.

        Idempotent: a no-op on an already-current DB, the full upgrade on an older one,
        and self-healing if a prior run crashed mid-migration (FTS is owned here, not by
        SCHEMA, so an absent captions_fts is recreated AND repopulated, not left empty).
        """
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(images)")}
        for name, decl in _EXTRA_COLUMNS.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE images ADD COLUMN {name} {decl}")

        # Create the FTS table if absent, or widen it if it's still the original
        # single-column shape. PRAGMA on a missing table returns no rows → empty set →
        # this branch runs and (re)builds it.
        fts_cols = {row[1] for row in self.conn.execute("PRAGMA table_info(captions_fts)")}
        if "reporting_marks" not in fts_cols:
            self.conn.executescript(
                """
                DROP TRIGGER IF EXISTS images_ai;
                DROP TRIGGER IF EXISTS images_au;
                DROP TRIGGER IF EXISTS images_ad;
                DROP TABLE IF EXISTS captions_fts;
                """
            )
            self.conn.executescript(_FTS_CREATE + _FTS_TRIGGERS)
            # Repopulate from the (possibly NULL) image columns; stale rows get
            # re-captioned on the next index pass and the triggers refresh them.
            self.conn.execute("INSERT INTO captions_fts(captions_fts) VALUES('rebuild')")

        self.conn.execute(
            "INSERT INTO config(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def get_hash(self, path: str) -> str | None:
        row = self.conn.execute(
            "SELECT hash FROM images WHERE path = ?", (path,)
        ).fetchone()
        return row[0] if row else None

    def get_caption_version(self, path: str) -> str | None:
        row = self.conn.execute(
            "SELECT caption_version FROM images WHERE path = ?", (path,)
        ).fetchone()
        return row[0] if row else None

    def get_hash_and_version(self, path: str) -> tuple[str | None, str | None]:
        """Fetch hash + caption_version in one query — the indexer's per-image skip
        check needs both, and one row fetch beats two on a large already-indexed tree."""
        row = self.conn.execute(
            "SELECT hash, caption_version FROM images WHERE path = ?", (path,)
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def upsert(
        self,
        path: str,
        hash_: str,
        caption: str,
        embedding: np.ndarray,
        thumbnail: bytes,
        *,
        reporting_marks: str = "",
        equipment: str = "",
        structured_json: str = "",
        is_railroad: int = 0,
        caption_version: str = "",
        view: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO images (
                path, hash, caption, reporting_marks, equipment,
                structured_json, is_railroad, caption_version, view, embedding, thumbnail
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                hash=excluded.hash,
                caption=excluded.caption,
                reporting_marks=excluded.reporting_marks,
                equipment=excluded.equipment,
                structured_json=excluded.structured_json,
                is_railroad=excluded.is_railroad,
                caption_version=excluded.caption_version,
                view=excluded.view,
                embedding=excluded.embedding,
                thumbnail=excluded.thumbnail,
                indexed_at=datetime('now')
            """,
            (path, hash_, caption, reporting_marks, equipment, structured_json,
             int(is_railroad), caption_version, view, _enc(embedding), thumbnail),
        )
        self.conn.commit()
        self._embedding_cache = None  # invalidate on write

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def count_stale_captions(self, current_version: str) -> int:
        """Rows whose captions were produced by an older model/prompt version."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM images "
            "WHERE caption_version IS NULL OR caption_version != ?",
            (current_version,),
        ).fetchone()[0]

    def all_embeddings(self) -> tuple[list[int], list[str], np.ndarray]:
        if self._embedding_cache is not None:
            return self._embedding_cache
        rows = self.conn.execute(
            "SELECT id, path, embedding FROM images WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return [], [], np.empty((0, 512), dtype=np.float32)
        ids = [r[0] for r in rows]
        paths = [r[1] for r in rows]
        matrix = np.stack([_dec(r[2]) for r in rows])
        self._embedding_cache = ids, paths, matrix
        return self._embedding_cache

    def fts_search(self, query: str, limit: int = 100) -> list[tuple[int, str, float]]:
        try:
            rows = self.conn.execute(
                """
                SELECT images.id, images.path, bm25(captions_fts, ?, ?, ?) AS score
                FROM captions_fts
                JOIN images ON images.id = captions_fts.rowid
                WHERE captions_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (*FTS_COLUMN_WEIGHTS, query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # malformed FTS query — treat as no results
            return []
        return rows

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, path, caption, thumbnail FROM images WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return [{"id": r[0], "path": r[1], "caption": r[2], "thumbnail": r[3]} for r in rows]

    def get_thumbnail(self, image_id: int) -> bytes | None:
        row = self.conn.execute(
            "SELECT thumbnail FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return row[0] if row else None

    def get_config(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_config(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO config(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def count_missing(self) -> int:
        """Count indexed entries whose files no longer exist, without deleting them."""
        rows = self.conn.execute("SELECT path FROM images").fetchall()
        return sum(1 for (path,) in rows if not Path(path).exists())

    def remove_missing(self) -> int:
        """Delete index entries whose files no longer exist. Returns count removed."""
        paths = self.conn.execute("SELECT id, path FROM images").fetchall()
        missing_ids = [row[0] for row in paths if not Path(row[1]).exists()]
        if missing_ids:
            placeholders = ",".join("?" * len(missing_ids))
            self.conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", missing_ids)
            self.conn.commit()
        return len(missing_ids)

    def count_unindexed(self, root: Path) -> int:
        """Count image files in root that are not yet in the index."""
        from .indexer import find_images, IMAGE_EXTENSIONS
        indexed = set(
            row[0] for row in self.conn.execute("SELECT path FROM images").fetchall()
        )
        return sum(1 for p in find_images(root) if str(p) not in indexed)

    def close(self) -> None:
        self.conn.close()
