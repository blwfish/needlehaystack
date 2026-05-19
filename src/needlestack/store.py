import io
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    hash        TEXT NOT NULL,
    caption     TEXT,
    embedding   BLOB,
    thumbnail   BLOB,
    indexed_at  TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(
    caption,
    content=images,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO captions_fts(rowid, caption) VALUES (new.id, new.caption);
END;

CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE OF caption ON images BEGIN
    INSERT INTO captions_fts(captions_fts, rowid, caption) VALUES ('delete', old.id, old.caption);
    INSERT INTO captions_fts(rowid, caption) VALUES (new.id, new.caption);
END;

CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO captions_fts(captions_fts, rowid, caption) VALUES ('delete', old.id, old.caption);
END;
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
        self.conn.commit()
        self._embedding_cache: tuple[list[int], list[str], "np.ndarray"] | None = None

    def get_hash(self, path: str) -> str | None:
        row = self.conn.execute(
            "SELECT hash FROM images WHERE path = ?", (path,)
        ).fetchone()
        return row[0] if row else None

    def upsert(
        self,
        path: str,
        hash_: str,
        caption: str,
        embedding: np.ndarray,
        thumbnail: bytes,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO images (path, hash, caption, embedding, thumbnail)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                hash=excluded.hash,
                caption=excluded.caption,
                embedding=excluded.embedding,
                thumbnail=excluded.thumbnail,
                indexed_at=datetime('now')
            """,
            (path, hash_, caption, _enc(embedding), thumbnail),
        )
        self.conn.commit()
        self._embedding_cache = None  # invalidate on write

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

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
                SELECT images.id, images.path, rank
                FROM captions_fts
                JOIN images ON images.id = captions_fts.rowid
                WHERE captions_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
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

    def get_config(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_config(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO config(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

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
