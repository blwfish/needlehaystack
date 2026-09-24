import io
import json as _json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from needlestack_core import taxonomy

_log = logging.getLogger(__name__)

# bm25 column weights for the multi-column FTS index: reporting marks (road numbers,
# heralds, builder's plates) are the highest-value railfan search tokens, so they
# outrank generic prose; equipment type/road name sits in between.
FTS_COLUMN_WEIGHTS = (1.0, 8.0, 4.0)  # (caption, reporting_marks, equipment)

# Bump when the on-disk schema changes; stamped into config by _migrate().
SCHEMA_VERSION = 4


def _embedding_dim() -> int:
    """Embedding vector width, sourced from Embedder.dim (a class attribute — no
    model load required) so an empty result matrix always matches what the actual
    embedder produces. Imported lazily: needlestack_core.embedder pulls in
    torch/open_clip at module scope, and Store must stay importable/usable (e.g. by
    doctor.py's no-query path) on a machine where those aren't installed."""
    from needlestack_core.embedder import Embedder
    return Embedder.dim

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
    # Raw + normalized EXIF metadata as a JSON blob (see indexer._extract_exif) —
    # LONGTEXT-equivalent (SQLite TEXT has no length cap) for an external, unbounded
    # source. Nothing observed in the file's EXIF is dropped: recognized fields are
    # promoted to named keys, everything else survives under "raw".
    "exif_json": "TEXT",
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


def _try_dec(blob: bytes, path: str) -> np.ndarray | None:
    """Decode an embedding BLOB, or None (logged) if it's corrupt. Single source of
    truth for the decode-or-skip check shared by all_embeddings() and
    count_corrupt_embeddings()."""
    try:
        return _dec(blob)
    except (ValueError, OSError):
        _log.warning("Corrupt embedding BLOB for %s — skipping", path)
        return None


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: server.py's /search and /thumbnail run on the
        # event-loop thread while /api/sync-status offloads count_missing() to a
        # separate worker thread via asyncio.to_thread -- both genuinely touch
        # this same Connection object from different OS threads, concurrently.
        # This is safe ONLY because sqlite3.threadsafety == 3 (SQLite compiled
        # "serialized": the C library itself is safe for concurrent multi-thread
        # use of one connection) -- verified below rather than assumed, since
        # that depends on how the platform's SQLite library was built and isn't
        # guaranteed across all Python/SQLite distributions. It covers concurrent
        # *reads* only: server.py deliberately never writes through this shared
        # connection from a background thread -- reindex_all/start_indexing open
        # their own separate writer Store/connection instead, so this connection
        # only ever has one writer (the main thread that constructed it) and
        # WAL's documented multi-connection model handles that writer's commits
        # becoming visible to this connection's next read.
        if sqlite3.threadsafety < 3:
            raise RuntimeError(
                "needlestack requires a SQLite library compiled in 'serialized' "
                "threading mode (sqlite3.threadsafety == 3) because the Store "
                "connection is shared read-only across threads; this Python's "
                f"sqlite3 module reports threadsafety={sqlite3.threadsafety}, "
                "which cannot safely support that."
            )
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()
        self._embedding_cache: tuple[list[int], list[str], "np.ndarray"] | None = None
        self._roots_cache: list[dict] | None = None

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

        stored_version = self.conn.execute(
            "SELECT value FROM config WHERE key='schema_version'"
        ).fetchone()
        if stored_version is not None and stored_version[0] != str(SCHEMA_VERSION):
            _log.info(
                "Migrating index schema v%s -> v%s", stored_version[0], SCHEMA_VERSION
            )
        self.conn.execute(
            "INSERT INTO config(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

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
        exif_json: str = "",
        commit: bool = True,
    ) -> None:
        """commit=False lets a caller doing many upserts in a row (bulk indexing)
        batch commits instead of fsyncing after every single row -- WAL mode
        already gives durable, isolated writes without a commit-per-row; the
        caller is responsible for eventually committing (directly on .conn, or
        via a later commit=True call)."""
        self.conn.execute(
            """
            INSERT INTO images (
                path, hash, caption, reporting_marks, equipment,
                structured_json, is_railroad, caption_version, view, exif_json,
                embedding, thumbnail
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                hash=excluded.hash,
                caption=excluded.caption,
                reporting_marks=excluded.reporting_marks,
                equipment=excluded.equipment,
                structured_json=excluded.structured_json,
                is_railroad=excluded.is_railroad,
                caption_version=excluded.caption_version,
                view=excluded.view,
                exif_json=excluded.exif_json,
                embedding=excluded.embedding,
                thumbnail=excluded.thumbnail,
                indexed_at=datetime('now')
            """,
            (path, hash_, caption, reporting_marks, equipment, structured_json,
             int(is_railroad), caption_version, view, exif_json, _enc(embedding), thumbnail),
        )
        if commit:
            self.conn.commit()
        self.invalidate_embedding_cache()

    def get_exif(self, path: str) -> str | None:
        row = self.conn.execute(
            "SELECT exif_json FROM images WHERE path = ?", (path,)
        ).fetchone()
        return row[0] if row else None

    def invalidate_embedding_cache(self) -> None:
        """Drop the cached embedding matrix. Needed whenever rows change through a
        path this Store instance didn't itself write through — e.g. a reindex run
        writing via a separate Store/connection pointed at the same db file (WAL
        makes the new rows visible to this connection, but not to this instance's
        in-process cache)."""
        self._embedding_cache = None

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
            return [], [], np.empty((0, _embedding_dim()), dtype=np.float32)
        ids = []
        paths = []
        vecs = []
        skipped = 0
        for r in rows:
            vec = _try_dec(r[2], r[1])
            if vec is None:
                skipped += 1
                continue
            vecs.append(vec)
            ids.append(r[0])
            paths.append(r[1])
        if skipped:
            # _try_dec already warned per-row; this summary line is what makes
            # "the matrix search()/doctor.py just got is missing N embeddings"
            # visible from a single call site instead of only from scattered
            # per-row log lines a caller would have to notice and count itself.
            _log.warning(
                "all_embeddings: skipped %d corrupt embedding(s) out of %d rows",
                skipped, len(rows),
            )
        if not ids:
            return [], [], np.empty((0, _embedding_dim()), dtype=np.float32)
        matrix = np.stack(vecs)
        self._embedding_cache = ids, paths, matrix
        return self._embedding_cache

    def count_corrupt_embeddings(self) -> int:
        """Count stored embeddings that fail to decode, without building the full
        matrix — a lightweight check for the doctor health report. Corrupt rows are
        neither NULL (so count_missing-style NULL checks miss them) nor usable, so
        they'd otherwise be invisible to every diagnostic."""
        rows = self.conn.execute(
            "SELECT path, embedding FROM images WHERE embedding IS NOT NULL"
        ).fetchall()
        return sum(1 for path, blob in rows if _try_dec(blob, path) is None)

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
        # Returns only the fields needed by search results and the thumbnail endpoint.
        # reporting_marks, equipment, view, is_railroad, structured_json are intentionally
        # omitted — they're stored for FTS weighting and re-indexing checks, not served
        # to the UI. Extend this SELECT if a caller ever needs them.
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

    @staticmethod
    def _is_verifiably_missing(path: str) -> bool:
        """True only if `path` is absolute AND doesn't exist.

        A relative stored path can't be safely judged "missing" -- it would be
        resolved against whatever the *current* process's cwd happens to be,
        which has no guaranteed relationship to the cwd `needlestack index` was
        run from. Indexing now always stores absolute paths (see cli.py's
        `resolve_path=True`), so this only matters for rows written before that
        fix; treating an unverifiable relative path as "present" rather than
        "missing" means a stale cwd can never cause files to be silently and
        permanently deleted from the index.
        """
        p = Path(path)
        if not p.is_absolute():
            return False
        try:
            return not p.exists()
        except OSError as e:
            # A transient OS-level error (encoding issue, permission denial,
            # unmounted network share) is not proof the file is gone -- treat
            # it as "can't verify, so don't delete" rather than letting the
            # exception propagate and abort count_missing/remove_missing for
            # every other row too.
            _log.warning("Could not check existence of %s: %s", path, e)
            return False

    def count_missing(self) -> int:
        """Count indexed entries whose files no longer exist, without deleting them."""
        rows = self.conn.execute("SELECT path FROM images").fetchall()
        return sum(1 for (path,) in rows if self._is_verifiably_missing(path))

    def remove_missing(self) -> int:
        """Delete index entries whose files no longer exist. Returns count removed."""
        paths = self.conn.execute("SELECT id, path FROM images").fetchall()
        missing_ids = [row[0] for row in paths if self._is_verifiably_missing(row[1])]
        if missing_ids:
            placeholders = ",".join("?" * len(missing_ids))
            self.conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", missing_ids)
            self.conn.commit()
            self.invalidate_embedding_cache()  # deleted rows may be in the cache
        return len(missing_ids)

    def get_roots(self) -> list[dict]:
        """Return all indexed roots as [{"path": str, "domain": str}].

        Falls back to the legacy indexed_root / indexed_domain keys so old
        single-root databases work without migration. Cached in-process (like
        all_embeddings()) since this data only changes via set_roots()/add_root(),
        but was previously re-read from config and re-JSON-parsed on every
        /search and /expand request.
        """
        if self._roots_cache is not None:
            return self._roots_cache
        raw = self.get_config("indexed_roots")
        if raw:
            try:
                self._roots_cache = _json.loads(raw)
            except _json.JSONDecodeError:
                _log.warning("indexed_roots config is malformed JSON — treating as empty")
                self._roots_cache = []
            return self._roots_cache
        root = self.get_config("indexed_root")
        if root:
            domain = self.get_config("indexed_domain", "railroad")
            self._roots_cache = [{"path": root, "domain": domain}]
            return self._roots_cache
        self._roots_cache = []
        return self._roots_cache

    def set_roots(self, roots: list[dict]) -> None:
        self.set_config("indexed_roots", _json.dumps(roots))
        self._roots_cache = None

    def add_root(self, path: str, domain: str) -> None:
        """Register a root directory. Updates domain if root is already known."""
        roots = self.get_roots()
        for r in roots:
            if r["path"] == path:
                r["domain"] = domain
                self.set_roots(roots)
                return
        roots.append({"path": path, "domain": domain})
        self.set_roots(roots)

    def domains(self) -> list[taxonomy.Domain]:
        """All distinct domains across indexed roots, preserving first-seen order.

        Delegates the "unrecognized domain name -> fall back to RAILROAD, logged"
        decision to taxonomy.resolve_domain() — the single source of truth for that
        fallback, so it can't drift out of sync between call sites the way it
        previously did across server.py's _primary_domain, _all_domains, and
        reindex_all (each re-implementing the same DOMAINS.get(..., RAILROAD)).
        """
        roots = self.get_roots()
        seen: set[str] = set()
        result: list[taxonomy.Domain] = []
        for r in roots:
            name = r["domain"]
            if name not in seen:
                seen.add(name)
                result.append(taxonomy.resolve_domain(name))
        return result or [taxonomy.RAILROAD]

    def primary_domain(self) -> taxonomy.Domain:
        """Domain for single-domain contexts: the first indexed root's domain."""
        return self.domains()[0]

    def count_unindexed(self, root: Path) -> int:
        """Count image files in root that are not yet in the index."""
        from .indexer import find_images, IMAGE_EXTENSIONS
        indexed = set(
            row[0] for row in self.conn.execute("SELECT path FROM images").fetchall()
        )
        return sum(1 for p in find_images(root) if str(p) not in indexed)

    def close(self) -> None:
        self.conn.close()
