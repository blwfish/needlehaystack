# Testing notes — known gaps

Status as of branch `strengthen-railroad-captions` (124 tests, ~60% line coverage).

The suite is strong on pure logic (`store.py`, `taxonomy.py`, `constants.py` 100%;
`search.py` 99%; `indexer.py` 94%; `captioner.py` 98%) and now covers the behavior
added by the railroad-caption work (structured captioning, version-stale re-caption,
weighted FTS, migration, sync-status stale reporting).

The items below are **deliberately left undone** — each needs a real model, a real
binary asset, a display/subprocess, or a thread harness, so none is a quick win.
Ordered by tractability.

## Tractable (no special hardware) — do these first

- **`doctor.py` — 0% coverage.** No `tests/test_doctor.py` exists. `doctor.run()` is the
  primary support/debugging tool and every error handler (Ollama unreachable, DB not
  found, embedding failures, query-trace exceptions) is unexercised. Testable with a
  temp DB + `patch("needlestack.doctor.httpx...")`. Highest value of the untested set.

- **`cli.py` — 0% coverage.** No `tests/test_cli.py`. Cover with click's `CliRunner`,
  mocking `Captioner`/`Store`/`Embedder`/`index_directory`. Priority paths:
  - `index`: `captioner.check()` fails → `sys.exit(1)` (cli.py ~44-45)
  - `serve`: port-in-use → reuse-or-find-next-free, no-free-port → exit (cli.py ~135-157)

- **`search.py:108` — CLIP-only merge arm.** No test constructs a result that is in
  `clip_scores` but NOT `fts_scores` (FTS matches nothing, CLIP still returns it). All
  search tests surface results via FTS. Needs an embedder whose vector matches a doc the
  FTS query misses.

- **Server `sync_status` sub-branches** (server.py ~195, ~200): the `no indexed_root
  config` and `root_missing` returns are still uncovered (the None-store and happy/stale
  paths are tested).

## Needs a real asset or environment

- **RAW image loading — `indexer.py:55-62` (0% of the rawpy branch).** The entire
  `.nef/.cr2/.arw/...` path, including the `half_size` large-sensor decision, is
  untested. Needs a small real RAW fixture committed to the repo (or a heavily mocked
  `rawpy.imread`, which would test little). A broken rawpy import/API change is currently
  invisible to CI.

- **`embedder.py` — real model never loaded (44%, all consumers use `MagicMock`).** If
  the open_clip API, device selection (mps/cuda/cpu), or normalization breaks, no test
  catches it. Suggest one `@pytest.mark.slow` smoke test that loads the real model and
  asserts `embed_text`/`embed_image` shape + unit norm.

## Needs thread / subprocess / display harnessing

- **Server background index thread — the `_run` closure.** The whole setup-wizard
  pipeline (Store/Captioner/Embedder construction, the `captioner.check()` early-exit,
  `index_directory`, config write, mode switch to search, and the `except` that sets
  `_index_state.error`) runs in an untested daemon thread.

- **Server `browse_folder` — `server.py:84-111`.** Native folder-picker dispatch
  (darwin/win32/other, cancelled dialog, exceptions) — entirely untested.

- **Server `open`/`reveal` subprocess dispatch — `server.py:283-297`.** Only the 503/404
  guards are tested; the actual `subprocess.Popen` per-platform branches are not. Also
  the real-results + thumbnail-base64 encoding path in `/search` (server.py ~236-241)
  and a 200 from `/thumbnail` are unexercised (test store is always empty).

## Resolved

- ~~Server test isolation~~ — fixtures now reset module globals in teardown
  (`_reset_server_globals`).
- ~~Stale fixture strings~~ — fixtures now use `constants.DEFAULT_MODEL`/`OLLAMA_URL`.
