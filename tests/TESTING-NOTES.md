# Testing notes — known gaps

Status as of the 2026-07-17 full-review fix pass (220 tests).

The suite now covers `doctor.py`, `cli.py`, `server.py`'s background indexing threads
(`start_indexing`/`reindex_all` via the shared `_run_indexing_loop`), `browse_folder`,
the RAW `half_size` threshold, the FTS/CLIP score-merge weights (including the
previously-unpinned CLIP-only/FTS-only merge arms), and EXIF/GPS extraction —
all previously 0% or unpinned. Ordered by remaining tractability.

## Needs a real asset or environment

- **`embedder.py` — real model never loaded (all consumers use `MagicMock`).** If
  the open_clip API, device selection (mps/cuda/cpu), or normalization breaks, no test
  catches it. Suggest one `@pytest.mark.slow` smoke test that loads the real model and
  asserts `embed_text`/`embed_image` shape + unit norm.

- **RAW image *decoding* itself** (as opposed to the `half_size` threshold decision,
  which is now tested via a mocked `rawpy.imread`) still has no real-file coverage.
  A broken rawpy API change that doesn't affect the threshold math would still be
  invisible to CI. Needs a small real RAW fixture committed to the repo.

## Needs thread / subprocess / display harnessing

- **Server `open`/`reveal` subprocess dispatch — `server.py`'s `open_image`/
  `reveal_image`.** Only the 503/404 guards (via `_resolve_image_path`) are tested;
  the actual `subprocess.Popen`/`subprocess.run` per-platform branches (darwin/win32)
  are not. Same shape as the `browse_folder` tests already added — platform-mocked via
  `monkeypatch.setattr(sys, "platform", ...)` plus a patched `subprocess.Popen`.

## Resolved

- ~~`doctor.py` — 0% coverage~~ — full `tests/test_doctor.py`, including domain
  awareness, corrupt-embedding reporting, and the model-presence prefix-match branch.
- ~~`cli.py` — 0% coverage~~ — `tests/test_cli.py` covers `index`/`doctor`/`serve`,
  including the port-scan width boundary (at/below/above) and the `--model`/`--preset`
  mutual exclusion on both `index` and `serve`.
- ~~`search.py` CLIP-only merge arm~~ — pinned together with the FTS-only arm and the
  FTS_WEIGHT/CLIP_WEIGHT values in one test (`test_score_merge_pins_weights_and_missing_side_defaults`),
  since a doc missing from `clip_scores` or `fts_scores` needed a controlled fixture in
  either direction.
- ~~Server `sync_status` sub-branches~~ — store-present-zero-roots is covered
  (`test_sync_status_store_present_zero_roots`); the None-store and happy/stale paths
  were already tested.
- ~~RAW image loading `half_size` threshold~~ — pinned at/below/above `MAX_PIXELS` via
  a mocked `rawpy.imread` (full decoding still open, see above).
- ~~Server background index thread~~ — `start_indexing`/`reindex_all` both now route
  through the shared `_run_indexing_loop`, with parity, failure-path, and
  connection-safety tests (`test_reindex_all_*`, `test_start_indexing_and_reindex_all_both_use_shared_loop`).
- ~~Server `browse_folder`~~ — cancel-vs-error, success, and unsupported-platform paths
  all covered.
- ~~Server test isolation~~ — fixtures reset module globals in teardown
  (`_reset_server_globals`).
- ~~Stale fixture strings~~ — fixtures use `constants.DEFAULT_MODEL`/`OLLAMA_URL`.
