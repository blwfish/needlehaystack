# needlestack

Find photos by describing what's in them — fully local, no cloud, no account.

> *"Show me the caboose on the siding at dusk"*  
> *"USS Enterprise alongside the pier, Cold War era"*  
> *"Tiger I with winter whitewash, Eastern Front"*  
> *"F-86 Sabre on the flight line, Korean War"*

needlestack indexes a folder of images using a local AI model that writes a rich description of each photo, then searches those descriptions by both keywords and visual similarity. Everything runs on your machine.

Works with **railroad**, **naval**, **armor (AFV)**, **aviation**, and **bird photography** collections — each domain brings its own vocabulary, identifier fields, and synonym expansion.

**Your photos never leave your computer.** All AI processing happens locally via [Ollama](https://ollama.com) — no account, no upload, no cloud. The AI reads your photos and generates captions; that's it. Nothing goes anywhere.

---

## How it works

1. **Index** — for each image, a local AI model looks at the scene and reads any text visible in it — road numbers, hull numbers, tail numbers, markings — and writes a detailed, searchable description matched to your collection type. The image's overall visual appearance is also captured as a compact signature. Both are saved on your machine.
2. **Search** — your query is automatically expanded with related terms and synonyms, then matched against the photo descriptions and visual signatures. Results are merged and ranked by relevance.
3. **Browse** — a small local webpage shows thumbnails; click to open in your default viewer, hover for the "show in Finder" button.

Runs on Apple Silicon (fast, using your Mac's built-in AI hardware), Intel Mac (slower), and Windows. Supports JPEG, PNG, TIFF, WebP, and common RAW formats (NEF, CR2, CR3, ARW, DNG, and more).

---

## Quick start

### Mac

```bash
# 1. Install Ollama — https://ollama.com — then:
./install-mac.sh
```

The installer checks prerequisites, downloads the AI model (~8 GB), sets up the environment, and creates a double-clickable launcher.

### Windows

```
install-windows.bat
```

### Manual (any platform with pixi)

```bash
pixi install
ollama pull qwen2.5vl:7b

# Index your photos (one-time, resumes if interrupted)
pixi run needlestack index /path/to/photos

# Start the search UI
pixi run needlestack serve
```

---

## Command line

needlestack also has a full command-line interface for scripting, bulk re-indexing, running on a server, or just personal preference. See [CLI.md](CLI.md) for all commands and options.

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) with `qwen2.5vl:7b` pulled
- ~12 GB disk space (model + environment + index)
- macOS 12+ or Windows 10+

[pixi](https://pixi.sh) is used for environment management and handles all Python dependencies including PyTorch automatically.

---

## For non-technical users

See [GUIDE.md](GUIDE.md) — plain English walkthrough from installation through searching, with a troubleshooting section covering every common failure.

**Found a bug?** Open an issue at [github.com/blwfish/needlehaystack/issues](https://github.com/blwfish/needlehaystack/issues) — click **New issue** and fill in the form. The more detail you can give (what you searched for, what you expected, what happened instead, your OS), the faster it gets fixed.

---

## For developers

```
src/needlestack/
  captioner.py   Ollama HTTP client — structured VLM captions per domain
  embedder.py    OpenCLIP ViT-B-32 — image and text embeddings (MPS/CUDA/CPU)
  indexer.py     File discovery, hashing, thumbnail generation, orchestration
  search.py      Query expansion, FTS5 search, cosine similarity, score fusion
  store.py       SQLite — schema, FTS5 triggers, embedding blobs, config
  taxonomy.py    Domain vocabulary — subject types, synonyms, prompt fragments
  server.py      FastAPI — search, setup wizard, open/reveal, sync-status
  cli.py         Click — index, serve, doctor commands
  ui/
    index.html   Search UI
    setup.html   First-run wizard (includes domain selector) + progress
```

Search scoring: `score = 0.6 × FTS5_rank + 0.4 × CLIP_cosine`, results below 0.38 are dropped.

```
tests/
  conftest.py               Shared fixtures — Ollama skip guard, Wikipedia photo downloads
  test_captioner.py         Captioner prompt construction and structured output parsing
  test_cli.py               CLI commands via click.testing.CliRunner (mocked dependencies)
  test_doctor.py            Doctor report generation and query tracing
  test_indexer.py           File discovery, hashing, thumbnail generation
  test_search.py            Query expansion, FTS5 scoring, score fusion, edge cases
  test_server.py            FastAPI endpoints, sync-status, domain resolution
  test_store.py             SQLite schema, FTS5 triggers, embedding storage, config
  test_taxonomy.py          Domain vocabulary, synonyms, field weights for all domains
  test_birds_integration.py Real Ollama + Wikipedia photos (skipped without Ollama)
```

**Tests:** 200+ unit tests (not lines — tests) cover captioning, indexing, search scoring, query expansion, the API, and every domain. Run the suite with:

```bash
pytest -m "not integration"
```

Integration tests (which caption real photos via Ollama) run with `pytest` alone — Ollama must be running.

**Contributing:** PRs are welcome for bug fixes and new domains. A few guidelines:

- *New domains* belong in `taxonomy.py` — follow the pattern of an existing domain, add it to `DOMAINS`, and add the option to `setup.html`.
- *Search weights or scoring changes* should open an issue for discussion first — these affect everyone's results.
- *All PRs must include tests.* The existing suite is the baseline; new behaviour without tests will not be merged.
