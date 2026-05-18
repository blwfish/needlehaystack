# needlestack

Find photos by describing what's in them — fully local, no cloud, no account.

> *"Show me the caboose on the siding at dusk"*  
> *"Red brick station with a covered platform"*  
> *"Steam locomotive with large driving wheels"*

needlestack indexes a folder of images using a local vision-language model to write a rich description of each photo, then searches those descriptions with a combination of full-text search and CLIP embeddings. Everything runs on your machine.

---

## How it works

1. **Index** — for each image, a VLM ([llava:13b](https://ollama.com/library/llava) via Ollama) writes a detailed caption; [OpenCLIP](https://github.com/mlfoundations/open_clip) generates a vector embedding. Both are stored in SQLite.
2. **Search** — your query is expanded to synonyms via the same LLM, then matched against captions with SQLite FTS5 (BM25) and against embeddings with cosine similarity. Results are merged and ranked.
3. **Browse** — a small local web UI shows thumbnails; click to open in your default viewer, hover for the "show in Finder" button.

Runs on Apple Silicon (fast, MPS-accelerated), Intel Mac (slower), and Windows. Supports JPEG, PNG, TIFF, WebP, and common RAW formats (NEF, CR2, CR3, ARW, DNG, and more).

---

## Quick start

### Mac

```bash
# 1. Install Ollama — https://ollama.com — then:
./install-mac.sh
```

The installer checks prerequisites, downloads the LLaVA model (~8 GB), sets up the Python environment, and creates a double-clickable launcher.

### Windows

> **Untested.** The Windows installer and launcher are written but have never been run on an actual Windows machine. Windows testing is coming — patient users should wait; the brave may proceed with low expectations and are encouraged to report what breaks.

```
install-windows.bat
```

### Manual (any platform with pixi)

```bash
pixi install
ollama pull llava:13b

# Index your photos (one-time, resumes if interrupted)
pixi run needlestack index /path/to/photos

# Start the search UI
pixi run needlestack serve
```

---

## CLI reference

```
needlestack index <directory>   Index a folder of images
needlestack serve               Start the web UI (default port 8484)
needlestack doctor              Diagnostic report — use --out report.txt to save
```

**`index` options**

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.needlestack/index.db` | SQLite database path |
| `--model` | `llava:13b` | Ollama vision model |
| `--ollama` | `http://localhost:11434` | Ollama base URL |
| `--force` | off | Re-caption already-indexed files |

**`serve` options**

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.needlestack/index.db` | SQLite database path |
| `--port` | `8484` | HTTP port |
| `--no-browser` | off | Don't open browser on start |

Indexing is incremental — re-running `index` on the same folder only processes new or changed files. The web UI detects new photos and offers a one-click re-index.

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) with `llava:13b` pulled
- ~14 GB disk space (model + environment + index)
- macOS 12+ or Windows 10+

[pixi](https://pixi.sh) is used for environment management and handles all Python dependencies including PyTorch automatically.

---

## Architecture

```
src/needlestack/
  captioner.py   Ollama HTTP client — generates captions via LLaVA
  embedder.py    OpenCLIP ViT-B-32 — image and text embeddings (MPS/CUDA/CPU)
  indexer.py     File discovery, hashing, thumbnail generation, orchestration
  search.py      Query expansion, FTS5 search, cosine similarity, score fusion
  store.py       SQLite — schema, FTS5 triggers, embedding blobs, config
  server.py      FastAPI — search, setup wizard, open/reveal, sync-status
  cli.py         Click — index, serve, doctor commands
  ui/
    index.html   Search UI
    setup.html   First-run wizard + re-index progress
```

Search scoring: `score = 0.6 × FTS5_rank + 0.4 × CLIP_cosine`, results below 0.38 are dropped.

---

## For non-technical users

See [GUIDE.md](GUIDE.md) — plain English walkthrough from installation through searching, with a troubleshooting section covering every common failure.
