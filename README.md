# needlestack

Find photos by describing what's in them — fully local, no cloud, no account.

> *"Show me the caboose on the siding at dusk"*  
> *"USS Enterprise alongside the pier, Cold War era"*  
> *"Tiger I with winter whitewash, Eastern Front"*  
> *"F-86 Sabre on the flight line, Korean War"*

needlestack indexes a folder of images using a local vision-language model to write a rich description of each photo, then searches those descriptions with a combination of full-text search and CLIP embeddings. Everything runs on your machine.

Works with **railroad**, **naval**, **armor (AFV)**, and **aviation** photo collections — each domain brings its own vocabulary, identifier fields, and synonym expansion.

**Your photos never leave your computer.** All AI processing happens locally via [Ollama](https://ollama.com) — no account, no upload, no cloud. The AI interprets your search query and generates captions; that's it. Nothing goes anywhere.

---

## How it works

1. **Index** — for each image, a VLM ([qwen2.5vl:7b](https://ollama.com/library/qwen2.5vl) via Ollama) reads the scene and any text in it — identifiers, markings, road numbers, hull numbers — into a structured, searchable caption matched to your collection's domain; [OpenCLIP](https://github.com/mlfoundations/open_clip) generates a vector embedding. Both are stored in SQLite.
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
ollama pull qwen2.5vl:7b

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
| `--domain` | `railroad` | Collection type: `railroad` / `naval` / `armor` / `aviation` |
| `--preset` | *(auto-chosen by installer)* | Model tier: `fast` / `balanced` / `quality` |
| `--model` | — | Exact Ollama model name (overrides `--preset`) |
| `--ollama` | `http://localhost:11434` | Ollama base URL |
| `--force` | off | Re-caption already-indexed files |
| `--thorough` | off | Add a dedicated OCR pass for identifiers (~2× slower) |

**Model presets**

The installer picks the right tier for your hardware automatically. You can override it any time:

| Preset | Model | Best for | Speed |
|--------|-------|----------|-------|
| `fast` | `minicpm-v:latest` | CPU-only / low-RAM machines | ~3–5s/photo |
| `balanced` | `qwen2.5vl:7b` | 16 GB+ Apple Silicon / mid GPU | ~4–6s/photo |
| `quality` | `qwen3-vl:32b` | 32 GB+ unified / high-VRAM GPU | ~90–120s/photo |

```bash
needlestack index /photos --preset quality   # switch to the quality tier
needlestack index /photos --model llava:13b  # use any Ollama model directly
```

Switching model triggers automatic re-captioning on the next index run — no `--force` needed.

**`serve` options**

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.needlestack/index.db` | SQLite database path |
| `--port` | `8484` | HTTP port |
| `--preset` | `balanced` | Model tier for query expansion |
| `--no-browser` | off | Don't open browser on start |

Indexing is incremental — re-running `index` on the same folder only processes new or changed files. The web UI detects new photos and offers a one-click re-index.

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) with `qwen2.5vl:7b` pulled
- ~12 GB disk space (model + environment + index)
- macOS 12+ or Windows 10+

[pixi](https://pixi.sh) is used for environment management and handles all Python dependencies including PyTorch automatically.

---

## Architecture

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

---

## For non-technical users

See [GUIDE.md](GUIDE.md) — plain English walkthrough from installation through searching, with a troubleshooting section covering every common failure.
