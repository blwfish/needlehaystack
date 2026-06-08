# needlestack CLI reference

```
needlestack index <directory>   Index a folder of images
needlestack serve               Start the web UI (default port 8484)
needlestack doctor              Diagnostic report — use --out report.txt to save
```

## `index` options

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.needlestack/index.db` | Database file location |
| `--domain` | `railroad` | Collection type: `railroad` / `naval` / `armor` / `aviation` |
| `--preset` | *(auto-chosen by installer)* | AI model tier: `fast` / `balanced` / `quality` |
| `--model` | — | Specific Ollama model name (overrides `--preset`) |
| `--ollama` | `http://localhost:11434` | Ollama address (if not running locally) |
| `--force` | off | Re-caption already-indexed files |
| `--thorough` | off | Add a dedicated pass to read identifiers more carefully (~2× slower) |

## Model presets

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

Switching models triggers automatic re-captioning on the next index run — no `--force` needed.

## `serve` options

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.needlestack/index.db` | Database file location |
| `--port` | `8484` | Port for the local webpage |
| `--preset` | `balanced` | AI model tier for query expansion |
| `--no-browser` | off | Don't open browser on start |

Indexing is incremental — re-running `index` on the same folder only processes new or changed files. The web UI detects new photos and offers a one-click re-index.
