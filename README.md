<h1 align="center">oMLX Lite</h1>
<p align="center"><b>LLM inference, optimized for your Mac</b><br>
Multi-model serving, continuous batching, and a two-tier KV cache — managed from your menu bar.</p>

<p align="center">
  <a href="mailto:junkim.dot@gmail.com">junkim.dot@gmail.com</a> · <a href="https://omlx.ai/me">https://omlx.ai/me</a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#features">Features</a> ·
  <a href="#api">API</a> ·
  <a href="#cli-reference">CLI</a> ·
  <a href="#models">Models</a> ·
  <a href="https://omlx.ai/benchmarks">Benchmarks</a> ·
  <a href="https://omlx.ai">oMLX.ai</a>
</p>

<p align="center">
  <b>English</b> · <a href="README.zh.md">中文</a>
</p>

---

## Why oMLX Lite

Local LLM servers usually force a choice between convenience and control. oMLX Lite is built around a simple idea: keep the models you use every day pinned in memory, swap the heavier ones in on demand, cap memory so the machine stays responsive — and run it all from a native menu bar app.

The part that makes local models practical for real coding work is the cache. oMLX Lite keeps the KV cache alive across a **hot in-memory tier** and a **cold SSD tier**, so past context stays reusable across requests even when the conversation changes — and even after the server restarts. Combined with continuous batching, it makes tools like Claude Code usable against a local model.

## Highlights

- **Multi-model serving** — LLMs, VLMs, OCR, embeddings, rerankers, and remote endpoints in one server, with LRU eviction, pinning, per-model TTL, and a process-wide memory guard.
- **Two-tier KV cache** — block-based prefix cache with Copy-on-Write, spanning RAM and SSD.
- **Continuous batching** — concurrent requests through mlx-lm's `BatchGenerator`, tunable concurrency.
- **OpenAI + Anthropic compatible** — chat, completions, messages, responses, embeddings, rerank, and optional audio, all from one endpoint.
- **Native macOS app** — SwiftUI menu bar app with usage history, auto-restart, and auto-update.
- **Admin dashboard** — real-time monitoring, model management, chat, downloads, benchmarks, remote models, and per-model settings.
- **Optimizations** — MoE expert offload, oQ dynamic quantization, DFlash / Lightning MTP speculative decoding, and optional native Metal kernels.
- **Experimental clusters** — split one model across multiple Macs (and, on the roadmap, CUDA nodes).

## Install

### macOS App

Download the `.dmg` from [Releases](https://github.com/jundot/omlx/releases) and drag it to Applications. The app includes in-app auto-update, and installs a small `~/.omlx/bin/omlx` CLI shim so terminal commands and Apple Shortcuts can drive the app-managed server.

### Homebrew

```bash
brew tap jundot/omlx https://github.com/jundot/omlx
brew install jundot/omlx/omlx

# Upgrade later
brew update && brew upgrade omlx
```

For the GLM-5.2 / MiniMax M3 / Qwen3.5 native kernels, build from `HEAD`:

```bash
brew install jundot/omlx/omlx --HEAD --with-custom-kernel
```

### From Source

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .          # core

# with native custom kernels
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .
```

Requires **macOS 15.0+ (Sequoia)**, **Python 3.11–3.13**, and **Apple Silicon** (M1–M5).

<details>
<summary><b>About native custom kernels</b></summary>

A plain `pip install -e .` does **not** build the native kernels, and the affected model families then fall back to much slower generic paths — for GLM-5.2 the fused DSA prefill is roughly 30x faster with kernels (measured 845 vs ~29 tok/s on an M3 Ultra), and the fallback also uses more memory. Building them requires the full Metal toolchain, which Command Line Tools alone do not provide (`xcrun: error: unable to find utility "metal"`): install full Xcode, or use the official DMG which ships them precompiled. Verify with:

```bash
python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
```

The optional kernels are: `bonsai`, `decode_fast`, `glm_moe_dsa`, `minimax_m3`, and `qwen35_prefill`. Whether each is active is also surfaced on `GET /api/status`.

</details>

## Quickstart

### macOS App

Launch oMLX Lite from Applications. The Welcome wizard covers three steps — an intro, a setup screen (storage path, model directory, port, and optional API key), and completion — then starts the server.

### CLI

```bash
# Managed background server (app or Homebrew install)
omlx start
omlx stop
omlx restart

# Foreground server attached to this terminal
omlx serve --model-dir ~/models
```

The server discovers LLMs, VLMs, OCR, embedding, reranker, and audio models from subdirectories automatically. Point any OpenAI-compatible client at `http://localhost:8000/v1`, or use the built-in chat UI at `http://localhost:8000/admin/chat`.

### As a Service

Via Homebrew, oMLX Lite can run as a managed background service. The `omlx start|stop|restart` commands are portable; Homebrew installs delegate them to `brew services`.

```bash
omlx start                    # start
omlx stop                     # stop
omlx restart                  # restart

brew services start omlx      # start (auto-restarts on crash)
brew services info omlx       # status
```

The service runs `omlx serve` with zero-config defaults (`~/.omlx/models`, port 8000). To customize, set environment variables (`OMLX_MODEL_DIR`, `OMLX_PORT`, …) or run `omlx serve --model-dir /your/path` once to persist settings to `~/.omlx/settings.json`.

Logs live in two places:

- **Service log** — `$(brew --prefix)/var/log/omlx.log` (stdout/stderr)
- **Server log** — `~/.omlx/logs/server.log` (structured application log)

## Features

### Inference Engine

**Continuous batching.** Concurrent requests are scheduled through mlx-lm's `BatchGenerator`; max concurrency is configurable from the CLI or the admin panel.

**Tiered KV cache (hot + cold).** Block-based cache management inspired by vLLM, with prefix sharing and Copy-on-Write:

- **Hot tier (RAM)** — frequently used blocks stay resident for fast resume.
- **Cold tier (SSD)** — when the hot cache fills, blocks are spilled to SSD as safetensors. A later request with a matching prefix restores them from disk instead of recomputing — even across server restarts.

Both tiers are opt-in: enable SSD spilling with `--paged-ssd-cache-dir` and the in-memory hot tier with `--hot-cache-max-size`.

**Multi-model serving.** Load any mix of LLMs, VLMs, embedding models, rerankers, and remote endpoints in one server:

- **LRU eviction** — least-recently-used models unload automatically under memory pressure.
- **Manual load/unload** — toggle models from the admin panel.
- **Pinning** — keep frequently used models resident.
- **Per-model TTL** — auto-unload a model after an idle period.
- **Memory guard** — a process-wide ceiling (default `balanced`) keeps the machine from OOMing. Tiers: `off`, `safe`, `balanced`, `aggressive`, or a custom GB value.

**Per-model settings.** Configure sampling parameters, chat template kwargs, TTL, alias, type override, and optimizations per model from the admin panel — applied immediately, no restart.

- **Alias** — a custom API-visible name; `/v1/models` returns it and both alias and directory name are accepted.
- **Type override** — force a model to be treated as LLM or VLM regardless of auto-detection.

### Model Support

Point `--model-dir` at a directory of MLX-format model subdirectories. Two-level layouts (`mlx-community/model-name/`) are supported too. Models are auto-detected by type; you can also download them from the dashboard.

| Type | Examples |
|------|----------|
| **LLM** | Anything supported by [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| **VLM** | Qwen3.5 series, GLM-4V, Pixtral, and other [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) models |
| **OCR** | DeepSeek-OCR, DOTS-OCR, GLM-OCR, Unlimited-OCR (auto-detected with tuned prompts) |
| **Embedding** | BERT, BGE-M3, ModernBERT, SigLIP |
| **Reranker** | ModernBERT, XLM-RoBERTa, Jina v3 |
| **Audio** | TTS / STT / STS via the optional `omlx[audio]` extra |

VLMs run on the same continuous batching and tiered cache stack as text models, with multi-image chat, base64/URL/file inputs, and tool calling with vision context.

### Optimizations

**MoE expert offload.** For Mixture-of-Experts models, keep a configurable fraction of each layer's experts resident and stream the rest from the checkpoint's own safetensors via mmap. Routing is computed exactly as shipped, so accuracy is preserved and the only cost is latency — letting you run MoE models larger than memory. See [`docs/MoE_Expert_Offload.md`](docs/MoE_Expert_Offload.md).

**oQ dynamic quantization.** A calibration-driven mixed-precision quantizer that measures each layer's sensitivity and allocates bits where they matter. Outputs standard mlx-lm-compatible models that work anywhere. See [`docs/oQ_Quantization.md`](docs/oQ_Quantization.md).

**Speculative decoding.** Per-model options for block-diffusion **DFlash** decoding ([dflash-mlx](https://github.com/bstnxbt/dflash-mlx)), **Lightning MTP** multi-token prediction, and VLM MTP, plus **SpecPrefill** draft-model prefill acceleration. Available from per-model settings.

**Native Metal kernels.** Optional fused kernels for Bonsai, GLM-5.2 (fused DSA prefill), MiniMax M3, Qwen3.5 prefill, and a generic decode path.

### APIs

Drop-in compatible with the OpenAI and Anthropic APIs. Supports streaming usage stats (`stream_options.include_usage`), Anthropic adaptive thinking, and vision inputs (base64 or URL).

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming) |
| `POST /v1/completions` | Text completions (streaming) |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/rerank` | Document reranking |
| `GET /v1/models` | List models |
| `POST /v1/models/{id}/load` · `/unload` | Manual model lifecycle |
| `POST /v1/audio/transcriptions` · `/speech` · `/process` | Audio (optional `omlx[audio]`) |
| `GET /health` · `GET /api/status` | Health and status |

**Tool calling & structured output.** Supports the function-calling formats in mlx-lm plus JSON Schema validation. Tool calling requires the model's chat template to accept `tools`. Auto-detected families:

| Model family | Format |
|---|---|
| Llama, Qwen, DeepSeek, etc. | JSON `<tool_call>` |
| Qwen3.5 series | XML `<function=...>` |
| Gemma | `<start_function_call>` |
| GLM (4.7, 5) | `<arg_key>` / `<arg_value>` XML |
| MiniMax | Namespaced `<minimax:tool_call>` |
| Mistral | `[TOOL_CALLS]` |
| IFM K2 Horizon | XML or JSON inside `<ifm\|tool_calls>` (native tool-name constraints use the optional `omlx[grammar]`) |
| Kimi K2 | `<\|tool_calls_section_begin\|>` |
| Longcat | `<longcat_tool_call>` |

Models not listed may still work if their chat template accepts `tools` and emits a recognized `<tool_call>` XML format. For tool-enabled streaming, assistant text is emitted incrementally while control markup is suppressed; structured tool calls are emitted once the turn is parsed.

### Admin Dashboard

A web UI at `/admin` for real-time monitoring, model management, chat, benchmarking, remote models, and per-model settings. Fully offline — all CDN dependencies are vendored. Available in English, Korean, Japanese, Chinese (Simplified and Traditional), French, Russian, Spanish, and Brazilian Portuguese.

**Built-in chat.** Talk to any loaded model, with conversation history, model switching, dark mode, reasoning output, and image upload for VLM/OCR.

**Model downloader.** Search and download MLX models from HuggingFace and ModelScope, view model cards and file sizes, and download with one click.

**Performance benchmark.** One-click prefill (PP) and text-generation (TG) throughput measurement, including partial prefix-cache hit tests for realistic numbers. A separate accuracy suite runs MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench evaluations with queued runs and results that persist until reset.

**Remote models.** Register OpenAI-compatible endpoints (other oMLX Lite instances, vLLM, OpenAI, …) and use them alongside local models, with connection testing and per-model enable/disable.

### macOS Menu Bar App

A native Swift / SwiftUI menu bar app — not Electron. Start, stop, and monitor the server without a terminal. It includes [local usage history](docs/usage-analytics.md) with per-model totals and an hourly heatmap, persistent serving stats, auto-restart on crash, and built-in auto-update.

### Experimental: Multi-Mac Inference

Source builds can split one downloaded model across unequal-memory Macs using MLX pipeline ranks over Ring or Thunderbolt RDMA/JACCL. Read-only peer discovery, byte-aware uneven shard planning, measured compute/link rebalancing, headroom-aware tuning, and a live shard/performance map are available on both Macs. A design for a single heterogeneous MLX + CUDA pool is also documented.

See [Distributed inference across Macs](docs/distributed-cluster.md) and [Heterogeneous clusters](docs/heterogeneous-cluster.md) for setup, security boundaries, and limitations.

## Models

```
~/models/
├── Step-3.5-Flash-8bit/
├── Qwen3-Coder-Next-8bit/
├── gpt-oss-120b-MXFP4-Q8/
├── Qwen3.5-122B-A10B-4bit/
└── bge-m3/
```

## CLI Reference

```bash
# Lifecycle (managed background server)
omlx start
omlx stop
omlx restart

# Serve (foreground)
omlx serve --model-dir ~/models

# Memory guard tier, or a custom GB ceiling
omlx serve --model-dir ~/models --memory-guard safe
omlx serve --model-dir ~/models --memory-guard-gb 48

# Two-tier cache
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache
omlx serve --model-dir ~/models --hot-cache-max-size 20%
omlx serve --model-dir ~/models --hot-cache-write-through

# Concurrency
omlx serve --model-dir ~/models --max-concurrent-requests 16

# Region / mirror endpoints
omlx serve --model-dir ~/models --hf-endpoint https://hf-mirror.com
omlx serve --model-dir ~/models --ms-endpoint https://modelscope.cn

# Authentication
omlx serve --model-dir ~/models --api-key your-secret-key
OMLX_API_KEY=your-secret-key omlx serve --model-dir ~/models --host 0.0.0.0
```

Other commands: `omlx diagnose` for installation/runtime diagnostics, and `omlx cluster ...` for distributed-inference tooling.

All settings are also editable from the admin panel and persisted to `~/.omlx/settings.json`; CLI flags take precedence. Before binding to a LAN address or `0.0.0.0`, set the main API key (or save both together) — oMLX Lite refuses to start on any non-loopback address without one, and API key verification can only be skipped for loopback-only binds.

## Architecture

<details>
<summary>Request path and components</summary>

```
FastAPI Server (OpenAI / Anthropic API)
    │
    ├── EnginePool (multi-model, LRU eviction, TTL, manual load/unload)
    │   ├── BatchedEngine (LLMs; DFlash and distributed variants)
    │   ├── VLMBatchedEngine (vision-language / OCR models)
    │   ├── EmbeddingEngine
    │   ├── RerankerEngine
    │   └── TTS / STT / STS engines (optional omlx[audio])
    │
    ├── ProcessMemoryEnforcer (total memory limit, TTL checks)
    │
    ├── Scheduler (FCFS, configurable concurrency)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack
        ├── PagedCacheManager (GPU, block-based, CoW, prefix sharing)
        ├── Hot Cache (in-memory tier, write-back)
        └── PagedSSDCacheManager (SSD cold tier, safetensors format)
```

</details>

## Development

### Server

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
pytest -m "not slow"
```

### macOS App

The SwiftUI app lives in `apps/omlx-mac/` and requires Xcode 26.5+ and Python 3.11+.

```bash
# Stage a runnable oMLX Lite.app (xcodebuild + venvstacks layers + ad-hoc sign)
apps/omlx-mac/Scripts/build.sh release

# Result lands at apps/omlx-mac/build/Stage/oMLX Lite.app
open apps/omlx-mac/build/Stage/oMLX Lite.app

# Force a fresh venvstacks rebuild (otherwise cached by fingerprint)
apps/omlx-mac/Scripts/build.sh release --rebuild-donor

# Include the optional native custom kernels
apps/omlx-mac/Scripts/build.sh release --with-custom-kernel
```

The first cold build takes 10–20 minutes (venvstacks layer assembly); later builds reuse `packaging/_export/` and finish in about 4 minutes. See [packaging/README.md](packaging/README.md) for layer config and [apps/omlx-mac/](apps/omlx-mac/) for the Swift sources.

## Contributing

Contributions are welcome — see the [Contributing Guide](docs/CONTRIBUTING.md). Bug fixes, performance work, and documentation improvements are all appreciated.

## License

[Apache 2.0](LICENSE)

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — vision-language inference on Apple Silicon
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) — oMLX Lite started from vllm-mlx v0.1.0 and grew multi-model serving, tiered KV caching, paginated-cache VLM, an admin panel, and a macOS menu bar app
- [venvstacks](https://venvstacks.lmstudio.ai) — portable Python environment layering for the app bundle
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — embedding model support for Apple Silicon
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) — block-diffusion speculative decoding on Apple Silicon
- [MTPLX](https://github.com/youssofal/mtplx) — Lightning MTP's verify-shape Metal kernels, which also inspired the depth-k pipeline
- [mlx-serve](https://github.com/ddalcu/mlx-serve) — the fused GDN verify prework kernel and the Qwen4 QSA 128-bit K/V staging kernel
- [SiliconScope](https://github.com/kennss/SiliconScope) — design and rendering approach for the menu bar statistics and energy-efficient re-render gating
