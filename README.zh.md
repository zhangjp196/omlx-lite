<h1 align="center">oMLX Lite</h1>
<p align="center"><b>LLM 推理，为你的 Mac 优化</b><br>
多模型服务、连续批处理与两级 KV 缓存 —— 直接在菜单栏中管理。</p>

<p align="center">
  <a href="mailto:junkim.dot@gmail.com">junkim.dot@gmail.com</a> · <a href="https://omlx.ai/me">https://omlx.ai/me</a>
</p>

<p align="center">
  <a href="#安装">安装</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#功能">功能</a> ·
  <a href="#api">API</a> ·
  <a href="#cli-参考">CLI</a> ·
  <a href="#模型">模型</a> ·
  <a href="https://omlx.ai/benchmarks">基准测试</a> ·
  <a href="https://omlx.ai">oMLX.ai</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>中文</b>
</p>

---

## 为什么选择 oMLX Lite

本地 LLM 服务器通常迫使你在便利性和可控性之间二选一。oMLX Lite 的核心思路很简单：把每天使用的模型固定在内存中，按需载入更重的模型，限制内存以免拖慢机器 —— 并通过原生菜单栏应用管理这一切。

让本地模型能真正用于编码工作的关键在缓存。oMLX Lite 让 KV 缓存在**热内存层**和**冷 SSD 层**之间持续存活，因此即使对话内容变化，历史上下文仍可跨请求复用 —— 甚至服务器重启后依然有效。配合连续批处理，Claude Code 之类的工具才真正能驱动本地模型。

## 亮点

- **多模型服务** —— 在同一个服务器中运行 LLM、VLM、OCR、嵌入、重排序模型和远程端点，支持 LRU 驱逐、模型固定、模型级 TTL 和进程级内存守卫。
- **两级 KV 缓存** —— 基于块、支持写时复制（CoW）的前缀缓存，横跨内存与 SSD。
- **连续批处理** —— 通过 mlx-lm 的 `BatchGenerator` 处理并发请求，并发数可调。
- **OpenAI + Anthropic 兼容** —— chat、completions、messages、responses、embeddings、rerank，以及可选的 audio，统一入口。
- **原生 macOS 应用** —— SwiftUI 菜单栏应用，包含使用历史、崩溃自动重启和自动更新。
- **管理后台** —— 实时监控、模型管理、聊天、下载、基准测试、远程模型和模型级设置。
- **性能优化** —— MoE 专家卸载、oQ 动态量化、DFlash / Lightning MTP 推测解码，以及可选的原生 Metal 内核。
- **实验性集群** —— 将单个模型拆分到多台 Mac（未来规划还包括 CUDA 节点）。

## 安装

### macOS 应用

从 [Releases](https://github.com/jundot/omlx/releases) 下载 `.dmg`，拖入 Applications 即可。应用支持应用内自动更新，并安装一个小巧的 `~/.omlx/bin/omlx` CLI shim，让终端命令和 Apple Shortcuts 也能驱动由应用管理的服务器。

### Homebrew

```bash
brew tap jundot/omlx https://github.com/jundot/omlx
brew install jundot/omlx/omlx

# 后续升级
brew update && brew upgrade omlx
```

如需 GLM-5.2 / MiniMax M3 / Qwen3.5 原生内核，请使用 `HEAD` 构建：

```bash
brew install jundot/omlx/omlx --HEAD --with-custom-kernel
```

### 从源码安装

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .          # 仅核心

# 包含原生自定义内核
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .
```

需要 **macOS 15.0+ (Sequoia)**、**Python 3.11–3.13** 和 **Apple Silicon**（M1–M5）。

<details>
<summary><b>关于原生自定义内核</b></summary>

直接执行 `pip install -e .` **不会**构建原生内核，受影响的模型系列会回退到慢得多的通用路径 —— 以 GLM-5.2 为例，启用内核后融合 DSA 预填充大约快 30 倍（M3 Ultra 上实测 845 vs ~29 tok/s），回退路径还会占用更多内存。构建内核需要完整的 Metal 工具链，仅安装 Command Line Tools 并不提供（`xcrun: error: unable to find utility "metal"`）：请安装完整版 Xcode，或使用已预编译内核的官方 DMG。验证方式：

```bash
python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
```

可选内核包括：`bonsai`、`decode_fast`、`glm_moe_dsa`、`minimax_m3` 和 `qwen35_prefill`。每个内核的启用状态也会在 `GET /api/status` 中展示。

</details>

## 快速开始

### macOS 应用

从 Applications 启动 oMLX Lite。欢迎向导分为三步 —— 介绍、设置（存储路径、模型目录、端口和可选 API key）和完成 —— 然后启动服务器。

### CLI

```bash
# 托管后台服务器（应用或 Homebrew 安装）
omlx start
omlx stop
omlx restart

# 附着在当前终端的前台服务器
omlx serve --model-dir ~/models
```

服务器会自动从子目录中发现 LLM、VLM、OCR、嵌入、重排序和音频模型。任何 OpenAI 兼容客户端都可以连接到 `http://localhost:8000/v1`，也可以使用内置聊天 UI：`http://localhost:8000/admin/chat`。

### 作为服务运行

通过 Homebrew 安装时，oMLX Lite 可以作为托管后台服务运行。`omlx start|stop|restart` 是可移植命令；Homebrew 安装会将其委托给 `brew services`。

```bash
omlx start                    # 启动
omlx stop                     # 停止
omlx restart                  # 重启

brew services start omlx      # 启动（崩溃时自动重启）
brew services info omlx       # 查看状态
```

服务使用零配置默认值运行 `omlx serve`（`~/.omlx/models`，端口 8000）。要自定义，可以设置环境变量（`OMLX_MODEL_DIR`、`OMLX_PORT` 等），或运行一次 `omlx serve --model-dir /your/path` 将设置持久化到 `~/.omlx/settings.json`。

日志有两个位置：

- **服务日志** —— `$(brew --prefix)/var/log/omlx.log`（stdout/stderr）
- **服务器日志** —— `~/.omlx/logs/server.log`（结构化应用日志）

## 功能

### 推理引擎

**连续批处理。** 并发请求通过 mlx-lm 的 `BatchGenerator` 调度；最大并发数可在 CLI 或管理面板中配置。

**两级 KV 缓存（热 + 冷）。** 借鉴 vLLM 的基于块的缓存管理，支持前缀共享和写时复制（CoW）：

- **热层（RAM）** —— 频繁访问的块保持常驻，快速恢复。
- **冷层（SSD）** —— 热缓存满时，块以 safetensors 格式溢出到 SSD。之后命中相同前缀的请求会直接从磁盘恢复，无需重算 —— 即使服务器重启也一样。

两层均为可选：用 `--paged-ssd-cache-dir` 启用 SSD 溢出，用 `--hot-cache-max-size` 启用内存热层。

**多模型服务。** 在一个服务器中混合加载 LLM、VLM、嵌入模型、重排序模型和远程端点：

- **LRU 驱逐** —— 内存紧张时自动卸载最近最少使用的模型。
- **手动加载/卸载** —— 在管理面板中切换。
- **模型固定** —— 让常用模型保持常驻。
- **模型级 TTL** —— 空闲一段时间后自动卸载。
- **内存守卫** —— 进程级上限（默认 `balanced`）避免机器 OOM。级别：`off`、`safe`、`balanced`、`aggressive`，或自定义 GB 值。

**模型级设置。** 在管理面板中为每个模型配置采样参数、聊天模板参数、TTL、别名、类型覆盖和优化选项 —— 立即生效，无需重启。

- **别名** —— 自定义 API 可见名称；`/v1/models` 返回别名，请求时别名和目录名都可用。
- **类型覆盖** —— 无论自动检测结果如何，强制按 LLM 或 VLM 处理。

### 模型支持

将 `--model-dir` 指向包含 MLX 格式模型子目录的目录，支持两级结构（如 `mlx-community/model-name/`）。模型会按类型自动识别；也可以直接从管理面板下载。

| 类型 | 示例 |
|------|----------|
| **LLM** | [mlx-lm](https://github.com/ml-explore/mlx-lm) 支持的所有模型 |
| **VLM** | Qwen3.5 系列、GLM-4V、Pixtral 及其他 [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 模型 |
| **OCR** | DeepSeek-OCR、DOTS-OCR、GLM-OCR、Unlimited-OCR（自动识别并使用调优提示词） |
| **嵌入** | BERT、BGE-M3、ModernBERT、SigLIP |
| **重排序** | ModernBERT、XLM-RoBERTa、Jina v3 |
| **音频** | 通过可选 extra `omlx[audio]` 提供 TTS / STT / STS |

VLM 与文本模型使用相同的连续批处理和分层缓存堆栈，支持多图聊天、base64/URL/文件输入，以及带视觉上下文的工具调用。

### 优化

**MoE 专家卸载。** 对于混合专家模型，保留每层中可配置比例的专家常驻，其余专家通过 mmap 从检查点自带的 safetensors 按需流式读取。路由计算与原始模型完全一致，因此精度不受影响，代价只是延迟 —— 这让你可以运行大于内存的 MoE 模型。见 [`docs/MoE_Expert_Offload.md`](docs/MoE_Expert_Offload.md)。

**oQ 动态量化。** 由校准驱动的混合精度量化器，逐层测量敏感度，把比特分配到真正重要的地方。输出标准的 mlx-lm 兼容模型，随处可用。见 [`docs/oQ_Quantization.md`](docs/oQ_Quantization.md)。

**推测解码。** 模型级选项支持块扩散 **DFlash** 解码（[dflash-mlx](https://github.com/bstnxbt/dflash-mlx)）、**Lightning MTP** 多 token 预测、VLM MTP，以及 **SpecPrefill** 草稿模型预填充加速。均可在模型级设置中启用。

**原生 Metal 内核。** 可选的融合内核，覆盖 Bonsai、GLM-5.2（融合 DSA 预填充）、MiniMax M3、Qwen3.5 预填充，以及通用解码路径。

### API

可直接替代 OpenAI 和 Anthropic API。支持流式使用统计（`stream_options.include_usage`）、Anthropic adaptive thinking，以及视觉输入（base64 或 URL）。

| 端点 | 说明 |
|----------|------|
| `POST /v1/chat/completions` | 聊天补全（流式） |
| `POST /v1/completions` | 文本补全（流式） |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/messages/count_tokens` | Anthropic token 计数 |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/embeddings` | 文本嵌入 |
| `POST /v1/rerank` | 文档重排序 |
| `GET /v1/models` | 列出模型 |
| `POST /v1/models/{id}/load` · `/unload` | 手动模型生命周期 |
| `POST /v1/audio/transcriptions` · `/speech` · `/process` | 音频（可选 `omlx[audio]`） |
| `GET /health` · `GET /api/status` | 健康与状态 |

**工具调用与结构化输出。** 支持 mlx-lm 中的函数调用格式以及 JSON Schema 验证。工具调用要求模型的聊天模板接受 `tools` 参数。自动识别的模型系列：

| 模型系列 | 格式 |
|---|---|
| Llama、Qwen、DeepSeek 等 | JSON `<tool_call>` |
| Qwen3.5 系列 | XML `<function=...>` |
| Gemma | `<start_function_call>` |
| GLM (4.7, 5) | `<arg_key>` / `<arg_value>` XML |
| MiniMax | 命名空间 `<minimax:tool_call>` |
| Mistral | `[TOOL_CALLS]` |
| IFM K2 Horizon | `<ifm\|tool_calls>` 内的 XML 或 JSON（原生工具名约束使用可选的 `omlx[grammar]`） |
| Kimi K2 | `<\|tool_calls_section_begin\|>` |
| Longcat | `<longcat_tool_call>` |

上表未列出的模型，只要聊天模板接受 `tools` 且输出可识别的 `<tool_call>` XML 格式，也可能正常工作。对于启用工具调用的流式请求，助手文本会增量输出，同时隐藏控制标记；结构化工具调用会在整个回合解析完成后发出。

### 管理后台

`/admin` 提供 Web UI，用于实时监控、模型管理、聊天、基准测试、远程模型和模型级设置。完全离线 —— 所有 CDN 依赖均已内置。支持英语、韩语、日语、简体中文与繁体中文、法语、俄语、西班牙语和巴西葡萄牙语。

**内置聊天。** 与任何已加载模型对话，支持对话历史、模型切换、深色模式、推理输出，以及 VLM/OCR 的图片上传。

**模型下载器。** 从 HuggingFace 和 ModelScope 搜索并下载 MLX 模型，查看模型卡片和文件大小，一键下载。

**性能基准测试。** 一键测量预填充（PP）和文本生成（TG）吞吐，包含部分前缀缓存命中测试，数据更贴近真实。另有精度评测套件，运行 MMLU、HellaSwag、TruthfulQA、GSM8K 和 LiveCodeBench，支持排队运行，结果持久保存直到手动重置。

**远程模型。** 注册 OpenAI 兼容端点（其他 oMLX Lite 实例、vLLM、OpenAI 等），与本地模型一起使用，支持连接测试和逐模型启用/禁用。

### macOS 菜单栏应用

原生 Swift / SwiftUI 菜单栏应用 —— 非 Electron。无需终端即可启动、停止和监控服务器。包含[本地使用历史](docs/usage-analytics.md)（按模型统计和小时级热力图）、持久化服务统计、崩溃自动重启和内建自动更新。

### 实验性：多 Mac 推理

源码构建可以通过 Ring 或 Thunderbolt RDMA/JACCL 上的 MLX pipeline rank，把单个模型拆分到内存不等的多台 Mac。两台 Mac 上都提供只读节点发现、按字节感知的不等分片规划、基于实测的计算/链路再平衡、考虑余量的调优，以及实时的分片/性能映射。此外还记录了单一异构 MLX + CUDA 资源池的设计。

设置、安全边界和限制请参阅[跨 Mac 分布式推理](docs/distributed-cluster.md)和[异构集群](docs/heterogeneous-cluster.md)。

## 模型

```
~/models/
├── Step-3.5-Flash-8bit/
├── Qwen3-Coder-Next-8bit/
├── gpt-oss-120b-MXFP4-Q8/
├── Qwen3.5-122B-A10B-4bit/
└── bge-m3/
```

## CLI 参考

```bash
# 生命周期（托管后台服务器）
omlx start
omlx stop
omlx restart

# 前台运行
omlx serve --model-dir ~/models

# 内存守卫级别，或自定义 GB 上限
omlx serve --model-dir ~/models --memory-guard safe
omlx serve --model-dir ~/models --memory-guard-gb 48

# 两级缓存
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache
omlx serve --model-dir ~/models --hot-cache-max-size 20%
omlx serve --model-dir ~/models --hot-cache-write-through

# 并发
omlx serve --model-dir ~/models --max-concurrent-requests 16

# 地区 / 镜像端点
omlx serve --model-dir ~/models --hf-endpoint https://hf-mirror.com
omlx serve --model-dir ~/models --ms-endpoint https://modelscope.cn

# 认证
omlx serve --model-dir ~/models --api-key your-secret-key
OMLX_API_KEY=your-secret-key omlx serve --model-dir ~/models --host 0.0.0.0
```

其他命令：`omlx diagnose` 用于安装/运行时诊断，`omlx cluster ...` 用于分布式推理工具。

所有设置都可以在管理面板中编辑，并持久化到 `~/.omlx/settings.json`；CLI 参数优先级更高。在绑定到 LAN 地址或 `0.0.0.0` 之前，请先设置主 API 密钥（或同时保存两项）—— oMLX Lite 在没有密钥时拒绝在任何非回环地址启动，且只有仅回环绑定才能跳过 API 密钥验证。

## 架构

<details>
<summary>请求路径与组件</summary>

```
FastAPI Server (OpenAI / Anthropic API)
    │
    ├── EnginePool (多模型、LRU 驱逐、TTL、手动加载/卸载)
    │   ├── BatchedEngine (LLM；DFlash 与分布式变体)
    │   ├── VLMBatchedEngine (视觉语言 / OCR 模型)
    │   ├── EmbeddingEngine
    │   ├── RerankerEngine
    │   └── TTS / STT / STS 引擎（可选 omlx[audio]）
    │
    ├── ProcessMemoryEnforcer (总内存限制、TTL 检查)
    │
    ├── Scheduler (FCFS，可配置并发数)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack
        ├── PagedCacheManager (GPU，基于块，CoW，前缀共享)
        ├── Hot Cache (内存缓存，write-back)
        └── PagedSSDCacheManager (SSD 冷缓存，safetensors 格式)
```

</details>

## 开发

### 服务器

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
pytest -m "not slow"
```

### macOS 应用

SwiftUI 应用位于 `apps/omlx-mac/`，需要 Xcode 26.5+ 和 Python 3.11+。

```bash
# 暂存可运行的 oMLX Lite.app（xcodebuild + venvstacks 层 + ad-hoc 签名）
apps/omlx-mac/Scripts/build.sh release

# 结果在 apps/omlx-mac/build/Stage/oMLX Lite.app
open apps/omlx-mac/build/Stage/oMLX Lite.app

# 强制重建 venvstacks（默认按指纹缓存）
apps/omlx-mac/Scripts/build.sh release --rebuild-donor

# 包含可选的原生自定义内核
apps/omlx-mac/Scripts/build.sh release --with-custom-kernel
```

首次 cold 构建需要 10–20 分钟（venvstacks 层组装）；后续构建复用 `packaging/_export/`，约 4 分钟完成。层配置见 [packaging/README.md](packaging/README.md)，Swift 源码见 [apps/omlx-mac/](apps/omlx-mac/)。

## 贡献

欢迎贡献 —— 详情见[贡献指南](docs/CONTRIBUTING.md)。Bug 修复、性能优化和文档改进都同样欢迎。

## 许可证

[Apache 2.0](LICENSE)

## 致谢

- [MLX](https://github.com/ml-explore/mlx) 和 [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) —— Apple Silicon 上的视觉语言模型推理
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) —— oMLX Lite 从 vllm-mlx v0.1.0 起步，逐步加入了多模型服务、分层 KV 缓存、完整分页缓存的 VLM、管理面板和 macOS 菜单栏应用
- [venvstacks](https://venvstacks.lmstudio.ai) —— 应用包的便携 Python 环境分层
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) —— Apple Silicon 嵌入模型支持
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) —— Apple Silicon 上的块扩散推测解码
- [MTPLX](https://github.com/youssofal/mtplx) —— Lightning MTP 的 verify-shape Metal 内核，也启发了 depth-k pipeline
- [mlx-serve](https://github.com/ddalcu/mlx-serve) —— 融合 GDN verify prework 内核与 Qwen4 QSA 128-bit K/V staging 内核
- [SiliconScope](https://github.com/kennss/SiliconScope) —— 菜单栏统计的设计与渲染方式，以及节能的重渲染门控
