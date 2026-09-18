# oMLX Lite 性能与优化总览

> 更新：2026-09-18
> 方法：静态代码走查 + 本机微基准（`benchmarks/` 端到端基线尚未建立，见 #12）。
> 结论：每 token Python 路径的两个大 O(n²) 热点已消除（#1/#2，实测）；#3 冷 SSD 预取经实测证伪、已回退；其余瓶颈分布在**性能天花板层（GPU/Metal 算子）**与若干**尚未系统评估的子系统**。

---

## 一、总体优化地图

按「性能天花板 → 可消除开销」自顶向下分层。状态：✅ 已处理 · 🔍 已评估 · ⏸️ 未系统评估（候选）。

| 层 | 子系统 | 主要位置 | 状态 | 备注 |
|---|---|---|---|---|
| **L0 GPU/Metal 算子** | 自定义 Metal kernel | `omlx/custom_kernels/`：`bonsai`、`decode_fast`、`qwen35_prefill`、`glm_moe_dsa`、`minimax_m3`、`nax` | 🔍 已评估·本地无法构建 native | **真正的性能天花板**；本地无 cmake/ninja/nanobind → 全走 MLX fallback；须 `OMLX_WITH_CUSTOM_KERNEL=1` 构建后测 |
| **L1 执行/调度** | 连续批处理、decode burst、异步 `store_cache`、相位计时 | `scheduler.py`、`engine/`、`engine_pool.py` | 🔍 高度优化 | 作者已大量异步化 + 计时；接近瓶颈 |
| **L2 KV 缓存** | 分页 SSD、前缀缓存、边界快照、块哈希链 | `cache/`（`paged_ssd_cache.py`、`prefix_cache.py`、`boundary_snapshot_store.py`） | 🔍 高度优化 | #3 预取已实测证伪并回退 |
| **L3 每 token Python 路径** | `scheduler.step()` 单 worker executor（持 GIL） | `scheduler.py`、`request.py` | ✅ 已消除大热点 | #1/#2 完成；仅剩 #4（不建议做） |
| **L4 内存管理** | 进程 enforcer（水位/轮询/驱逐）、内存监控 | `process_memory_enforcer.py`、`memory_monitor.py`、`cluster/memory_guard.py` | 🔍 已评估·开销可忽略 | 每 tick ~6µs syscall；唯一发现：3 个 admin 路由未缓存 6.2ms `sysctl`（见 4.4） |
| **L5 分布式集群** | planner、张量/流水线策略、JACCL/NCCL、遥测、带宽感知 | `omlx/cluster/`（50+ 文件） | ⛔ 忽略 | 用户决定不做 |
| **L6 量化/压缩** | oQ 数据驱动混合精度、MoE expert offload | `omlx/oq.py`、`docs/oQ_Quantization.md`、`docs/MoE_Expert_Offload.md` | ⏸️ 有公开质量基准，速度未评估 | bpw ↔ 质量/速度/内存权衡 |
| **L7 模型生命周期** | 加载/切换/LRU、渐进加载、模型发现 | `engine_pool.py`、`model_discovery.py`、`cluster/progressive_loading.py` | 🔍 已评估 | 加载持 pool 锁 → 切换时队头阻塞（刻意权衡，修复高风险）；见 4.7 |
| **L8 多模态** | VLM/视觉、STT/TTS、embedding、reranker | `engine/vlm.py`、`engine/{stt,tts,embedding,reranker,sts}.py` | ⏸️ 未系统评估 | 独立引擎 |
| **L9 服务/API** | HTTP、流式、指标、工具调用 | `server.py`、`api/`、`admin/` | 🔍 部分已确认无问题 | 序列化/流式开销 |
| **L10 推测解码** | draft/target、投机 prefill、接受率 | `speculative/`、`specprefill/` | ⏸️ 未系统评估 | draft 开销 vs 加速比 |
| **L11 ANE 卸载** | Apple Neural Engine | `admin/ane_tuning.py`、`benchmarks/qwen35_ane_*` | ⛔ 忽略 | 用户决定不做（POC） |

**总体判断：**

- 原 12 项集中在 **L2/L3 + 工程面**；**L0、L4–L11 此前未系统覆盖**，是「其它方面」的候选池。
- Python 侧每 token 路径（L3）的两个大 O(n²) 已修完（#1/#2），该层基本收尾。
- 真正的收益天花板在 **L0（算子）**，但**必须先有基线**（#12）才能量化。
- **根本约束**：模型 GPU 时间是固定的，Python 侧每 token 开销直接吃掉解码吞吐（L3 已处理）；更上层收益须靠算子优化或量化。

---

## 二、逐项清单（原 12 项）

### 性能项

| # | 优先级 | 位置 | 问题 | 调整方案 | 预期收益 | 风险/工作量 | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | P1 | `omlx/scheduler.py:11470` | 每 token `output_token_ids=list(request.output_token_ids)` 全量拷贝 → O(n²)。中间态无消费者 | 仅在 `is_finished` 时物化，中间步骤传空列表 | 长生成省 80ms–1.5s+ | 低 / 1 行 | ✅ 已完成（实测 32k 省 ~918ms） |
| 2 | P1 | `omlx/scheduler.py:11380`、`:11513` | 协议解析请求每 token `str +=` → O(n²)；属性态字符串 refcount≥2，不触发原地扩展 | `_output_text_parts` 累积 + `"".join()` 读取 | agent 长回复省 0.1–2s | 中 / 3–4 处读取点 | ✅ 已完成（实测 32k 省 ~62ms） |
| 3 | P2 | `omlx/cache/prefix_cache.py:2951` → `paged_ssd_cache.py:3832` | 冷前缀重建逐块同步 `mx.load`（**实测证伪：`mx.load` 懒加载 0.028ms/块，不阻塞**） | 曾试纯 I/O 预读 raw bytes + read-ahead | 实测为回退 | 已回退 | ↩️ 已回退（实测为回退） |
| 4 | P2 | `omlx/scheduler.py:11841` | 请求结束时 owner 线程对全上下文 `mx.eval(*pre_eval_arrays)` | 复用已物化 boundary snapshot；或 `async_eval` + 引擎流 fence | 长上下文完成期停顿消除/降低 | 高 / 跨线程 stream 语义 | 🔍 已评估·不建议做 |
| 5 | P3 | `omlx/scheduler.py:7088-7093` | 块边界 `mx.synchronize` + 全量 `extract_cache` 周期停顿 | 只对 non-sliceable 层快照；评估 `async_eval` | 该类模型解码抖动下降 | 中 / 模型相关 | 🔍 已评审·无需改动 |
| 6 | P3 | `omlx/scheduler.py:3725`、`:3785` | 每 prefill chunk 两次 `get_phys_footprint()` syscall | 按 100–500ms 节流缓存采样 | 微秒级，可忽略 | 低 / 几行 | 🔍 已实施·已回退 |
| 7 | P4 | `omlx/engine_core.py:200-212` | burst 预算使流式按 ~0.1s 批量下发，交互延迟抖动 | 调 `OMLX_DECODE_BURST_*`；或单请求按步 flush | 交互平滑度提升 | 低 / 配置层 | 🔍 已评审·保持默认 |

### 工程 / 维护项

| # | 位置 | 问题 | 方案 | 工作量 | 状态 |
|---|---|---|---|---|---|
| 8 | `scheduler.py`(13.9k)、`server.py`(8.1k)、`oq.py`(9.3k) | 巨型单文件，定位/改动成本高 | 按职责拆分（调度/缓存/解析/采样），保留兼容入口 | 高 | ⏸️ 单独立项 |
| 9 | `omlx/patches/`、scheduler 内 `_patched_*` | 大量 monkey-patch mlx-lm，升级脆弱 | 收敛到单一适配层 + ABI/版本断言 + 冒烟测试 | 中高 | 🟡 部分完成（版本断言+冒烟；全量收敛待做） |
| 10 | `OMLX_*` 环境变量散落 | 调参入口分散、缺文档 | 统一到 `config.py`/文档，env 仅作覆盖 | 中 | 🟡 部分完成（清单+漂移守卫；调用点迁移待做；实测 134 变量/49 文件） |
| 11 | `scheduler._check_memory_pressure`(pass)、`_get_current_memory_usage`(恒 0)、`_get_process_rss` | 死路径；MemoryMonitor 已转为纯估算器 | 删除或标注 deprecated | 低 | ✅ 已完成 |
| 12 | `benchmarks/` | GPU 算子层缺 CI 基线 | 固定模型/序列建立回归基线并接入 CI | 中 | ✅ 已完成（合成算子软门 + Python 硬门） |

### 已确认无问题（复核通过，勿动）

- `get_phys_footprint` 非热路径调用（enforcer 自适应轮询 1s/10s/30s）
- SSD 索引纯内存 `dict` + `OrderedDict` LRU，无每块落盘 I/O（`paged_ssd_cache.py:1076-1098`）
- 指标限速落盘（`server_metrics.py:151`）、usage history 异步 SQLite（`usage_history.py`）
- SSE keepalive 事件驱动（`server.py:2430`）、输出 collector 单槽聚合（`output_collector.py`）
- 块哈希链式 O(1)/块（`paged_cache.py:78`）、stop 串只扫尾部（`scheduler.py:11414-11431`）
- 已文档化的权衡：非编译采样器、每 1024 token 的 `mx.synchronize`+`clear_cache`（`scheduler.py:12634`）、MiniMax-M3 周期性 cache eval（`scheduler.py:12597`）、每请求全量 KV 落盘

---

## 三、实测结果（本机微基准，非估计）

- **#1 `output_token_ids`**：8k / 16k / 32k token 分别省 ~48ms / ~212ms / **~918ms**（O(N²)→O(N)）→ 保留。
- **#2 `output_text`**：8k / 16k / 32k token 分别省 ~2.5ms / ~15.6ms / ~62ms（O(N²)→O(N)）→ 保留（绝对量小于原估计 0.1–2s）。
- **#3 预取**：16 块 × 8MB，serial 1.75ms vs 预取 9.34ms（**慢 5.3×**），预取命中 **0/16** → 回退。根因：`mx.load` 懒加载（0.028ms/块），Python raw-bytes 读取慢 ~36×，worker 追不上循环，预取纯浪费。

---

## 四、待调整总清单

> 汇总全部结论。**任何结论须先实测**（#3 的教训）。优先级顺序见第五节。

### 4.1 立即可做 · 低危

| # | 位置 | 问题 | 调整 | 工作量 |
|---|---|---|---|---|
| **A1** | `omlx/admin/routes.py:3568`、`:6106`、`:6511` | 3 个 admin 路由同步 spawn `sysctl` = 6.2ms，阻塞事件循环 | 给 `get_iogpu_wired_limit_bytes` 加 30s TTL 缓存 + `force_refresh` 绕过 | ✅ 已完成（11.4ms → 0.002ms） |

### 4.2 前置项（必须先做）

| # | 位置 | 问题 | 调整 | 工作量 |
|---|---|---|---|---|
| **B1 = #12** | `benchmarks/` | GPU 算子层无 CI 基线，无法证伪任何优化 | 合成算子基准 + 基线 JSON + CI 软门；Python 侧 CI 硬门 | ✅ 已完成 |

### 4.3 工程 / 维护（单独立项，无直接运行时收益）

| # | 位置 | 问题 | 调整 | 工作量 |
|---|---|---|---|---|
| **C1 = #10** | `OMLX_*` env（实测 **134 变量 / 49 文件**，非原估 299/167） | 调参入口分散、缺索引 | ① 自动生成 `docs/ENV_VARS.md` + 漂移守卫 ✅ ② 调用点迁移到 `config.py` ⏸️ | ① 低 ✅ / ② 中高 |
| **C2 = #8** | `scheduler.py`(13.9k)/`server.py`(8.1k)/`oq.py`(9.3k) | 巨型文件 | 按职责拆分 | 高 |
| **C3 = #9** | `omlx/patches/`（~60 模块）+ `_patched_*` | monkey-patch 升级脆弱 | ① 中央版本 pins + 运行时断言 + 冒烟测试 ✅ ② 全量收敛单一适配层 ⏸️（巨型重构） | ① 低 ✅ / ② 很高 |

### 4.4 候选层（需调研 + 实测，范围大）

| 层 | 入口 | 关注点 |
|---|---|---|
| **D1 = L0** | `custom_kernels/`、`benchmarks/*.py` | 🔍 已评估：**本地无 cmake/ninja/nanobind，native kernel 未构建** → 全走 MLX fallback（已实测并记入基线）；须 `OMLX_WITH_CUSTOM_KERNEL=1` 构建后测 native |
| ~~D2 = L5~~ | `cluster/`、`tensor_strategies.py`、`runtime_optimizations.py`、`telemetry.py` | ⛔ 忽略（用户决定，多节点成本过高） |
| **D3 = L6** | `oq.py`、MoE offload、`docs/oQ_Quantization.md`、`docs/MoE_Expert_Offload.md` | 🔍 已评估：offload 的 TTFT 瓶颈 = **同步 fetch on miss**；作者已在 `docs/MoE_Expert_Offload.md` 记录后续项（prefill 可精确预算专家调度 + decode 预取 +17pp 命中），**非新发现** |
| **D4 = L7** | `engine_pool.py`、`model_discovery.py`、`cluster/progressive_loading.py` | 🔍 已评估（见 4.7）：加载持 pool 锁致切换队头阻塞，刻意权衡 |
| **D5 = L8** | `engine/vlm.py`、`engine/{stt,tts,embedding,reranker}.py`、`utils/image.py` | 视觉编码器 prefill 开销；图像解码缓存 |
| **D6 = L10** | `speculative/`、`specprefill/` | 🔍 已评估：opt-in；`MTPProcessingSampler.sample_target` 每 slot 重扫全 `_history`（无状态 penalty）+ 每 slot 一次 `item()` GPU sync；SpecPrefill 准入策略简单。无安全快赢 |
| ~~D7 = L11~~ | `admin/ane_tuning.py`、`benchmarks/qwen35_ane_*` | ⛔ 忽略（用户决定，需 ANE 硬件） |

### 4.5 已关闭（不动）

- **已完成**：#1（实测 32k 省 ~918ms）、#2（实测 32k 省 ~62ms）、#11。
- **已回退**：#3（实测为回退，见第七节）。
- **不建议做**：#4（full-eval 是防 `SIGABRT` 的刻意设计）。
- **无需改动**：#5 / #6 / #7（已评审）。
- **L4 内存管理**：已深入评估，开销可忽略，仅剩 A1（详见 4.6）。

### 4.6 L4 内存管理（已深入评估，实测）

- **架构**：2 水位（soft/hard）+ 紧急制动；自适应轮询 1s（活跃/压力）/10s（loaded idle）/30s（unloaded idle）；`ceiling = min(static, dynamic, metal_cap)`。
- **每 tick 开销（实测）**：`get_phys_footprint` 0.96µs（~3×/tick）、`get_macos_vm_stats` 1.37µs（~2×/tick）、`get_effective_metal_cap` 6.2ms 但**已缓存**（仅 start/切 tier/custom 变更刷新）、遍历 entries O(N) µs 级 → 每 tick ~6µs，1s 间隔 ≈0.001% CPU。**开销可忽略。**
- **唯一可动项（= A1，低危）**：`get_iogpu_wired_limit_bytes()` spawn `sysctl` = **6.2ms**。enforcer 内部已缓存，但 3 个 admin 路由**同步调未缓存**的 `get_effective_metal_cap_bytes()`：`GET /api/global-settings`（`admin/routes.py:3568`）、`/api/hf/recommended`（:6106）、`/api/ms/recommended`（:6511）→ 每次请求**阻塞事件循环 6.2ms**（期间流式推理回调停）。修复：改读 enforcer 缓存值，或给 sysctl 读加 TTL 缓存。
- **可忽略微冗余（不建议改）**：`_get_ceiling_breakdown()` 每 tick 算 2 次 + `_current_usage_bytes()` 再算 1 次 `get_phys_footprint`（省 ~4µs/tick）；`_propagate_memory_limit()` 无条件遍历 + 设 15 属性；`_walk_store_cache_caps()` 每 tick 步进。
- **行为（刻意设计，非 bug）**：soft（85–92.5%）时若 >1 非 pinned 模型则 abort+evict LRU（反抖动）；soft 暂停新 admission（in-flight 继续）。

### 4.7 L7 模型生命周期（已深入评估）

- **架构**：`EnginePool` 管多模型 LRU 驱逐 + 预加载内存检查；请求经 `server.get_engine_for_model` → `pool.get_engine(_lease=True)` 取 engine 租约。
- **发现（真实，刻意权衡）**：`get_engine`（`engine_pool.py:1737`）在 `async with self._lock`（`asyncio.Lock`）内 `await self._load_engine(...)`（`:2000`）——加载（大模型数秒：分片读 + 反量化 + warmup）期间**整个 pool 锁被持有**，而 `_load_engine` 自身不取锁（靠调用方）。由于**每个请求**都走 `get_engine` 取租约（`server.py:3363/3907/6266/6635/6754`），一次加载会**阻塞所有并发请求**（含已加载的其他模型）→ 多模型动态切换 / L4 驱逐重载时出现队头阻塞延迟尖峰。
  - 修复非平凡：需「按模型 loading 状态 + 锁外加载 + 锁内仅做准入/安装」，且准入/驱逐需与锁原子，有竞态风险。当前串行化是**加载安全的刻意权衡**（避免重复加载 / 内存超配 / 驱逐竞态）。
- **可忽略**：每请求 `_engine_runtime_signature`（`settings.to_dict` + 配置遍历）与 `_current_ceiling`（仅 qwen4/deepseek，2 次 syscall）——µs 级。
- **启动期**：`discover_models` 目录扫描、`preload_pinned_models` 串行预加载（仅启动一次）。

---

## 五、建议执行顺序

```
A1 → B1(#12 基线) → D1(L0 算子实测) → D4/D3(L7/L6) → D2/D6(L5/L10) → C1(#10) → C2/C3(#8/#9)
```

- **A1 顺手**：低危小改，先清掉。
- **B1(#12) 先行**：没有基线，任何「优化」都无法证伪（见 #3 教训）。
- **D1 算子**：收益天花板最高。
- **D4/D3**：跨模型/请求的延迟与内存，用户可感知。
- **D2/D6**：特定场景（多节点/长上下文）大，但范围大。
- **#4 关闭**，除非出现线程流安全的新方案。

---

## 六、验证方法

| # | 验证 |
|---|---|
| 1 | 长输出（≥32k token）单请求基准，比较 decode tok/s 与端到端耗时 |
| 2 | 工具调用模型长回复流式基准，比较累计耗时；断言 `output_text` 最终值不变 |
| 3 | 冷 SSD 缓存 + 长前缀，测量 TTFT；确认无 `no Stream(gpu, N)` / SIGABRT（**已实测证伪，勿重启**） |
| 4 | 长上下文请求完成期停顿（`store_cache_main_dispatch` 相位计时） |
| 5 | Mamba/GDN 类模型解码抖动（p99 inter-token latency） |

> 所有改动前先跑 `benchmarks/` 建立基线；涉及缓存的改动需跑 `tests/` 全量回归。**先实测假设，再实现**（#3 教训）。

---

## 七、实施记录（2026-09-18）

### 已完成

- **#1**：`omlx/scheduler.py:11470` — `output_token_ids=list(request.output_token_ids)` 改为 `list(request.output_token_ids) if is_finished else []`。中间步不再 O(n) 拷贝，finish 路径仍物化完整列表（`output_collector.py:143` merge 取 latest、`scheduler.py:6090` 读取，均安全）。
- **#2**：`omlx/request.py` 将 `output_text` 由 dataclass 字段改为 property（backed by `_output_text_parts: list[str]`），新增 `append_output_text()`；`omlx/scheduler.py:11380/11518` 两处 `output_text +=` 改为 `append_output_text()`。消除每 token `str +=` 的 O(n²)。
- **#11**：删除 `scheduler.py` 的 `_check_memory_pressure`（pass）；删除 `memory_monitor.py` 的 `_get_process_rss`（未用）；内联 `_get_current_memory_usage()` → `used = 0`。
- **A1**（L4 内存）：`process_memory_enforcer.py` 的 `get_iogpu_wired_limit_bytes`（`sysctl` 子进程，实测 ~6–11ms）加 30s TTL 缓存并拆出未缓存的 `_read_iogpu_wired_limit_bytes`；`get_effective_metal_cap_bytes` 与 enforcer `_refresh_effective_metal_cap_bytes`、`_apply_metal_wired_limit` 加 `force_refresh`，保留「配置变更时立即读最新」语义。修复 3 个 admin 路由（`/api/global-settings`、`/api/hf/recommended`、`/api/ms/recommended`）每次请求同步 spawn `sysctl` 阻塞事件循环的问题。实测 **11.4ms → 0.002ms**（缓存命中），3 个新测试（`TestIOGPUWiredLimitCache`）。
- **B1 = #12**（基线）：
  - **Python CI 硬门** `tests/test_perf_regression.py`：机器无关的复杂度比例断言（#2 `append_output_text`+join 在 4× 尺寸下 <9×，线性≈4 / 二次≈16）+ A1 缓存命中；另含两个**源码形态 tripwire**，直接守卫 scheduler 的 #1（`output_token_ids=list(...) if is_finished else []`）与 #2（无 `request.output_text +=`）调用点（这两处需加载模型才能计时）。共 5 测试、约 2.3s，进默认 CI。
  - **GPU CI 硬门** `benchmarks/operator_baseline.py` + `benchmarks/operator_baseline.json`：合成算子固定 shape 计时，与提交基线按 **归一化比例** 对比，默认容差 1.5×；并记录 `native`（各 custom kernel 包 `has_native()`）以免拿 fallback 与 native 基线对比。**机器无关**：每算子用同轮 compute 参考 matmul（4096² bf16）归一化，基线存 `op_ms/ref_ms` 比例——CI runner（macos-14 = M1）与开发机（M3 Max）绝对耗时差 5–6×，但比例稳定（实测同机 4 轮 ±10%）。`ci.yml` 的 `operator-benchmark` 任务（macos-14）已**去掉 `continue-on-error`，为硬门**；模拟 3× 回归实测正确 exit 1。本机基线：ref 10.8ms；qmv 0.17 / qmm 11.9 / sdpa_d 0.39 / sdpa_p 24.3 / moe 47.8 ms（M3 Max, mlx 0.32.2）。
- **C1 = #10**（部分）：新增 `scripts/gen_env_docs.py` 从 `omlx/` 扫描 `OMLX_*` 静态字面量生成 `docs/ENV_VARS.md`（实测 **134 变量 / 49 文件**，修正原「299/167」估计）；新增 `tests/test_env_var_inventory.py` 漂移守卫（文档与代码不一致即失败，进默认 CI）。**仅做清单/文档（低风险 additive）；调用点迁移到 `config.py` 待做（中高风险）。**

- **C3 = #9**（部分）：新增 `omlx/patches/_compat.py` 集中 `REQUIRED_VERSIONS`（mlx 0.32.2 / mlx-lm 0.31.3 / mlx-vlm 0.7.1 / mlx-embeddings 0.1.0）+ `check_pins()` / `assert_pins_match()`；`omlx/patches/__init__.py` 在导入时调用（版本漂移即告警，非致命）；新增 `tests/test_patches_compat.py`（pins vs 运行环境 + 代表性 patch 生效冒烟：`llama4_attention` 标记）。**仅做版本断言 + 冒烟（低风险）；~60 个 patch 模块的全量收敛为单一适配层待做（巨型重构）。**

### D1 = L0 算子层（↩️ 本地受阻）

- **结论**：自定义 Metal kernel（`bonsai`/`decode_fast`/`minimax_m3`/`qwen35_prefill`/`glm_moe_dsa`）经 CMake+nanobind 构建，本环境**无 cmake/ninja/nanobind、无已编译 `.so`/`.metallib`**，`has_native()` 全为 False → 所有模型路径走 **stock MLX fallback**，无法在此测 native 性能。
- **已做**：`benchmarks/operator_baseline.py` 增 `_native_status()`，基线 JSON 记录各包原生可用性并每次运行打印；`bench_m5_sorted_gather_chunk.py` 实测 MoE 路径已调优（`segmented`≈`sorted`：27.8 / 49.3ms per layer @ chunk 2048 / 4096，远快于 `unsorted` 91 / 183ms）。
- **native L0 实测前置**：`OMLX_WITH_CUSTOM_KERNEL=1 python setup.py build_ext`（需 cmake≥3.27 + ninja + `nanobind==2.15.0` + MLX 头文件）或装带 kernel 的 release wheel，再用现有 per-kernel harness（`bonsai_decode_bench.py`、`qwen35_ane_*`、`deepseek_v41_offload_bench.py`）以模型 shape 实测。

### 已评审·无需改动

- **#5**：块边界快照的核心优化已实现——`_detect_boundary_snapshot_need`（`scheduler.py:6982`）已限制仅 hybrid 模型，`:7117-7126` 已过滤 non-sliceable 层；`extract_cache` 是 core BatchGenerator 方法（外部/patched），改动风险高。
- **#6**：实施 `get_phys_footprint` 节流后破坏 `test_prefill_oom_graceful.py`（测试 mock 不同 pre/post 值，节流使 delta=0）；收益微秒级可忽略，已回退。
- **#7**：decode burst 是有意的吞吐优化（80 vs 74 tok/s），0.1s 单请求预算给 ~10 updates/sec 已足够平滑；配置已暴露（`OMLX_DECODE_BURST_*`）且文档化，改默认值会回退作者有意优化。

### 单独立项（范围大，本次不做）

- **#8 / #9**：巨型文件拆分、monkey-patch 收敛，均为长期工程项。

### P2 深入评估（#3 / #4）

- **#3 冷 SSD 重建（↩️ 已回退：实测为回退，非收益）**：
  - 原假设「同步 `mx.load`（文件 I/O + 数组构造）逐块阻塞，长冷前缀 TTFT 可达数秒」经**实测证伪**。
  - 曾实现（`_read_safetensors_no_mx` 纯 Python 读 + 单 worker 纯 I/O 预取 executor + `load_block_with_metadata` 优先取预取 + 重建循环 read-ahead），提交后基准测试显示为**回退**，已撤销该本地未推送提交（`5d2cf40` 已从 `master` 移除，HEAD 回到 `66b5524`）。
  - **实测数据**（16 块 × 8MB = 128MB，page-warm，本机）：

    | 路径 | 总耗时 | 每块 |
    |---|---|---|
    | serial（现状 `mx.load`） | 1.75 ms | 0.109 ms |
    | 预取版 | 9.34 ms | 0.583 ms |
    | 预取命中 | **0 / 16** | — |

    组件：`mx.load` 0.028ms/块（**懒加载**）；`mx.load`+`mx.eval` 0.358ms/块（真正物化被推迟）；裸 `f.read()` 0.576ms/块；Python no-mx 读取器 1.004ms/块；no-mx 读+重建 1.282ms/块。
  - **根因**：`mx.load` 懒加载、返回几乎零成本，真正物化推迟到模型前向；Python 层 raw-bytes 读取（读+解析+拷贝）比懒 `mx.load` 慢 ~36×。预取 worker（1.0ms/块）远慢于循环（0.11ms/块），循环从不等它 → 预取 0 命中、全部回退 `mx.load`，且后台读取白占 CPU/GIL，把重建拖慢 **5.3×**。
  - **教训**：「`mx.load` 阻塞」是未经验证的静态假设；涉及 GPU/IO 的优化必须先实测再实现。当时的 5 个单测（`TestRawPrefetch`）只验证机制正确性，未验证收益。
- **#4 store_cache 全量 eval（不建议做）**：
  - `scheduler.py:11825-11846` 作者**已刻意选 full eval 而非 async_eval**，注释给出崩溃根因：KV 数组携带线程局部 `self._stream`，lazy 状态被 worker 物化会 `There is no Stream(gpu, N)` → `SIGABRT`。**已把 host memcpy + 磁盘写移到 worker**（`:11842-11845`），owner 线程只留 GPU 完成 fence。
  - 两个优化选项均不理想：复用已物化 boundary snapshot 需触碰已精细处理的 Metal 线程局部流语义（高风险）；`async_eval` 正是 `SIGABRT` 根因（作者已否决）。
  - 结论：当前 full-eval 是带崩溃分析的成熟安全措施，剩余优化收益不确定 + 高风险，维持现状。

### 验证

- 核心测试：`test_scheduler.py` + `test_request.py` + `test_output_collector.py` + `test_memory_monitor.py` = **436 passed**。
- #3 回退后：`test_paged_ssd_cache.py` + `test_prefix_cache.py` = **301 passed**。
- 全量回归：6 文件子集（含 scheduler/server/engine_pool）stash 前后均 **142 failed / 30 errors**，完全一致 → **本次改动零回归**。
- 全量套件 284 failed / 41 errors 为预存失败（`ImportError: cannot import name 'hc_fused' from 'mlx_vlm.models.qwen4_exp'` 等依赖/环境问题），与本次改动无关。
