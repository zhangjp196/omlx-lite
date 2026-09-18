# OMLX_* Environment Variables

Auto-generated from `omlx/` by `scripts/gen_env_docs.py` — 134 variables across 49 files. Do not edit by hand; run the generator and commit the result.

| Variable | Occurrences | Locations |
|---|---|---|
| `OMLX_API_KEY` | 1 | `omlx/settings.py:1009` |
| `OMLX_BASE_PATH` | 5 | `omlx/cluster/discovery.py:806`, `omlx/cluster/identity.py:40`, `omlx/cluster/registry.py:207`, `omlx/cluster/worker_shim.py:46` (+1) |
| `OMLX_BONJOUR` | 1 | `omlx/server.py:452` |
| `OMLX_CACHE_ENABLED` | 1 | `omlx/settings.py:971` |
| `OMLX_CHUNK_SNAP` | 1 | `omlx/scheduler.py:1561` |
| `OMLX_CLUSTER_CONTROL_PROXY_PYTHON` | 1 | `omlx/cluster/system_socket_proxy.py:96` |
| `OMLX_CLUSTER_CONTROL_TRANSPORT` | 2 | `omlx/cluster/control_plane.py:315`, `omlx/cluster/system_socket_proxy.py:106` |
| `OMLX_CLUSTER_LAUNCHER_LEASE` | 1 | `omlx/cluster/inference_worker.py:836` |
| `OMLX_CLUSTER_PEER_ABORT_GRACE` | 1 | `omlx/cluster/inference_worker.py:810` |
| `OMLX_CLUSTER_SIGNAL_CLEAR_TIMEOUT` | 1 | `omlx/cluster/inference_worker.py:588` |
| `OMLX_CLUSTER_SSH_HOST_PUBLIC_KEY` | 1 | `omlx/cluster/pairing.py:745` |
| `OMLX_CLUSTER_STATE_DIR` | 1 | `omlx/cluster/performance_worker.py:122` |
| `OMLX_CONTENDED_PREFILL_CHUNK` | 1 | `omlx/scheduler.py:1543` |
| `OMLX_CONTINUOUS_BATCHING` | 1 | `omlx/config.py:240` |
| `OMLX_DECODE_BURST_BUDGET_S` | 1 | `omlx/engine_core.py:210` |
| `OMLX_DECODE_BURST_BUDGET_SINGLE_S` | 2 | `omlx/engine_core.py:205`, `omlx/settings.py:161` |
| `OMLX_DECODE_BURST_MAX_STEPS` | 2 | `omlx/engine_core.py:201`, `omlx/settings.py:160` |
| `OMLX_DECODE_EVAL_KV_CACHE_INTERVAL` | 1 | `omlx/scheduler.py:2104` |
| `OMLX_DECODE_FAIR_SHARE` | 1 | `omlx/scheduler.py:1533` |
| `OMLX_DECODE_STALL_TARGET_MS` | 1 | `omlx/scheduler.py:1540` |
| `OMLX_DEEPSEEK_AFFINE_BLOCK_MIN_ROUTES` | 1 | `omlx/patches/deepseek_v4/switch_layers.py:28` |
| `OMLX_DEEPSEEK_MOE_NAX` | 1 | `omlx/patches/deepseek_v4/switch_layers.py:42` |
| `OMLX_DEEPSEEK_MOE_NAX_MIN_ROUTES` | 1 | `omlx/patches/deepseek_v4/switch_layers.py:44` |
| `OMLX_DEEPSEEK_MXFP4_LARGE_BLOCK_MIN_ROUTES` | 1 | `omlx/patches/deepseek_v4/switch_layers.py:33` |
| `OMLX_DEEPSEEK_SORT_MIN_ROUTES` | 1 | `omlx/patches/deepseek_v4/switch_layers.py:26` |
| `OMLX_DISABLE_PRESSURE_RECLAIM` | 2 | `omlx/process_memory_enforcer.py:1541`, `omlx/process_memory_enforcer.py:1574` |
| `OMLX_DISCOVERY` | 1 | `omlx/server.py:477` |
| `OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT` | 1 | `omlx/engine/distributed.py:189` |
| `OMLX_DSV4F_M2_MMA_SCORE` | 1 | `omlx/patches/deepseek_v4/deepseek_v4_model.py:489` |
| `OMLX_DSV4_WSDPA` | 1 | `omlx/patches/deepseek_v4/wsdpa_attention.py:24` |
| `OMLX_DSV4_WSDPA_TOPK` | 1 | `omlx/patches/deepseek_v4/wsdpa_attention.py:158` |
| `OMLX_EMBEDDING_BATCH_SIZE` | 1 | `omlx/settings.py:962` |
| `OMLX_EMBEDDING_COMPILE` | 1 | `omlx/models/embedding.py:673` |
| `OMLX_FA256_DEBUG` | 1 | `omlx/patches/qwen35_fa256_attention.py:218` |
| `OMLX_FA256_DISPATCH_BUDGET` | 1 | `omlx/patches/qwen35_fa256_attention.py:219` |
| `OMLX_FA256_K_BLOCK` | 1 | `omlx/patches/qwen35_fa256_attention.py:217` |
| `OMLX_FA256_MIN_KV_LEN` | 1 | `omlx/patches/qwen35_fa256_attention.py:215` |
| `OMLX_FA256_Q_BLOCK` | 1 | `omlx/patches/qwen35_fa256_attention.py:216` |
| `OMLX_FA256_STEEL` | 1 | `omlx/patches/qwen35_fa256_attention.py:196` |
| `OMLX_GDN_BLOCK_T` | 1 | `omlx/custom_kernels/qwen35_prefill/gdn.py:535` |
| `OMLX_GDN_FUSED_G_BETA` | 1 | `omlx/patches/qwen35_gdn_chunked.py:56` |
| `OMLX_GDN_IMPL` | 1 | `omlx/patches/qwen35_gdn_chunked.py:65` |
| `OMLX_GDN_KERNEL` | 1 | `omlx/patches/qwen35_gdn_chunked.py:43` |
| `OMLX_GDN_MIN_T` | 1 | `omlx/patches/qwen35_gdn_chunked.py:55` |
| `OMLX_GDN_SIDECAR_STATE_DTYPE` | 2 | `omlx/config.py:227`, `omlx/settings.py:998` |
| `OMLX_GDN_SNAPSHOT_STORAGE` | 2 | `omlx/config.py:212`, `omlx/settings.py:985` |
| `OMLX_GDN_SSD_PENDING_MAX_SIZE` | 2 | `omlx/config.py:223`, `omlx/settings.py:996` |
| `OMLX_GDN_SSD_SPLIT_ENABLED` | 3 | `omlx/config.py:218`, `omlx/config.py:220`, `omlx/settings.py:990` |
| `OMLX_GDN_STUB` | 1 | `omlx/patches/qwen35_gdn_chunked.py:57` |
| `OMLX_HF_CACHE_ENABLED` | 1 | `omlx/settings.py:1015` |
| `OMLX_HF_ENDPOINT` | 1 | `omlx/settings.py:1013` |
| `OMLX_HOST` | 2 | `omlx/config.py:192`, `omlx/settings.py:930` |
| `OMLX_HOT_CACHE_ONLY` | 2 | `omlx/config.py:211`, `omlx/settings.py:977` |
| `OMLX_HOT_CACHE_WRITE_THROUGH` | 1 | `omlx/settings.py:979` |
| `OMLX_INITIAL_CACHE_BLOCKS` | 1 | `omlx/settings.py:1000` |
| `OMLX_INKLING_MTP_FINAL_NORM` | 1 | `omlx/patches/mlx_vlm_mtp/inkling_vlm_runtime.py:168` |
| `OMLX_INKLING_MTP_PRIME_WINDOW` | 1 | `omlx/patches/mlx_vlm_mtp/inkling_vlm_runtime.py:157` |
| `OMLX_INKLING_SLIDING_SLICE` | 1 | `omlx/patches/mlx_vlm_inkling_compat/vendor/mlx_vlm/models/inkling/language.py:27` |
| `OMLX_JACCL_PYTHON_SIDE_CHANNEL` | 1 | `omlx/cluster/jaccl_side_channel.py:164` |
| `OMLX_JACCL_SIDE_CHANNEL_PYTHON` | 1 | `omlx/cluster/jaccl_side_channel.py:333` |
| `OMLX_JACCL_SIDE_CHANNEL_TIMEOUT_SECONDS` | 2 | `omlx/cluster/jaccl_side_channel.py:242`, `omlx/cluster/jaccl_side_channel.py:367` |
| `OMLX_JACCL_SIDE_CHANNEL_TRACE` | 1 | `omlx/cluster/jaccl_side_channel.py:229` |
| `OMLX_JACCL_SIDE_CHANNEL_TRANSPORT` | 1 | `omlx/cluster/jaccl_side_channel.py:341` |
| `OMLX_LAGUNA_COMPILED_FUSIONS` | 1 | `omlx/patches/laguna/laguna_model.py:37` |
| `OMLX_LAGUNA_FUSED_ROUTED_GATE_UP` | 1 | `omlx/patches/laguna/laguna_model.py:118` |
| `OMLX_LAGUNA_FUSED_SHARED_GATE_UP` | 1 | `omlx/patches/laguna/laguna_model.py:124` |
| `OMLX_LOG_DIR` | 1 | `omlx/settings.py:1028` |
| `OMLX_LOG_LEVEL` | 2 | `omlx/config.py:194`, `omlx/settings.py:937` |
| `OMLX_LOG_RETENTION_DAYS` | 1 | `omlx/settings.py:1030` |
| `OMLX_M5_GATHER_QMM_FIX` | 1 | `omlx/patches/m5_gather_qmm.py:182` |
| `OMLX_MAX_AUDIO_UPLOAD_SIZE` | 1 | `omlx/settings.py:943` |
| `OMLX_MAX_CONCURRENT_REQUESTS` | 1 | `omlx/settings.py:952` |
| `OMLX_MAX_NUM_SEQS` | 1 | `omlx/settings.py:953` |
| `OMLX_MAX_TOKENS` | 1 | `omlx/config.py:204` |
| `OMLX_MODEL` | 1 | `omlx/config.py:197` |
| `OMLX_MODEL_DIR` | 1 | `omlx/settings.py:947` |
| `OMLX_MOE_EXPERT_OFFLOAD` | 6 | `omlx/engine/vlm.py:1835`, `omlx/engine_pool.py:424`, `omlx/engine_pool.py:548`, `omlx/patches/moe_expert_offload.py:517` (+2) |
| `OMLX_MS_ENDPOINT` | 1 | `omlx/settings.py:1024` |
| `OMLX_MTP_PRIME_WINDOW` | 1 | `omlx/patches/mlx_lm_mtp/prompt_priming.py:99` |
| `OMLX_MTP_PROMPT_PRIMING` | 1 | `omlx/patches/mlx_lm_mtp/prompt_priming.py:81` |
| `OMLX_MTP_ROWWISE_BATCH` | 1 | `omlx/patches/mlx_lm_mtp/batch_generator.py:429` |
| `OMLX_NAX` | 1 | `omlx/custom_kernels/qwen35_prefill/fast.py:933` |
| `OMLX_OQ_A8` | 3 | `omlx/patches/qwen35_oq_a8.py:78`, `omlx/patches/qwen35_oq_a8.py:85`, `omlx/patches/qwen35_oq_a8.py:89` |
| `OMLX_OQ_A8_ACT_MODE` | 1 | `omlx/custom_kernels/qwen35_prefill/fast.py:1203` |
| `OMLX_OQ_A8_MIN_TOKENS` | 1 | `omlx/patches/qwen35_oq_a8.py:371` |
| `OMLX_OQ_A8_Q4_ACT_MODE` | 1 | `omlx/patches/qwen35_oq_a8.py:137` |
| `OMLX_OQ_A8_Q4_VARIANT` | 1 | `omlx/patches/qwen35_oq_a8.py:160` |
| `OMLX_OQ_A8_Q5_ACT_MODE` | 1 | `omlx/patches/qwen35_oq_a8.py:136` |
| `OMLX_OQ_A8_Q5_VARIANT` | 1 | `omlx/patches/qwen35_oq_a8.py:159` |
| `OMLX_OQ_A8_VARIANT` | 2 | `omlx/custom_kernels/qwen35_prefill/fast.py:1202`, `omlx/patches/qwen35_oq_a8.py:157` |
| `OMLX_PAGED_SSD_CACHE_DIR` | 1 | `omlx/config.py:230` |
| `OMLX_PAGED_SSD_CACHE_MAX_SIZE` | 1 | `omlx/config.py:235` |
| `OMLX_PORT` | 2 | `omlx/config.py:193`, `omlx/settings.py:932` |
| `OMLX_PRESERVE_MID_SYSTEM_CACHE` | 1 | `omlx/settings.py:939` |
| `OMLX_QWEN35_ANE_BANK_MAX_BYTES` | 2 | `omlx/patches/qwen35_ane_prefill.py:2185`, `omlx/patches/qwen35_ane_prefill.py:2308` |
| `OMLX_QWEN35_ANE_COMPILE_CACHE` | 3 | `omlx/admin/routes.py:4234`, `omlx/admin/routes.py:4236`, `omlx/cli.py:113` |
| `OMLX_QWEN35_ANE_DOWN_COMBINED_BANK` | 1 | `omlx/patches/qwen35_ane_prefill.py:2593` |
| `OMLX_QWEN35_ANE_DOWN_LAYER_STRIDE` | 1 | `omlx/patches/qwen35_ane_prefill.py:2581` |
| `OMLX_QWEN35_ANE_PREFILL` | 1 | `omlx/patches/qwen35_ane_prefill.py:3168` |
| `OMLX_QWEN35_MOE_GATE_UP` | 1 | `omlx/patches/qwen35_moe_gate_up.py:204` |
| `OMLX_QWEN35_MOE_WEIGHTED_SUM` | 2 | `omlx/patches/qwen35_moe_weighted_sum.py:52`, `omlx/patches/qwen35_moe_weighted_sum.py:174` |
| `OMLX_QWEN35_MOE_WEIGHTED_SUM_MIN_TOKENS` | 1 | `omlx/patches/qwen35_moe_weighted_sum.py:181` |
| `OMLX_QWEN35_Q4_LINEAR` | 3 | `omlx/patches/qwen35_q4_mlp.py:372`, `omlx/patches/qwen35_q4_mlp.py:405`, `omlx/patches/qwen35_q4_mlp.py:418` |
| `OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS` | 3 | `omlx/patches/qwen35_ane_prefill.py:309`, `omlx/patches/qwen35_q4_mlp.py:393`, `omlx/patches/qwen35_q4_mlp.py:466` |
| `OMLX_QWEN35_Q4_LINEAR_VARIANT` | 2 | `omlx/patches/qwen35_q4_mlp.py:392`, `omlx/patches/qwen35_q4_mlp.py:465` |
| `OMLX_QWEN35_Q4_LM_LINEAR` | 3 | `omlx/patches/qwen35_q4_mlp.py:454`, `omlx/patches/qwen35_q4_mlp.py:476`, `omlx/patches/qwen35_q4_mlp.py:593` |
| `OMLX_QWEN35_Q4_MLP` | 4 | `omlx/patches/qwen35_q4_mlp.py:272`, `omlx/patches/qwen35_q4_mlp.py:328`, `omlx/patches/qwen35_q4_mlp.py:721`, `omlx/patches/qwen35_q4_mlp.py:798` |
| `OMLX_QWEN35_Q4_MLP_ALLOW_GS128` | 1 | `omlx/patches/qwen35_q4_mlp.py:142` |
| `OMLX_QWEN35_Q4_MLP_MIN_TOKENS` | 2 | `omlx/patches/qwen35_q4_mlp.py:335`, `omlx/patches/qwen35_q4_mlp.py:805` |
| `OMLX_QWEN35_Q4_MLP_VARIANT` | 2 | `omlx/patches/qwen35_q4_mlp.py:334`, `omlx/patches/qwen35_q4_mlp.py:804` |
| `OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS` | 4 | `omlx/patches/qwen35_ane_prefill.py:311`, `omlx/patches/qwen35_q4_mlp.py:249`, `omlx/patches/qwen35_q4_mlp.py:395`, `omlx/patches/qwen35_q4_mlp.py:468` |
| `OMLX_QWEN35_Q8_MLP_MIN_TOKENS` | 3 | `omlx/patches/qwen35_ane_prefill.py:957`, `omlx/patches/qwen35_q4_mlp.py:337`, `omlx/patches/qwen35_q4_mlp.py:807` |
| `OMLX_QWEN35_QMM_NAX` | 1 | `omlx/custom_kernels/qwen35_prefill/fast.py:956` |
| `OMLX_QWEN35_QMM_NAX_VARIANT` | 1 | `omlx/custom_kernels/qwen35_prefill/fast.py:841` |
| `OMLX_QWEN4_EAGER_DISPATCH` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:1036` |
| `OMLX_QWEN4_GATHERED_MIN_QUERY` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:97` |
| `OMLX_QWEN4_HC_FUSED` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py:29` |
| `OMLX_QWEN4_HC_HYBRID` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_projection.py:48` |
| `OMLX_QWEN4_PLE_MODE` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:191` |
| `OMLX_QWEN4_QSA_GATHERED_VERIFY` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:1045` |
| `OMLX_QWEN4_QSA_NATIVE_MAIN_MIN_ROWS` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py:68` |
| `OMLX_QWEN4_QSA_NATIVE_SCORE_MIN_ROWS` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py:54` |
| `OMLX_QWEN4_QSA_NATIVE_TOPK_MIN_ROWS` | 1 | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py:61` |
| `OMLX_QWEN4_STEP_TEXT_POSITIONS` | 1 | `omlx/models/vlm.py:36` |
| `OMLX_QWEN4_STEP_TEXT_POSITIONS_MIN_CONTEXT` | 1 | `omlx/models/vlm.py:41` |
| `OMLX_SDPA256_TILED` | 1 | `omlx/patches/sdpa256_attention.py:104` |
| `OMLX_SECRET_KEY` | 2 | `omlx/admin/auth.py:26`, `omlx/admin/auth.py:47` |
| `OMLX_SSD_CACHE_DIR` | 1 | `omlx/settings.py:973` |
| `OMLX_SSD_CACHE_MAX_SIZE` | 1 | `omlx/settings.py:975` |
| `OMLX_SUPERVISED` | 1 | `omlx/admin/routes.py:3523` |
| `OMLX_TAILSCALE_CLI` | 1 | `omlx/cluster/discovery.py:1037` |
| `OMLX_TEMPERATURE` | 1 | `omlx/config.py:207` |
| `OMLX_TRUST_REMOTE_CODE` | 1 | `omlx/config.py:199` |
| `OMLX_USAGE_HISTORY` | 1 | `omlx/settings.py:1037` |
