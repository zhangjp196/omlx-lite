// PR 13 — Benchmark DTOs (throughput + accuracy).
//
// Throughput mirrors omlx/admin/routes.py:/api/bench/* (start @4619,
// stream @4679, cancel @4724, results @4748). Accuracy mirrors
// /api/bench/accuracy/* (queue/add @4460, queue/status @4507, results
// @4531, reset @4547, cancel @4558, stream @4569).
//
// Both feature areas use *polling* in the Swift screens (the HTML uses
// SSE). The shape of `GET /api/bench/{id}/results` and
// `GET /api/bench/accuracy/results` matches what each respective SSE
// stream emits once aggregated, so polling produces functionally
// equivalent state — just without per-token granularity.

import Foundation

// =============================================================================
// MARK: - Device info (shared by both screens)
// =============================================================================

/// Response from `GET /admin/api/device-info`. Surfaced as a small chip
/// at the top of the Throughput screen so the user can sanity-check
/// which hardware they're benchmarking.
struct DeviceInfoDTO: Codable, Sendable {
    let chipName: String?
    let chipVariant: String?
    let memoryGb: Int?
    let gpuCores: Int?
}

// =============================================================================
// MARK: - Throughput bench
// =============================================================================

enum BenchmarkContextProfile: String, Codable, CaseIterable, Sendable {
    case codePython = "code_python"
    case codeMixed = "code_mixed"
    case novelKorean = "novel_ko"
    case novelEnglish = "novel_en"
    case novelJapanese = "novel_ja"
}

enum BenchmarkWarmupMode: String, Codable, CaseIterable, Sendable {
    case quick
    case ane2048 = "ane_2048"
}

/// Body for `POST /admin/api/bench/start`. `prompt_lengths` and
/// `batch_sizes` are server-validated against a known whitelist
/// (1024…200000 / 2…8). `generation_length` is free-form int.
struct BenchStartRequest: Encodable, Sendable {
    let modelId: String
    let contextProfile: BenchmarkContextProfile
    let warmupMode: BenchmarkWarmupMode
    let alignPromptToAne: Bool
    let promptLengths: [Int]
    let generationLength: Int
    let batchSizes: [Int]
}

struct BenchStartResponse: Codable, Sendable {
    let benchId: String
    let status: String
    let totalTests: Int
}

/// Single + batch results share one envelope; `testType` is the
/// discriminator. We use optional fields so a single struct can decode
/// both variants without resorting to an enum-with-associated-values
/// (which would have to be hand-coded for Codable).
struct BenchResultDTO: Codable, Equatable, Sendable {
    let testType: String

    // Single-only
    let pp: Int?
    let tg: Int?
    let ttftMs: Double?
    let tpotMs: Double?
    let processingTps: Double?
    let genTps: Double?
    let e2eLatencyS: Double?
    let totalThroughput: Double?
    let peakMemoryBytes: Int64?

    // Batch-only
    let batchSize: Int?
    let tgTps: Double?
    let ppTps: Double?
    let avgTtftMs: Double?

    /// Host load during this test's window. Optional so results from a server
    /// that predates host sampling still decode.
    let systemMetrics: BenchSystemMetricsDTO?
}

/// Aggregated host telemetry for one test. Every field is optional: the server
/// adds and renames nested keys over time, and a missing reading has to render
/// as "unknown" rather than break decoding.
struct BenchSystemMetricsDTO: Codable, Equatable, Sendable {
    struct CPU: Codable, Equatable, Sendable {
        let totalAvg: Double?
        let totalMax: Double?
        let pAvg: Double?
        let eAvg: Double?
    }

    struct GPU: Codable, Equatable, Sendable {
        let utilAvg: Double?
        let utilMax: Double?
    }

    /// Raw OSThermalPressureLevel: 0 nominal, 1 moderate, 2 heavy,
    /// 3 trapping, 4 sleeping. Five-valued — Foundation's
    /// ProcessInfo.ThermalState collapses the last two into `.critical`.
    struct Thermal: Codable, Equatable, Sendable {
        let start: Int?
        let max: Int?
    }

    /// All values are GiB, matching the leaderboard's other memory fields.
    struct Memory: Codable, Equatable, Sendable {
        let physFootprintPeak: Double?
        let mlxActivePeak: Double?
        let mlxCachePeak: Double?
        let systemUsedPeak: Double?
        let systemWiredPeak: Double?
        let totalRam: Double?
    }

    let sampleCount: Int?
    let intervalS: Double?
    let cpu: CPU?
    let gpu: GPU?
    let thermal: Thermal?
    let memory: Memory?
}

struct BenchResultsResponse: Codable, Sendable {
    let benchId: String
    let status: String
    let contextProfile: BenchmarkContextProfile?
    let results: [BenchResultDTO]
    let error: String?
}

struct BenchCancelResponse: Codable, Sendable {
    let status: String
    let benchId: String?
}

// =============================================================================
// MARK: - Qwen ANE/GPU split tuner
// =============================================================================

struct ANETuningStartRequest: Encodable, Sendable {
    let modelId: String
    let sequenceLength: Int
    let repeats: Int
    let allowCpu: Bool
    let allowCpuGate: Bool
    let allowCpuDown: Bool
    let allowAneGdn: Bool
    let allowCpuGdn: Bool
    let allowCpuSharedResource: Bool
}

struct ANETuningStartResponse: Codable, Sendable {
    let tuningId: String
    let status: String
    let total: Int
}

struct ANETuningCandidateDTO: Codable, Equatable, Identifiable, Sendable {
    let label: String
    let detail: String?
    let stage: String?
    let enabled: Bool
    let mlpFraction: Double?
    let gdnEnabled: Bool
    let gdnFraction: Double?
    let cpuEnabled: Bool?
    let cpuFraction: Double?
    let cpuDownFraction: Double?
    let cpuGdnFraction: Double?
    let fusedDown: Bool?
    let state: String?
    let processingTps: Double?
    let latencyMs: Double?
    let samples: [Double]
    let speedupPercent: Double?
    let error: String?

    var id: String { label }
}

struct ANETuningRecommendationDTO: Codable, Equatable, Sendable {
    var backend: String? = nil
    var sharedFraction: Double? = nil
    let enabled: Bool
    let mlpFraction: Double?
    let gdnEnabled: Bool
    let gdnFraction: Double?
    let cpuEnabled: Bool?
    let cpuFraction: Double?
    let cpuDownFraction: Double?
    let cpuGdnFraction: Double?
    let fusedDown: Bool?
    let cpuThreads: Int?
    let cpuSharedResource: Bool?
    // Null when the tuner returned a verdict without measuring, e.g. the
    // GPU-only preflight on machines without the ANE compiler (#3067).
    let processingTps: Double?
    let speedupPercent: Double?
    let sequenceLength: Int
    let tailPaddingMinTokens: Int?
}

struct ANETuningStatusResponse: Codable, Sendable {
    let tuningId: String
    let modelId: String
    let status: String
    let phase: String
    let message: String
    let current: Int
    let total: Int
    let results: [ANETuningCandidateDTO]
    let recommendation: ANETuningRecommendationDTO?
    let error: String?
    let terminationReason: String?
}

struct ANETuningCancelResponse: Codable, Sendable {
    let status: String
    let tuningId: String
}

// =============================================================================
// MARK: - Accuracy bench
// =============================================================================

/// Body for `POST /admin/api/bench/accuracy/queue/add`. `benchmarks` is
/// a dict of benchmark-key → sample-size; the catalog of valid keys
/// lives in AccuracyBenchScreen.swift to keep the DTO server-agnostic.
struct AccuracyQueueAddRequest: Encodable, Sendable {
    let modelId: String
    let benchmarks: [String: Int]
    let batchSize: Int
    let enableThinking: Bool
}

struct AccuracyQueueItem: Codable, Equatable, Sendable {
    let modelId: String
    let benchmarks: [String]
}

/// Mirrors the queue-status snapshot returned by every accuracy
/// queue/add, queue/status, queue/remove endpoint. `lastProgress` is
/// the most recent progress event the SSE stream emitted; polling
/// callers use it to drive the in-flight progress message.
///
/// `phase` is finer-grained than `running` — it distinguishes
/// "evaluating" (still scoring) from "unloading" (post-result cleanup),
/// so the UI can hide the running row once the result card has
/// appeared. Values: "pending" | "loading" | "evaluating" |
/// "unloading" | "completed" | "cancelled" | "error". Nullable when no
/// run is in flight.
struct AccuracyQueueStatus: Codable, Sendable {
    let running: Bool
    let currentModel: String
    let currentBenchId: String
    let lastProgress: AccuracyProgressDTO?
    let phase: String?
    let queue: [AccuracyQueueItem]

    /// True only while the run is actively producing results. Hides the
    /// running row during post-result unload and after the bench task
    /// finishes but before `_queue_running` flips back.
    var isActivelyEvaluating: Bool {
        guard running else { return false }
        switch phase {
        case "evaluating", "loading", "pending", nil: return true
        default: return false
        }
    }
}

struct AccuracyProgressDTO: Codable, Equatable, Sendable {
    let modelId: String?
    let message: String?
    let current: Int?
    let total: Int?
    let benchCurrent: Int?
    let benchTotal: Int?
    let benchmark: String?
}

struct AccuracyResultDTO: Codable, Equatable, Sendable, Identifiable {
    let benchmark: String
    let modelId: String
    let accuracy: Double
    let correct: Int
    let total: Int
    let timeS: Double
    let thinkingUsed: Bool
    let categoryScores: [String: Double]?

    /// Synthetic ID — the server doesn't emit one and `(benchmark,
    /// model)` is unique within an accAllResults array.
    var id: String { "\(benchmark)::\(modelId)" }
}

struct AccuracyResultsResponse: Codable, Sendable {
    let results: [AccuracyResultDTO]
    let running: Bool
    let currentModel: String
    let currentBenchId: String
}
