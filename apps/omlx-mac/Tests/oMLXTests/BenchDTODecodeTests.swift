// Covers the bench results envelope's decode contract, in both directions of
// the app/server version skew.
//
// The forward case matters because each result gained `system_metrics` when
// accelerated runs started carrying host telemetry.
//
// The backward case matters more: results from an older server must still
// decode even when optional keys are absent.

import XCTest
@testable import oMLX

final class BenchDTODecodeTests: XCTestCase {

    /// Mirrors the `OMLXClient` decoder configuration.
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    /// Same source-relative resolution DTOFixtureTests uses — the fixtures are
    /// git artifacts, not bundle resources.
    private func loadFixture(_ name: String) throws -> Data {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("\(name).json")
        return try Data(contentsOf: url)
    }

    // MARK: Current server

    func testEncodeBenchmarkContextProfile() throws {
        let request = BenchStartRequest(
            modelId: "model",
            contextProfile: .novelKorean,
            warmupMode: .ane2048,
            alignPromptToAne: true,
            promptLengths: [1024],
            generationLength: 128,
            batchSizes: [2]
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoder.encode(request))
                as? [String: Any]
        )
        XCTAssertEqual(object["context_profile"] as? String, "novel_ko")
        XCTAssertEqual(object["warmup_mode"] as? String, "ane_2048")
        XCTAssertEqual(object["align_prompt_to_ane"] as? Bool, true)
    }

    func testDecodeResultsContextProfile() throws {
        let json = """
        {
            "bench_id": "b-1",
            "status": "completed",
            "context_profile": "code_mixed",
            "results": []
        }
        """.data(using: .utf8)!
        let response = try decoder.decode(BenchResultsResponse.self, from: json)
        XCTAssertEqual(response.contextProfile, .codeMixed)
    }

    func testDecodeSystemMetrics() throws {
        let response = try decoder.decode(
            BenchResultsResponse.self, from: try loadFixture("bench-results")
        )
        let metrics = try XCTUnwrap(response.results.first?.systemMetrics)

        XCTAssertEqual(metrics.sampleCount, 42)
        XCTAssertEqual(metrics.intervalS, 1.0)
        XCTAssertEqual(metrics.cpu?.totalAvg, 38.2)
        XCTAssertEqual(metrics.cpu?.pAvg, 55.1)
        XCTAssertEqual(metrics.gpu?.utilMax, 99.0)
        // Raw OSThermalPressureLevel, not the four-valued Foundation enum.
        XCTAssertEqual(metrics.thermal?.start, 0)
        XCTAssertEqual(metrics.thermal?.max, 1)
        XCTAssertEqual(metrics.memory?.physFootprintPeak, 44.87)
        XCTAssertEqual(metrics.memory?.totalRam, 128)
    }

    func testBatchResultCarriesNoSystemMetricsWhenAbsent() throws {
        let response = try decoder.decode(
            BenchResultsResponse.self, from: try loadFixture("bench-results")
        )
        let batch = try XCTUnwrap(response.results.first { $0.testType == "batch" })
        XCTAssertNil(batch.systemMetrics)
        XCTAssertEqual(batch.batchSize, 4)
    }

    // MARK: Older server

    func testDecodeLegacyResponseWithoutNewKeys() throws {
        let response = try decoder.decode(
            BenchResultsResponse.self, from: try loadFixture("bench-results-legacy")
        )
        XCTAssertEqual(response.status, "completed")
        XCTAssertNil(response.results.first?.systemMetrics)
    }
}
