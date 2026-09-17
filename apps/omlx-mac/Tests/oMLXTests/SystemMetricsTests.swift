// The menu bar's System block surfaces derived values that each have a
// single point of failure: the thermal-state → severity mapping and the
// bytes → GB formatter used in the row labels. These tests pin those
// mappings so an SDK roll or a stray locale tweak can't change what the
// UI prints.

import XCTest
@testable import oMLX

final class SystemMetricsTests: XCTestCase {

    // MARK: - Thermal severity mapping

    func testThermalSeverityNominal() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .nominal),
            .nominal
        )
    }

    func testThermalSeverityFair() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .fair),
            .fair
        )
    }

    func testThermalSeveritySerious() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .serious),
            .serious
        )
    }

    func testThermalSeverityCritical() {
        XCTAssertEqual(
            SystemMetricsPoller.severity(for: .critical),
            .critical
        )
    }

    func testThermalLabelsMatchSeverity() {
        XCTAssertEqual(SystemMetricsPoller.label(for: .nominal),  "Nominal")
        XCTAssertEqual(SystemMetricsPoller.label(for: .fair),     "Fair")
        XCTAssertEqual(SystemMetricsPoller.label(for: .serious),  "Serious")
        XCTAssertEqual(SystemMetricsPoller.label(for: .critical), "Critical")
    }

    // MARK: - Byte formatters

    func testFormatBytesAsGbRoundsToOneDecimal() {
        // 34.6 GB in decimal bytes = 34_600_000_000.
        let bytes: UInt64 = 34_600_000_000
        XCTAssertEqual(SystemMetricsPoller.formatBytesAsGB(bytes), "34.6")
    }

    func testFormatBytesAsGbZero() {
        XCTAssertEqual(SystemMetricsPoller.formatBytesAsGB(0), "0.0")
    }

    func testFormatBytesAsGibUsesBinaryUnits() {
        XCTAssertEqual(SystemMetricsPoller.formatBytesAsGiB(51_539_607_552), "48.0")
        XCTAssertEqual(SystemMetricsPoller.formatBytesAsGiB(2_684_354_560), "2.5")
    }

    func testFormatBytesAsGbRoundsHalfUp() {
        // 12.55 GB → "12.5" (banker's) or "12.6" (away). printf %.1f on
        // Darwin rounds half to even at the binary level — pin whichever
        // string we actually produce so a future libc swap is visible.
        let bytes: UInt64 = 12_550_000_000
        let out = SystemMetricsPoller.formatBytesAsGB(bytes)
        XCTAssertTrue(out == "12.5" || out == "12.6",
                      "Unexpected rounding output: \(out)")
    }
}
