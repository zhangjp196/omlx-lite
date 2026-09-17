// Shared runtime services. Wires AppDelegate-owned objects (ServerProcess,
// AppConfig) to the app's consumers (Welcome wizard, menubar) and
// republishes `serverState` on every ServerProcess state change so callers
// can observe it as a source of truth. ServerProcess itself stays
// NSNotification-driven (no Combine retrofit).

import Foundation
import SwiftUI

@MainActor
@Observable
final class AppServices: NSObject {
    var config: AppConfig
    var serverState: ServerProcess.State = .stopped

    let client: OMLXClient
    let updates: UpdateController

    @ObservationIgnored
    private weak var server: ServerProcess?

    init(config: AppConfig = .default, server: ServerProcess? = nil) {
        self.config = config
        self.client = OMLXClient(host: config.host, port: config.port, apiKey: config.apiKey)
        self.updates = UpdateController()
        super.init()
        self.bind(server: server)
        // Wire Sparkle (or its stub) on the next runloop so any user prefs
        // saved on disk are applied before the first background check.
        DispatchQueue.main.async { [weak self] in
            self?.updates.bootstrap()
        }
    }

    func bind(server: ServerProcess?) {
        // Detach from the previous server (if any) before re-attaching.
        if self.server != nil {
            NotificationCenter.default.removeObserver(
                self,
                name: ServerProcess.stateDidChangeNotification,
                object: nil
            )
        }
        self.server = server
        if let server {
            self.serverState = server.state
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(serverStateDidChange(_:)),
                name: ServerProcess.stateDidChangeNotification,
                object: server
            )
        }
    }

    @objc private func serverStateDidChange(_ note: Notification) {
        guard let proc = note.object as? ServerProcess, proc === server else { return }
        // ServerProcess posts on the main queue (via DispatchQueue.main.async
        // in terminationHandler / @MainActor health-check Task), so we're
        // already on the main thread here.
        serverState = proc.state
        var updated = config
        updated.bindAddress = proc.bindAddress
        updated.port = proc.port
        updateConfig(updated)
    }

    func updateConfig(_ next: AppConfig) {
        self.config = next
        client.configure(host: next.host, port: next.port, apiKey: next.apiKey)
    }

    // MARK: - Server lifecycle (proxied to ServerProcess)

    var hasServer: Bool { server != nil }

    @discardableResult
    func startServer() throws -> ServerProcess.StartResult? {
        try server?.start()
    }

    func stopServer() async {
        await server?.stop()
    }

    func restartServer() async throws {
        await server?.stop()
        _ = try server?.start()
    }

    // MARK: - Base path relocation (pure path helpers)

    /// If `path` is inside `oldBase`, swap the prefix to `newBase`.
    /// Returns the input unchanged when it's empty or sits outside the
    /// migrated tree. Internal so unit tests can drive it directly. Pure —
    /// `nonisolated` so it's callable without bouncing onto MainActor.
    nonisolated static func relocate(path: String, oldBase: String, newBase: String) -> String {
        guard !path.isEmpty else { return path }
        let normalized = normalize(path)
        let oldRoot = oldBase
        if normalized == oldRoot {
            return newBase
        }
        let oldPrefix = oldRoot.hasSuffix("/") ? oldRoot : oldRoot + "/"
        if normalized.hasPrefix(oldPrefix) {
            let suffix = String(normalized.dropFirst(oldPrefix.count))
            return URL(fileURLWithPath: newBase, isDirectory: true)
                .appendingPathComponent(suffix).path
        }
        return path
    }

    /// Rewrite path-bearing fields in `<basePath>/settings.json` that may
    /// contain old-base absolute paths. Paths outside the migrated tree are
    /// left alone.
    nonisolated static func relocateOrphanPaths(in url: URL, oldBase: String, newBase: String) throws {
        NSLog("oMLX: relocateOrphanPaths in=%@ old=%@ new=%@",
              url.path, oldBase, newBase)
        guard FileManager.default.fileExists(atPath: url.path) else {
            NSLog("oMLX: relocateOrphanPaths skipped — file does not exist")
            return
        }
        let data = try Data(contentsOf: url)
        guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            NSLog("oMLX: relocateOrphanPaths skipped — root is not an object")
            return
        }

        if var model = json["model"] as? [String: Any] {
            if let dirs = model["model_dirs"] as? [String] {
                model["model_dirs"] = dirs.map {
                    Self.relocate(path: $0, oldBase: oldBase, newBase: newBase)
                }
            }
            if let dir = model["model_dir"] as? String, !dir.isEmpty {
                model["model_dir"] = Self.relocate(path: dir, oldBase: oldBase, newBase: newBase)
            }
            json["model"] = model
        }

        if var cache = json["cache"] as? [String: Any] {
            if let dir = cache["ssd_cache_dir"] as? String, !dir.isEmpty {
                cache["ssd_cache_dir"] = Self.relocate(path: dir, oldBase: oldBase, newBase: newBase)
            }
            json["cache"] = cache
        }

        if var logging = json["logging"] as? [String: Any] {
            if let dir = logging["log_dir"] as? String, !dir.isEmpty {
                logging["log_dir"] = Self.relocate(path: dir, oldBase: oldBase, newBase: newBase)
            }
            json["logging"] = logging
        }

        let out = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted])
        try out.write(to: url, options: [.atomic])
        NSLog("oMLX: relocateOrphanPaths wrote %d bytes", out.count)
    }

    nonisolated private static func normalize(_ path: String) -> String {
        ((path as NSString).expandingTildeInPath as NSString).standardizingPath
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}
