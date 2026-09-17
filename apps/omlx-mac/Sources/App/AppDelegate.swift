// Application delegate: sequences activation policy, menubar, server
// bootstrap, and signal handlers.
//
// Boot flow
//   applicationWillFinishLaunching  → setActivationPolicy(.regular)
//                                     (Dock icon shows briefly during launch)
//   applicationDidFinishLaunching   → load AppConfig
//                                     → install NSWindow observers (drive
//                                       the dock-icon toggle)
//                                     → if first run (no settings.json):
//                                         • show Welcome window (wizard
//                                           persists config + spawns server
//                                           only after Start Server)
//                                     else (returning user):
//                                         • resolve PythonRuntime
//                                         • spawn ServerProcess
//                                         • create MenubarController
//                                         • install POSIX SignalHandlers
//                                         • flip to .accessory next tick
//                                           (Dock icon hides; menubar stays)
//   applicationWillTerminate        → await server.stop(timeout: 10)
//
// Dock-icon toggle
//   Any time an in-app NSWindow becomes main → .regular (Dock icon shows).
//   When the last visible app window closes → .accessory (Dock icon hides).
//   Server + menubar are untouched by the toggle.

import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) var server: ServerProcess?
    private var menubar: MenubarController?
    private var controlServer: AppControlServer?
    let services = AppServices()

    private var welcomeController: WelcomeWindowController?
    private var welcomeCloseObserver: NSObjectProtocol?
    private var updateConfirmationController: UpdateConfirmationWindowController?

    /// Set true by `requestQuit()` to permit a real terminate. Cmd-Q / Dock
    /// Quit / "Quit oMLX" from the application menu all route through
    /// `applicationShouldTerminate`, which (when this flag is false) closes
    /// any visible app window instead of terminating — preserving the
    /// menubar status item + the running server. The menubar's own "Quit"
    /// item flips this flag before triggering termination.
    private var explicitQuitRequested: Bool = false

    /// Set true by `hideWindowsAndDropDockIcon()` so the willCloseNotification
    /// observer knows this close was app-initiated (Cmd-Q / Dock Quit) and
    /// should drop the Dock icon. When false, the close came from the user
    /// clicking the red traffic-light button — leave the Dock icon up so
    /// the user can click it to bring the window back.
    private var dropDockIconOnNextClose: Bool = false

    func requestQuit() {
        explicitQuitRequested = true
        NSApp.terminate(nil)
    }

    /// Cmd-Q / Dock → Quit path: hide every titled window AND set
    /// `.accessory` so the Dock icon vanishes. Server + menubar stay alive.
    func hideWindowsAndDropDockIcon() {
        dropDockIconOnNextClose = true
        var hidAny = false
        for win in NSApp.windows where win.styleMask.contains(.titled) && win.isVisible {
            win.close()
            if win.isVisible { win.orderOut(nil) }
            hidAny = hidAny || !win.isVisible
        }
        // If close() was vetoed and only orderOut hid the window,
        // willCloseNotification didn't fire — drop policy explicitly.
        let stillVisible = NSApp.windows.contains { $0.styleMask.contains(.titled) && $0.isVisible }
        if !stillVisible {
            NSApp.setActivationPolicy(.accessory)
        }
        dropDockIconOnNextClose = false
        _ = hidAny
    }

    /// Opens the browser-based web admin dashboard with auto-login. Gated on
    /// the server being up — the admin UI lives on the live server port.
    func openWebAdmin() {
        guard let server, case .running = server.state else { return }
        let host = MenubarController.displayHost(server: server, fallback: services.config.host)
        let port = MenubarController.displayPort(server: server, fallback: services.config.port)
        guard let url = MenubarController.webAdminURL(
            host: host,
            port: port,
            apiKey: services.config.apiKey
        ) else { return }
        NSWorkspace.shared.open(url)
    }

    /// Presents the update confirmation window (owned here so the menubar
    /// app can show it without any window scene). No-op when the controller
    /// has nothing to confirm.
    private func presentUpdateConfirmation() {
        if updateConfirmationController == nil {
            updateConfirmationController = UpdateConfirmationWindowController(updates: services.updates)
        }
        updateConfirmationController?.present()
    }

    nonisolated func applicationWillFinishLaunching(_ notification: Notification) {
        // Regular policy until the status item registers; we flip to Accessory
        // after creating the menubar (next runloop tick).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        installWindowObservers()
        services.updates.setTerminateForUpdate { [weak self] in
            if let self {
                self.requestQuit()
            } else {
                NSApp.terminate(nil)
            }
        }
        services.updates.setPresentUpdateConfirmation { [weak self] in
            self?.presentUpdateConfirmation()
        }
        if !isRunningUnitTests {
            do {
                let cliResult = try ShellEnvWriter.ensureCLIShim()
                handleCLISetupResult(cliResult)
            } catch {
                NSLog("oMLX: CLI shim setup failed — \(error)")
            }
            startControlServer()
        }

        let config = AppConfig.load()
        services.updateConfig(config)

        if AppConfig.hasExistingConfig {
            // Returning user. AppConfig.load() picks the highest-priority
            // file (`~/.omlx/settings.json` first, Library config.json
            // second) and stamps `config.source` so future saves route to
            // the same file. No re-write needed here.
            bootstrapServer(config: config)
            scheduleAccessoryPolicyFlip()
        } else {
            // First run: show the wizard only. Do not create the menubar or
            // persist settings until the user clicks Start Server.
            NSApp.activate(ignoringOtherApps: true)
            presentWelcome()
        }
    }

    private func handleCLISetupResult(_ result: ShellEnvWriter.CLISetupResult) {
        guard case .needsShellPathPrompt(let reason) = result else { return }
        guard !ShellEnvWriter.shouldSuppressCLIPathPrompt() else { return }
        promptForShellPathExport(reason: reason)
    }

    private func promptForShellPathExport(reason: String) {
        let alert = NSAlert()
        alert.messageText = "Enable `omlx` in Terminal?"
        alert.informativeText = """
        oMLX could not create a public `omlx` command in /opt/homebrew/bin or /usr/local/bin.

        To make `omlx` available in new Terminal sessions, oMLX can add a small PATH block to your shell init file. This only happens if you choose Update Shell File.

        \(reason)
        """
        alert.addButton(withTitle: "Update Shell File")
        alert.addButton(withTitle: "Dismiss Now")
        alert.addButton(withTitle: "Don't Ask Again")
        alert.window.level = .floating

        switch alert.runModal() {
        case .alertFirstButtonReturn:
            do {
                try ShellEnvWriter.ensureShellPathExport()
            } catch {
                NSLog("oMLX: CLI shell path setup failed — \(error)")
            }
        case .alertThirdButtonReturn:
            ShellEnvWriter.suppressCLIPathPromptForever()
        default:
            break
        }
    }

    private var isRunningUnitTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    /// All three MenubarController construction sites (first-run, returning
    /// user success, returning user failure) capture the same callbacks and
    /// differ only in `server`/`lastError`.
    private func makeMenubar(
        server: ServerProcess?,
        config: AppConfig,
        lastError: Error? = nil
    ) -> MenubarController {
        MenubarController(
            server: server,
            config: config,
            updates: services.updates,
            lastError: lastError,
            client: services.client,
            openModelSettings: { [weak self] _ in self?.openWebAdmin() },
            requestQuit:  { [weak self] in self?.requestQuit() }
        )
    }

    private func bootstrapServer(config: AppConfig) {
        do {
            let runtime = try PythonRuntime.resolve()
            let server = ServerProcess(
                runtime: runtime,
                bindAddress: config.bindAddress,
                port: config.port,
                basePath: URL(fileURLWithPath: config.basePath, isDirectory: true)
            )
            self.server = server
            self.menubar = makeMenubar(server: server, config: config)
            services.bind(server: server)

            // Install signal handlers BEFORE the spawn so a fast crash of
            // the parent during startup still reaps any child we managed
            // to spawn.
            SignalHandlers.shared.install { [weak server] in
                server?.reapSync()
            }

            if config.autoStartOnLaunch {
                switch try server.start() {
                case .started, .alreadyRunning:
                    break
                case .portConflict:
                    // ServerProcess already posted .portConflictNotification +
                    // updated state to .failed; MenubarController will surface
                    // it on next click.
                    break
                }
            }
        } catch {
            // Surface the failure in the menubar header so the user has a
            // recovery affordance without needing to dig through logs.
            self.menubar = makeMenubar(server: nil, config: config, lastError: error)
            NSLog("oMLX: server bootstrap failed — \(error)")
        }
    }

    private func scheduleAccessoryPolicyFlip() {
        // Defer the policy flip so the status item has time to register
        // with WindowServer before we hide the Dock icon (mirrors
        // switchToAccessoryPolicy_ in app.py:324-327).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.accessory)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    // MARK: - Dock-icon toggle via NSWindow observers

    /// Wire NSWindow lifecycle notifications so the Dock icon follows the
    /// "any app window visible → .regular, none → .accessory" rule.
    /// Both the Welcome wizard and the update confirmation window
    /// participate; the menubar status item is not an NSWindow and is
    /// unaffected.
    ///
    /// Uses the selector-based observer API (not the closure-based one) so
    /// the non-Sendable Notification + NSWindow values don't need to cross
    /// an actor boundary. NSWindow.* notifications are delivered on the
    /// main thread per Apple's documented contract, so the AppDelegate's
    /// @MainActor methods receive them safely.
    private func installWindowObservers() {
        let center = NotificationCenter.default
        center.addObserver(self,
                           selector: #selector(windowDidBecomeMainNotification(_:)),
                           name: NSWindow.didBecomeMainNotification,
                           object: nil)
        center.addObserver(self,
                           selector: #selector(windowWillCloseNotification(_:)),
                           name: NSWindow.willCloseNotification,
                           object: nil)
    }

    @objc private func windowDidBecomeMainNotification(_ notif: Notification) {
        guard let win = notif.object as? NSWindow, isAppOwnedWindow(win) else { return }
        if NSApp.activationPolicy() != .regular {
            NSApp.setActivationPolicy(.regular)
        }
    }

    @objc private func windowWillCloseNotification(_ notif: Notification) {
        guard let win = notif.object as? NSWindow, isAppOwnedWindow(win) else { return }
        let shouldDropDockIcon = dropDockIconOnNextClose
        // The closing window is still in NSApp.windows at notification time;
        // defer the visible-count check so it reflects post-close state.
        DispatchQueue.main.async {
            let stillVisible = NSApp.windows.contains { other in
                other !== win && other.isVisible && self.isAppOwnedWindow(other)
            }
            // Only drop to .accessory when the app initiated the close (Cmd-Q /
            // Dock Quit / Welcome wizard finish). Red-button close keeps the
            // Dock icon up so clicking it can re-open the window via
            // applicationShouldHandleReopen.
            if !stillVisible, shouldDropDockIcon {
                NSApp.setActivationPolicy(.accessory)
            }
        }
    }

    /// True for windows we own — excludes Sparkle's update windows, panel
    /// chrome from system services, etc. Heuristic: must be titled
    /// (so panel popovers don't count) and not excluded from the windows
    /// menu (so system status windows don't count).
    private func isAppOwnedWindow(_ win: NSWindow) -> Bool {
        guard win.styleMask.contains(.titled) else { return false }
        guard !win.isExcludedFromWindowsMenu else { return false }
        return true
    }

    // MARK: - Welcome wizard

    private func presentWelcome() {
        // First-run only — once `<basePath>/settings.json` exists,
        // `applicationDidFinishLaunching` takes the returning-user path and
        // this is never reached again.
        let controller = WelcomeWindowController(
            services: services,
            server: server,
            didFinish: { [weak self] _, finishedServer in
                // The wizard returns the spawned ServerProcess. Adopt it so
                // applicationWillTerminate can clean up correctly.
                guard let self else { return }
                self.server = finishedServer
                if let proc = finishedServer {
                    SignalHandlers.shared.install { [weak proc] in
                        proc?.reapSync()
                    }
                }
                self.menubar = self.makeMenubar(
                    server: finishedServer,
                    config: self.services.config
                )
            }
        )
        self.welcomeController = controller

        welcomeCloseObserver = NotificationCenter.default.addObserver(
            forName: WelcomeWindowController.willCloseNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.welcomeDidClose()
            }
        }

        controller.show()
    }

    private func welcomeDidClose() {
        if let observer = welcomeCloseObserver {
            NotificationCenter.default.removeObserver(observer)
            welcomeCloseObserver = nil
        }
        welcomeController = nil

        // Close before Start Server is cancellation: no menubar, no settings,
        // no base directory. Quit completely so relaunch shows Welcome again.
        guard server != nil else {
            explicitQuitRequested = true
            NSApp.terminate(nil)
            return
        }

        // The wizard spawned the server itself. Rebuild the menubar with the
        // running server state and switch to menubar-only mode.
        if let server, menubar != nil {
            self.menubar = makeMenubar(server: server, config: services.config)
        }
        scheduleAccessoryPolicyFlip()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Graceful stop. SIGKILL fallback is inside ServerProcess.stop().
        // We can't await indefinitely here — AppKit will eventually time
        // us out — so we run a short synchronous reap as belt-and-suspenders
        // (SignalHandlers also covers most external-kill paths).
        NotificationCenter.default.removeObserver(self)
        controlServer?.stop()
        controlServer = nil

        guard let server else { return }
        let group = DispatchGroup()
        group.enter()
        Task { @MainActor in
            await server.stop(timeout: 8)
            group.leave()
        }
        _ = group.wait(timeout: .now() + 9)
        server.reapSync(timeout: 1)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Menubar app — never quit on window close.
        false
    }

    /// Intercept terminate so Cmd-Q / Dock → Quit *only* close any visible
    /// window. The single real-quit path is the menubar status item's
    /// "Quit oMLX", which routes through `requestQuit()` to set the
    /// explicit flag.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if explicitQuitRequested { return .terminateNow }
        hideWindowsAndDropDockIcon()
        return .terminateCancel
    }

    private func startControlServer() {
        let control = AppControlServer()
        control.handler = self
        do {
            try control.start()
            self.controlServer = control
        } catch {
            NSLog("oMLX: app-control server failed to start — \(error)")
        }
    }

    /// Dock icon click while no window is visible: open the web dashboard
    /// when the server is up (the app's main surface now), otherwise just
    /// activate the app.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            if case .running = server?.state {
                openWebAdmin()
            } else {
                NSApp.activate(ignoringOtherApps: true)
            }
        }
        return true
    }
}

extension AppDelegate: AppControlHandling {
    func handleAppControl(_ command: AppControlServer.Command) async -> AppControlServer.Response {
        guard let server else {
            return .failure(
                status: "unavailable",
                state: .stopped,
                server: nil,
                message: "Managed server is unavailable. Complete the oMLX first-run setup in the app."
            )
        }

        switch command {
        case .status:
            return .success(status: "ok", state: server.state, server: server)

        case .start:
            do {
                switch try server.start() {
                case .started:
                    return .success(status: "starting", state: server.state, server: server)
                case .alreadyRunning:
                    return .success(status: "running", state: server.state, server: server)
                case .portConflict(let conflict):
                    let pid = conflict.pid.map(String.init) ?? "unknown"
                    return .failure(
                        status: "port_conflict",
                        state: server.state,
                        server: server,
                        message: "Port \(server.port) is in use by PID \(pid)."
                    )
                }
            } catch {
                return .failure(
                    status: "error",
                    state: server.state,
                    server: server,
                    message: String(describing: error)
                )
            }

        case .stop:
            await server.stop()
            return .success(
                status: "stopped",
                state: server.state,
                server: server,
                message: "oMLX stopped"
            )

        case .restart:
            await server.stop()
            do {
                switch try server.start() {
                case .started:
                    return .success(status: "starting", state: server.state, server: server)
                case .alreadyRunning:
                    return .success(status: "running", state: server.state, server: server)
                case .portConflict(let conflict):
                    let pid = conflict.pid.map(String.init) ?? "unknown"
                    return .failure(
                        status: "port_conflict",
                        state: server.state,
                        server: server,
                        message: "Port \(server.port) is in use by PID \(pid)."
                    )
                }
            } catch {
                return .failure(
                    status: "error",
                    state: server.state,
                    server: server,
                    message: String(describing: error)
                )
            }
        }
    }
}
