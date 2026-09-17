// SwiftUI shell. The app is menubar-first: no visible window scenes. The
// (suppressed) Window scene exists only because a SwiftUI App needs a scene;
// it never materialises an NSWindow — the Welcome wizard and the update
// confirmation window are manual NSWindow controllers owned by AppDelegate.

import Darwin
import SwiftUI

@main
enum OMLXEntryPoint {
    static func main() {
        do {
            if let request = try UpdateInstaller.workerRequest(
                from: CommandLine.arguments
            ) {
                exit(UpdateInstaller.runWorker(request))
            }
        } catch {
            NSLog("oMLX: invalid updater worker invocation: %@", error.localizedDescription)
            exit(EXIT_FAILURE)
        }

        OMLXApp.main()
    }
}

struct OMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("", id: "main") {
            EmptyView()
        }
        .defaultLaunchBehavior(.suppressed)
    }
}
