// Standalone update confirmation window. Hosts the release-notes review UI
// (notes, download size, Install & Relaunch / Later) in its own NSWindow so
// the menubar app can present it without the old AppView shell. AppDelegate
// owns one controller and asks it to present whenever UpdateController
// raises `confirmationUpdate`.

import AppKit
import MarkdownUI
import SwiftUI

@MainActor
final class UpdateConfirmationWindowController: NSWindowController {
    private let updates: UpdateController

    init(updates: UpdateController) {
        self.updates = updates
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 680, height: 560),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = String(localized: "update.confirm.window_title",
                              defaultValue: "oMLX Update",
                              comment: "Title bar of the update confirmation window")
        window.isReleasedWhenClosed = false
        super.init(window: window)
        window.delegate = self
        window.center()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    /// Presents the window with the controller's current confirmation update.
    /// No-op when there is nothing to confirm.
    func present() {
        guard let update = updates.confirmationUpdate else { return }
        let view = UpdateConfirmationView(
            update: update,
            updates: updates,
            onLater: { [weak self] in
                self?.updates.deferUpdate(update)
                self?.close()
            },
            onConfirm: { [weak self] in
                self?.updates.confirmUpdate(update)
                self?.close()
            }
        )
        .omlxThemed()
        let hosting = NSHostingController(rootView: view)
        hosting.sizingOptions = [.preferredContentSize]
        window?.contentViewController = hosting
        window?.center()
        showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

extension UpdateConfirmationWindowController: NSWindowDelegate {
    nonisolated func windowWillClose(_ notification: Notification) {
        Task { @MainActor in
            self.updates.dismissUpdateConfirmation()
        }
    }
}

// MARK: - Update confirmation

@MainActor
struct UpdateConfirmationView: View {
    let update: AvailableUpdate
    let updates: UpdateController
    let onLater: () -> Void
    let onConfirm: () -> Void

    @Environment(\.omlxTheme) private var theme

    private var trimmedNotes: String {
        update.notes.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var noteBlocks: [ReleaseNotesBlock] {
        ReleaseNotesHTML.blocks(from: trimmedNotes)
    }

    private var isStaged: Bool {
        if case .ready(let ready) = updates.state {
            return ready.version == update.version
        }
        return false
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            notesBody
            Divider()
            footer
        }
        .frame(width: 680, height: 560)
        .background(theme.windowBg)
    }

    private var header: some View {
        HStack(spacing: 12) {
            Squircle(systemSymbol: "arrow.down.circle.fill",
                     size: 34,
                     gradient: SquircleGradient.update)
            VStack(alignment: .leading, spacing: 4) {
                Text(String(localized: "update.confirm.title",
                            defaultValue: "oMLX \(update.version) is available",
                            comment: "Update confirmation sheet title; placeholder is the version"))
                    .font(.omlxText(17, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(String(localized: "update.confirm.subtitle",
                            defaultValue: "Review the release notes before downloading and relaunching.",
                            comment: "Subtitle for the update confirmation sheet"))
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }
            Spacer()
            Button {
                NSWorkspace.shared.open(update.htmlURL)
            } label: {
                Image(systemName: "arrow.up.right.square")
                    .font(.system(size: 13, weight: .semibold))
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .help(String(localized: "update.confirm.view_release",
                         defaultValue: "View release on GitHub",
                         comment: "Tooltip for the release link button in the update confirmation sheet"))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }

    @ViewBuilder
    private var notesBody: some View {
        if trimmedNotes.isEmpty {
            VStack(spacing: 10) {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.system(size: 30, weight: .light))
                    .foregroundStyle(theme.textTertiary)
                Text(String(localized: "update.confirm.empty_notes",
                            defaultValue: "This release does not include detailed notes.",
                            comment: "Empty state when a GitHub release has no release notes"))
                    .font(.omlxText(13))
                    .foregroundStyle(theme.textSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(24)
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(noteBlocks) { block in
                        switch block {
                        case .markdown(let text):
                            Markdown(text)
                                .markdownTheme(.docC)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        case .imageGroup(let images):
                            ReleaseNotesImageGroup(images: images)
                        }
                    }
                }
                .padding(20)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(String(localized: "update.confirm.size",
                            defaultValue: "Download size: \(update.sizeText ?? "Unknown")",
                            comment: "Update confirmation download size line; placeholder is a formatted byte size or Unknown"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                Text(String(localized: "update.confirm.restart_notice",
                            defaultValue: "oMLX will quit, install the update, and relaunch.",
                            comment: "Notice explaining what happens after confirming an update"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }
            Spacer()
            Button(String(localized: "update.confirm.later",
                          defaultValue: "Later",
                          comment: "Dismiss button in the update confirmation sheet")) {
                onLater()
            }
            .buttonStyle(.omlx(.normal))
            Button(primaryButtonTitle) {
                onConfirm()
            }
            .buttonStyle(.omlx(.primary))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var primaryButtonTitle: String {
        if isStaged {
            return String(localized: "update.confirm.install_ready",
                          defaultValue: "Install & Relaunch",
                          comment: "Primary button when the update is already staged")
        }
        return String(localized: "update.confirm.install",
                      defaultValue: "Download, Install & Relaunch",
                      comment: "Primary button to download, install, and relaunch")
    }
}

private enum ReleaseNotesBlock: Identifiable {
    case markdown(String)
    case imageGroup([ReleaseNotesImage])

    var id: String {
        switch self {
        case .markdown(let text):
            return "markdown:\(text.hashValue)"
        case .imageGroup(let images):
            return "images:\(images.map(\.id).joined(separator: ","))"
        }
    }
}

private struct ReleaseNotesImage: Identifiable, Equatable {
    let url: URL
    let alt: String

    var id: String { "\(url.absoluteString):\(alt)" }
}

private enum ReleaseNotesHTML {
    private static let imageParagraphPattern = #"(?is)<p\b[^>]*>\s*((?:<img\b[^>]*>\s*)+)</p>"#
    private static let imagePattern = #"(?is)<img\b([^>]*)>"#
    private static let attributePattern = #"([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(['"])(.*?)\2"#

    static func blocks(from raw: String) -> [ReleaseNotesBlock] {
        guard let paragraphRegex = try? NSRegularExpression(pattern: imageParagraphPattern) else {
            return markdownBlocks(raw)
        }

        var blocks: [ReleaseNotesBlock] = []
        var cursor = raw.startIndex
        let fullRange = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        let matches = paragraphRegex.matches(in: raw, range: fullRange)

        for match in matches {
            guard let matchRange = Range(match.range, in: raw) else { continue }
            appendMarkdown(String(raw[cursor..<matchRange.lowerBound]), to: &blocks)

            if match.numberOfRanges > 1,
               let bodyRange = Range(match.range(at: 1), in: raw) {
                let images = extractImages(from: String(raw[bodyRange]))
                if images.isEmpty {
                    appendMarkdown(String(raw[matchRange]), to: &blocks)
                } else {
                    blocks.append(.imageGroup(images))
                }
            }

            cursor = matchRange.upperBound
        }

        appendMarkdown(String(raw[cursor..<raw.endIndex]), to: &blocks)
        return blocks.isEmpty ? markdownBlocks(raw) : blocks
    }

    private static func markdownBlocks(_ raw: String) -> [ReleaseNotesBlock] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? [] : [.markdown(trimmed)]
    }

    private static func appendMarkdown(_ raw: String, to blocks: inout [ReleaseNotesBlock]) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            blocks.append(.markdown(trimmed))
        }
    }

    private static func extractImages(from html: String) -> [ReleaseNotesImage] {
        guard let imageRegex = try? NSRegularExpression(pattern: imagePattern) else { return [] }
        let fullRange = NSRange(html.startIndex..<html.endIndex, in: html)
        return imageRegex.matches(in: html, range: fullRange).compactMap { match in
            guard match.numberOfRanges > 1,
                  let attrsRange = Range(match.range(at: 1), in: html)
            else { return nil }

            let attrs = attributes(from: String(html[attrsRange]))
            guard let src = attrs["src"],
                  let url = URL(string: decodeHTMLEntities(src))
            else { return nil }

            return ReleaseNotesImage(
                url: url,
                alt: attrs["alt"].map(decodeHTMLEntities) ?? ""
            )
        }
    }

    private static func attributes(from raw: String) -> [String: String] {
        guard let attrRegex = try? NSRegularExpression(pattern: attributePattern) else { return [:] }
        var attrs: [String: String] = [:]
        let fullRange = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        for match in attrRegex.matches(in: raw, range: fullRange) {
            guard match.numberOfRanges > 3,
                  let keyRange = Range(match.range(at: 1), in: raw),
                  let valueRange = Range(match.range(at: 3), in: raw)
            else { continue }
            attrs[String(raw[keyRange]).lowercased()] = String(raw[valueRange])
        }
        return attrs
    }

    private static func decodeHTMLEntities(_ raw: String) -> String {
        guard let data = raw.data(using: .utf8),
              let decoded = try? NSAttributedString(
                data: data,
                options: [
                    .documentType: NSAttributedString.DocumentType.html,
                    .characterEncoding: String.Encoding.utf8.rawValue,
                ],
                documentAttributes: nil
              ).string
        else { return raw }
        return decoded
    }
}

private struct ReleaseNotesImageGroup: View {
    let images: [ReleaseNotesImage]

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            ForEach(images) { image in
                AsyncImage(url: image.url) { phase in
                    switch phase {
                    case .empty:
                        ProgressView()
                            .frame(maxWidth: .infinity, minHeight: 120)
                    case .success(let rendered):
                        rendered
                            .resizable()
                            .scaledToFit()
                            .accessibilityLabel(image.alt)
                    case .failure:
                        VStack(spacing: 8) {
                            Image(systemName: "photo")
                                .font(.system(size: 24, weight: .light))
                                .foregroundStyle(theme.textTertiary)
                            if !image.alt.isEmpty {
                                Text(image.alt)
                                    .font(.omlxText(11))
                                    .foregroundStyle(theme.textSecondary)
                                    .multilineTextAlignment(.center)
                            }
                        }
                        .frame(maxWidth: .infinity, minHeight: 120)
                    @unknown default:
                        EmptyView()
                    }
                }
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(theme.groupBorder.opacity(0.55), lineWidth: 1)
                )
            }
        }
        .frame(maxWidth: .infinity)
    }
}
