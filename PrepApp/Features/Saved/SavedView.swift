import SwiftData
import SwiftUI

struct SavedView: View {
    private enum Tab: String, CaseIterable { case bookmarks = "Bookmarks", flagged = "Flagged", notes = "Notes" }

    @Environment(ContentStore.self) private var content
    @Environment(\.modelContext) private var ctx
    @Environment(\.theme) private var theme
    @Query private var progress: [TopicProgress]
    @Query(sort: \FlagRecord.date, order: .reverse) private var flags: [FlagRecord]
    @Query private var states: [CardState]
    @State private var tab: Tab = .bookmarks
    @State private var showResolved = false

    private var stateByItem: [String: CardState] {
        Dictionary(states.map { ($0.itemID, $0) }, uniquingKeysWith: { a, _ in a })
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Space.m) {
                    ScreenHeader(title: "Saved") {
                        if tab == .flagged, !openFlags.isEmpty {
                            ShareLink(item: exportText, subject: Text("Flagged prep items")) {
                                Image(systemName: "square.and.arrow.up").font(.title3).foregroundStyle(theme.accent)
                            }
                            .accessibilityLabel("Export flagged items")
                        }
                    }
                    ThemedSegmented(selection: $tab, options: Tab.allCases.map { ($0, $0.rawValue) })
                    switch tab {
                    case .bookmarks: bookmarks
                    case .flagged: flagged
                    case .notes: notes
                    }
                }
                .padding()
            }
            .screenBackground()
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: Topic.self) { TopicDetailView(topic: $0) }
        }
    }

    private func empty(_ title: String, _ icon: String, _ body: String) -> some View {
        EmptyState(icon: icon, title: title, message: body)
    }

    @ViewBuilder private var bookmarks: some View {
        let topics = content.topics.filter { t in progress.contains { $0.topicID == t.id && $0.bookmarked } }
        if topics.isEmpty {
            empty("No bookmarks", "bookmark", "Tap the bookmark icon on any topic to save it here.")
        } else {
            PanelList(topics) { t in
                NavigationLink(value: t) {
                    TopicRow(topic: t, states: stateByItem, progress: progress.first { $0.topicID == t.id })
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var openFlags: [FlagRecord] { flags.filter { !$0.resolved } }

    @ViewBuilder private var flagged: some View {
        let shown = showResolved ? flags : openFlags
        if flags.contains(where: \.resolved) {
            ToggleRow(title: "Show resolved", isOn: $showResolved).panel(padding: 12)
        }
        if shown.isEmpty {
            empty("Nothing flagged", "flag", "Use the flag button during a session to report a wrong or unclear item, then export the list to get it fixed.")
        } else {
            PanelList(shown) { f in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SectionLabel(f.reason.rawValue, color: f.resolved ? theme.textMuted : theme.warning)
                        Spacer()
                        Text(f.date.formatted(date: .abbreviated, time: .omitted)).font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                    }
                    Text(content.itemsByID[f.itemID].map { String($0.promptText.prefix(140)) } ?? "(item no longer exists)")
                        .font(theme.font(.callout)).foregroundStyle(theme.text).lineLimit(3)
                        .strikethrough(f.resolved)
                    if !f.note.isEmpty { Text(f.note).font(theme.font(.caption)).foregroundStyle(theme.textSecondary) }
                    HStack {
                        Text(f.itemID).font(theme.font(.label)).foregroundStyle(theme.textMuted)
                        Spacer()
                        Button(theme.labelText(f.resolved ? "Reopen" : "Resolve")) { f.resolved.toggle(); try? ctx.save() }
                            .font(theme.font(.label)).foregroundStyle(theme.success)
                        Button(theme.labelText("Delete")) { ctx.delete(f); try? ctx.save() }
                            .font(theme.font(.label)).foregroundStyle(theme.danger)
                            .padding(.leading, 10)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    @ViewBuilder private var notes: some View {
        let withNotes = progress.filter { !$0.note.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && content.topicsByID[$0.topicID] != nil }
        if withNotes.isEmpty {
            empty("No notes yet", "note.text", "Add notes at the bottom of any topic page.")
        } else {
            PanelList(withNotes) { p in
                if let t = content.topicsByID[p.topicID] {
                    NavigationLink(value: t) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(t.title).font(theme.font(.bodyStrong)).foregroundStyle(theme.text)
                            Text(p.note).font(theme.font(.callout)).foregroundStyle(theme.textSecondary).lineLimit(4)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    /// Markdown list to paste into Claude for content fixes.
    private var exportText: String {
        var lines = ["# Flagged items (\(openFlags.count))", ""]
        for f in openFlags {
            let item = content.itemsByID[f.itemID]
            let file = content.topicsByID[f.topicID].map { "PrepApp/Content/\($0.area)/\($0.id).yaml" } ?? f.topicID
            lines.append("- **\(f.itemID)** — \(f.reason.rawValue)  (`\(file)`)")
            if let item { lines.append("  - Prompt: \(item.promptText.replacingOccurrences(of: "\n", with: " ").prefix(200))") }
            if !f.note.isEmpty { lines.append("  - Note: \(f.note)") }
        }
        return lines.joined(separator: "\n")
    }
}
