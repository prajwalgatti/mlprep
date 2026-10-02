import SwiftData
import SwiftUI

struct TopicDetailView: View {
    let topic: Topic
    @Environment(ContentStore.self) private var content
    @Environment(\.modelContext) private var ctx
    @Environment(\.theme) private var theme
    @Query private var states: [CardState]
    @State private var active: ActiveSession?
    @State private var progress: TopicProgress?

    private var tint: Color { theme.tint(for: content.area(topic.area)) }
    private var stateByItem: [String: CardState] {
        Dictionary(states.filter { $0.topicID == topic.id }.map { ($0.itemID, $0) }, uniquingKeysWith: { a, _ in a })
    }

    var body: some View {
        let mastery = Study.mastery(of: topic, states: stateByItem)
        let started = progress?.lessonCompletedAt != nil
        let due = stateByItem.values.filter { $0.due <= .now }.compactMap { content.itemsByID[$0.itemID] }

        ScrollView {
            VStack(alignment: .leading, spacing: Space.m) {
                // Header
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 8) {
                        SectionLabel(content.area(topic.area)?.title ?? topic.area, color: tint)
                        if let level = topic.level { LevelBadge(level: level) }
                    }
                    HStack(alignment: .top, spacing: 12) {
                        Text(topic.title).font(theme.font(.display)).foregroundStyle(theme.text)
                        Spacer()
                        MasteryRing(value: mastery, tint: tint, lineWidth: 4, showsLabel: true).frame(width: 46, height: 46)
                    }
                    Text(topic.summary).font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
                }

                ThemedButton(started ? "Practice all \(topic.items.count) items" : "Start lesson") {
                    active = started ? .review(topic.items.shuffled(), source: .practice, title: "Practice complete") : .lesson(topic)
                }
                HStack(spacing: 10) {
                    ThemedButton("Read explainer", prominent: false) { active = .explainer(topic) }
                    if !due.isEmpty {
                        ThemedButton("\(due.count) due", prominent: false) {
                            active = .review(due, source: .review, title: "Review complete")
                        }
                    }
                }

                if let prereqs = topic.prereqs, !prereqs.isEmpty {
                    SectionLabel("Builds on").sectionGap()
                    PanelList(prereqs) { id in
                        if let t = content.topicsByID[id] {
                            NavigationLink(value: t) {
                                HStack {
                                    Text(t.title).font(theme.font(.body)).foregroundStyle(theme.text)
                                    Spacer()
                                    Image(systemName: "chevron.right").font(.caption).foregroundStyle(theme.textMuted)
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        } else {
                            Text(id).font(theme.font(.mono)).foregroundStyle(theme.textMuted)
                        }
                    }
                }

                SectionLabel("Explainer").sectionGap()
                PanelList(Array(topic.explainer.enumerated())) { pair in
                    Button { active = .explainer(topic) } label: {
                        HStack(spacing: 10) {
                            Text(String(format: "%02d", pair.offset + 1)).font(theme.font(.label)).foregroundStyle(theme.label)
                            Text(pair.element.title).font(theme.font(.body)).foregroundStyle(theme.text)
                            Spacer()
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }

                SectionLabel("Items · \(topic.items.count)").sectionGap()
                PanelList(topic.items) { item in
                    ItemRow(item: item, state: stateByItem[item.id])
                }

                if let reading = topic.reading, !reading.isEmpty {
                    SectionLabel("Further reading").sectionGap()
                    PanelList(reading) { r in
                        if let url = URL(string: r.url) {
                            Link(destination: url) {
                                HStack(alignment: .top, spacing: 12) {
                                    Image(systemName: (r.kind ?? .paper).symbol).foregroundStyle(tint).frame(width: 20)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(r.title).font(theme.font(.body)).foregroundStyle(theme.text).multilineTextAlignment(.leading)
                                        if let note = r.note {
                                            Text(note).font(theme.font(.caption)).foregroundStyle(theme.textSecondary).multilineTextAlignment(.leading)
                                        }
                                    }
                                    Spacer(minLength: 0)
                                    Image(systemName: "arrow.up.right").font(.caption).foregroundStyle(theme.textMuted)
                                }
                            }
                        }
                    }
                }

                if let progress {
                    SectionLabel("My notes").sectionGap()
                    NotesField(progress: progress)
                }
            }
            .padding()
        }
        .screenBackground()
        .themedNavBar(topic.title)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    progress?.bookmarked.toggle()
                    try? ctx.save()
                    Haptics.tap()
                } label: {
                    Image(systemName: progress?.bookmarked == true ? "bookmark.fill" : "bookmark")
                        .foregroundStyle(progress?.bookmarked == true ? theme.warning : theme.textSecondary)
                }
                .accessibilityLabel("Bookmark")
            }
        }
        .sessionCover($active)
        .onAppear {
            let p = Study.progress(topic.id, in: ctx)
            p.lastOpened = .now
            try? ctx.save()
            progress = p
        }
    }
}

private struct ItemRow: View {
    @Environment(\.theme) private var theme
    let item: StudyItem
    let state: CardState?
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(.snappy) { expanded.toggle() }
            } label: {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: icon).font(.callout).foregroundStyle(theme.textMuted).frame(width: 18)
                    Text(Self.plain(item.promptText)).font(theme.font(.callout)).lineLimit(expanded ? nil : 2)
                        .foregroundStyle(theme.text).multilineTextAlignment(.leading)
                    Spacer(minLength: 0)
                    if let state, !state.isNew {
                        Text(state.due <= .now ? theme.labelText("due") : FSRS.format(days: state.stability))
                            .font(theme.font(.label)).foregroundStyle(state.due <= .now ? theme.warning : theme.textSecondary)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if expanded {
                switch item {
                case .mcq(let q, _):
                    RichText(q.prompt, role: .callout)
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "checkmark").foregroundStyle(theme.success)
                        RichText(q.correct, role: .callout)
                    }
                    if let e = q.explanation { RichText(e, role: .caption, color: theme.textSecondary) }
                case .flash(let c, _):
                    RichText(c.front, role: .callout)
                    Rectangle().fill(theme.border).frame(height: 1)
                    RichText(c.back, role: .callout)
                }
            }
        }
    }

    private var icon: String {
        switch item {
        case .mcq: "list.bullet"
        case .flash(let c, _): c.isOpen ? "waveform" : "rectangle.on.rectangle"
        }
    }

    /// Strips code fences and math delimiters for compact list display.
    static func plain(_ s: String) -> String {
        var out = s
        if let fence = out.range(of: "```") { out = String(out[..<fence.lowerBound]) + " [code]" }
        return out.replacingOccurrences(of: "$$", with: "").replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}

private struct NotesField: View {
    @Bindable var progress: TopicProgress
    @Environment(\.modelContext) private var ctx
    @Environment(\.theme) private var theme

    var body: some View {
        TextField("", text: $progress.note, prompt: Text("Add notes, mnemonics, follow-ups…").foregroundStyle(theme.textMuted), axis: .vertical)
            .lineLimit(3...12)
            .font(theme.font(.body))
            .foregroundStyle(theme.text)
            .panel()
            .onChange(of: progress.note) { try? ctx.save() }
    }
}
