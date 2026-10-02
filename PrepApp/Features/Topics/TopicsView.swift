import SwiftData
import SwiftUI

struct TopicsView: View {
    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    @Query private var states: [CardState]
    @Query private var progress: [TopicProgress]
    @State private var query = ""

    private var stateByItem: [String: CardState] {
        Dictionary(states.map { ($0.itemID, $0) }, uniquingKeysWith: { a, _ in a })
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Space.m) {
                    ScreenHeader("Topics")
                    ThemedSearchField(text: $query, prompt: "Search topics, e.g. attention")
                    if query.isEmpty {
                        let live = content.areas.filter { !content.topics(in: $0.id).isEmpty }
                        let soon = content.areas.filter { content.topics(in: $0.id).isEmpty }
                        PanelList(live) { area in
                            let topics = content.topics(in: area.id)
                            NavigationLink(value: area) {
                                AreaRow(area: area, topics: topics, mastery: areaMastery(topics))
                            }
                            .buttonStyle(.plain)
                        }
                        if !soon.isEmpty {
                            SectionLabel("Coming soon").sectionGap()
                            LazyVGrid(columns: [GridItem(.flexible(), spacing: Space.s), GridItem(.flexible())], spacing: Space.s) {
                                ForEach(soon) { SoonChip(area: $0) }
                            }
                        }
                    } else {
                        let results = content.search(query)
                        if results.isEmpty {
                            EmptyState(icon: "magnifyingglass", title: "No matches",
                                       message: "Nothing matches “\(query)”. Try a broader word, like “loss” or “attention”.")
                        } else {
                            PanelList(results) { topic in
                                NavigationLink(value: topic) {
                                    TopicRow(topic: topic, states: stateByItem, progress: progress.first { $0.topicID == topic.id })
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }
                .padding(.horizontal, Space.l)
                .padding(.bottom, Space.l)
            }
            .scrollDismissesKeyboard(.immediately)
            .screenBackground()
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: Area.self) { AreaView(area: $0) }
            .navigationDestination(for: Topic.self) { TopicDetailView(topic: $0) }
        }
    }

    private func areaMastery(_ topics: [Topic]) -> Double {
        guard !topics.isEmpty else { return 0 }
        let states = stateByItem
        return topics.map { Study.mastery(of: $0, states: states) }.reduce(0, +) / Double(topics.count)
    }
}

struct AreaRow: View {
    @Environment(\.theme) private var theme
    let area: Area
    let topics: [Topic]
    let mastery: Double

    var body: some View {
        let tint = theme.tint(for: area)
        HStack(spacing: 14) {
            Image(systemName: area.icon)
                .font(.body)
                .frame(width: 38, height: 38)
                .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: min(theme.radius, 10)))
                .foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 3) {
                Text(area.title).font(theme.font(.bodyStrong)).foregroundStyle(theme.text)
                Text(topics.isEmpty ? area.blurb : "\(topics.count) topic\(topics.count == 1 ? "" : "s")")
                    .font(theme.font(.caption)).foregroundStyle(theme.textSecondary).lineLimit(1)
            }
            Spacer()
            if !topics.isEmpty {
                MasteryRing(value: mastery, tint: tint, lineWidth: 3, showsLabel: true).frame(width: 34, height: 34)
                Image(systemName: "chevron.right").font(.caption).foregroundStyle(theme.textMuted)
            }
        }
        .contentShape(Rectangle())
        .opacity(topics.isEmpty ? 0.5 : 1)
    }
}

/// Compact tile for an area that has no content yet.
private struct SoonChip: View {
    @Environment(\.theme) private var theme
    let area: Area

    var body: some View {
        HStack(spacing: Space.s) {
            Image(systemName: area.icon).font(.footnote).foregroundStyle(theme.tint(for: area).opacity(0.7))
                .frame(width: 18)
            Text(area.title).font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                .lineLimit(2).minimumScaleFactor(0.85)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .frame(maxHeight: .infinity)
        .panel(padding: 10)
        .accessibilityElement(children: .combine)
        .accessibilityHint(area.blurb)
    }
}

struct TopicRow: View {
    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    let topic: Topic
    let states: [String: CardState]
    let progress: TopicProgress?

    var body: some View {
        let tint = theme.tint(for: content.area(topic.area))
        HStack(spacing: 12) {
            MasteryRing(value: Study.mastery(of: topic, states: states), tint: tint, lineWidth: 3)
                .frame(width: 24, height: 24)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(topic.title).font(theme.font(.bodyStrong)).foregroundStyle(theme.text)
                    if progress?.bookmarked == true {
                        Image(systemName: "bookmark.fill").font(.caption2).foregroundStyle(theme.warning)
                    }
                }
                Text(topic.summary).font(theme.font(.caption)).foregroundStyle(theme.textSecondary).lineLimit(2)
            }
            Spacer(minLength: 0)
            if progress?.lessonCompletedAt == nil {
                Text(theme.labelText("New")).font(theme.font(.label)).foregroundStyle(tint)
            }
            Image(systemName: "chevron.right").font(.caption).foregroundStyle(theme.textMuted)
        }
        .contentShape(Rectangle())
    }
}

struct AreaView: View {
    let area: Area
    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    @Query private var states: [CardState]
    @Query private var progress: [TopicProgress]
    @State private var active: ActiveSession?

    var body: some View {
        let topics = content.topics(in: area.id)
        let stateByItem = Dictionary(states.map { ($0.itemID, $0) }, uniquingKeysWith: { a, _ in a })
        let studied = topics.flatMap(\.items).filter { stateByItem[$0.id] != nil }
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(area.title).font(theme.font(.display)).foregroundStyle(theme.text)
                    Text(area.blurb).font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
                }
                ThemedButton("Practice studied items", prominent: false) {
                    active = .review(Array(studied.shuffled().prefix(20)), source: .practice, title: "Practice complete")
                }
                .disabled(studied.isEmpty)
                SectionLabel("Topics").padding(.top, 4)
                PanelList(topics) { topic in
                    NavigationLink(value: topic) {
                        TopicRow(topic: topic, states: stateByItem, progress: progress.first { $0.topicID == topic.id })
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding()
        }
        .screenBackground()
        .themedNavBar(area.title)
        .sessionCover($active)
    }
}
