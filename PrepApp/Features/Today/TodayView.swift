import SwiftData
import SwiftUI

struct TodayView: View {
    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    @Query private var states: [CardState]
    @Query private var progress: [TopicProgress]
    @Query(sort: \ReviewLog.date, order: .reverse) private var logs: [ReviewLog]
    @AppStorage("dailyGoal") private var dailyGoal = 20
    @AppStorage("sessionSize") private var sessionSize = 20
    @State private var active: ActiveSession?
    @State private var now = Date.now

    private var dueStates: [CardState] {
        states.filter { $0.due <= now && content.itemsByID[$0.itemID] != nil }.sorted { $0.due < $1.due }
    }
    private var progressByTopic: [String: TopicProgress] {
        Dictionary(progress.map { ($0.topicID, $0) }, uniquingKeysWith: { a, _ in a })
    }
    private var counts: [Date: Int] { Study.dailyCounts(logs) }
    private var todayCount: Int { counts[Calendar.current.startOfDay(for: now)] ?? 0 }
    private var nextDue: Date? {
        states.filter { $0.due > now && content.itemsByID[$0.itemID] != nil }.map(\.due).min()
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Space.m) {
                    SchmidhuberMascotGreeting(dueCount: dueStates.count, goalDone: todayCount >= dailyGoal, streak: Study.streak(counts))  // unlockable mascot (SchmidhuberMascot.swift)
                    ScreenHeader(title: "Today") {
                        HStack(spacing: Space.xs) {
                            streakBadge
                            NavigationLink { SettingsView() } label: {
                                Image(systemName: "gearshape").font(.title3).foregroundStyle(theme.textSecondary)
                                    .frame(width: Space.tap, height: Space.tap)
                                    .contentShape(Rectangle())
                            }
                            .accessibilityLabel("Settings")
                        }
                        .padding(.trailing, -Space.s)
                    }
                    .schmidhuberSecretTaps()  // easter egg: tap the title 5× (SchmidhuberEasterEgg.swift)
                    statusLine
                        .padding(.top, -Space.s)
                        .padding(.bottom, Space.xs)
                    // The one thing to do now goes first and gets the prominent button.
                    if dueStates.isEmpty {
                        learnCard.entrance(0)
                        reviewCard.entrance(1)
                    } else {
                        reviewCard.entrance(0)
                        learnCard.entrance(1)
                    }
                    goalCard.entrance(2)
                    examCard.entrance(3)
                    weakSpots.entrance(4)
                }
                .padding(.horizontal, Space.l)
                .padding(.bottom, Space.l)
            }
            .screenBackground()
            .toolbar(.hidden, for: .navigationBar)
            .sessionCover($active) { now = .now }
            .onAppear { now = .now }
            .refreshable { now = .now }
        }
    }

    // MARK: Sections

    private var streakBadge: some View {
        let streak = Study.streak(counts)
        return Group {
            switch theme.id {
            case .workbench:
                Text("STREAK \(String(format: "%02d", streak))").font(theme.font(.label)).tracking(1.2)
                    .foregroundStyle(streak > 0 ? theme.label : theme.textMuted)
            case .terminal:
                Text("\(streak)d ▲").font(theme.font(.label)).foregroundStyle(streak > 0 ? theme.accent : theme.textMuted)
            case .editorial:
                Label("\(streak)", systemImage: "flame.fill").font(theme.font(.caption))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(theme.accent.opacity(0.18), in: Capsule())
                    .foregroundStyle(streak > 0 ? theme.accent : theme.textMuted)
            }
        }
    }

    /// One line under the title that says what today looks like.
    private var statusLine: some View {
        let text: String = if !dueStates.isEmpty {
            "Reviews first, while they're fresh. Then something new."
        } else if states.isEmpty {
            "Start with a short lesson. About 5 minutes."
        } else if todayCount >= dailyGoal {
            "Goal done for today. Anything more is a bonus."
        } else {
            "Nothing due. A good moment to learn something new."
        }
        return Text(text).font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
    }

    /// Rough session length: ~12 s per review item.
    private static func minutes(items: Int) -> Int { max(1, Int((Double(items) * 12 / 60).rounded())) }

    private var goalCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionLabel(todayCount >= dailyGoal ? "Daily goal reached" : "Daily goal", icon: "target")
            GoalMeter(count: todayCount, goal: dailyGoal)
        }
        .panel()
    }

    @ViewBuilder private var reviewCard: some View {
        if dueStates.isEmpty {
            // Quiet status row; the review deck only gets a big card when something is due.
            if !states.isEmpty {
                HStack(spacing: Space.m) {
                    Image(systemName: "checkmark.circle").font(.title3).foregroundStyle(theme.success)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Reviews all caught up").font(theme.font(.bodyStrong)).foregroundStyle(theme.text)
                        if let nextDue {
                            Text("Next review \(nextDue.formatted(.relative(presentation: .named)))")
                                .font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                        }
                    }
                    Spacer(minLength: 0)
                }
                .panel()
            }
        } else {
            VStack(alignment: .leading, spacing: Space.m) {
                SectionLabel("Reviews due", icon: "rectangle.stack")
                HStack(alignment: .firstTextBaseline, spacing: Space.s) {
                    Text("\(dueStates.count)").font(theme.font(.hero)).foregroundStyle(theme.text)
                        .contentTransition(.numericText())
                    Text(dueStates.count == 1 ? "card" : "cards").font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
                    Spacer()
                    Text("≈ \(Self.minutes(items: min(dueStates.count, sessionSize))) min")
                        .font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                }
                ThemedButton("Start review" + (dueStates.count > sessionSize ? " (\(sessionSize))" : "")) {
                    let items = dueStates.prefix(sessionSize).compactMap { content.itemsByID[$0.itemID] }
                    active = .review(Array(items), source: .review, title: "Review complete")
                }
                .accessibilityIdentifier("startReview")
            }
            .panel(padding: Space.l, raised: true, stroke: theme.accent.opacity(0.45))
        }
    }

    @ViewBuilder private var learnCard: some View {
        if let topic = Study.nextTopic(content: content, progress: progressByTopic) {
            let area = content.area(topic.area)
            let tint = theme.tint(for: area)
            let hero = dueStates.isEmpty
            VStack(alignment: .leading, spacing: Space.m) {
                SectionLabel("Learn next", icon: "sparkles")
                VStack(alignment: .leading, spacing: Space.xs + 2) {
                    Text(topic.title).font(theme.font(hero ? .display : .title)).foregroundStyle(theme.text)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(topic.summary).font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
                        .lineLimit(hero ? 3 : 2)
                    HStack(spacing: 6) {
                        Circle().fill(tint).frame(width: 6, height: 6)
                        Text("\(area?.title ?? "") · \(topic.explainer.count) cards · \(topic.items.count) items")
                    }
                    .font(theme.font(.caption)).foregroundStyle(theme.textMuted)
                    .padding(.top, 2)
                }
                ThemedButton("Start lesson", prominent: hero) {
                    active = .lesson(topic)
                }
                .accessibilityIdentifier("startLesson")
            }
            .panel(padding: hero ? Space.l : 14, raised: hero, stroke: hero ? theme.accent.opacity(0.45) : nil)
        } else if !content.topics.isEmpty {
            Text("You've started every topic. Time to add more content.")
                .font(theme.font(.callout)).foregroundStyle(theme.textSecondary).panel()
        }
    }

    private var examCard: some View {
        Button {
            active = .exam(ExamConfig(areaIDs: Set(content.areas.map(\.id)), count: 10, timed: false, onlyStudied: false))
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "timer").font(.title3).foregroundStyle(theme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Quick mixed quiz").font(theme.font(.bodyStrong)).foregroundStyle(theme.text)
                    Text("10 questions across all topics").font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(theme.textMuted)
            }
            .panel()
        }
        .buttonStyle(.plain)
        .disabled(content.topics.isEmpty)
    }

    @ViewBuilder private var weakSpots: some View {
        let weak = WeakSpots.compute(logs: logs, content: content).prefix(3)
        if !weak.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                SectionLabel("Weak spots", icon: "exclamationmark.triangle", color: theme.warning)
                ForEach(Array(weak), id: \.topic.id) { w in
                    Button {
                        active = .review(w.topic.items, source: .practice, title: "Practice complete")
                    } label: {
                        HStack {
                            Text(w.topic.title).font(theme.font(.callout)).foregroundStyle(theme.text)
                            Spacer()
                            Text("\(Int(w.accuracy * 100))%").font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                            Image(systemName: "play.fill").font(.caption).foregroundStyle(theme.warning)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .panel()
        }
    }
}

enum WeakSpots {
    struct Entry { let topic: Topic; let accuracy: Double; let answered: Int }

    /// Topics with the lowest accuracy over the last 30 days (min 3 answers, below 85%).
    static func compute(logs: [ReviewLog], content: ContentStore, now: Date = .now) -> [Entry] {
        let cutoff = now.addingTimeInterval(-30 * 86_400)
        var tally: [String: (Int, Int)] = [:]
        for log in logs where log.date >= cutoff {
            let t = tally[log.topicID] ?? (0, 0)
            tally[log.topicID] = (t.0 + (log.correct ? 1 : 0), t.1 + 1)
        }
        return tally.compactMap { id, t in
            guard let topic = content.topicsByID[id], t.1 >= 3 else { return nil }
            return Entry(topic: topic, accuracy: Double(t.0) / Double(t.1), answered: t.1)
        }
        .filter { $0.accuracy < 0.85 }
        .sorted { $0.accuracy < $1.accuracy }
    }
}
