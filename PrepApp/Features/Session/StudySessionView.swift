import SwiftData
import SwiftUI

struct SessionSummary {
    var answered = 0
    var correct = 0
    var started = Date.now
    var missedItemIDs: [String] = []
    /// Distinct items seen, and how many of those were right on the first attempt
    /// (re-queued retries don't drag the headline score down).
    var seenItemIDs: Set<String> = []
    var firstTryCorrect = 0

    var accuracy: Double { answered == 0 ? 0 : Double(correct) / Double(answered) }
    var firstTryAccuracy: Double { seenItemIDs.isEmpty ? 0 : Double(firstTryCorrect) / Double(seenItemIDs.count) }
}

/// Close button, progress and trailing accessory shared by all full-screen sessions.
struct SessionTopBar<Trailing: View>: View {
    @Environment(\.theme) private var theme
    var progress: Double
    var tint: Color?
    var onClose: () -> Void
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: Space.xs) {
            IconButton("xmark", label: "Close", action: onClose)
            if theme.progress == .ascii {
                Text(AsciiBar.make(progress, width: 18))
                    .font(theme.font(.mono)).foregroundStyle(tint ?? theme.accent)
                    .lineLimit(1).minimumScaleFactor(0.5)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ThinBar(value: progress, tint: tint, height: theme.id == .editorial ? 8 : 4)
            }
            trailing
        }
        .padding(.horizontal, Space.xs).padding(.vertical, Space.xs)
        .animation(.snappy, value: progress)
    }
}

/// Runs a queue of MCQs and flashcards, recording every answer into the scheduler.
/// Items answered wrong / rated "Again" are re-queued at the end of the session.
struct StudySessionView: View {
    let source: ReviewSource
    var onClose: () -> Void
    var onFinish: (SessionSummary) -> Void

    @Environment(\.modelContext) private var ctx
    @Environment(\.theme) private var theme
    @State private var queue: [StudyItem]
    @State private var index = 0
    @State private var summary = SessionSummary()
    @State private var itemStart = Date.now
    @State private var flagging: StudyItem?
    private let initialCount: Int

    init(items: [StudyItem], source: ReviewSource, onClose: @escaping () -> Void,
         onFinish: @escaping (SessionSummary) -> Void) {
        self.source = source
        self.onClose = onClose
        self.onFinish = onFinish
        self.initialCount = items.count
        _queue = State(initialValue: items)
    }

    var body: some View {
        VStack(spacing: 0) {
            SessionTopBar(progress: Double(min(index, queue.count)) / Double(max(queue.count, 1)), onClose: onClose) {
                Text("\(min(index + 1, queue.count))/\(queue.count)")
                    .font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .padding(.leading, Space.s)
                IconButton("flag", label: "Flag item") {
                    if index < queue.count { flagging = queue[index] }
                }
            }
            if index < queue.count {
                itemView(queue[index])
                    .id("\(index)-\(queue[index].id)")
                    .transition(.asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading)).combined(with: .opacity))
            } else {
                Spacer()
            }
        }
        .screenBackground()
        .sheet(item: $flagging) { FlagSheet(item: $0) }
        .onAppear { if queue.isEmpty { onFinish(summary) } }
    }

    @ViewBuilder
    private func itemView(_ item: StudyItem) -> some View {
        switch item {
        case .mcq(let q, _):
            MCQView(question: q, mode: .practice) { correct in
                answer(item, rating: correct ? .good : .again, correct: correct)
            }
        case .flash(let c, _):
            let state = Study.cardState(item.id, in: ctx)
            FlashcardView(card: c, previews: Study.scheduler.previews(memory: Study.memory(of: state), lastReview: state?.lastReview)) { r in
                answer(item, rating: r, correct: r != .again)
            }
        }
    }

    private func answer(_ item: StudyItem, rating: Rating, correct: Bool) {
        let ms = Int(Date.now.timeIntervalSince(itemStart) * 1000)
        Study.record(item, rating: rating, correct: correct, source: source, durationMs: ms, in: ctx)
        if !summary.seenItemIDs.contains(item.id) {
            summary.seenItemIDs.insert(item.id)
            if correct { summary.firstTryCorrect += 1 }
        }
        summary.answered += 1
        if correct { summary.correct += 1 } else { summary.missedItemIDs.append(item.id) }
        if correct { SchmidhuberCenter.shared.correctAnswer(item) }  // easter egg (SchmidhuberEasterEgg.swift)
        // Re-queue misses, but cap the session so it can't grow without bound.
        if rating == .again && queue.count < initialCount * 3 { queue.append(item) }
        itemStart = .now
        withAnimation(.snappy) { index += 1 }
        if index >= queue.count { onFinish(summary) }
    }
}

struct SessionSummaryView: View {
    let summary: SessionSummary
    var title = "Session complete"
    var subtitle: String?
    var onDone: () -> Void

    @Environment(\.theme) private var theme
    @Query private var logs: [ReviewLog]
    @AppStorage("dailyGoal") private var dailyGoal = 20

    private var score: Double { summary.firstTryAccuracy }
    private var scoreTint: Color { score >= 0.8 ? theme.success : score >= 0.5 ? theme.accent : theme.warning }

    private var verdict: String {
        guard summary.answered > 0 else { return "Nice. That's one more topic in your head." }
        switch score {
        case 0.9...: return "Sharp recall. These won't be back for a while."
        case 0.7...: return "Solid. The ones you missed will come back sooner."
        default: return "Good reps. Misses come back sooner, which is how they stick."
        }
    }

    var body: some View {
        let counts = Study.dailyCounts(logs)
        let today = counts[Calendar.current.startOfDay(for: .now)] ?? 0
        VStack(spacing: Space.xl) {
            Spacer(minLength: Space.l)

            Group {
                if summary.answered > 0 {
                    ScoreDial(value: score, tint: scoreTint, caption: "First try")
                } else if theme.id == .terminal {
                    Text("exit 0  ✓").font(theme.font(.number)).foregroundStyle(theme.accent)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Image(systemName: "checkmark.circle")
                        .font(.system(size: 76, weight: .light))
                        .foregroundStyle(theme.success)
                        .symbolEffect(.bounce, value: summary.answered)
                }
            }
            .entrance(0)

            VStack(alignment: theme.id == .terminal ? .leading : .center, spacing: Space.s) {
                theme.titleText(title).font(theme.font(.display)).foregroundStyle(theme.text)
                if let subtitle {
                    Text(subtitle).font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
                }
                Text(verdict).font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
            }
            .multilineTextAlignment(theme.id == .terminal ? .leading : .center)
            .frame(maxWidth: .infinity, alignment: theme.id == .terminal ? .leading : .center)
            .entrance(1)
            SchmidhuberMascotCheer(accuracy: score, answered: summary.answered)  // unlockable mascot (SchmidhuberMascot.swift)

            Spacer(minLength: Space.l)

            VStack(spacing: Space.m) {
                StatStrip(items: [
                    StatItem(value: "\(summary.seenItemIDs.count)", label: "Items"),
                    StatItem(value: Self.clock(Date.now.timeIntervalSince(summary.started)), label: "Time"),
                    StatItem(value: "\(Study.streak(counts))", label: "Streak",
                             tint: Study.streak(counts) > 0 ? theme.warning : nil),
                ])
                VStack(alignment: .leading, spacing: Space.s) {
                    SectionLabel(today >= dailyGoal ? "Daily goal reached" : "Daily goal", icon: "target",
                                 color: today >= dailyGoal ? theme.success : nil)
                    GoalMeter(count: today, goal: dailyGoal)
                }
                .panel()
            }
            .entrance(2)

            ThemedButton("Done", action: onDone)
                .accessibilityIdentifier("summaryDone")
        }
        .padding(.horizontal, Space.l)
        .padding(.bottom, Space.s)
        .screenBackground()
        .onAppear { if summary.answered > 0 { Haptics.celebrate() } }
    }

    private static func clock(_ t: TimeInterval) -> String {
        let s = max(Int(t), 0)
        return s >= 3600 ? "\(s / 3600)h\(s % 3600 / 60)m" : String(format: "%d:%02d", s / 60, s % 60)
    }
}

/// Full-screen review session: a queue of items, then a summary.
struct ReviewScreen: View {
    let items: [StudyItem]
    var source: ReviewSource = .review
    var title = "Review complete"

    @Environment(\.dismiss) private var dismiss
    @State private var finished: SessionSummary?

    var body: some View {
        if let finished {
            SessionSummaryView(summary: finished, title: title) { dismiss() }
        } else {
            StudySessionView(items: items, source: source, onClose: { dismiss() }) { s in
                withAnimation { finished = s }
            }
        }
    }
}
