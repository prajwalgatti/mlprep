import SwiftData
import SwiftUI

struct ExamConfig: Identifiable, Hashable {
    let id = UUID()
    var areaIDs: Set<String>
    var count: Int
    var timed: Bool
    var onlyStudied: Bool
    /// Seconds per question when timed.
    var secondsPerQuestion = 60

    var title: String { "\(count)-question \(timed ? "timed " : "")exam" }
}

/// A themed on/off row.
struct ToggleRow: View {
    @Environment(\.theme) private var theme
    let title: String
    var icon: String?
    var tint: Color?
    @Binding var isOn: Bool

    var body: some View {
        Toggle(isOn: $isOn) {
            HStack(spacing: 10) {
                if let icon { Image(systemName: icon).foregroundStyle(tint ?? theme.textSecondary).frame(width: 22) }
                Text(title).font(theme.font(.body)).foregroundStyle(theme.text)
            }
        }
        .tint(tint ?? theme.accent)
    }
}

// MARK: - Setup tab

struct ExamTab: View {
    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    @Query private var progress: [TopicProgress]
    @Query(sort: \ExamRecord.date, order: .reverse) private var history: [ExamRecord]
    @State private var areaIDs: Set<String> = []
    @State private var count = 20
    @State private var timed = true
    @State private var onlyStudied = false
    @State private var active: ActiveSession?

    private var areasWithContent: [Area] {
        content.areas.filter { a in content.topics(in: a.id).contains { !$0.mcqs.isEmpty } }
    }
    private var studiedIDs: Set<String> {
        Set(progress.filter { $0.lessonCompletedAt != nil }.map(\.topicID))
    }
    private var config: ExamConfig {
        ExamConfig(areaIDs: areaIDs, count: count, timed: timed, onlyStudied: onlyStudied)
    }
    private var available: Int {
        ExamScreen.pool(content: content, config: config, studied: studiedIDs).count
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Space.m) {
                    ScreenHeader("Mock exam")
                    Text("Questions are drawn at random. Feedback comes at the end, like a real interview round.")
                        .font(theme.font(.callout)).foregroundStyle(theme.textSecondary)

                    SectionLabel("Areas").sectionGap()
                    PanelList(areasWithContent) { area in
                        ToggleRow(title: area.title, icon: area.icon, tint: theme.tint(for: area), isOn: Binding(
                            get: { areaIDs.contains(area.id) },
                            set: { on in if on { areaIDs.insert(area.id) } else { areaIDs.remove(area.id) } }
                        ))
                    }

                    SectionLabel("Format").sectionGap()
                    ThemedSegmented(selection: $count, options: [10, 20, 30, 50].map { ($0, "\($0)") })
                    VStack(spacing: 0) {
                        ToggleRow(title: "Timed (1 min per question)", isOn: $timed).padding(.horizontal, 14).padding(.vertical, 11)
                        PanelDivider()
                        ToggleRow(title: "Only topics I've studied", isOn: $onlyStudied).padding(.horizontal, 14).padding(.vertical, 11)
                    }
                    .panel(padding: 0)

                    ThemedButton(available < count ? "Start (\(available) available)" : "Start exam") {
                        active = .exam(config)
                    }
                    .disabled(available == 0)
                    .accessibilityIdentifier("startExam")
                    .padding(.top, 4)

                    if !history.isEmpty {
                        SectionLabel("History").sectionGap()
                        PanelList(history) { r in
                            NavigationLink { ExamRecordDetail(record: r) } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(r.title).font(theme.font(.body)).foregroundStyle(theme.text)
                                        Text(r.date.formatted(date: .abbreviated, time: .shortened))
                                            .font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                                    }
                                    Spacer()
                                    Text("\(r.correct)/\(r.total)").font(theme.font(.bodyStrong))
                                        .foregroundStyle(scoreColor(r.score))
                                    Image(systemName: "chevron.right").font(.caption).foregroundStyle(theme.textMuted)
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding()
            }
            .screenBackground()
            .toolbar(.hidden, for: .navigationBar)
            .onAppear { if areaIDs.isEmpty { areaIDs = Set(areasWithContent.map(\.id)) } }
            .sessionCover($active)
        }
    }

    private func scoreColor(_ s: Double) -> Color {
        s >= 0.8 ? theme.success : s >= 0.6 ? theme.warning : theme.danger
    }
}

// MARK: - Running an exam

struct ExamScreen: View {
    let config: ExamConfig

    @Environment(ContentStore.self) private var content
    @Environment(\.modelContext) private var ctx
    @Environment(\.dismiss) private var dismiss
    @Environment(\.theme) private var theme
    @Query private var progress: [TopicProgress]

    @State private var questions: [StudyItem] = []
    @State private var index = 0
    @State private var answers: [String: Bool] = [:]
    @State private var started = Date.now
    @State private var remaining = 0
    @State private var record: ExamRecord?
    @State private var confirmQuit = false

    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    static func pool(content: ContentStore, config: ExamConfig, studied: Set<String>) -> [StudyItem] {
        content.topics
            .filter { config.areaIDs.contains($0.area) && (!config.onlyStudied || studied.contains($0.id)) }
            .flatMap { t in t.mcqs.map { StudyItem.mcq($0, topicID: t.id) } }
    }

    var body: some View {
        Group {
            if let record {
                NavigationStack {
                    ExamRecordDetail(record: record)
                        .toolbar { Button("Done") { dismiss() } }
                }
            } else if index < questions.count, case .mcq(let q, _) = questions[index] {
                VStack(spacing: 0) {
                    SessionTopBar(progress: Double(index) / Double(questions.count), onClose: { confirmQuit = true }) {
                        Group {
                            if config.timed {
                                Text(Self.clock(remaining))
                                    .foregroundStyle(remaining < 60 ? theme.danger : theme.textSecondary)
                            } else {
                                Text("\(index + 1)/\(questions.count)").foregroundStyle(theme.textSecondary)
                            }
                        }
                        .font(theme.font(.label))
                        .monospacedDigit()
                        .padding(.leading, Space.s)
                        .padding(.trailing, Space.m)
                    }
                    MCQView(question: q, mode: .exam) { correct in
                        answers[questions[index].id] = correct
                        if index + 1 >= questions.count { finish() } else { index += 1 }
                    }
                    .id(index)
                }
                .screenBackground()
            } else {
                ProgressView().screenBackground()
            }
        }
        .onAppear(perform: start)
        .onReceive(timer) { _ in
            guard config.timed, record == nil, !questions.isEmpty else { return }
            remaining -= 1
            if remaining <= 0 { finish() }
        }
        .confirmationDialog("End exam?", isPresented: $confirmQuit) {
            Button("Submit answers so far") { finish() }
            Button("Discard exam", role: .destructive) { dismiss() }
        }
    }

    private func start() {
        guard questions.isEmpty else { return }
        let studied = Set(progress.filter { $0.lessonCompletedAt != nil }.map(\.topicID))
        questions = Array(Self.pool(content: content, config: config, studied: studied).shuffled().prefix(config.count))
        remaining = questions.count * config.secondsPerQuestion
        started = .now
    }

    private func finish() {
        guard record == nil else { return }
        var breakdown: [String: [Int]] = [:]
        var wrong: [String] = []
        for item in questions {
            let correct = answers[item.id] ?? false
            var b = breakdown[item.topicID] ?? [0, 0]
            b[0] += correct ? 1 : 0
            b[1] += 1
            breakdown[item.topicID] = b
            if !correct { wrong.append(item.id) }
            // Unanswered (timed out) questions count as wrong but don't touch the schedule.
            if answers[item.id] != nil {
                Study.recordExam(item, correct: correct, durationMs: 0, in: ctx)
            }
        }
        let r = ExamRecord(title: config.title, total: questions.count, correct: questions.count - wrong.count,
                           durationSec: Int(Date.now.timeIntervalSince(started)), breakdown: breakdown, wrongItemIDs: wrong)
        ctx.insert(r)
        try? ctx.save()
        withAnimation { record = r }
    }

    static func clock(_ s: Int) -> String {
        String(format: "%d:%02d", max(s, 0) / 60, max(s, 0) % 60)
    }
}

// MARK: - Results

struct ExamRecordDetail: View {
    let record: ExamRecord
    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    @State private var active: ActiveSession?

    var body: some View {
        let wrongItems = record.wrongItemIDs.compactMap { content.itemsByID[$0] }
        let rows = record.breakdown.sorted { ratio($0.value) < ratio($1.value) }
        ScrollView {
            VStack(alignment: .leading, spacing: Space.m) {
                VStack(spacing: 6) {
                    Text("\(Int((record.score * 100).rounded()))%")
                        .font(theme.font(.hero))
                        .foregroundStyle(scoreColor(record.score))
                    Text("\(record.correct) of \(record.total) correct · \(ExamScreen.clock(record.durationSec))")
                        .font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
                }
                .frame(maxWidth: .infinity)
                .panel(padding: 20)

                SectionLabel("By topic").sectionGap()
                PanelList(rows) { row in
                    HStack(spacing: 10) {
                        Text(content.topicsByID[row.key]?.title ?? row.key).font(theme.font(.callout)).foregroundStyle(theme.text).lineLimit(1)
                        Spacer()
                        ThinBar(value: ratio(row.value), tint: scoreColor(ratio(row.value))).frame(width: 60)
                        Text("\(row.value[0])/\(row.value[1])").font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                            .frame(width: 34, alignment: .trailing)
                    }
                }

                if !wrongItems.isEmpty {
                    ThemedButton("Retry \(wrongItems.count) missed", prominent: false) {
                        active = .review(wrongItems, source: .practice, title: "Mistakes reviewed")
                    }
                    .padding(.top, 4)
                    SectionLabel("Missed").sectionGap()
                    PanelList(wrongItems) { item in
                        if case .mcq(let q, _) = item {
                            MissedRow(question: q)
                        }
                    }
                }
            }
            .padding()
        }
        .screenBackground()
        .themedNavBar("Results")
        .sessionCover($active)
    }

    private func ratio(_ v: [Int]) -> Double { v.count == 2 && v[1] > 0 ? Double(v[0]) / Double(v[1]) : 0 }
    private func scoreColor(_ s: Double) -> Color { s >= 0.8 ? theme.success : s >= 0.6 ? theme.warning : theme.danger }
}

private struct MissedRow: View {
    @Environment(\.theme) private var theme
    let question: MCQ
    @State private var open = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button { withAnimation(.snappy) { open.toggle() } } label: {
                HStack(alignment: .top) {
                    RichText(question.prompt, role: .callout)
                    Image(systemName: open ? "chevron.up" : "chevron.down").font(.caption).foregroundStyle(theme.textMuted)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if open {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "checkmark").foregroundStyle(theme.success)
                    RichText(question.correct, role: .callout)
                }
                if let e = question.explanation { RichText(e, role: .caption, color: theme.textSecondary) }
            }
        }
    }
}
