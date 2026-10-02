import Charts
import SwiftData
import SwiftUI

struct StatsView: View {
    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    @Query private var states: [CardState]
    @Query(sort: \ReviewLog.date, order: .reverse) private var logs: [ReviewLog]

    private var stateByItem: [String: CardState] {
        Dictionary(states.map { ($0.itemID, $0) }, uniquingKeysWith: { a, _ in a })
    }

    var body: some View {
        let counts = Study.dailyCounts(logs)
        let recent = logs.filter { $0.date > .now.addingTimeInterval(-30 * 86_400) }
        let accuracy = recent.isEmpty ? 0 : Double(recent.filter(\.correct).count) / Double(recent.count)
        let mastered = content.topics.filter { Study.mastery(of: $0, states: stateByItem) >= 0.9 }.count
        let streak = Study.streak(counts)

        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Space.m) {
                    ScreenHeader("Progress")
                    if logs.isEmpty {
                        EmptyState(icon: "chart.bar", title: "No activity yet",
                                   message: "Finish a lesson and your streak, accuracy and review forecast will show up here.")
                    } else {
                        StatStrip(items: [
                            StatItem(value: "\(streak)", label: "Streak", tint: streak > 0 ? theme.warning : nil),
                            StatItem(value: "\(logs.count)", label: "Answers"),
                            StatItem(value: recent.isEmpty ? "–" : "\(Int(accuracy * 100))%", label: "Acc 30d"),
                            StatItem(value: "\(mastered)/\(content.topics.count)", label: "Mastered"),
                        ])

                        VStack(alignment: .leading, spacing: Space.m) {
                            header("Activity", trailing: activeDays(counts))
                            Heatmap(counts: counts)
                        }
                        .panel()

                        dueChart

                        masteryPanel

                        let weak = WeakSpots.compute(logs: logs, content: content)
                        if !weak.isEmpty {
                            VStack(alignment: .leading, spacing: Space.s) {
                                header("Weakest topics", trailing: "last 30 days", color: theme.warning)
                                ForEach(weak.prefix(5), id: \.topic.id) { w in
                                    HStack(spacing: Space.s) {
                                        Text(w.topic.title).font(theme.font(.callout)).foregroundStyle(theme.text).lineLimit(1)
                                        Spacer()
                                        Text("\(Int(w.accuracy * 100))%").font(theme.font(.label))
                                            .foregroundStyle(w.accuracy < 0.6 ? theme.danger : theme.warning)
                                        Text("of \(w.answered)").font(theme.font(.label)).foregroundStyle(theme.textMuted)
                                    }
                                    .padding(.vertical, 2)
                                }
                            }
                            .panel()
                        }
                    }
                }
                .padding(.horizontal, Space.l)
                .padding(.bottom, Space.l)
            }
            .screenBackground()
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    private func activeDays(_ counts: [Date: Int]) -> String {
        let n = counts.keys.filter { $0 > .now.addingTimeInterval(-112 * 86_400) }.count
        return "\(n) active day\(n == 1 ? "" : "s")"
    }

    private func header(_ title: String, trailing: String? = nil, color: Color? = nil) -> some View {
        HStack(alignment: .firstTextBaseline) {
            SectionLabel(title, color: color)
            Spacer()
            if let trailing {
                Text(trailing).font(theme.font(.caption)).foregroundStyle(theme.textMuted)
            }
        }
    }

    private var dueChart: some View {
        let days = forecast()
        let cal = Calendar.current
        return VStack(alignment: .leading, spacing: Space.m) {
            header("Due this week", trailing: "\(days.map(\.count).reduce(0, +)) cards")
            Chart(days, id: \.day) { d in
                BarMark(x: .value("Day", d.day, unit: .day), y: .value("Due", d.count))
                    .foregroundStyle(cal.isDateInToday(d.day) ? theme.accent : theme.accent.opacity(0.55))
                    .cornerRadius(theme.radius == 0 ? 0 : 3)
                    .annotation(position: .top, spacing: 4) {
                        if d.count > 0 {
                            Text("\(d.count)").font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                        }
                    }
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .day)) { value in
                    AxisValueLabel {
                        if let date = value.as(Date.self) {
                            Text(cal.isDateInToday(date) ? theme.labelText("Today") : date.formatted(.dateTime.weekday(.abbreviated)))
                                .font(theme.font(.label))
                                .foregroundStyle(cal.isDateInToday(date) ? theme.text : theme.textSecondary)
                        }
                    }
                }
            }
            .chartYAxis(.hidden)
            .frame(height: 120)
        }
        .panel()
    }

    private var masteryPanel: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            header("Mastery by area")
            ForEach(content.areas) { area in
                let topics = content.topics(in: area.id)
                if !topics.isEmpty {
                    let m = topics.map { Study.mastery(of: $0, states: stateByItem) }.reduce(0, +) / Double(topics.count)
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(area.title).font(theme.font(.callout)).foregroundStyle(theme.text)
                            Spacer()
                            Text("\(Int(m * 100))%").font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                        }
                        if theme.progress == .ascii {
                            Text(AsciiBar.make(m, width: 24)).font(theme.font(.mono)).foregroundStyle(theme.accent)
                                .lineLimit(1).minimumScaleFactor(0.5)
                        } else {
                            ThinBar(value: m, tint: theme.tint(for: area), height: theme.id == .editorial ? 8 : 4)
                        }
                    }
                }
            }
            Text("Mastery = how long you'd remember each item, relative to \(Int(Study.masteryDays)) days.")
                .font(theme.font(.caption)).foregroundStyle(theme.textMuted)
        }
        .panel()
    }

    private struct DayCount { let day: Date; let count: Int }

    private func forecast() -> [DayCount] {
        let cal = Calendar.current
        let today = cal.startOfDay(for: .now)
        let live = states.filter { content.itemsByID[$0.itemID] != nil }
        return (0..<7).map { offset in
            let day = cal.date(byAdding: .day, value: offset, to: today)!
            let end = cal.date(byAdding: .day, value: 1, to: day)!
            // Today's bar includes everything overdue.
            let n = live.filter { $0.due < end && (offset == 0 || $0.due >= day) }.count
            return DayCount(day: day, count: n)
        }
    }
}

/// GitHub-style activity grid for the last 16 weeks.
struct Heatmap: View {
    @Environment(\.theme) private var theme
    let counts: [Date: Int]
    var weeks = 16

    var body: some View {
        let cal = Calendar.current
        let today = cal.startOfDay(for: .now)
        let weekday = cal.component(.weekday, from: today) - 1   // 0 = Sunday
        let start = cal.date(byAdding: .day, value: -(weeks - 1) * 7 - weekday, to: today)!
        let maxCount = max(counts.values.max() ?? 1, 1)

        HStack(alignment: .top, spacing: 3) {
            ForEach(0..<weeks, id: \.self) { w in
                VStack(spacing: 3) {
                    ForEach(0..<7, id: \.self) { d in
                        let day = cal.date(byAdding: .day, value: w * 7 + d, to: start)!
                        let n = counts[day] ?? 0
                        let cell = RoundedRectangle(cornerRadius: theme.radius == 0 ? 0 : (theme.id == .editorial ? 4 : 2))
                        cell
                            .fill(day > today ? Color.clear
                                  : n == 0 ? theme.border
                                  : theme.accent.opacity(0.3 + 0.7 * Double(n) / Double(maxCount)))
                            .overlay(cell.strokeBorder(day == today ? theme.text.opacity(0.6) : .clear, lineWidth: 1))
                            .aspectRatio(1, contentMode: .fit)
                    }
                }
            }
        }
    }
}
