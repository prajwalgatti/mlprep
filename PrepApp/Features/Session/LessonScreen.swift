import SwiftData
import SwiftUI

/// Explainer cards → quiz + flashcards → summary. Marks the topic's lesson complete.
struct LessonScreen: View {
    let topic: Topic
    /// When true, only show the explainer (re-reading a finished topic).
    var explainerOnly = false

    private enum Phase { case explainer, practice, done }

    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var ctx
    @State private var phase: Phase = .explainer
    @State private var summary = SessionSummary()

    var body: some View {
        switch phase {
        case .explainer:
            ExplainerPager(topic: topic, finishLabel: explainerOnly || topic.items.isEmpty ? "Done" : "Start practice",
                           onClose: { dismiss() }) {
                if explainerOnly { dismiss() }
                else if topic.items.isEmpty { complete(SessionSummary()) }
                else { withAnimation { phase = .practice } }
            }
        case .practice:
            StudySessionView(items: topic.items, source: .lesson, onClose: { dismiss() }) { s in
                complete(s)
            }
        case .done:
            SessionSummaryView(summary: summary, title: "Lesson complete",
                               subtitle: "\(topic.items.count) items from \(topic.title) are now in your review deck.") {
                dismiss()
            }
        }
    }

    private func complete(_ s: SessionSummary) {
        let p = Study.progress(topic.id, in: ctx)
        if p.lessonCompletedAt == nil { p.lessonCompletedAt = .now }
        p.lastQuizScore = s.accuracy
        try? ctx.save()
        summary = s
        withAnimation { phase = .done }
    }
}

struct ExplainerPager: View {
    let topic: Topic
    var finishLabel = "Done"
    var onClose: () -> Void
    var onFinish: () -> Void

    @Environment(ContentStore.self) private var content
    @Environment(\.theme) private var theme
    @State private var page = 0

    private var tint: Color { theme.tint(for: content.area(topic.area)) }
    private var isLast: Bool { page >= topic.explainer.count - 1 }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: Space.xs) {
                IconButton("xmark", label: "Close", action: onClose)
                HStack(spacing: 4) {
                    ForEach(topic.explainer.indices, id: \.self) { i in
                        RoundedRectangle(cornerRadius: theme.radius == 0 ? 0 : 2)
                            .fill(i <= page ? tint : theme.border).frame(height: 4)
                    }
                }
                .animation(.snappy, value: page)
                Text("\(page + 1)/\(topic.explainer.count)")
                    .font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                    .monospacedDigit()
                    .padding(.leading, Space.s)
            }
            .padding(.leading, Space.xs).padding(.trailing, Space.l).padding(.vertical, Space.xs)

            TabView(selection: $page) {
                ForEach(Array(topic.explainer.enumerated()), id: \.offset) { i, card in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            if i == 0 {
                                SectionLabel(topic.title, color: tint)
                            }
                            Text(card.title).font(theme.font(.title)).foregroundStyle(theme.text)
                            RichText(card.body)
                            if let src = card.source { SourceLine(src).padding(.top, 4) }
                        }
                        .padding()
                        .padding(.bottom, 20)
                    }
                    .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))

            HStack(spacing: 10) {
                if page > 0 {
                    ThemedButton("Back", prominent: false) { withAnimation { page -= 1 } }
                        .frame(width: 96)
                }
                ThemedButton(isLast ? finishLabel : "Next") {
                    if isLast { onFinish() } else { withAnimation { page += 1 } }
                }
                .accessibilityIdentifier("explainerNext")
            }
            .actionBar()
        }
        .screenBackground()
    }
}
