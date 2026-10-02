import SwiftUI

// MARK: - Multiple choice

struct MCQView: View {
    enum Mode { case practice, exam }

    let question: MCQ
    var mode: Mode = .practice
    var onDone: (_ correct: Bool) -> Void

    @Environment(\.theme) private var theme
    @State private var choices: [String]
    @State private var selected: Int?
    @State private var revealed = false
    @State private var shake: CGFloat = 0
    @State private var pulse = false

    init(question: MCQ, mode: Mode = .practice, onDone: @escaping (_ correct: Bool) -> Void) {
        self.question = question
        self.mode = mode
        self.onDone = onDone
        _choices = State(initialValue: ([question.correct] + question.wrong).shuffled())
    }

    private var correctIndex: Int { choices.firstIndex(of: question.correct) ?? 0 }
    private var isCorrect: Bool { selected == correctIndex }

    var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                RichText(question.prompt, role: .title)

                VStack(spacing: Space.s + 2) {
                    ForEach(choices.indices, id: \.self) { i in
                        Button { choose(i) } label: { choiceRow(i) }
                            .buttonStyle(.plain)
                            .allowsHitTesting(!revealed)
                            .accessibilityIdentifier("choice-\(i)")
                    }
                }

                if revealed {
                    let tint = isCorrect ? theme.success : theme.warning
                    VStack(alignment: .leading, spacing: Space.s) {
                        SectionLabel(isCorrect ? "Correct" : "Not quite",
                                     icon: isCorrect ? "checkmark.seal.fill" : "lightbulb.fill",
                                     color: tint)
                        if let e = question.explanation, !e.isEmpty {
                            RichText(e, role: .callout)
                        }
                        if let src = question.source { SourceLine(src) }
                    }
                    .panel(raised: true, stroke: tint.opacity(0.35))
                    .id("feedback")
                    .transition(.opacity.combined(with: .offset(y: 16)))
                }
            }
            .padding()
        }
        .onChange(of: revealed) { _, now in
            guard now else { return }
            // Bring the explanation into view so it isn't hidden behind the Continue button.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                withAnimation(.easeInOut(duration: 0.35)) { proxy.scrollTo("feedback", anchor: .bottom) }
            }
        }
        }
        .safeAreaInset(edge: .bottom) { bottomBar }
    }

    // MARK: Rows

    private enum RowState { case idle, selected, correct, wrong, dimmed }

    private func rowState(_ i: Int) -> RowState {
        if revealed {
            if i == correctIndex { return .correct }
            if i == selected { return .wrong }
            return .dimmed
        }
        return selected == i ? .selected : .idle
    }

    private func accent(_ s: RowState) -> Color {
        switch s {
        case .correct: theme.success
        case .wrong: theme.danger
        case .selected: theme.accent
        default: theme.textSecondary
        }
    }

    private func choiceRow(_ i: Int) -> some View {
        let state = rowState(i)
        let c = accent(state)
        let corner = theme.id == .editorial ? 14 : theme.radius
        let shape = RoundedRectangle(cornerRadius: corner, style: .continuous)
        let fill: Color = switch state {
        case .correct: theme.successBg
        case .wrong: theme.dangerBg
        case .selected: theme.accent.opacity(0.1)
        default: theme.surface
        }
        let stroke: Color = (state == .idle || state == .dimmed) ? theme.border : c

        return HStack(alignment: .top, spacing: Space.m) {
            Text(theme.id == .terminal ? (state == .idle || state == .dimmed ? " " : ">") : String(UnicodeScalar(65 + i)!))
                .font(theme.font(.mono))
                .frame(width: 24, height: 24)
                .background(theme.id == .terminal ? .clear : c.opacity(0.15), in: RoundedRectangle(cornerRadius: min(corner, 12)))
                .foregroundStyle(c)
            RichText(choices[i], role: .body, color: state == .dimmed ? theme.textSecondary : theme.text, spacing: 6)
                .multilineTextAlignment(.leading)
            if state == .correct {
                Image(systemName: "checkmark").font(.body.weight(.bold)).foregroundStyle(theme.success)
                    .transition(.scale(scale: 0.4).combined(with: .opacity))
            } else if state == .wrong {
                Image(systemName: "xmark").font(.body.weight(.bold)).foregroundStyle(theme.danger)
                    .transition(.scale(scale: 0.4).combined(with: .opacity))
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(fill, in: shape)
        .overlay(shape.strokeBorder(stroke, style: StrokeStyle(lineWidth: stroke == theme.border ? 1 : 1.5,
                                                               dash: theme.dashed && stroke == theme.border ? [4, 3] : [])))
        .contentShape(shape)
        .opacity(state == .dimmed ? 0.7 : 1)
        .scaleEffect(state == .correct && pulse ? 1.025 : 1)
        .modifier(ShakeEffect(travel: state == .wrong ? shake : 0))
    }

    @ViewBuilder private var bottomBar: some View {
        let show = mode == .practice ? revealed : selected != nil
        if show {
            ThemedButton(mode == .exam ? "Next" : "Continue",
                         tint: mode == .exam ? nil : (isCorrect ? theme.success : theme.warning)) {
                onDone(isCorrect)
            }
            .accessibilityIdentifier("mcqContinue")
            .actionBar()
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    private func choose(_ i: Int) {
        switch mode {
        case .exam:
            Haptics.tap()
            selected = i
        case .practice:
            selected = i
            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { revealed = true }
            if i == correctIndex {
                Haptics.hit()
                withAnimation(.spring(response: 0.22, dampingFraction: 0.5)) { pulse = true }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.22) {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) { pulse = false }
                }
            } else {
                Haptics.miss()
                withAnimation(.linear(duration: 0.4)) { shake += 1 }
            }
        }
    }
}

// MARK: - Flashcard

struct FlashcardView: View {
    let card: Flashcard
    var previews: [Rating: String] = [:]
    var onRate: (Rating) -> Void

    @Environment(\.theme) private var theme
    @State private var revealed = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if card.isOpen {
                    SectionLabel("Say it out loud, then reveal", icon: "waveform", color: theme.warning)
                }
                RichText(card.front, role: .title)
                if revealed {
                    Rectangle().fill(theme.border).frame(height: 1)
                    if card.isOpen { SectionLabel("Model answer") }
                    RichText(card.back)
                        .transition(.opacity.combined(with: .offset(y: 10)))
                    if let src = card.source { SourceLine(src) }
                }
            }
            .padding()
        }
        .safeAreaInset(edge: .bottom) {
            Group {
                if revealed {
                    HStack(spacing: 8) {
                        ForEach(Rating.allCases) { r in
                            Button {
                                Haptics.tap()
                                onRate(r)
                            } label: {
                                VStack(spacing: 2) {
                                    Text(theme.labelText(r.label))
                                    if let p = previews[r] { Text(p).font(theme.font(.label)).opacity(0.75) }
                                }
                            }
                            .buttonStyle(ThemedButtonStyle(prominent: true, tint: color(r)))
                            .accessibilityIdentifier("rate-\(r.label)")
                            .transition(.opacity.combined(with: .offset(y: 12)))
                        }
                    }
                } else {
                    ThemedButton(card.isOpen ? "Reveal answer" : "Show answer") {
                        Haptics.tap()
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) { revealed = true }
                    }
                    .accessibilityIdentifier("showAnswer")
                }
            }
            .actionBar()
        }
    }

    private func color(_ r: Rating) -> Color {
        switch r {
        case .again: theme.danger
        case .hard: theme.warning
        case .good: theme.success
        case .easy: theme.info
        }
    }
}

// MARK: - Flag sheet

struct FlagSheet: View {
    let item: StudyItem
    @Environment(\.modelContext) private var ctx
    @Environment(\.dismiss) private var dismiss
    @Environment(\.theme) private var theme
    @State private var reason: FlagReason = .wrong
    @State private var note = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    SectionLabel("What's wrong?")
                    PanelList(FlagReason.allCases) { r in
                        Button { reason = r } label: {
                            HStack {
                                Text(r.rawValue).font(theme.font(.body)).foregroundStyle(theme.text)
                                Spacer()
                                if reason == r { Image(systemName: "checkmark").foregroundStyle(theme.accent) }
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                    SectionLabel("Details (optional)")
                    TextField("", text: $note, prompt: Text("The correct answer should be…").foregroundStyle(theme.textMuted), axis: .vertical)
                        .lineLimit(3...6)
                        .font(theme.font(.body))
                        .foregroundStyle(theme.text)
                        .panel()
                    Text(item.id).font(theme.font(.label)).foregroundStyle(theme.textMuted)
                }
                .padding()
            }
            .screenBackground()
            .themedNavBar("Flag item")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Flag") {
                        ctx.insert(FlagRecord(itemID: item.id, topicID: item.topicID, reason: reason, note: note))
                        try? ctx.save()
                        Haptics.success()
                        dismiss()
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}
