import SwiftUI

// MARK: - Containers

struct PanelModifier: ViewModifier {
    @Environment(\.theme) private var theme
    var padding: CGFloat
    var raised: Bool
    var stroke: Color?

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: theme.radius, style: .continuous)
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(raised ? theme.surfaceRaised : theme.surface, in: shape)
            .overlay(shape.strokeBorder(stroke ?? theme.border, style: StrokeStyle(lineWidth: 1, dash: theme.dashed && stroke == nil ? [4, 3] : [])))
    }
}

extension View {
    func panel(padding: CGFloat = 14, raised: Bool = false, stroke: Color? = nil) -> some View {
        modifier(PanelModifier(padding: padding, raised: raised, stroke: stroke))
    }

    /// Full-screen themed background.
    func screenBackground() -> some View {
        modifier(ScreenBackground())
    }

    /// Inline navigation bar styled for the theme (for pushed screens).
    func themedNavBar(_ title: String) -> some View {
        modifier(ThemedNavBar(title: title))
    }
}

private struct ScreenBackground: ViewModifier {
    @Environment(\.theme) private var theme
    func body(content: Content) -> some View {
        content
            .background(theme.bg.ignoresSafeArea())
            // Opaque strip behind the status bar so scrolled content doesn't collide with the clock.
            .safeAreaInset(edge: .top, spacing: 0) {
                Color.clear.frame(height: 0)
                    .background(theme.bg.ignoresSafeArea(edges: .top))
                    .allowsHitTesting(false)
            }
    }
}

private struct ThemedNavBar: ViewModifier {
    @Environment(\.theme) private var theme
    let title: String
    func body(content: Content) -> some View {
        content
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Text(theme.labelText(title)).font(theme.font(theme.labelUsesMono ? .label : .heading))
                        .tracking(theme.labelCase == .upper ? 1.4 : 0)
                        .foregroundStyle(theme.text)
                        .lineLimit(1)
                }
            }
            .toolbarBackground(theme.bg, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
    }
}

/// A list of rows inside one panel, separated by hairlines.
struct PanelList<Data: RandomAccessCollection, Row: View>: View {
    @Environment(\.theme) private var theme
    let data: Data
    @ViewBuilder let row: (Data.Element) -> Row

    init(_ data: Data, @ViewBuilder row: @escaping (Data.Element) -> Row) {
        self.data = data
        self.row = row
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(data.enumerated()), id: \.offset) { i, element in
                if i > 0 { PanelDivider() }
                row(element).padding(.horizontal, 14).padding(.vertical, 11)
            }
        }
        .panel(padding: 0)
    }
}

struct PanelDivider: View {
    @Environment(\.theme) private var theme
    var body: some View {
        Rectangle().fill(theme.border).frame(height: 1).padding(.leading, 14)
    }
}

// MARK: - Text

struct SectionLabel: View {
    @Environment(\.theme) private var theme
    let text: String
    var icon: String?
    var color: Color?

    init(_ text: String, icon: String? = nil, color: Color? = nil) {
        self.text = text
        self.icon = icon
        self.color = color
    }

    var body: some View {
        HStack(spacing: 6) {
            if let icon, !theme.labelUsesMono { Image(systemName: icon).font(.caption.weight(.semibold)) }
            Text(theme.labelText(text))
                .font(theme.font(.label))
                .tracking(theme.labelCase == .upper ? 1.4 : 0)
        }
        .foregroundStyle(color ?? theme.label)
    }
}

/// Large themed screen title for root tabs (replaces the system large title).
struct ScreenHeader<Trailing: View>: View {
    @Environment(\.theme) private var theme
    let title: String
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            theme.titleText(title)
                .font(theme.font(.display))
                .foregroundStyle(theme.text)
            Spacer()
            trailing
        }
        .padding(.top, 8)
    }
}

extension ScreenHeader where Trailing == EmptyView {
    init(_ title: String) {
        self.title = title
        self.trailing = EmptyView()
    }
}

struct LevelBadge: View {
    @Environment(\.theme) private var theme
    let level: TopicLevel
    var body: some View {
        let c = level.color(theme)
        Text(theme.labelText(level.label))
            .font(theme.font(.label))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(c.opacity(0.14), in: RoundedRectangle(cornerRadius: min(theme.radius, 8)))
            .foregroundStyle(c)
    }
}

// MARK: - Buttons

struct ThemedButtonStyle: ButtonStyle {
    var prominent = true
    var tint: Color?
    var compact = false

    func makeBody(configuration: Configuration) -> some View {
        ThemedButtonBody(configuration: configuration, prominent: prominent, tint: tint, compact: compact)
    }
}

private struct ThemedButtonBody: View {
    @Environment(\.theme) private var theme
    @Environment(\.isEnabled) private var isEnabled
    let configuration: ButtonStyleConfiguration
    let prominent: Bool
    let tint: Color?
    let compact: Bool

    var body: some View {
        let c = tint ?? theme.accent
        let shape = RoundedRectangle(cornerRadius: theme.buttonRadius, style: .continuous)
        let solid = prominent && theme.buttonFill == .solid
        configuration.label
            .font(theme.font(.button))
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .frame(maxWidth: compact ? nil : .infinity)
            .padding(.vertical, compact ? 8 : 13)
            .padding(.horizontal, compact ? 12 : 14)
            .foregroundStyle(solid ? theme.onAccent : (prominent ? c : theme.text))
            .background(solid ? c : (prominent ? c.opacity(0.1) : theme.surfaceRaised), in: shape)
            .overlay(shape.strokeBorder(solid ? .clear : (prominent ? c : theme.border), lineWidth: 1))
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.4)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.snappy(duration: 0.15), value: configuration.isPressed)
    }
}

/// A full-width themed button whose label follows the theme's voice ("▸ START", "[ start ]", "Start").
struct ThemedButton: View {
    @Environment(\.theme) private var theme
    let title: String
    var prominent = true
    var tint: Color?
    var compact = false
    let action: () -> Void

    init(_ title: String, prominent: Bool = true, tint: Color? = nil, compact: Bool = false, action: @escaping () -> Void) {
        self.title = title
        self.prominent = prominent
        self.tint = tint
        self.compact = compact
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(theme.buttonText(title, prominent: prominent))
        }
        .buttonStyle(ThemedButtonStyle(prominent: prominent, tint: tint, compact: compact))
    }
}

// MARK: - Controls

struct ThemedSegmented<T: Hashable>: View {
    @Environment(\.theme) private var theme
    @Binding var selection: T
    let options: [(value: T, label: String)]

    var body: some View {
        let inner = max(theme.buttonRadius == 99 ? 99 : theme.radius - 2, 0)
        HStack(spacing: 4) {
            ForEach(options.indices, id: \.self) { i in
                let selected = options[i].value == selection
                Button {
                    Haptics.tap()
                    withAnimation(.snappy) { selection = options[i].value }
                } label: {
                    Text(theme.labelText(options[i].label))
                        .font(theme.font(theme.labelUsesMono ? .label : .callout))
                        .tracking(theme.labelCase == .upper ? 1.2 : 0)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .foregroundStyle(selected ? (theme.buttonFill == .solid ? theme.onAccent : theme.accent) : theme.textSecondary)
                        .background(selected ? (theme.buttonFill == .solid ? theme.accent : theme.accent.opacity(0.12)) : .clear,
                                    in: RoundedRectangle(cornerRadius: inner, style: .continuous))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .panel(padding: 0)
    }
}

struct ThemedSearchField: View {
    @Environment(\.theme) private var theme
    @Binding var text: String
    var prompt: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(theme.textMuted)
            TextField("", text: $text, prompt: Text(prompt).foregroundStyle(theme.textMuted))
                .font(theme.font(.callout))
                .foregroundStyle(theme.text)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            if !text.isEmpty {
                Button { text = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(theme.textMuted) }
            }
        }
        .panel(padding: 11)
    }
}

// MARK: - Progress

struct MasteryRing: View {
    @Environment(\.theme) private var theme
    var value: Double
    var tint: Color?
    var lineWidth: CGFloat = 4
    var showsLabel = false

    var body: some View {
        let c = tint ?? theme.accent
        ZStack {
            Circle().stroke(c.opacity(0.18), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: min(value, 1))
                .stroke(c, style: StrokeStyle(lineWidth: lineWidth, lineCap: theme.radius == 0 ? .butt : .round))
                .rotationEffect(.degrees(-90))
            if showsLabel {
                Text("\(Int((value * 100).rounded()))%")
                    .font(theme.font(.label))
                    .foregroundStyle(theme.text)
                    .lineLimit(1).minimumScaleFactor(0.6)
                    .padding(lineWidth + 2)
            }
        }
        .animation(.easeOut, value: value)
    }
}

struct ThinBar: View {
    @Environment(\.theme) private var theme
    var value: Double
    var tint: Color?
    var height: CGFloat = 4

    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(theme.border)
                Capsule().fill(tint ?? theme.accent).frame(width: g.size.width * min(max(value, 0), 1))
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: theme.radius == 0 ? 0 : height / 2))
    }
}

enum AsciiBar {
    static func make(_ value: Double, width: Int = 14) -> String {
        let filled = Int((min(max(value, 0), 1) * Double(width)).rounded())
        return "[" + String(repeating: "#", count: filled) + String(repeating: ".", count: width - filled) + "]"
    }
}

/// Progress towards a goal, drawn in the theme's style (bar / ASCII / ring).
struct GoalMeter: View {
    @Environment(\.theme) private var theme
    let count: Int
    let goal: Int

    var body: some View {
        let v = Double(count) / Double(max(goal, 1))
        switch theme.progress {
        case .bar:
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("\(count) / \(goal)")
                    Spacer()
                    Text(count >= goal ? theme.labelText("✓ Done") : "\(Int(v * 100))%")
                }
                .font(theme.font(.label))
                .foregroundStyle(theme.textSecondary)
                ThinBar(value: v)
            }
        case .ascii:
            Text("\(AsciiBar.make(v)) \(count)/\(goal)")
                .font(theme.font(.mono))
                .foregroundStyle(theme.accent)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        case .ring:
            HStack(spacing: 12) {
                MasteryRing(value: v, lineWidth: 6).frame(width: 44, height: 44)
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(count) of \(goal)").font(theme.font(.bodyStrong)).foregroundStyle(theme.text)
                    Text(count >= goal ? "Goal reached" : "\(goal - count) to go today")
                        .font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                }
            }
        }
    }
}

/// Small citation line under a card: where the material comes from.
struct SourceLine: View {
    @Environment(\.theme) private var theme
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: "book.closed").font(.caption2)
            Text(text).font(theme.font(.caption)).multilineTextAlignment(.leading)
        }
        .foregroundStyle(theme.textMuted)
        .accessibilityLabel("Source: \(text)")
    }
}
