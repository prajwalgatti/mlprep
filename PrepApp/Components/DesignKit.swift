import SwiftUI

// Shared building blocks for layout rhythm, feedback motion and reward moments.
// Everything here reads colors, fonts and shapes from `Theme`, so all three themes keep their voice.

// MARK: - Bottom action bar

private struct ActionBar: ViewModifier {
    @Environment(\.theme) private var theme

    func body(content: Content) -> some View {
        content
            .padding(.horizontal, Space.l)
            .padding(.top, Space.s)
            .padding(.bottom, Space.s)
            .frame(maxWidth: .infinity)
            .background(theme.bg.ignoresSafeArea(edges: .bottom))
            // Soft fade so scrolling content dissolves into the bar instead of being cut off.
            .background(alignment: .top) {
                LinearGradient(colors: [theme.bg.opacity(0), theme.bg], startPoint: .top, endPoint: .bottom)
                    .frame(height: Space.xl)
                    .offset(y: -Space.xl)
                    .allowsHitTesting(false)
            }
    }
}

extension View {
    /// Styles a bottom button row: gutter padding, opaque background, and a fade above it.
    func actionBar() -> some View { modifier(ActionBar()) }

    /// Extra space above a section label so sections read as groups.
    func sectionGap() -> some View { padding(.top, Space.m) }

    /// Fades and lifts a view in when it first appears. `index` staggers siblings.
    func entrance(_ index: Int = 0, enabled: Bool = true) -> some View {
        modifier(Entrance(delay: Double(index) * 0.06, enabled: enabled))
    }
}

private struct Entrance: ViewModifier {
    let delay: Double
    let enabled: Bool
    @State private var shown = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content
            .opacity(shown || !enabled ? 1 : 0)
            .offset(y: shown || !enabled || reduceMotion ? 0 : 10)
            .onAppear {
                guard enabled, !shown else { return }
                withAnimation(.spring(response: 0.45, dampingFraction: 0.85).delay(delay)) { shown = true }
            }
    }
}

// MARK: - Motion

/// Horizontal shake for a wrong answer. Animate `travel` by whole numbers.
struct ShakeEffect: GeometryEffect {
    var travel: CGFloat
    var amplitude: CGFloat = 6
    var animatableData: CGFloat {
        get { travel }
        set { travel = newValue }
    }

    func effectValue(size: CGSize) -> ProjectionTransform {
        ProjectionTransform(CGAffineTransform(translationX: amplitude * sin(travel * .pi * 3), y: 0))
    }
}

// MARK: - Icon button

/// SF Symbol button with a full 44pt tap target (the glyph itself stays small).
struct IconButton: View {
    @Environment(\.theme) private var theme
    let systemName: String
    var label: String
    var tint: Color?
    let action: () -> Void

    init(_ systemName: String, label: String, tint: Color? = nil, action: @escaping () -> Void) {
        self.systemName = systemName
        self.label = label
        self.tint = tint
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(tint ?? theme.textSecondary)
                .frame(width: Space.tap, height: Space.tap)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

// MARK: - Stat strip

struct StatItem: Identifiable {
    let value: String
    let label: String
    var tint: Color?
    var id: String { label }
}

/// A row of key numbers in one panel, separated by hairlines. Denser and calmer than a grid of tiles.
struct StatStrip: View {
    @Environment(\.theme) private var theme
    let items: [StatItem]

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(items.enumerated()), id: \.element.id) { i, item in
                if i > 0 { Rectangle().fill(theme.border).frame(width: 1).padding(.vertical, Space.xs) }
                VStack(alignment: .leading, spacing: Space.xs) {
                    Text(item.value)
                        .font(theme.font(.number))
                        .foregroundStyle(item.tint ?? theme.text)
                        .lineLimit(1).minimumScaleFactor(0.6)
                        .contentTransition(.numericText())
                    Text(theme.labelText(item.label))
                        .font(theme.font(.label))
                        .tracking(theme.labelCase == .upper ? 1 : 0)
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1).minimumScaleFactor(0.7)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, i > 0 ? Space.m : 0)
                .accessibilityElement(children: .combine)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .panel()
    }
}

// MARK: - Empty state

/// Friendly placeholder for an empty list or screen, drawn in the theme's voice.
struct EmptyState: View {
    @Environment(\.theme) private var theme
    let icon: String
    let title: String
    let message: String

    var body: some View {
        VStack(spacing: Space.m) {
            if theme.id == .terminal {
                Text("$ ls\n(empty)")
                    .font(theme.font(.mono))
                    .foregroundStyle(theme.textMuted)
                    .multilineTextAlignment(.center)
            } else {
                Image(systemName: icon)
                    .font(.system(size: 24, weight: .regular))
                    .foregroundStyle(theme.accent)
                    .frame(width: 56, height: 56)
                    .background(theme.accent.opacity(0.12),
                                in: RoundedRectangle(cornerRadius: theme.radius >= 12 ? 28 : theme.radius + 4, style: .continuous))
            }
            VStack(spacing: Space.xs + 2) {
                Text(title).font(theme.font(.title)).foregroundStyle(theme.text)
                Text(message).font(theme.font(.callout)).foregroundStyle(theme.textSecondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Space.xl)
        .padding(.horizontal, Space.s)
        .panel()
    }
}

// MARK: - Score dial

/// Big animated score for the end of a session: a ring (or ASCII bar) that fills while the number counts up.
struct ScoreDial: View {
    @Environment(\.theme) private var theme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let value: Double
    var tint: Color
    var caption: String

    @State private var shown = 0.0

    var body: some View {
        Group {
            if theme.progress == .ascii {
                VStack(alignment: .leading, spacing: Space.s) {
                    Text(percent)
                        .font(theme.font(.hero))
                        .foregroundStyle(tint)
                        .contentTransition(.numericText(value: shown))
                    Text(AsciiBar.make(shown, width: 20))
                        .font(theme.font(.mono)).foregroundStyle(tint)
                        .lineLimit(1).minimumScaleFactor(0.6)
                    Text(caption).font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ZStack {
                    Circle().stroke(tint.opacity(0.15), lineWidth: 10)
                    Circle()
                        .trim(from: 0, to: shown)
                        .stroke(tint, style: StrokeStyle(lineWidth: 10, lineCap: theme.radius == 0 ? .butt : .round))
                        .rotationEffect(.degrees(-90))
                    VStack(spacing: 0) {
                        Text(percent)
                            .font(theme.font(.hero))
                            .foregroundStyle(theme.text)
                            .contentTransition(.numericText(value: shown))
                            .lineLimit(1).minimumScaleFactor(0.5)
                        Text(theme.labelText(caption))
                            .font(theme.font(.label))
                            .tracking(theme.labelCase == .upper ? 1.2 : 0)
                            .foregroundStyle(theme.textSecondary)
                    }
                    .padding(Space.l)
                }
                .frame(width: 176, height: 176)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(percent) \(caption)")
        .onAppear {
            if reduceMotion { shown = value; return }
            withAnimation(.easeOut(duration: 0.9).delay(0.15)) { shown = value }
        }
    }

    private var percent: String { "\(Int((shown * 100).rounded()))%" }
}
