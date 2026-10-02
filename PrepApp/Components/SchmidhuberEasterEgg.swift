import SwiftUI
import UIKit

// An affectionate easter egg: a pixel-art Jürgen Schmidhuber pops up from the bottom edge to point out
// that he did it first. Everything lives in this file; the rest of the app only has one-line hooks:
//
//   • TodayView: `.schmidhuberSecretTaps()` on the screen header (tap the title 5× quickly to summon him).
//   • StudySessionView.answer: `SchmidhuberCenter.shared.correctAnswer(item)` after a correct answer.
//   • SettingsView: `SchmidhuberSettingsSection()` (the toggle, the mascot unlock, the hidden "Hall of priority").
// The unlockable mascot lives in SchmidhuberMascot.swift.
//
// He's drawn in a separate pass-through UIWindow above everything, so he also shows over full-screen
// lesson / review covers without touching their view code. Only his own frame takes touches.

enum SchmidhuberKey {
    /// "Schmidhuber mode": lets him pop in after correct answers. Summoning him by hand always works.
    static let enabled = "schmidhuber.enabled"
    /// How many times he has claimed credit (shown in Settings as the "Hall of priority").
    static let claims = "schmidhuber.claims"
}

// MARK: - Quotes

enum SchmidhuberQuotes {
    /// Topics he has an opinion about. The keywords are matched as whole words against an item's
    /// topic ID, item ID, topic title and tags (case and punctuation insensitive).
    enum Subject: String, CaseIterable {
        case lstm, rnn, gan, transformer, highway, metaLearning, curiosity, worldModel

        var keywords: [String] {
            switch self {
            case .lstm: ["lstm", "lstms", "long short-term memory", "vanishing gradient", "vanishing gradients"]
            case .rnn: ["rnn", "rnns", "recurrent", "recurrent neural network", "bptt"]
            case .gan: ["gan", "gans", "generative adversarial", "adversarial", "minimax"]
            case .transformer: ["attention", "self-attention", "transformer", "transformers", "linear attention",
                                "fast weight", "fast weights", "fast-weight"]
            case .highway: ["highway", "residual", "residuals", "resnet", "resnets", "skip connection", "skip connections"]
            case .metaLearning: ["meta-learning", "metalearning", "meta learning", "learning to learn", "maml"]
            case .curiosity: ["curiosity", "intrinsic motivation", "intrinsic reward", "artificial curiosity"]
            case .worldModel: ["world model", "world models", "model-based rl", "dreamer"]
            }
        }
    }

    static let general = [
        "I published this in 1991.",
        "Have you cited me yet?",
        "Every idea is just compression progress.",
        "My 1990 tech report says hello.",
        "Credit assignment is my favourite problem. Especially the credit part.",
        "Don't worry, I'll add it to the survey.",
        "We ran this on a 1991 workstation. Slowly.",
    ]

    /// Only make sense right after a correct answer (the random trigger).
    static let afterCorrect = [
        "Correct! Now please check the references section.",
        "A fine answer. Rediscovered, but fine.",
        "Right answer. Wrong citation.",
    ]

    static let bySubject: [Subject: [String]] = [
        .lstm: ["Ah yes, LSTM. You're welcome.",
                "Vanishing gradients? Sepp's 1991 thesis. Look it up."],
        .rnn: ["Neural history compressor, 1991. Very deep learning.",
               "Recurrent nets are general computers. I mentioned this."],
        .gan: ["GANs? See: Artificial Curiosity, 1990.",
               "Two nets playing a minimax game? 1990 called."],
        .transformer: ["Transformers are just fast weight programmers (1992).",
                       "Linear attention? Fast weights, 1991.",
                       "Attention is all you need. I needed it in 1991."],
        .highway: ["Highway networks came first, ResNet.",
                   "A ResNet is a Highway net with the gates held open."],
        .metaLearning: ["Meta-learning? My 1987 diploma thesis.",
                        "Learning to learn to learn... since 1987."],
        .curiosity: ["Curiosity is just compression progress.",
                     "Artificial Curiosity, 1990. You're welcome."],
        .worldModel: ["World models? Making the world differentiable, 1990.",
                      "Planning inside a learned world model. 1990, again."],
    ]

    static var all: [String] { general + Subject.allCases.flatMap { bySubject[$0] ?? [] } }

    /// Lowercased, punctuation turned into spaces, padded so whole-word matching is `contains(" word ")`.
    static func normalize(_ s: String) -> String {
        let mapped = s.lowercased().unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
        return " " + String(mapped).split(separator: " ").joined(separator: " ") + " "
    }

    /// The subjects mentioned in some text (IDs, titles, tags...).
    static func subjects(in texts: [String]) -> [Subject] {
        let haystack = normalize(texts.joined(separator: " "))
        return Subject.allCases.filter { s in s.keywords.contains { haystack.contains(normalize($0)) } }
    }

    static func subjects(for item: StudyItem, topic: Topic?) -> [Subject] {
        subjects(in: [item.topicID, item.localID, topic?.title ?? ""] + (topic?.tags ?? []))
    }
}

// MARK: - Center (state + triggers)

@MainActor
@Observable
final class SchmidhuberCenter {
    static let shared = SchmidhuberCenter()

    private init() { SchmidhuberMascotState.debugResetIfRequested() }

    struct Appearance: Identifiable, Equatable {
        let id = UUID()
        let quote: String
    }

    /// Chance of a topic-matched appearance after a correct answer (at most once per study session).
    static let contextChance = 0.25
    /// Chance of a random appearance after any correct answer.
    static let randomChance = 0.02
    /// A study "session" for the once-per-session cap ends after this long without answers.
    static let sessionGap: TimeInterval = 20 * 60
    static let visibleSeconds: Double = 4

    private(set) var current: Appearance?
    /// How far above the screen's bottom edge he stands: clear of the tab bar when one is showing.
    private(set) var bottomInset: CGFloat = 0
    /// One-time banner shown after he's dismissed (the mascot unlock).
    private(set) var banner: String?

    @ObservationIgnored weak var content: ContentStore?
    /// Where the sprite + bubble are on screen (window coordinates); only this area takes touches.
    @ObservationIgnored var hitRect: CGRect = .zero
    @ObservationIgnored var random: () -> Double = { Double.random(in: 0..<1) }
    @ObservationIgnored private var window: SchmidhuberWindow?
    @ObservationIgnored private var hideTask: Task<Void, Never>?
    @ObservationIgnored private var lastAnswerAt: Date?
    @ObservationIgnored private(set) var contextShownThisSession = false
    @ObservationIgnored private var recentQuotes: [String] = []
    @ObservationIgnored private var pendingBanner: String?

    /// UI tests drive the app by tapping; never let him wander in uninvited there.
    private let automaticAllowed = !ProcessInfo.processInfo.arguments.contains("-uiTesting")

    var isEnabled: Bool { UserDefaults.standard.object(forKey: SchmidhuberKey.enabled) as? Bool ?? true }

    // MARK: Triggers

    /// Call after an item is answered correctly. Topic-matched: 25%, once per session. Otherwise 2% at random.
    func correctAnswer(_ item: StudyItem, now: Date = .now) {
        if let last = lastAnswerAt, now.timeIntervalSince(last) > Self.sessionGap { contextShownThisSession = false }
        lastAnswerAt = now
        guard automaticAllowed, isEnabled, current == nil else { return }

        let subjects = SchmidhuberQuotes.subjects(for: item, topic: content?.topicsByID[item.topicID])
        if let subject = subjects.randomElement(), !contextShownThisSession, random() < Self.contextChance {
            contextShownThisSession = true
            appear(quote: pickQuote(from: SchmidhuberQuotes.bySubject[subject] ?? SchmidhuberQuotes.general))
        } else if random() < Self.randomChance {
            appear(quote: pickQuote(from: SchmidhuberQuotes.general + SchmidhuberQuotes.afterCorrect))
        }
    }

    /// The secret gesture (and the Hall of priority row): always works, even with Schmidhuber mode off.
    func summon() {
        appear(quote: pickQuote(from: SchmidhuberQuotes.all))
    }

    // MARK: Showing

    func appear(quote: String) {
        hideTask?.cancel()
        guard installWindow() else { return }
        bottomInset = Self.obstructedBottomInset(scene: window?.windowScene)
        window?.isHidden = false
        var quote = quote
        if SchmidhuberMascotState.registerClaim() {
            quote = SchmidhuberMascotState.unlockLine
            pendingBanner = SchmidhuberMascotState.unlockBanner
        }
        Haptics.tap()
        // Let the window come on screen first so the insertion transition actually animates.
        DispatchQueue.main.async { [self] in
            withAnimation(Self.animation(appearing: true)) { current = Appearance(quote: quote) }
            if UIAccessibility.isVoiceOverRunning {
                UIAccessibility.post(notification: .announcement, argument: "Jürgen Schmidhuber pops up: \(quote)")
            }
        }
        let seconds = UIAccessibility.isVoiceOverRunning ? Self.visibleSeconds * 2 : Self.visibleSeconds
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self?.dismiss()
        }
    }

    func dismiss() {
        hideTask?.cancel()
        guard current != nil else { return }
        withAnimation(Self.animation(appearing: false)) { current = nil }
        hitRect = .zero
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(450))
            guard !Task.isCancelled, let self, self.current == nil else { return }
            if let text = self.pendingBanner {
                self.pendingBanner = nil
                Haptics.success()
                withAnimation(Self.animation(appearing: true)) { self.banner = text }
                if UIAccessibility.isVoiceOverRunning { UIAccessibility.post(notification: .announcement, argument: text) }
                try? await Task.sleep(for: .seconds(3.5))
                withAnimation(Self.animation(appearing: false)) { self.banner = nil }
                try? await Task.sleep(for: .milliseconds(450))
                guard !Task.isCancelled, self.current == nil else { return }
            }
            self.window?.isHidden = true
        }
    }

    /// Bottom inset that keeps him clear of the app's tab bar: the tab bar's top edge (+8pt) when the main window shows
    /// one, otherwise the bottom safe area (full-screen sessions, sheets).
    static func obstructedBottomInset(scene: UIWindowScene?) -> CGFloat {
        guard let main = scene?.windows.first(where: { !($0 is SchmidhuberWindow) && $0.isKeyWindow })
                ?? scene?.windows.first(where: { !($0 is SchmidhuberWindow) }),
              let root = main.rootViewController else { return 0 }
        let safe = main.safeAreaInsets.bottom
        var top = root
        while let presented = top.presentedViewController, !presented.isBeingDismissed { top = presented }
        guard top === root, let tabBar = tabBarController(in: root)?.tabBar,
              !tabBar.isHidden, tabBar.alpha > 0.01, tabBar.window === main else { return safe }
        let frame = tabBar.convert(tabBar.bounds, to: main)
        guard frame.minY > main.bounds.midY else { return safe }
        return max(safe, main.bounds.maxY - frame.minY + 8)
    }

    private static func tabBarController(in vc: UIViewController) -> UITabBarController? {
        if let tab = vc as? UITabBarController { return tab }
        for child in vc.children { if let found = tabBarController(in: child) { return found } }
        return nil
    }

    static func animation(appearing: Bool) -> Animation {
        if UIAccessibility.isReduceMotionEnabled { return .easeInOut(duration: 0.2) }
        return appearing ? .spring(response: 0.5, dampingFraction: 0.55) : .easeIn(duration: 0.25)
    }

    /// Rotate through quotes, avoiding the last few shown.
    private func pickQuote(from pool: [String]) -> String {
        let fresh = pool.filter { !recentQuotes.contains($0) }
        let quote = (fresh.isEmpty ? pool : fresh).randomElement() ?? "I published this in 1991."
        recentQuotes.append(quote)
        if recentQuotes.count > 5 { recentQuotes.removeFirst() }
        return quote
    }

    private func installWindow() -> Bool {
        if window != nil { return true }
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        guard let scene = scenes.first(where: { $0.activationState == .foregroundActive }) ?? scenes.first else { return false }
        let w = SchmidhuberWindow(windowScene: scene)
        w.hitRect = { [weak self] in self?.hitRect ?? .zero }
        w.windowLevel = .normal + 1
        w.backgroundColor = .clear
        let host = SchmidhuberHostingController(rootView: SchmidhuberOverlayRoot())
        host.view.backgroundColor = .clear
        w.rootViewController = host
        w.isHidden = true
        window = w
        return true
    }
}

/// A window that only takes touches inside the sprite's frame; everything else falls through to the app.
private final class SchmidhuberWindow: UIWindow {
    var hitRect: () -> CGRect = { .zero }
    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        hitRect().contains(point) ? super.hitTest(point, with: event) : nil
    }
}

private final class SchmidhuberHostingController: UIHostingController<SchmidhuberOverlayRoot> {
    override var preferredStatusBarStyle: UIStatusBarStyle { .lightContent }
}

// MARK: - Overlay

/// Root of the overlay window. Mirrors RootView's theme so he matches the app.
struct SchmidhuberOverlayRoot: View {
    @AppStorage(AppearanceKey.theme) private var themeID = ThemeID.workbench.rawValue
    @AppStorage(AppearanceKey.headline) private var headline = ""
    @AppStorage(AppearanceKey.body) private var bodyFont = ""
    @AppStorage(AppearanceKey.mono) private var mono = ""
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private let center = SchmidhuberCenter.shared

    private var theme: Theme {
        Theme.named(themeID).with(headline: FontFamily(rawValue: headline), body: FontFamily(rawValue: bodyFont),
                                  mono: FontFamily(rawValue: mono))
    }

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Color.clear
            if let appearance = center.current {
                SchmidhuberPopup(quote: appearance.quote) { center.dismiss() }
                    .id(appearance.id)
                    .onGeometryChange(for: CGRect.self) { $0.frame(in: .global) } action: { center.hitRect = $0 }
                    .padding(.bottom, center.bottomInset)
                    .transition(reduceMotion ? .opacity : .move(edge: .bottom).combined(with: .opacity))
            }
            if let banner = center.banner {
                SchmidhuberBanner(text: banner)
                    .frame(maxHeight: .infinity, alignment: .top)
                    .transition(reduceMotion ? .opacity : .move(edge: .top).combined(with: .opacity))
            }
        }
        .ignoresSafeArea(edges: .bottom)
        .environment(\.theme, theme)
        .preferredColorScheme(.dark)
    }
}

/// The sprite peeking up from the bottom edge, with a speech bubble to its left.
struct SchmidhuberPopup: View {
    let quote: String
    var onTap: () -> Void = {}

    @Environment(\.theme) private var theme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var bob = false
    @State private var bubbleShown = false

    var body: some View {
        HStack(alignment: .bottom, spacing: 0) {
            SchmidhuberBubble(text: quote)
                // Bubble sits beside his face, above the bottom bar, rather than on top of it.
                .padding(.bottom, 50)
                .scaleEffect(bubbleShown || reduceMotion ? 1 : 0.4, anchor: .bottomTrailing)
                .opacity(bubbleShown || reduceMotion ? 1 : 0)
            SchmidhuberSprite()
                // 2-frame idle: hop up by exactly one art pixel and back, no tweening.
                .offset(y: bob ? -SchmidhuberSprite.pointsPerPixel : 0)
        }
        .padding(.leading, 16)
        .padding(.trailing, 18)
        .contentShape(Rectangle())
        .onTapGesture {
            Haptics.tap()
            onTap()
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Jürgen Schmidhuber says: \(quote)")
        .accessibilityHint("Double-tap to dismiss.")
        .accessibilityAddTraits(.isButton)
        .task {
            guard !reduceMotion else { return }
            try? await Task.sleep(for: .milliseconds(180))
            withAnimation(.spring(response: 0.35, dampingFraction: 0.6)) { bubbleShown = true }
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(650))
                bob.toggle()
            }
        }
    }
}

// MARK: - Sprite

/// Pixel-art head-and-shoulders portrait (Assets: Schmidhuber.imageset), a native-resolution PNG cleaned up from
/// design/easter-egg-portrait.jpg. Drawn at a whole number of device pixels per art pixel with no smoothing.
struct SchmidhuberSprite: View {
    /// Native size of the art in pixels.
    static let artSize = CGSize(width: 98, height: 121)
    /// 1pt per art pixel = 3 device pixels on current iPhones, so every art pixel stays a crisp square.
    static let pointsPerPixel: CGFloat = 1

    var body: some View {
        Image("Schmidhuber")
            .resizable()
            .interpolation(.none)
            .antialiased(false)
            .aspectRatio(contentMode: .fit)
            .frame(width: Self.artSize.width * Self.pointsPerPixel, height: Self.artSize.height * Self.pointsPerPixel)
            .accessibilityHidden(true)
    }
}

// MARK: - Bubble

/// Speech bubble with stepped "pixel" corners and a staircase tail pointing right, toward his face.
struct SchmidhuberBubble: View {
    let text: String
    @Environment(\.theme) private var theme

    private let step: CGFloat = 3

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(text)
                .font(theme.font(.mono))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
            Text(theme.labelText("J. Schmidhuber"))
                .font(theme.font(.label))
                .foregroundStyle(theme.label)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .frame(maxWidth: 220, alignment: .leading)
        .background(PixelBubbleShape(step: step).fill(theme.surfaceRaised))
        .overlay(PixelBubbleShape(step: step).stroke(theme.accent, lineWidth: step))
        .overlay(alignment: .bottomTrailing) {
            PixelTail(step: step)
                .fill(theme.accent)
                .frame(width: step * 4, height: step * 4)
                .offset(x: step * 4 - step / 2, y: -10)
        }
        .padding(.trailing, step * 4)
    }
}

/// A rectangle whose corners are cut into a single pixel step, like an 8-bit dialogue box.
struct PixelBubbleShape: Shape {
    var step: CGFloat
    func path(in r: CGRect) -> Path {
        let s = step
        var p = Path()
        p.move(to: CGPoint(x: r.minX + 2 * s, y: r.minY))
        p.addLine(to: CGPoint(x: r.maxX - 2 * s, y: r.minY))
        p.addLine(to: CGPoint(x: r.maxX - 2 * s, y: r.minY + s))
        p.addLine(to: CGPoint(x: r.maxX - s, y: r.minY + s))
        p.addLine(to: CGPoint(x: r.maxX - s, y: r.minY + 2 * s))
        p.addLine(to: CGPoint(x: r.maxX, y: r.minY + 2 * s))
        p.addLine(to: CGPoint(x: r.maxX, y: r.maxY - 2 * s))
        p.addLine(to: CGPoint(x: r.maxX - s, y: r.maxY - 2 * s))
        p.addLine(to: CGPoint(x: r.maxX - s, y: r.maxY - s))
        p.addLine(to: CGPoint(x: r.maxX - 2 * s, y: r.maxY - s))
        p.addLine(to: CGPoint(x: r.maxX - 2 * s, y: r.maxY))
        p.addLine(to: CGPoint(x: r.minX + 2 * s, y: r.maxY))
        p.addLine(to: CGPoint(x: r.minX + 2 * s, y: r.maxY - s))
        p.addLine(to: CGPoint(x: r.minX + s, y: r.maxY - s))
        p.addLine(to: CGPoint(x: r.minX + s, y: r.maxY - 2 * s))
        p.addLine(to: CGPoint(x: r.minX, y: r.maxY - 2 * s))
        p.addLine(to: CGPoint(x: r.minX, y: r.minY + 2 * s))
        p.addLine(to: CGPoint(x: r.minX + s, y: r.minY + 2 * s))
        p.addLine(to: CGPoint(x: r.minX + s, y: r.minY + s))
        p.addLine(to: CGPoint(x: r.minX + 2 * s, y: r.minY + s))
        p.closeSubpath()
        return p
    }
}

/// A 4-step staircase triangle pointing right.
struct PixelTail: Shape {
    var step: CGFloat
    func path(in r: CGRect) -> Path {
        var p = Path()
        for i in 0..<4 {
            let h = CGFloat(4 - i) * step
            p.addRect(CGRect(x: r.minX + CGFloat(i) * step, y: r.minY + (r.height - h) / 2, width: step, height: h))
        }
        return p
    }
}

// MARK: - Hooks

extension View {
    /// Tap the leading part of this view (the screen title) 5 times quickly to summon him.
    func schmidhuberSecretTaps() -> some View { modifier(SchmidhuberSecretTaps()) }
}

private struct SchmidhuberSecretTaps: ViewModifier {
    @Environment(ContentStore.self) private var store
    @State private var taps: [Date] = []
    @State private var width: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .contentShape(Rectangle())
            .simultaneousGesture(SpatialTapGesture().onEnded { value in
                // Only the title side counts, so the streak badge / gear on the right don't.
                guard width == 0 || value.location.x < width * 0.55 else { return }
                let now = Date.now
                taps = taps.filter { now.timeIntervalSince($0) < 2 } + [now]
                if taps.count >= 5 {
                    taps = []
                    SchmidhuberCenter.shared.summon()
                }
            })
            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { width = $0 }
            // Gives the center access to topic titles/tags for the context-aware trigger.
            .onAppear { SchmidhuberCenter.shared.content = store }
    }
}

/// Settings rows: the "Schmidhuber mode" toggle, the mascot (locked/unlocked) and, once he's shown up, the "Hall of priority".
struct SchmidhuberSettingsSection: View {
    @Environment(\.theme) private var theme
    @AppStorage(SchmidhuberKey.enabled) private var enabled = true
    @AppStorage(SchmidhuberKey.claims) private var claims = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel("Fun").padding(.top, 6)
            VStack(spacing: 0) {
                Toggle(isOn: $enabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Schmidhuber mode").font(theme.font(.body)).foregroundStyle(theme.text)
                        Text("Occasional reminders of who did it first.")
                            .font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                    }
                }
                .tint(theme.accent)
                .padding(.horizontal, 14).padding(.vertical, 10)
                PanelDivider()
                SchmidhuberMascotSettingsRow()
                if claims > 0 {
                    PanelDivider()
                    Button { SchmidhuberCenter.shared.summon() } label: {
                        HStack {
                            Text("Hall of priority").font(theme.font(.body)).foregroundStyle(theme.text)
                            Spacer()
                            Text("\(claims) \(claims == 1 ? "claim" : "claims") of credit")
                                .font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                        }
                        .padding(.horizontal, 14).padding(.vertical, 12)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("hallOfPriority")
                }
            }
            .panel(padding: 0)
        }
    }
}
