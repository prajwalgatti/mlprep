import CoreText
import SwiftUI
import UIKit

// MARK: - Fonts

/// A font family the user can pick. Bundled fonts live in PrepApp/Fonts (all SIL OFL).
enum FontFamily: String, CaseIterable, Identifiable {
    case system, newYork, rounded, sfMono
    case instrumentSerif, fraunces, spaceGrotesk, inter, jetbrainsMono, plexMono

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .system: "SF Pro"
        case .newYork: "New York"
        case .rounded: "SF Rounded"
        case .sfMono: "SF Mono"
        case .instrumentSerif: "Instrument Serif"
        case .fraunces: "Fraunces"
        case .spaceGrotesk: "Space Grotesk"
        case .inter: "Inter"
        case .jetbrainsMono: "JetBrains Mono"
        case .plexMono: "IBM Plex Mono"
        }
    }

    var isMono: Bool { self == .sfMono || self == .jetbrainsMono || self == .plexMono }
    /// Instrument Serif only has a regular weight, so it's not offered for body text.
    var isDisplayOnly: Bool { self == .instrumentSerif }
    static var headlineChoices: [FontFamily] { allCases }
    static var bodyChoices: [FontFamily] { allCases.filter { !$0.isDisplayOnly } }
    static var monoChoices: [FontFamily] { allCases.filter(\.isMono) }

    /// Instrument Serif has a small x-height; bump it so it sits level with the others.
    private var sizeScale: CGFloat { self == .instrumentSerif ? 1.18 : 1 }

    /// Weight used for big headlines in this family.
    var displayWeight: UIFont.Weight {
        switch self {
        case .instrumentSerif: .regular
        case .fraunces, .newYork: .semibold
        case .jetbrainsMono, .plexMono, .sfMono: .medium
        default: .bold
        }
    }

    func uiFont(size rawSize: CGFloat, weight: UIFont.Weight) -> UIFont {
        let size = rawSize * sizeScale
        switch self {
        case .system:
            return .systemFont(ofSize: size, weight: weight)
        case .newYork, .rounded:
            let base = UIFont.systemFont(ofSize: size, weight: weight)
            let design: UIFontDescriptor.SystemDesign = self == .newYork ? .serif : .rounded
            return base.fontDescriptor.withDesign(design).map { UIFont(descriptor: $0, size: size) } ?? base
        case .sfMono:
            return .monospacedSystemFont(ofSize: size, weight: weight)
        case .instrumentSerif:
            return UIFont(name: "InstrumentSerif-Regular", size: size) ?? .systemFont(ofSize: size, weight: weight)
        case .plexMono:
            let name = weight >= .semibold ? "IBMPlexMono-SemiBold" : weight >= .medium ? "IBMPlexMono-Medium" : "IBMPlexMono-Regular"
            return UIFont(name: name, size: size) ?? .monospacedSystemFont(ofSize: size, weight: weight)
        case .fraunces:
            return Self.variable("Fraunces-9ptBlack", size: size, axes: [
                Self.wghtTag: Self.wght(weight), Self.opszTag: min(max(size, 9), 144), Self.softTag: 0, Self.wonkTag: 0,
            ]) ?? .systemFont(ofSize: size, weight: weight)
        case .inter:
            return Self.variable("Inter-Regular", size: size, axes: [
                Self.wghtTag: Self.wght(weight), Self.opszTag: min(max(size, 14), 32),
            ]) ?? .systemFont(ofSize: size, weight: weight)
        case .spaceGrotesk:
            return Self.variable("SpaceGrotesk-Light", size: size, axes: [Self.wghtTag: min(max(Self.wght(weight), 300), 700)])
                ?? .systemFont(ofSize: size, weight: weight)
        case .jetbrainsMono:
            return Self.variable("JetBrainsMono-Regular", size: size, axes: [Self.wghtTag: min(Self.wght(weight), 800)])
                ?? .monospacedSystemFont(ofSize: size, weight: weight)
        }
    }

    // Variable-font axis tags (four-char codes).
    private static let wghtTag = 0x7767_6874, opszTag = 0x6F70_737A, softTag = 0x534F_4654, wonkTag = 0x574F_4E4B

    private static func wght(_ w: UIFont.Weight) -> CGFloat {
        switch w {
        case .ultraLight: 200
        case .thin: 100
        case .light: 300
        case .medium: 500
        case .semibold: 600
        case .bold: 700
        case .heavy: 800
        case .black: 900
        default: 400
        }
    }

    private static func variable(_ postScriptName: String, size: CGFloat, axes: [Int: CGFloat]) -> UIFont? {
        guard let base = UIFont(name: postScriptName, size: size) else { return nil }
        let key = UIFontDescriptor.AttributeName(rawValue: kCTFontVariationAttribute as String)
        let variation = Dictionary(uniqueKeysWithValues: axes.map { (NSNumber(value: $0.key), NSNumber(value: Double($0.value))) })
        return UIFont(descriptor: base.fontDescriptor.addingAttributes([key: variation]), size: size)
    }

    /// Register every bundled font file once at launch (no Info.plist entry needed).
    static func registerBundledFonts() {
        let urls = (Bundle.main.urls(forResourcesWithExtension: "ttf", subdirectory: nil) ?? [])
            + (Bundle.main.urls(forResourcesWithExtension: "otf", subdirectory: nil) ?? [])
        for url in urls { CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil) }
    }
}

// MARK: - Theme

enum ThemeID: String, CaseIterable, Identifiable {
    case workbench, terminal, editorial
    var id: String { rawValue }
}

/// Spacing scale. Use these instead of ad-hoc numbers so screens share one rhythm.
enum Space {
    /// 4: icon-to-text, tight stacks.
    static let xs: CGFloat = 4
    /// 8: rows inside a card.
    static let s: CGFloat = 8
    /// 12: between cards in a stack; card inner gaps.
    static let m: CGFloat = 12
    /// 16: screen gutter and card padding.
    static let l: CGFloat = 16
    /// 24: before a new section label.
    static let xl: CGFloat = 24
    /// 32: around hero moments.
    static let xxl: CGFloat = 32
    /// Minimum tap target (Apple HIG).
    static let tap: CGFloat = 44
}

enum TextRole {
    case hero, display, title, heading, body, bodyStrong, callout, caption, label, mono, button, number

    var size: CGFloat {
        switch self {
        case .hero: 56
        case .display: 34
        case .title: 22
        case .heading, .body, .bodyStrong: 17
        case .callout: 15
        case .caption: 12
        case .label: 11
        case .mono: 13
        case .button: 15
        case .number: 28
        }
    }

    var textStyle: UIFont.TextStyle {
        switch self {
        case .hero, .display: .largeTitle
        case .title, .number: .title2
        case .heading: .headline
        case .body, .bodyStrong: .body
        case .callout: .callout
        case .caption, .label: .caption1
        case .mono: .footnote
        case .button: .subheadline
        }
    }
}

struct Theme: Equatable {
    enum LabelCase { case upper, lower, sentence }
    enum ButtonFill { case outline, solid }
    enum Progress { case bar, ascii, ring }

    let id: ThemeID
    let name: String
    let tagline: String

    // Colors
    let bg, surface, surfaceRaised, border: Color
    let text, textSecondary, textMuted: Color
    let accent, onAccent, label: Color
    let success, successBg, danger, dangerBg, warning, info: Color
    let codeBg: Color
    let codeKeyword, codeString, codeNumber, codeComment, codeLib: Color
    private let areaColors: [String: Color]

    // Shape
    let radius: CGFloat
    let buttonRadius: CGFloat
    let dashed: Bool

    // Voice
    let labelCase: LabelCase
    let labelUsesMono: Bool
    let buttonFill: ButtonFill
    let buttonUsesMono: Bool
    let progress: Progress

    // Typography (after user overrides)
    var headlineFont: FontFamily
    var bodyFont: FontFamily
    var monoFont: FontFamily

    func with(headline: FontFamily?, body: FontFamily?, mono: FontFamily?) -> Theme {
        var t = self
        if let headline { t.headlineFont = headline }
        if let body { t.bodyFont = body }
        if let mono { t.monoFont = mono }
        return t
    }

    // MARK: Fonts

    func family(for role: TextRole) -> FontFamily {
        switch role {
        case .hero, .display, .title, .number: headlineFont
        case .mono: monoFont
        case .label: labelUsesMono ? monoFont : bodyFont
        case .button: buttonUsesMono ? monoFont : bodyFont
        default: bodyFont
        }
    }

    func uiFont(_ role: TextRole) -> UIFont {
        let family = family(for: role)
        let weight: UIFont.Weight = switch role {
        case .hero, .display, .title, .number: family.displayWeight
        case .heading, .bodyStrong, .button: family.isMono ? .medium : .semibold
        case .label: .medium
        default: .regular
        }
        let base = family.uiFont(size: role.size, weight: weight)
        return UIFontMetrics(forTextStyle: role.textStyle).scaledFont(for: base)
    }

    func font(_ role: TextRole) -> Font { Font(uiFont(role)) }

    // MARK: Voice

    func labelText(_ s: String) -> String {
        switch labelCase {
        case .upper: s.uppercased()
        case .lower: s.lowercased()
        case .sentence: s
        }
    }

    func buttonText(_ s: String, prominent: Bool) -> String {
        switch id {
        case .workbench: prominent ? "▸ " + s.uppercased() : s.uppercased()
        case .terminal: prominent ? "[ \(s.lowercased()) ]" : s.lowercased()
        case .editorial: s
        }
    }

    /// Screen titles get a little personality per theme.
    func titleText(_ s: String) -> Text {
        switch id {
        case .workbench: Text(s) + Text(".").foregroundColor(label)
        case .terminal: Text("~/" + s.lowercased().replacingOccurrences(of: " ", with: "-")) + Text("_").foregroundColor(accent)
        case .editorial: Text(s)
        }
    }

    func tint(for area: Area?) -> Color {
        guard let area else { return accent }
        return areaColors[area.color] ?? accent
    }

    static func == (a: Theme, b: Theme) -> Bool {
        a.id == b.id && a.headlineFont == b.headlineFont && a.bodyFont == b.bodyFont && a.monoFont == b.monoFont
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255, blue: Double(hex & 0xFF) / 255)
    }
}

extension Theme {
    static func named(_ raw: String) -> Theme {
        switch ThemeID(rawValue: raw) ?? .workbench {
        case .workbench: .workbench
        case .terminal: .terminal
        case .editorial: .editorial
        }
    }

    static let all: [Theme] = [.workbench, .terminal, .editorial]

    /// Pro-tool look: hairline panels, amber mono labels, serif headlines.
    static let workbench = Theme(
        id: .workbench, name: "Workbench", tagline: "Hairline panels, amber labels, serif headlines",
        bg: Color(hex: 0x0E1014), surface: Color(hex: 0x151821), surfaceRaised: Color(hex: 0x1B1F2A), border: Color(hex: 0x262B36),
        text: Color(hex: 0xE6E8EE), textSecondary: Color(hex: 0x9AA0AE), textMuted: Color(hex: 0x5F6573),
        accent: Color(hex: 0x8C9BFF), onAccent: Color(hex: 0x0E1014), label: Color(hex: 0xE8A23A),
        success: Color(hex: 0x3FB97A), successBg: Color(hex: 0x12251C), danger: Color(hex: 0xF0616D), dangerBg: Color(hex: 0x2A1418),
        warning: Color(hex: 0xE8A23A), info: Color(hex: 0x5FB4FF),
        codeBg: Color(hex: 0x10131A),
        codeKeyword: Color(hex: 0xC792EA), codeString: Color(hex: 0xC3E88D), codeNumber: Color(hex: 0xF78C6C),
        codeComment: Color(hex: 0x5F6573), codeLib: Color(hex: 0x82AAFF),
        areaColors: [
            "blue": Color(hex: 0x6FA8FF), "teal": Color(hex: 0x4FC1B0), "purple": Color(hex: 0xA98BFF), "pink": Color(hex: 0xF07AAE),
            "orange": Color(hex: 0xF59E4C), "green": Color(hex: 0x5CC98A), "indigo": Color(hex: 0x8C9BFF), "red": Color(hex: 0xF0616D),
            "yellow": Color(hex: 0xE8C547), "gray": Color(hex: 0x9AA0AE),
        ],
        radius: 6, buttonRadius: 4, dashed: false,
        labelCase: .upper, labelUsesMono: true, buttonFill: .outline, buttonUsesMono: true, progress: .bar,
        headlineFont: .instrumentSerif, bodyFont: .inter, monoFont: .jetbrainsMono)

    /// All-mono, lime on black, ASCII bars.
    static let terminal = Theme(
        id: .terminal, name: "Terminal", tagline: "All mono, lime on black, ASCII bars",
        bg: Color(hex: 0x0A0B0A), surface: Color(hex: 0x0F110E), surfaceRaised: Color(hex: 0x151913), border: Color(hex: 0x2E3529),
        text: Color(hex: 0xD5DECB), textSecondary: Color(hex: 0x8C9683), textMuted: Color(hex: 0x5A6253),
        accent: Color(hex: 0xB8F15A), onAccent: Color(hex: 0x0A0B0A), label: Color(hex: 0x6E7666),
        success: Color(hex: 0xB8F15A), successBg: Color(hex: 0x1A2410), danger: Color(hex: 0xFF6B5B), dangerBg: Color(hex: 0x2A1210),
        warning: Color(hex: 0xF5C451), info: Color(hex: 0x6BC7FF),
        codeBg: Color(hex: 0x0D0F0C),
        codeKeyword: Color(hex: 0xB8F15A), codeString: Color(hex: 0xF5C451), codeNumber: Color(hex: 0x6BC7FF),
        codeComment: Color(hex: 0x5A6253), codeLib: Color(hex: 0x9FE3C4),
        areaColors: [:],
        radius: 0, buttonRadius: 0, dashed: true,
        labelCase: .lower, labelUsesMono: true, buttonFill: .solid, buttonUsesMono: true, progress: .ascii,
        headlineFont: .plexMono, bodyFont: .plexMono, monoFont: .plexMono)

    /// Warm dark, rounded cards, terracotta accent.
    static let editorial = Theme(
        id: .editorial, name: "Night editorial", tagline: "Warm dark, rounded cards, terracotta accent",
        bg: Color(hex: 0x16130F), surface: Color(hex: 0x211C16), surfaceRaised: Color(hex: 0x2A241D), border: Color(hex: 0x2A241D),
        text: Color(hex: 0xEFE6D8), textSecondary: Color(hex: 0xA3968A), textMuted: Color(hex: 0x6F6559),
        accent: Color(hex: 0xE0714F), onAccent: Color(hex: 0x1E120C), label: Color(hex: 0xA3968A),
        success: Color(hex: 0x8FC48A), successBg: Color(hex: 0x1E2A1E), danger: Color(hex: 0xE86A5A), dangerBg: Color(hex: 0x2E1915),
        warning: Color(hex: 0xE8B04F), info: Color(hex: 0x8DB3D6),
        codeBg: Color(hex: 0x1B1712),
        codeKeyword: Color(hex: 0xE0714F), codeString: Color(hex: 0x8FC48A), codeNumber: Color(hex: 0xE8B04F),
        codeComment: Color(hex: 0x6F6559), codeLib: Color(hex: 0x8DB3D6),
        areaColors: [
            "blue": Color(hex: 0x8DB3D6), "teal": Color(hex: 0x7DBFB2), "purple": Color(hex: 0xBE9CC7), "pink": Color(hex: 0xE59AAF),
            "orange": Color(hex: 0xE0714F), "green": Color(hex: 0x8FC48A), "indigo": Color(hex: 0xA3A6DB), "red": Color(hex: 0xE86A5A),
            "yellow": Color(hex: 0xE8B04F), "gray": Color(hex: 0xA3968A),
        ],
        radius: 18, buttonRadius: 99, dashed: false,
        labelCase: .sentence, labelUsesMono: false, buttonFill: .solid, buttonUsesMono: false, progress: .ring,
        headlineFont: .fraunces, bodyFont: .spaceGrotesk, monoFont: .jetbrainsMono)
}

private struct ThemeKey: EnvironmentKey {
    static let defaultValue = Theme.workbench
}

extension EnvironmentValues {
    var theme: Theme {
        get { self[ThemeKey.self] }
        set { self[ThemeKey.self] = newValue }
    }
}

extension TopicLevel {
    var label: String { rawValue.capitalized }
    func color(_ theme: Theme) -> Color {
        switch self {
        case .core: theme.success
        case .intermediate: theme.warning
        case .advanced: theme.danger
        }
    }
}

enum Haptics {
    static func success() { UINotificationFeedbackGenerator().notificationOccurred(.success) }
    static func error() { UINotificationFeedbackGenerator().notificationOccurred(.error) }
    static func tap() { UIImpactFeedbackGenerator(style: .light).impactOccurred() }
    /// A gentle bump for a wrong answer: noticeable, not scolding.
    static func miss() { UIImpactFeedbackGenerator(style: .soft).impactOccurred(intensity: 0.9) }
    /// Two-beat "nice" for a correct answer.
    static func hit() {
        UIImpactFeedbackGenerator(style: .rigid).impactOccurred(intensity: 0.7)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.09) {
            UIImpactFeedbackGenerator(style: .light).impactOccurred(intensity: 0.9)
        }
    }
    /// End-of-session reward.
    static func celebrate() {
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }
}
