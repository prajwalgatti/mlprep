import SwiftUI
import UIKit

// "Schmidhuber as mascot": unlocked once he has claimed credit 5 times (Hall of priority >= 5).
// Hooks elsewhere are one-liners:
//   • TodayView: `SchmidhuberMascotGreeting(...)` above the title.
//   • SessionSummaryView: `SchmidhuberMascotCheer(...)` under the verdict.
//   • Settings: rows live in `SchmidhuberSettingsSection`; the app icon option comes from `extraAppIcons`.
// Debug builds can force-unlock with the launch argument `-schmidhuberUnlocked YES` (or relock: `-schmidhuberResetUnlock YES`).

extension SchmidhuberKey {
    /// Set once at the 5th claim. Never cleared.
    static let unlocked = "schmidhuber.unlocked"
    /// The "Mascot" toggle; switched on at the moment of unlock.
    static let mascot = "schmidhuber.mascot"
}

enum SchmidhuberMascotState {
    static let threshold = 5
    static let unlockLine = "Fine. I'll supervise your lab now."
    static let unlockBanner = "Schmidhuber unlocked as mascot. Toggle it in Settings."
    static let appIconName = "AppIcon-Schmidhuber"

    /// DEBUG only: `-schmidhuberUnlocked YES` unlocks him (and defaults the mascot on) without touching saved state.
    static var debugForced: Bool {
        #if DEBUG
        return UserDefaults.standard.bool(forKey: "schmidhuberUnlocked")
        #else
        return false
        #endif
    }

    static func isUnlocked(_ stored: Bool) -> Bool { stored || debugForced }

    /// DEBUG only: `-schmidhuberResetUnlock YES` relocks him at launch (UI tests of the unlock flow).
    static func debugResetIfRequested() {
        #if DEBUG
        guard UserDefaults.standard.bool(forKey: "schmidhuberResetUnlock") else { return }
        UserDefaults.standard.removeObject(forKey: SchmidhuberKey.unlocked)
        UserDefaults.standard.removeObject(forKey: SchmidhuberKey.mascot)
        #endif
    }

    /// Counts one claim of credit. Returns true exactly once: on the claim that unlocks the mascot.
    @discardableResult
    static func registerClaim(defaults: UserDefaults = .standard) -> Bool {
        let claims = defaults.integer(forKey: SchmidhuberKey.claims) + 1
        defaults.set(claims, forKey: SchmidhuberKey.claims)
        guard claims >= threshold, !defaults.bool(forKey: SchmidhuberKey.unlocked) else { return false }
        defaults.set(true, forKey: SchmidhuberKey.unlocked)
        defaults.set(true, forKey: SchmidhuberKey.mascot)
        return true
    }

    /// Extra entries for Settings → App icon (only once unlocked).
    static var extraAppIcons: [(name: String?, label: String, preview: String)] {
        isUnlocked(UserDefaults.standard.bool(forKey: SchmidhuberKey.unlocked))
            ? [(appIconName, "Schmidhuber", "Preview-\(appIconName)")] : []
    }

    /// Today's line: reacts to state first, otherwise rotates daily.
    static func greeting(dueCount: Int, goalDone: Bool, streak: Int, date: Date = .now) -> String {
        if dueCount > 0 { return "\(dueCount) review\(dueCount == 1 ? "" : "s"). I did these in 1991." }
        if goalDone { return "Good. Now cite me." }
        if streak == 0 { return "Even LSTMs need a first step." }
        let daily = [
            "Day \(streak). Respectable, by 1991 standards.",
            "Learn something new. I'll claim it later.",
            "Today's plan: compression progress.",
            "Curiosity is a reward signal. Use it.",
            "Remember: credit assignment includes credit.",
            "Your streak is a recurrent net. Keep it going.",
            "Small steps. Like gradient descent, but with tea.",
        ]
        let day = Calendar.current.ordinality(of: .day, in: .era, for: date) ?? 0
        return daily[day % daily.count]
    }

    /// Session-complete line, by first-try accuracy.
    static func cheer(accuracy: Double, answered: Int) -> String {
        guard answered > 0 else { return "A whole lesson! I wrote that one too." }
        switch accuracy {
        case 0.9...: return "Excellent. Almost as fast as my 1991 net."
        case 0.7...: return "Solid work. I'll allow it."
        default: return "Mistakes are just compression not yet achieved."
        }
    }
}

/// The small pixel head (Assets: SchmidhuberHead), drawn at 2 device pixels per art pixel.
struct SchmidhuberHead: View {
    static let artSize = CGSize(width: 69, height: 80)
    var pointsPerPixel: CGFloat = 2.0 / 3.0

    var body: some View {
        Image("SchmidhuberHead")
            .resizable()
            .interpolation(.none)
            .antialiased(false)
            .aspectRatio(contentMode: .fit)
            .frame(width: Self.artSize.width * pointsPerPixel, height: Self.artSize.height * pointsPerPixel)
            .accessibilityHidden(true)
    }
}

/// Reads the unlock + toggle state; renders its content only when the mascot is on.
private struct MascotGate<Content: View>: View {
    @AppStorage(SchmidhuberKey.unlocked) private var unlocked = false
    @AppStorage(SchmidhuberKey.mascot) private var mascotOn = SchmidhuberMascotState.debugForced
    @ViewBuilder var content: Content

    var body: some View {
        if SchmidhuberMascotState.isUnlocked(unlocked) && mascotOn { content }
    }
}

/// A head with a little pixel speech line beside it. Tapping him summons the full pop-up with a random quote.
private struct MascotLine: View {
    let text: String
    @Environment(\.theme) private var theme

    var body: some View {
        Button {
            Haptics.tap()
            SchmidhuberCenter.shared.summon()
        } label: {
            HStack(alignment: .center, spacing: 6) {
                SchmidhuberHead()
                HStack(spacing: 0) {
                    PixelTail(step: 2)
                        .fill(theme.accent)
                        .frame(width: 8, height: 8)
                        .rotationEffect(.degrees(180))
                    Text(text)
                        .font(theme.font(.mono))
                        .foregroundStyle(theme.text)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 10).padding(.vertical, 7)
                        .background(PixelBubbleShape(step: 2).fill(theme.surfaceRaised))
                        .overlay(PixelBubbleShape(step: 2).stroke(theme.accent, lineWidth: 2))
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Mascot Jürgen Schmidhuber: \(text)")
        .accessibilityHint("Shows another quote.")
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("schmidhuberMascot")
    }
}

/// Today screen: the mascot and a state-aware line of the day.
struct SchmidhuberMascotGreeting: View {
    let dueCount: Int
    let goalDone: Bool
    let streak: Int

    var body: some View {
        MascotGate {
            MascotLine(text: SchmidhuberMascotState.greeting(dueCount: dueCount, goalDone: goalDone, streak: streak))
        }
    }
}

/// Session-complete screen: a congratulating line.
struct SchmidhuberMascotCheer: View {
    let accuracy: Double
    let answered: Int

    var body: some View {
        MascotGate {
            MascotLine(text: SchmidhuberMascotState.cheer(accuracy: accuracy, answered: answered))
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// One-time banner at the top of the screen (shown by the overlay window after the unlock line).
struct SchmidhuberBanner: View {
    let text: String
    @Environment(\.theme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            SchmidhuberHead(pointsPerPixel: 1.0 / 3.0)
            VStack(alignment: .leading, spacing: 2) {
                Text(theme.labelText("Unlocked")).font(theme.font(.label)).foregroundStyle(theme.label)
                Text(text).font(theme.font(.mono)).foregroundStyle(theme.text)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(PixelBubbleShape(step: 3).fill(theme.surfaceRaised))
        .overlay(PixelBubbleShape(step: 3).stroke(theme.accent, lineWidth: 3))
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .accessibilityElement(children: .combine)
    }
}

/// Settings rows for the mascot: a mysterious locked row until unlocked, then the "Mascot" toggle.
struct SchmidhuberMascotSettingsRow: View {
    @Environment(\.theme) private var theme
    @AppStorage(SchmidhuberKey.unlocked) private var unlocked = false
    @AppStorage(SchmidhuberKey.mascot) private var mascotOn = SchmidhuberMascotState.debugForced
    @AppStorage(SchmidhuberKey.claims) private var claims = 0

    var body: some View {
        if SchmidhuberMascotState.isUnlocked(unlocked) {
            Toggle(isOn: $mascotOn) {
                HStack(spacing: 10) {
                    SchmidhuberHead(pointsPerPixel: 1.0 / 3.0)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Mascot").font(theme.font(.body)).foregroundStyle(theme.text)
                        Text("He watches over Today and your finished sessions.")
                            .font(theme.font(.caption)).foregroundStyle(theme.textSecondary)
                    }
                }
            }
            .tint(theme.accent)
            .padding(.horizontal, 14).padding(.vertical, 10)
            .accessibilityIdentifier("mascotToggle")
        } else {
            HStack {
                // SF Symbol rather than the 🔒 emoji: the bundled fonts have no emoji fallback.
                Label("???", systemImage: "lock.fill").font(theme.font(.body)).foregroundStyle(theme.textSecondary)
                Spacer()
                Text("\(min(claims, SchmidhuberMascotState.threshold))/\(SchmidhuberMascotState.threshold) priority claims")
                    .font(theme.font(.label)).foregroundStyle(theme.textMuted)
            }
            .padding(.horizontal, 14).padding(.vertical, 12)
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Locked. \(min(claims, SchmidhuberMascotState.threshold)) of \(SchmidhuberMascotState.threshold) priority claims.")
            .accessibilityIdentifier("mascotLocked")
        }
    }
}
