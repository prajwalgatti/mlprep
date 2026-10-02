import SwiftData
import SwiftUI

/// UserDefaults keys for appearance, shared with RootView.
enum AppearanceKey {
    static let theme = "theme"
    static let headline = "font.headline"
    static let body = "font.body"
    static let mono = "font.mono"
}

struct SettingsView: View {
    @Environment(ContentStore.self) private var content
    @Environment(\.modelContext) private var ctx
    @Environment(\.theme) private var theme
    @AppStorage("dailyGoal") private var dailyGoal = 20
    @AppStorage("sessionSize") private var sessionSize = 20
    @AppStorage(AppearanceKey.theme) private var themeID = ThemeID.workbench.rawValue
    @AppStorage(AppearanceKey.headline) private var headline = ""
    @AppStorage(AppearanceKey.body) private var bodyFont = ""
    @AppStorage(AppearanceKey.mono) private var mono = ""
    @State private var confirmReset = false

    private var base: Theme { Theme.named(themeID) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                SectionLabel("Theme")
                HStack(spacing: 10) {
                    ForEach(Theme.all, id: \.id) { t in
                        ThemeCard(theme: t, selected: t.id.rawValue == themeID) {
                            Haptics.tap()
                            withAnimation(.easeInOut(duration: 0.25)) { themeID = t.id.rawValue }
                        }
                    }
                }

                SectionLabel("App icon").padding(.top, 6)
                AppIconPicker()

                SectionLabel("Fonts").padding(.top, 6)
                VStack(spacing: 0) {
                    fontRow("Headlines", selection: $headline, choices: FontFamily.headlineChoices, defaultFamily: base.headlineFont, role: .title)
                    PanelDivider()
                    fontRow("Body", selection: $bodyFont, choices: FontFamily.bodyChoices, defaultFamily: base.bodyFont, role: .body)
                    PanelDivider()
                    fontRow("Code and labels", selection: $mono, choices: FontFamily.monoChoices, defaultFamily: base.monoFont, role: .mono)
                }
                .panel(padding: 0)
                if !(headline.isEmpty && bodyFont.isEmpty && mono.isEmpty) {
                    ThemedButton("Use theme fonts", prominent: false) { headline = ""; bodyFont = ""; mono = "" }
                }

                SectionLabel("Preview").padding(.top, 6)
                VStack(alignment: .leading, spacing: 10) {
                    SectionLabel("Learn next", icon: "sparkles")
                    Text("Bias–Variance Tradeoff").font(theme.font(.title)).foregroundStyle(theme.text)
                    RichText("Expected error splits into $\\text{Bias}^2 + \\text{Var} + \\sigma^2$, and **capacity** trades the first two off.",
                             role: .callout)
                    CodeBlock(code: "attn = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # scale", language: "python")
                    ThemedButton("Start lesson") {}
                }
                .panel()

                SectionLabel("Study").padding(.top, 6)
                VStack(spacing: 0) {
                    stepperRow("Daily goal", value: $dailyGoal, range: 5...200, unit: "answers")
                    PanelDivider()
                    stepperRow("Review session", value: $sessionSize, range: 5...100, unit: "items")
                }
                .panel(padding: 0)

                SectionLabel("Content").padding(.top, 6)
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("\(content.topics.count) topics · \(content.totalItemCount) items").font(theme.font(.callout)).foregroundStyle(theme.text)
                        Spacer()
                        Button(theme.labelText("Reload")) {
                            content.reload()
                            Study.reconcile(content: content, in: ctx)
                        }
                        .font(theme.font(.label)).foregroundStyle(theme.accent)
                    }
                    if content.issues.isEmpty {
                        Label("All content loaded", systemImage: "checkmark.circle").font(theme.font(.caption)).foregroundStyle(theme.success)
                    } else {
                        ForEach(content.issues, id: \.self) { Text($0).font(theme.font(.label)).foregroundStyle(theme.warning) }
                    }
                    Text("Content is bundled YAML (see CONTENT_GUIDE.md). Files in Documents/ContentOverride override bundled ones.")
                        .font(theme.font(.caption)).foregroundStyle(theme.textMuted)
                }
                .panel()

                SchmidhuberSettingsSection()  // "Schmidhuber mode" toggle + Hall of priority (SchmidhuberEasterEgg.swift)

                ThemedButton("Reset all progress", prominent: false, tint: theme.danger) { confirmReset = true }
                    .padding(.top, 6)
                Text("Clears review history, schedules and exam results. Bookmarks, notes and flags are kept.")
                    .font(theme.font(.caption)).foregroundStyle(theme.textMuted)

                HStack {
                    Text("v\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "–")")
                    Spacer()
                    Text("FSRS-5 · 90% retention")
                }
                .font(theme.font(.label)).foregroundStyle(theme.textMuted)
                .padding(.top, 6)
            }
            .padding()
        }
        .screenBackground()
        .themedNavBar("Settings")
        .confirmationDialog("Reset all progress?", isPresented: $confirmReset, titleVisibility: .visible) {
            Button("Reset", role: .destructive) { Study.resetAll(in: ctx) }
        }
    }

    private func fontRow(_ title: String, selection: Binding<String>, choices: [FontFamily], defaultFamily: FontFamily,
                         role: TextRole) -> some View {
        let current = FontFamily(rawValue: selection.wrappedValue) ?? defaultFamily
        return NavigationLink {
            FontPickerView(title: title, selection: selection, choices: choices, defaultFamily: defaultFamily, role: role)
        } label: {
            HStack {
                Text(title).font(theme.font(.body)).foregroundStyle(theme.text)
                Spacer()
                Text(current.displayName)
                    .font(Font(current.uiFont(size: 15, weight: .regular)))
                    .foregroundStyle(theme.textSecondary)
                Image(systemName: "chevron.right").font(.caption).foregroundStyle(theme.textMuted)
            }
            .padding(.horizontal, 14).padding(.vertical, 12)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func stepperRow(_ title: String, value: Binding<Int>, range: ClosedRange<Int>, unit: String) -> some View {
        Stepper(value: value, in: range, step: 5) {
            HStack(spacing: 6) {
                Text(title).font(theme.font(.body)).foregroundStyle(theme.text)
                Text("\(value.wrappedValue) \(unit)").font(theme.font(.label)).foregroundStyle(theme.textSecondary)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
    }
}

/// Switches between the bundled app icons (one per theme; see tools/make_icons.py).
private struct AppIconPicker: View {
    @Environment(\.theme) private var theme
    @State private var current: String? = UIApplication.shared.alternateIconName

    private var options: [(name: String?, label: String, preview: String)] { [
        (nil, "Workbench", "Preview-AppIcon"),
        ("AppIcon-Terminal", "Terminal", "Preview-AppIcon-Terminal"),
        ("AppIcon-Editorial", "Night editorial", "Preview-AppIcon-Editorial"),
    ] + SchmidhuberMascotState.extraAppIcons }  // + "Schmidhuber" once unlocked (SchmidhuberMascot.swift)

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            ForEach(options.indices, id: \.self) { i in
                let option = options[i]
                let selected = option.name == current
                Button {
                    Haptics.tap()
                    Task { @MainActor in
                        try? await UIApplication.shared.setAlternateIconName(option.name)
                        current = UIApplication.shared.alternateIconName
                    }
                } label: {
                    VStack(spacing: 8) {
                        Image(option.preview)
                            .resizable()
                            .frame(width: 64, height: 64)
                            .clipShape(RoundedRectangle(cornerRadius: 14.5, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 14.5, style: .continuous)
                                .strokeBorder(selected ? theme.accent : .clear, lineWidth: 2.5).padding(-4))
                        Text(option.label)
                            .font(theme.font(.caption))
                            .foregroundStyle(selected ? theme.text : theme.textSecondary)
                            .lineLimit(1).minimumScaleFactor(0.8)
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(option.label) app icon")
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(.vertical, 16)
        .panel(padding: 8)
    }
}

/// Mini preview of a theme, rendered with that theme's own colors and fonts.
private struct ThemeCard: View {
    let theme: Theme
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                theme.titleText("Today")
                    .font(Font(theme.headlineFont.uiFont(size: 20, weight: theme.headlineFont.displayWeight)))
                    .foregroundStyle(theme.text)
                    .lineLimit(1).minimumScaleFactor(0.6)
                Text(theme.labelText("Reviews"))
                    .font(Font(theme.family(for: .label).uiFont(size: 10, weight: .medium)))
                    .tracking(theme.labelCase == .upper ? 1 : 0)
                    .foregroundStyle(theme.label)
                RoundedRectangle(cornerRadius: min(theme.buttonRadius, 10))
                    .fill(theme.buttonFill == .solid ? theme.accent : theme.accent.opacity(0.12))
                    .overlay(RoundedRectangle(cornerRadius: min(theme.buttonRadius, 10)).strokeBorder(theme.accent, lineWidth: theme.buttonFill == .solid ? 0 : 1))
                    .frame(height: 18)
                Spacer(minLength: 0)
                Text(theme.name)
                    .font(Font(theme.bodyFont.uiFont(size: 12, weight: .medium)))
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(1).minimumScaleFactor(0.7)
            }
            .padding(10)
            .frame(maxWidth: .infinity, minHeight: 130, alignment: .topLeading)
            .background(theme.surface, in: RoundedRectangle(cornerRadius: min(theme.radius, 12)))
            .overlay(
                RoundedRectangle(cornerRadius: min(theme.radius, 12))
                    .strokeBorder(selected ? theme.accent : theme.border,
                                  style: StrokeStyle(lineWidth: selected ? 2 : 1, dash: theme.dashed && !selected ? [4, 3] : []))
            )
            .background(theme.bg, in: RoundedRectangle(cornerRadius: min(theme.radius, 12)))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(theme.name) theme")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// Full list of fonts, each row rendered in its own typeface.
private struct FontPickerView: View {
    let title: String
    @Binding var selection: String
    let choices: [FontFamily]
    let defaultFamily: FontFamily
    let role: TextRole
    @Environment(\.theme) private var theme

    private var sample: String {
        switch role {
        case .mono: "softmax(q @ k.T / √d)"
        case .body: "Averaging reduces variance, not bias."
        default: "Bias–Variance Tradeoff"
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                PanelList([""] + choices.map(\.rawValue)) { raw in
                    let family = FontFamily(rawValue: raw) ?? defaultFamily
                    Button {
                        Haptics.tap()
                        selection = raw
                    } label: {
                        HStack(alignment: .center) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(raw.isEmpty ? "Theme default · \(defaultFamily.displayName)" : family.displayName)
                                    .font(theme.font(.label)).foregroundStyle(theme.textSecondary)
                                Text(sample)
                                    .font(Font(family.uiFont(size: role == .title ? 22 : 17,
                                                             weight: role == .title ? family.displayWeight : .regular)))
                                    .foregroundStyle(theme.text)
                            }
                            Spacer()
                            if selection == raw { Image(systemName: "checkmark").foregroundStyle(theme.accent) }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding()
        }
        .screenBackground()
        .themedNavBar(title)
    }
}
