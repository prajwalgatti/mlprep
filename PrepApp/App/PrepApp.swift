import SwiftData
import SwiftUI

@main
struct PrepApp: App {
    @State private var content = ContentStore()
    private let container: ModelContainer

    init() {
        FontFamily.registerBundledFonts()
        let schema = Schema([CardState.self, TopicProgress.self, ReviewLog.self, FlagRecord.self, ExamRecord.self])
        // UI tests launch with -uiTesting to start from a clean, throwaway store.
        let inMemory = ProcessInfo.processInfo.arguments.contains("-uiTesting")
        do {
            container = try ModelContainer(for: schema, configurations: ModelConfiguration(isStoredInMemoryOnly: inMemory))
        } catch {
            fatalError("Could not open the progress database: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(content)
        }
        .modelContainer(container)
    }
}

struct RootView: View {
    @Environment(ContentStore.self) private var content
    @Environment(\.modelContext) private var ctx
    @AppStorage(AppearanceKey.theme) private var themeID = ThemeID.workbench.rawValue
    @AppStorage(AppearanceKey.headline) private var headline = ""
    @AppStorage(AppearanceKey.body) private var bodyFont = ""
    @AppStorage(AppearanceKey.mono) private var mono = ""

    private var theme: Theme {
        Theme.named(themeID).with(headline: FontFamily(rawValue: headline), body: FontFamily(rawValue: bodyFont),
                                  mono: FontFamily(rawValue: mono))
    }

    var body: some View {
        TabView {
            TodayView()
                .tabItem { Label("Today", systemImage: "sun.max") }
            TopicsView()
                .tabItem { Label("Topics", systemImage: "square.grid.2x2") }
            ExamTab()
                .tabItem { Label("Exam", systemImage: "timer") }
            StatsView()
                .tabItem { Label("Progress", systemImage: "chart.bar") }
            SavedView()
                .tabItem { Label("Saved", systemImage: "bookmark") }
        }
        .environment(\.theme, theme)
        .tint(theme.accent)
        .preferredColorScheme(.dark)
        .task { Study.reconcile(content: content, in: ctx) }
    }
}
