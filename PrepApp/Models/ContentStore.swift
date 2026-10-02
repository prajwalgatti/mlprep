import Foundation
import Observation
import Yams

/// Loads every topic YAML shipped in the app bundle. Files in Documents/ContentOverride
/// (same filenames) take precedence, so content can later be updated without a rebuild.
@Observable
final class ContentStore {
    private(set) var areas: [Area] = []
    private(set) var topics: [Topic] = []
    private(set) var topicsByID: [String: Topic] = [:]
    private(set) var itemsByID: [String: StudyItem] = [:]
    /// Human-readable load problems, shown in Settings → Content.
    private(set) var issues: [String] = []

    init(directories: [URL]? = nil) {
        load(from: directories ?? Self.defaultDirectories)
    }

    static var overrideDirectory: URL {
        URL.documentsDirectory.appending(path: "ContentOverride", directoryHint: .isDirectory)
    }

    static var defaultDirectories: [URL] {
        [Bundle.main.resourceURL, overrideDirectory].compactMap { $0 }
    }

    func reload() {
        load(from: Self.defaultDirectories)
    }

    func load(from directories: [URL]) {
        var issues: [String] = []
        var areaFile: URL?
        var topicFiles: [String: URL] = [:]   // filename -> url (later directories win)

        for dir in directories {
            for url in Self.yamlFiles(in: dir) {
                if url.lastPathComponent == "areas.yaml" {
                    areaFile = url
                } else {
                    topicFiles[url.lastPathComponent] = url
                }
            }
        }

        let decoder = YAMLDecoder()
        var areas: [Area] = []
        if let areaFile {
            do {
                areas = try decoder.decode([Area].self, from: String(contentsOf: areaFile, encoding: .utf8))
            } catch {
                issues.append("areas.yaml: \(Self.describe(error))")
            }
        } else {
            issues.append("areas.yaml not found in bundle")
        }
        let areaIDs = Set(areas.map(\.id))

        var topics: [Topic] = []
        var seen = Set<String>()
        for (name, url) in topicFiles.sorted(by: { $0.key < $1.key }) {
            do {
                let topic = try decoder.decode(Topic.self, from: String(contentsOf: url, encoding: .utf8))
                guard !seen.contains(topic.id) else {
                    issues.append("\(name): duplicate topic id \(topic.id)")
                    continue
                }
                if !areaIDs.contains(topic.area) {
                    issues.append("\(name): unknown area '\(topic.area)'")
                }
                seen.insert(topic.id)
                topics.append(topic)
            } catch {
                issues.append("\(name): \(Self.describe(error))")
            }
        }

        let areaOrder = Dictionary(uniqueKeysWithValues: areas.map { ($0.id, $0.order) })
        topics.sort {
            let a0 = areaOrder[$0.area] ?? 99, a1 = areaOrder[$1.area] ?? 99
            return a0 != a1 ? a0 < a1 : ($0.sortOrder, $0.title) < ($1.sortOrder, $1.title)
        }

        var items: [String: StudyItem] = [:]
        for topic in topics {
            for item in topic.items {
                if items[item.id] != nil { issues.append("duplicate item id \(item.id)") }
                items[item.id] = item
            }
        }

        self.areas = areas.sorted { $0.order < $1.order }
        self.topics = topics
        self.topicsByID = Dictionary(uniqueKeysWithValues: topics.map { ($0.id, $0) })
        self.itemsByID = items
        self.issues = issues
    }

    // MARK: Queries

    func topics(in areaID: String) -> [Topic] {
        topics.filter { $0.area == areaID }
    }

    func area(_ id: String) -> Area? {
        areas.first { $0.id == id }
    }

    func search(_ query: String) -> [Topic] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return [] }
        return topics.filter { t in
            t.title.lowercased().contains(q)
                || t.summary.lowercased().contains(q)
                || (t.tags ?? []).contains { $0.lowercased().contains(q) }
        }
    }

    var totalItemCount: Int { itemsByID.count }

    // MARK: Helpers

    private static func yamlFiles(in dir: URL) -> [URL] {
        guard let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil) else { return [] }
        return e.compactMap { $0 as? URL }.filter { $0.pathExtension == "yaml" || $0.pathExtension == "yml" }
    }

    private static func describe(_ error: Error) -> String {
        switch error {
        case DecodingError.keyNotFound(let key, let ctx):
            return "missing '\(key.stringValue)' at \(path(ctx))"
        case DecodingError.typeMismatch(_, let ctx), DecodingError.valueNotFound(_, let ctx),
             DecodingError.dataCorrupted(let ctx):
            return "\(ctx.debugDescription) at \(path(ctx))"
        default:
            return String(describing: error)
        }
    }

    private static func path(_ ctx: DecodingError.Context) -> String {
        ctx.codingPath.map { $0.intValue.map(String.init) ?? $0.stringValue }.joined(separator: ".")
    }
}
