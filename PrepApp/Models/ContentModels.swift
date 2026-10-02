import Foundation

// Read-only study content, decoded from the YAML files in Content/. See CONTENT_GUIDE.md.

struct Area: Codable, Identifiable, Hashable {
    let id: String
    let title: String
    let icon: String
    let color: String
    let blurb: String
    let order: Int
}

enum TopicLevel: String, Codable, Hashable {
    case core, intermediate, advanced
}

struct Topic: Codable, Identifiable, Hashable {
    let id: String
    let area: String
    let title: String
    let summary: String
    let order: Int?
    let level: TopicLevel?
    let tags: [String]?
    let prereqs: [String]?
    let explainer: [ExplainerCard]
    let quiz: [MCQ]?
    let cards: [Flashcard]?
    let reading: [ReadingItem]?

    var mcqs: [MCQ] { quiz ?? [] }
    var flashcards: [Flashcard] { cards ?? [] }
    var sortOrder: Int { order ?? 1000 }

    /// Every reviewable item in this topic, quiz first.
    var items: [StudyItem] {
        mcqs.map { .mcq($0, topicID: id) } + flashcards.map { .flash($0, topicID: id) }
    }
}

struct ExplainerCard: Codable, Hashable {
    let title: String
    let body: String
    /// Where this card's material comes from, e.g. "Goodfellow et al., Deep Learning §8.5.3".
    let source: String?
}

struct MCQ: Codable, Hashable {
    let id: String
    let prompt: String
    let correct: String
    let wrong: [String]
    let explanation: String?
    let source: String?
    let rev: Int?
}

enum FlashcardType: String, Codable, Hashable {
    case flash, open
}

struct Flashcard: Codable, Hashable {
    let id: String
    let type: FlashcardType?
    let front: String
    let back: String
    let source: String?
    let rev: Int?

    var isOpen: Bool { type == .open }
}

enum ReadingKind: String, Codable, Hashable {
    case paper, blog, book, video, course, docs

    var symbol: String {
        switch self {
        case .paper: "doc.text"
        case .blog: "text.alignleft"
        case .book: "book.closed"
        case .video: "play.rectangle"
        case .course: "graduationcap"
        case .docs: "doc.plaintext"
        }
    }
}

struct ReadingItem: Codable, Hashable {
    let title: String
    let url: String
    let kind: ReadingKind?
    let note: String?
}

/// A single reviewable unit, identified globally by "topicID/itemID".
enum StudyItem: Hashable, Identifiable {
    case mcq(MCQ, topicID: String)
    case flash(Flashcard, topicID: String)

    var id: String { "\(topicID)/\(localID)" }

    var topicID: String {
        switch self {
        case .mcq(_, let t), .flash(_, let t): t
        }
    }

    var localID: String {
        switch self {
        case .mcq(let q, _): q.id
        case .flash(let c, _): c.id
        }
    }

    var rev: Int {
        switch self {
        case .mcq(let q, _): q.rev ?? 1
        case .flash(let c, _): c.rev ?? 1
        }
    }

    var isMCQ: Bool {
        if case .mcq = self { return true }
        return false
    }

    /// Short plain-ish text used in lists and exports.
    var promptText: String {
        switch self {
        case .mcq(let q, _): q.prompt
        case .flash(let c, _): c.front
        }
    }
}
