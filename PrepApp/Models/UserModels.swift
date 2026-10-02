import Foundation
import SwiftData

// Persistent user state. Everything is keyed by stable content IDs ("topicID/itemID"),
// so editing or adding content never wipes progress.

@Model
final class CardState {
    @Attribute(.unique) var itemID: String
    var topicID: String
    var stability: Double
    var difficulty: Double
    var due: Date
    var lastReview: Date?
    var reps: Int
    var lapses: Int
    /// Content revision this schedule belongs to; bumping `rev` in YAML resets the card.
    var rev: Int

    init(itemID: String, topicID: String, rev: Int) {
        self.itemID = itemID
        self.topicID = topicID
        self.stability = 0
        self.difficulty = 0
        self.due = .now
        self.lastReview = nil
        self.reps = 0
        self.lapses = 0
        self.rev = rev
    }

    var isNew: Bool { reps == 0 }
}

@Model
final class TopicProgress {
    @Attribute(.unique) var topicID: String
    var lessonCompletedAt: Date?
    var lastQuizScore: Double?
    var bookmarked: Bool
    var note: String
    var lastOpened: Date?

    init(topicID: String) {
        self.topicID = topicID
        self.lessonCompletedAt = nil
        self.lastQuizScore = nil
        self.bookmarked = false
        self.note = ""
        self.lastOpened = nil
    }
}

enum ReviewSource: String, Codable {
    case lesson, review, practice, exam
}

@Model
final class ReviewLog {
    var itemID: String
    var topicID: String
    var date: Date
    var rating: Int
    var correct: Bool
    var sourceRaw: String
    var durationMs: Int

    init(itemID: String, topicID: String, date: Date = .now, rating: Int, correct: Bool,
         source: ReviewSource, durationMs: Int) {
        self.itemID = itemID
        self.topicID = topicID
        self.date = date
        self.rating = rating
        self.correct = correct
        self.sourceRaw = source.rawValue
        self.durationMs = durationMs
    }

    var source: ReviewSource { ReviewSource(rawValue: sourceRaw) ?? .review }
}

enum FlagReason: String, CaseIterable, Codable, Identifiable {
    case wrong = "Incorrect"
    case unclear = "Unclear"
    case formatting = "Formatting / math"
    case tooEasy = "Too easy / trivial"
    case other = "Other"
    var id: String { rawValue }
}

@Model
final class FlagRecord {
    var itemID: String
    var topicID: String
    var reasonRaw: String
    var note: String
    var date: Date
    var resolved: Bool

    init(itemID: String, topicID: String, reason: FlagReason, note: String) {
        self.itemID = itemID
        self.topicID = topicID
        self.reasonRaw = reason.rawValue
        self.note = note
        self.date = .now
        self.resolved = false
    }

    var reason: FlagReason { FlagReason(rawValue: reasonRaw) ?? .other }
}

@Model
final class ExamRecord {
    var date: Date
    var title: String
    var total: Int
    var correct: Int
    var durationSec: Int
    /// JSON-encoded [topicID: [correct, total]]
    var breakdownData: Data
    var wrongItemIDs: [String]

    init(date: Date = .now, title: String, total: Int, correct: Int, durationSec: Int,
         breakdown: [String: [Int]], wrongItemIDs: [String]) {
        self.date = date
        self.title = title
        self.total = total
        self.correct = correct
        self.durationSec = durationSec
        self.breakdownData = (try? JSONEncoder().encode(breakdown)) ?? Data()
        self.wrongItemIDs = wrongItemIDs
    }

    var breakdown: [String: [Int]] {
        (try? JSONDecoder().decode([String: [Int]].self, from: breakdownData)) ?? [:]
    }

    var score: Double { total == 0 ? 0 : Double(correct) / Double(total) }
}
