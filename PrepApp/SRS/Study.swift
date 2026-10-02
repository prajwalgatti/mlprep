import Foundation
import SwiftData

/// All reads/writes of study progress go through here.
@MainActor
enum Study {
    static var scheduler = FSRS()
    /// A card counts as "mastered" once its stability reaches this many days.
    static let masteryDays = 21.0

    // MARK: Card state

    static func cardState(_ itemID: String, in ctx: ModelContext) -> CardState? {
        var d = FetchDescriptor<CardState>(predicate: #Predicate { $0.itemID == itemID })
        d.fetchLimit = 1
        return try? ctx.fetch(d).first
    }

    static func memory(of state: CardState?) -> MemoryState? {
        guard let s = state, !s.isNew else { return nil }
        return MemoryState(stability: s.stability, difficulty: s.difficulty)
    }

    /// Record an answer: updates the FSRS schedule and appends a log entry.
    @discardableResult
    static func record(_ item: StudyItem, rating: Rating, correct: Bool, source: ReviewSource,
                       durationMs: Int, in ctx: ModelContext, now: Date = .now) -> CardState {
        let state = cardState(item.id, in: ctx) ?? {
            let s = CardState(itemID: item.id, topicID: item.topicID, rev: item.rev)
            ctx.insert(s)
            return s
        }()
        let result = scheduler.schedule(memory: memory(of: state), lastReview: state.lastReview, rating: rating, now: now)
        state.stability = result.memory.stability
        state.difficulty = result.memory.difficulty
        state.due = result.due
        state.lastReview = now
        state.reps += 1
        if rating == .again && state.reps > 1 { state.lapses += 1 }
        ctx.insert(ReviewLog(itemID: item.id, topicID: item.topicID, date: now, rating: rating.rawValue,
                             correct: correct, source: source, durationMs: durationMs))
        try? ctx.save()
        return state
    }

    /// Exam answers: a miss sends the card back for relearning; a hit leaves the schedule alone.
    static func recordExam(_ item: StudyItem, correct: Bool, durationMs: Int, in ctx: ModelContext) {
        if correct {
            ctx.insert(ReviewLog(itemID: item.id, topicID: item.topicID, rating: Rating.good.rawValue,
                                 correct: true, source: .exam, durationMs: durationMs))
            try? ctx.save()
        } else {
            record(item, rating: .again, correct: false, source: .exam, durationMs: durationMs, in: ctx)
        }
    }

    /// Items due for review now, most overdue first. Skips cards whose content no longer exists.
    static func dueItems(content: ContentStore, in ctx: ModelContext, now: Date = .now, limit: Int = 50) -> [StudyItem] {
        let d = FetchDescriptor<CardState>(predicate: #Predicate { $0.due <= now }, sortBy: [SortDescriptor(\.due)])
        let states = (try? ctx.fetch(d)) ?? []
        return Array(states.compactMap { content.itemsByID[$0.itemID] }.prefix(limit))
    }

    static func dueCount(content: ContentStore, in ctx: ModelContext, now: Date = .now) -> Int {
        dueItems(content: content, in: ctx, now: now, limit: .max).count
    }

    /// Reset cards whose content `rev` was bumped. Call on launch.
    static func reconcile(content: ContentStore, in ctx: ModelContext) {
        let states = (try? ctx.fetch(FetchDescriptor<CardState>())) ?? []
        var changed = false
        for s in states {
            guard let item = content.itemsByID[s.itemID], item.rev > s.rev else { continue }
            s.rev = item.rev
            s.stability = 0
            s.difficulty = 0
            s.reps = 0
            s.lastReview = nil
            s.due = .now
            changed = true
        }
        if changed { try? ctx.save() }
    }

    // MARK: Topic progress

    static func progress(_ topicID: String, in ctx: ModelContext) -> TopicProgress {
        var d = FetchDescriptor<TopicProgress>(predicate: #Predicate { $0.topicID == topicID })
        d.fetchLimit = 1
        if let p = try? ctx.fetch(d).first { return p }
        let p = TopicProgress(topicID: topicID)
        ctx.insert(p)
        return p
    }

    /// 0...1: average of each item's stability relative to the mastery threshold.
    static func mastery(of topic: Topic, states: [String: CardState]) -> Double {
        let items = topic.items
        guard !items.isEmpty else { return 0 }
        let total = items.reduce(0.0) { acc, item in
            guard let s = states[item.id], !s.isNew else { return acc }
            return acc + min(s.stability / masteryDays, 1)
        }
        return total / Double(items.count)
    }

    /// Suggested next topic to learn: first unfinished topic whose prereqs are done (falls back to any unfinished).
    static func nextTopic(content: ContentStore, progress: [String: TopicProgress]) -> Topic? {
        let done = Set(progress.values.filter { $0.lessonCompletedAt != nil }.map(\.topicID))
        let remaining = content.topics.filter { !done.contains($0.id) }
        return remaining.first { ($0.prereqs ?? []).allSatisfy { done.contains($0) || content.topicsByID[$0] == nil } }
            ?? remaining.first
    }

    // MARK: Streaks & activity

    static func dailyCounts(_ logs: [ReviewLog], calendar: Calendar = .current) -> [Date: Int] {
        var out: [Date: Int] = [:]
        for log in logs { out[calendar.startOfDay(for: log.date), default: 0] += 1 }
        return out
    }

    /// Consecutive days with activity, ending today (or yesterday if nothing yet today).
    static func streak(_ counts: [Date: Int], now: Date = .now, calendar: Calendar = .current) -> Int {
        var day = calendar.startOfDay(for: now)
        if counts[day] == nil { day = calendar.date(byAdding: .day, value: -1, to: day)! }
        var n = 0
        while counts[day] != nil {
            n += 1
            day = calendar.date(byAdding: .day, value: -1, to: day)!
        }
        return n
    }

    // MARK: Reset

    static func resetAll(in ctx: ModelContext) {
        try? ctx.delete(model: CardState.self)
        try? ctx.delete(model: ReviewLog.self)
        try? ctx.delete(model: ExamRecord.self)
        let progress = (try? ctx.fetch(FetchDescriptor<TopicProgress>())) ?? []
        for p in progress {
            p.lessonCompletedAt = nil
            p.lastQuizScore = nil
        }
        try? ctx.save()
    }
}
