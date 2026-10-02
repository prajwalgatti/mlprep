import Foundation

enum Rating: Int, CaseIterable, Identifiable {
    case again = 1, hard, good, easy
    var id: Int { rawValue }

    var label: String {
        switch self {
        case .again: "Again"
        case .hard: "Hard"
        case .good: "Good"
        case .easy: "Easy"
        }
    }
}

/// Memory state of one card as FSRS sees it.
struct MemoryState: Equatable {
    var stability: Double   // days until retrievability drops to 90%
    var difficulty: Double  // 1 (easy) ... 10 (hard)
}

/// The outcome of scheduling a review.
struct ScheduleResult: Equatable {
    var memory: MemoryState
    var due: Date
    var intervalDays: Double  // 0 means "relearn within this session"
}

/// FSRS-5 scheduler (https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
/// with the default parameters. Same-day reviews use the short-term stability formula.
struct FSRS {
    static let defaultWeights: [Double] = [
        0.4072, 1.1829, 3.1262, 15.4722, 7.2102, 0.5316, 1.0651, 0.0234, 1.616, 0.1544,
        1.0824, 1.9813, 0.0953, 0.2975, 2.2042, 0.2407, 2.9466, 0.5034, 0.6567,
    ]
    static let decay = -0.5
    static let factor = 19.0 / 81.0  // makes R(S, S) = 0.9

    var w: [Double] = FSRS.defaultWeights
    var desiredRetention: Double = 0.9
    var maximumIntervalDays: Double = 365
    /// Delay before a failed card comes back (it is also re-queued in the current session).
    var relearnDelay: TimeInterval = 10 * 60

    // MARK: Core formulas

    static func retrievability(elapsedDays t: Double, stability s: Double) -> Double {
        pow(1 + factor * t / s, decay)
    }

    func interval(stability s: Double) -> Double {
        s / Self.factor * (pow(desiredRetention, 1 / Self.decay) - 1)
    }

    func initialStability(_ g: Rating) -> Double {
        max(w[g.rawValue - 1], 0.1)
    }

    func initialDifficulty(_ g: Rating) -> Double {
        clampD(w[4] - exp(w[5] * Double(g.rawValue - 1)) + 1)
    }

    func nextDifficulty(_ d: Double, _ g: Rating) -> Double {
        let delta = -w[6] * Double(g.rawValue - 3)
        let damped = d + delta * (10 - d) / 9
        // Mean reversion towards D0(Easy).
        return clampD(w[7] * initialDifficulty(.easy) + (1 - w[7]) * damped)
    }

    func recallStability(d: Double, s: Double, r: Double, _ g: Rating) -> Double {
        let hardPenalty = g == .hard ? w[15] : 1
        let easyBonus = g == .easy ? w[16] : 1
        return s * (exp(w[8]) * (11 - d) * pow(s, -w[9]) * (exp(w[10] * (1 - r)) - 1) * hardPenalty * easyBonus + 1)
    }

    func forgetStability(d: Double, s: Double, r: Double) -> Double {
        let sNew = w[11] * pow(d, -w[12]) * (pow(s + 1, w[13]) - 1) * exp(w[14] * (1 - r))
        return min(sNew, s)
    }

    func shortTermStability(s: Double, _ g: Rating) -> Double {
        s * exp(w[17] * (Double(g.rawValue) - 3 + w[18]))
    }

    private func clampD(_ d: Double) -> Double { min(max(d, 1), 10) }

    // MARK: Scheduling

    /// Schedule a review. `memory == nil` means the card has never been reviewed.
    func schedule(memory: MemoryState?, lastReview: Date?, rating g: Rating, now: Date = .now) -> ScheduleResult {
        var next: MemoryState
        if let m = memory, let last = lastReview, m.stability > 0 {
            let elapsed = max(0, now.timeIntervalSince(last) / 86_400)
            let d = nextDifficulty(m.difficulty, g)
            let s: Double
            if elapsed < 1 {
                s = shortTermStability(s: m.stability, g)
            } else {
                let r = Self.retrievability(elapsedDays: elapsed, stability: m.stability)
                s = g == .again
                    ? forgetStability(d: m.difficulty, s: m.stability, r: r)
                    : recallStability(d: m.difficulty, s: m.stability, r: r, g)
            }
            next = MemoryState(stability: max(s, 0.1), difficulty: d)
        } else {
            next = MemoryState(stability: initialStability(g), difficulty: initialDifficulty(g))
        }

        if g == .again {
            return ScheduleResult(memory: next, due: now.addingTimeInterval(relearnDelay), intervalDays: 0)
        }
        let days = min(max(interval(stability: next.stability).rounded(), 1), maximumIntervalDays)
        return ScheduleResult(memory: next, due: now.addingTimeInterval(days * 86_400), intervalDays: days)
    }

    /// Human-readable preview of the next interval for each rating (for the rating buttons).
    func previews(memory: MemoryState?, lastReview: Date?, now: Date = .now) -> [Rating: String] {
        var out: [Rating: String] = [:]
        for g in Rating.allCases {
            let r = schedule(memory: memory, lastReview: lastReview, rating: g, now: now)
            out[g] = Self.format(days: r.intervalDays)
        }
        return out
    }

    static func format(days: Double) -> String {
        switch days {
        case ..<1: "10m"
        case ..<30: "\(Int(days))d"
        case ..<365: String(format: "%.1fmo", days / 30)
        default: String(format: "%.1fy", days / 365)
        }
    }
}
