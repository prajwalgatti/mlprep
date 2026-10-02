import XCTest
@testable import PrepApp

final class FSRSTests: XCTestCase {
    let f = FSRS()
    let t0 = Date(timeIntervalSince1970: 1_800_000_000)

    func testRetrievabilityIsNinetyPercentAtStability() {
        XCTAssertEqual(FSRS.retrievability(elapsedDays: 10, stability: 10), 0.9, accuracy: 1e-9)
        XCTAssertEqual(f.interval(stability: 10), 10, accuracy: 1e-9)
    }

    func testFirstReviewUsesInitialStability() {
        let good = f.schedule(memory: nil, lastReview: nil, rating: .good, now: t0)
        XCTAssertEqual(good.memory.stability, FSRS.defaultWeights[2], accuracy: 1e-9)
        XCTAssertEqual(good.intervalDays, 3)

        let again = f.schedule(memory: nil, lastReview: nil, rating: .again, now: t0)
        XCTAssertEqual(again.intervalDays, 0)
        XCTAssertEqual(again.due, t0.addingTimeInterval(600))
    }

    func testRatingsAreMonotonic() {
        let m = MemoryState(stability: 5, difficulty: 5)
        let later = t0.addingTimeInterval(5 * 86_400)
        let s = Rating.allCases.map { f.schedule(memory: m, lastReview: t0, rating: $0, now: later).memory.stability }
        XCTAssertLessThan(s[0], s[1])
        XCTAssertLessThan(s[1], s[2])
        XCTAssertLessThan(s[2], s[3])
        let d = Rating.allCases.map { f.schedule(memory: m, lastReview: t0, rating: $0, now: later).memory.difficulty }
        XCTAssertGreaterThan(d[0], d[3])
    }

    func testLapseNeverIncreasesStability() {
        let m = MemoryState(stability: 30, difficulty: 6)
        let r = f.schedule(memory: m, lastReview: t0, rating: .again, now: t0.addingTimeInterval(40 * 86_400))
        XCTAssertLessThanOrEqual(r.memory.stability, 30)
    }

    func testSuccessfulReviewsGrowIntervals() {
        var m: MemoryState?
        var last: Date?
        var now = t0
        var intervals: [Double] = []
        for _ in 0..<5 {
            let r = f.schedule(memory: m, lastReview: last, rating: .good, now: now)
            intervals.append(r.intervalDays)
            m = r.memory; last = now; now = r.due
        }
        XCTAssertEqual(intervals, intervals.sorted())
        XCTAssertGreaterThan(intervals.last!, 30)
    }

    func testDifficultyStaysInRange() {
        var m = MemoryState(stability: 2, difficulty: 9.9)
        for _ in 0..<20 { m = f.schedule(memory: m, lastReview: t0, rating: .again, now: t0.addingTimeInterval(86_400)).memory }
        XCTAssertLessThanOrEqual(m.difficulty, 10)
        for _ in 0..<50 { m = f.schedule(memory: m, lastReview: t0, rating: .easy, now: t0.addingTimeInterval(86_400)).memory }
        XCTAssertGreaterThanOrEqual(m.difficulty, 1)
    }
}
