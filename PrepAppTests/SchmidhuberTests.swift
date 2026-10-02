import UIKit
import XCTest
@testable import PrepApp

/// The pixel-art easter egg: bundled portrait, topic matching and the once-per-session cap.
@MainActor
final class SchmidhuberTests: XCTestCase {
    func testPortraitAssetIsBundledAtNativeSize() throws {
        let image = try XCTUnwrap(UIImage(named: "Schmidhuber", in: Bundle(for: SchmidhuberCenter.self), with: nil))
        XCTAssertEqual(image.size.width * image.scale, SchmidhuberSprite.artSize.width)
        XCTAssertEqual(image.size.height * image.scale, SchmidhuberSprite.artSize.height)
    }

    func testSubjectMatchingIsWholeWord() {
        typealias Q = SchmidhuberQuotes
        XCTAssertEqual(Q.subjects(in: ["llm.self-attention", "Scaled Dot-Product & Multi-Head Attention"]), [.transformer])
        XCTAssertEqual(Q.subjects(in: ["dl.lstm", "LSTMs and GRUs"]).first, .lstm)
        XCTAssertTrue(Q.subjects(in: ["Residual connections"]).contains(.highway))
        XCTAssertTrue(Q.subjects(in: ["tags: meta-learning"]).contains(.metaLearning))
        XCTAssertTrue(Q.subjects(in: ["World Models"]).contains(.worldModel))
        XCTAssertTrue(Q.subjects(in: ["GANs"]).contains(.gan))
        // Substrings inside other words don't count.
        XCTAssertTrue(Q.subjects(in: ["organ", "began", "fund.bias-variance", "Adam"]).isEmpty)
        for s in Q.Subject.allCases { XCTAssertFalse(Q.bySubject[s, default: []].isEmpty, "\(s) has no quotes") }
    }

    func testContextTriggerShowsAtMostOncePerSession() {
        let center = SchmidhuberCenter.shared
        // appear() counts a claim (and may unlock the mascot); put the real values back afterwards.
        let keys = [SchmidhuberKey.claims, SchmidhuberKey.unlocked, SchmidhuberKey.mascot]
        let saved = keys.map { UserDefaults.standard.object(forKey: $0) }
        defer { for (k, v) in zip(keys, saved) { UserDefaults.standard.set(v, forKey: k) } }
        let item = StudyItem.mcq(MCQ(id: "q1", prompt: "?", correct: "a", wrong: ["b"], explanation: nil, source: nil, rev: nil),
                                 topicID: "llm.self-attention")
        let t0 = Date.now
        center.random = { 0.99 }   // dice say no
        center.correctAnswer(item, now: t0)
        XCTAssertFalse(center.contextShownThisSession)

        center.random = { 0 }      // dice say yes
        center.correctAnswer(item, now: t0.addingTimeInterval(60))
        XCTAssertTrue(center.contextShownThisSession)

        // A long gap starts a new session and clears the cap.
        center.random = { 0.99 }
        center.correctAnswer(item, now: t0.addingTimeInterval(60 + SchmidhuberCenter.sessionGap + 1))
        XCTAssertFalse(center.contextShownThisSession)
        center.dismiss()
        center.random = { Double.random(in: 0..<1) }
    }

    func testMascotUnlocksOnFifthClaimExactlyOnce() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "SchmidhuberTests"))
        defaults.removePersistentDomain(forName: "SchmidhuberTests")
        defer { defaults.removePersistentDomain(forName: "SchmidhuberTests") }
        for _ in 1..<SchmidhuberMascotState.threshold {
            XCTAssertFalse(SchmidhuberMascotState.registerClaim(defaults: defaults))
        }
        XCTAssertFalse(defaults.bool(forKey: SchmidhuberKey.unlocked))
        XCTAssertTrue(SchmidhuberMascotState.registerClaim(defaults: defaults), "5th claim unlocks")
        XCTAssertTrue(defaults.bool(forKey: SchmidhuberKey.unlocked))
        XCTAssertTrue(defaults.bool(forKey: SchmidhuberKey.mascot), "mascot defaults on at unlock")
        XCTAssertFalse(SchmidhuberMascotState.registerClaim(defaults: defaults), "only once")
        XCTAssertEqual(defaults.integer(forKey: SchmidhuberKey.claims), 6)
    }

    func testMascotLinesReactToState() {
        typealias M = SchmidhuberMascotState
        XCTAssertEqual(M.greeting(dueCount: 3, goalDone: true, streak: 4), "3 reviews. I did these in 1991.")
        XCTAssertEqual(M.greeting(dueCount: 0, goalDone: true, streak: 4), "Good. Now cite me.")
        XCTAssertEqual(M.greeting(dueCount: 0, goalDone: false, streak: 0), "Even LSTMs need a first step.")
        let day = 86_400.0
        let lines = Set((0..<7).map { M.greeting(dueCount: 0, goalDone: false, streak: 2, date: Date(timeIntervalSince1970: 1_800_000_000 + Double($0) * day)) })
        XCTAssertGreaterThan(lines.count, 1, "the idle line changes from day to day")
        XCTAssertNotEqual(M.cheer(accuracy: 0.95, answered: 10), M.cheer(accuracy: 0.3, answered: 10))
    }
}
