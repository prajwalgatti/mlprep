import XCTest

/// End-to-end walk through the main flows with screenshots attached to the test result.
final class SmokeTests: XCTestCase {
    var app: XCUIApplication!

    override func setUp() {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["-uiTesting"]
        app.launch()
    }

    private func snap(_ name: String) {
        let a = XCTAttachment(screenshot: app.screenshot())
        a.name = name
        a.lifetime = .keepAlways
        add(a)
    }

    /// Answers whatever item is on screen. Returns false when no item is showing.
    @discardableResult
    private func answerCurrentItem(rate: String = "Good", snapFlashcard: Bool = false) -> Bool {
        let choice = app.buttons["choice-0"]
        let show = app.buttons["showAnswer"]
        if choice.waitForExistence(timeout: 2) {
            choice.tap()
            let cont = app.buttons["mcqContinue"]
            XCTAssertTrue(cont.waitForExistence(timeout: 3))
            cont.tap()
            return true
        }
        if show.exists {
            show.tap()
            let r = app.buttons["rate-\(rate)"]
            XCTAssertTrue(r.waitForExistence(timeout: 3))
            if snapFlashcard { snap("04-flashcard-revealed") }
            r.tap()
            return true
        }
        return false
    }

    func testLessonReviewExamAndTabs() {
        snap("01-today-fresh")

        // Lesson: explainer → practice → summary
        let start = app.buttons["startLesson"]
        XCTAssertTrue(start.waitForExistence(timeout: 5))
        start.tap()
        let next = app.buttons["explainerNext"]
        XCTAssertTrue(next.waitForExistence(timeout: 5))
        snap("02-explainer")
        for _ in 0..<10 where next.exists { next.tap() }
        XCTAssertTrue(app.buttons["choice-0"].waitForExistence(timeout: 5))
        snap("03-mcq")

        var snappedCard = false
        for _ in 0..<80 {
            if app.buttons["summaryDone"].exists { break }
            if app.buttons["showAnswer"].exists && !snappedCard {
                snappedCard = true
                answerCurrentItem(snapFlashcard: true)
                continue
            }
            if !answerCurrentItem() { _ = app.buttons["summaryDone"].waitForExistence(timeout: 2) }
        }
        XCTAssertTrue(app.buttons["summaryDone"].waitForExistence(timeout: 5), "lesson summary never appeared")
        snap("05-lesson-summary")
        app.buttons["summaryDone"].tap()
        snap("06-today-after-lesson")

        // Topics
        app.tabBars.buttons["Topics"].tap()
        snap("07-topics")
        app.staticTexts["ML Fundamentals"].firstMatch.tap()
        snap("08-area")
        let topic = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'Bayes'")).firstMatch
        XCTAssertTrue(topic.waitForExistence(timeout: 3))
        topic.tap()
        _ = app.navigationBars.firstMatch.waitForExistence(timeout: 2)
        snap("09-topic-detail")

        // Exam
        app.tabBars.buttons["Exam"].tap()
        snap("10-exam-setup")
        let startExam = app.buttons["startExam"]
        XCTAssertTrue(startExam.waitForExistence(timeout: 3))
        startExam.tap()
        XCTAssertTrue(app.buttons["choice-0"].waitForExistence(timeout: 5))
        snap("11-exam-question")
        for _ in 0..<60 {
            if app.navigationBars["Results"].exists { break }
            if !answerCurrentItem() { _ = app.navigationBars["Results"].waitForExistence(timeout: 2) }
        }
        XCTAssertTrue(app.navigationBars["Results"].waitForExistence(timeout: 5), "exam results never appeared")
        snap("12-exam-results")
        app.buttons["Done"].tap()

        // Progress + Saved
        app.tabBars.buttons["Progress"].tap()
        snap("13-progress")
        app.tabBars.buttons["Saved"].tap()
        snap("14-saved")
    }
}
