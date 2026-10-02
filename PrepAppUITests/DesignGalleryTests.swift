import XCTest

/// Walks every main screen in every theme and screenshots it, for design review.
/// Screenshots are attached to the test result and written as PNGs to `DESIGN_SHOT_DIR`
/// (pass `TEST_RUNNER_DESIGN_SHOT_DIR=/some/dir` to xcodebuild). docs/design-before and docs/design-after came from this.
final class DesignGalleryTests: XCTestCase {
    override func setUp() { continueAfterFailure = true }

    /// Opt-in because it takes ~10 minutes: run with `TEST_RUNNER_DESIGN_SHOT_DIR=/abs/dir`
    /// (optionally `TEST_RUNNER_DESIGN_THEMES=workbench,terminal`).
    func testCaptureEveryScreen() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["DESIGN_SHOT_DIR"] == nil, "Set DESIGN_SHOT_DIR to capture the design gallery")
        let themes = (ProcessInfo.processInfo.environment["DESIGN_THEMES"] ?? "workbench,terminal,editorial")
            .split(separator: ",").map(String.init)
        for theme in themes { capture(theme: theme) }
    }

    /// Opt-in (DESIGN_SEED=1): finishes one lesson against the persistent store, so that ~10 minutes later
    /// the missed items are due and the Today "Reviews due" state can be screenshotted by hand.
    func testSeedPersistentLesson() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DESIGN_SEED"] == "1", "Seeding is opt-in")
        let app = XCUIApplication()
        app.launchArguments = ["-theme", "workbench"]
        app.launch()
        XCTAssertTrue(app.buttons["startLesson"].waitForExistence(timeout: 8))
        app.buttons["startLesson"].tap()
        let next = app.buttons["explainerNext"]
        XCTAssertTrue(next.waitForExistence(timeout: 5))
        for _ in 0..<12 where next.exists { next.tap(); usleep(300_000) }
        for _ in 0..<80 {
            if app.buttons["summaryDone"].exists { break }
            if app.buttons["choice-0"].waitForExistence(timeout: 2) {
                app.buttons["choice-0"].tap()
                if app.buttons["mcqContinue"].waitForExistence(timeout: 3) { app.buttons["mcqContinue"].tap() }
            } else if app.buttons["showAnswer"].exists {
                app.buttons["showAnswer"].tap()
                if app.buttons["rate-Again"].waitForExistence(timeout: 3) { app.buttons["rate-Again"].tap() }
            }
        }
        if app.buttons["summaryDone"].waitForExistence(timeout: 5) { app.buttons["summaryDone"].tap() }
    }

    private func capture(theme: String) {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTesting", "-theme", theme]
        app.launch()
        let outDir = ProcessInfo.processInfo.environment["DESIGN_SHOT_DIR"]

        func snap(_ name: String) {
            let shot = app.screenshot()
            let a = XCTAttachment(screenshot: shot)
            a.name = "\(theme)-\(name)"
            a.lifetime = .keepAlways
            add(a)
            if let outDir {
                let url = URL(fileURLWithPath: outDir).appendingPathComponent("\(theme)-\(name).png")
                try? shot.pngRepresentation.write(to: url)
            }
        }
        func button(containing s: String) -> XCUIElement {
            app.buttons.matching(NSPredicate(format: "label CONTAINS[c] %@", s)).firstMatch
        }
        func feedbackShown(_ s: String) -> Bool {
            app.staticTexts.matching(NSPredicate(format: "label ==[c] %@", s)).firstMatch.exists
        }

        // Today, fresh
        XCTAssertTrue(app.buttons["startLesson"].waitForExistence(timeout: 8))
        sleep(1)
        snap("01-today-fresh")

        // Lesson explainer
        app.buttons["startLesson"].tap()
        let next = app.buttons["explainerNext"]
        XCTAssertTrue(next.waitForExistence(timeout: 5))
        sleep(1)
        snap("02-explainer-1")
        next.tap()
        sleep(1)
        snap("03-explainer-2")
        for _ in 0..<12 where next.exists { next.tap(); usleep(300_000) }

        // Practice: capture MCQ idle / correct / wrong, flashcard front / back, then finish.
        var gotIdle = false, gotRight = false, gotWrong = false, gotFlash = false
        for _ in 0..<80 {
            if app.buttons["summaryDone"].exists { break }
            let choice0 = app.buttons["choice-0"]
            if choice0.waitForExistence(timeout: 2) {
                if !gotIdle { gotIdle = true; snap("04-mcq-idle") }
                // Try to land on whichever state we still need.
                let pick = (gotRight && !gotWrong) ? 1 : 0
                let target = app.buttons["choice-\(pick)"].exists ? app.buttons["choice-\(pick)"] : choice0
                target.tap()
                let cont = app.buttons["mcqContinue"]
                XCTAssertTrue(cont.waitForExistence(timeout: 3))
                usleep(900_000)
                if feedbackShown("correct"), !gotRight { gotRight = true; snap("05-mcq-correct") }
                else if feedbackShown("not quite"), !gotWrong { gotWrong = true; snap("06-mcq-wrong") }
                cont.tap()
                continue
            }
            let show = app.buttons["showAnswer"]
            if show.exists {
                if !gotFlash { usleep(500_000); snap("07-flashcard-front") }
                show.tap()
                let good = app.buttons["rate-Good"]
                XCTAssertTrue(good.waitForExistence(timeout: 3))
                if !gotFlash { gotFlash = true; usleep(500_000); snap("08-flashcard-back") }
                good.tap()
                continue
            }
            _ = app.buttons["summaryDone"].waitForExistence(timeout: 2)
        }
        XCTAssertTrue(app.buttons["summaryDone"].waitForExistence(timeout: 5))
        sleep(1)
        snap("09-session-complete")
        app.buttons["summaryDone"].tap()
        sleep(1)
        snap("10-today-after")

        // Topics → area → topic detail
        app.tabBars.buttons["Topics"].tap()
        sleep(1)
        snap("11-topics")
        app.staticTexts["ML Fundamentals"].firstMatch.tap()
        sleep(1)
        snap("12-area")
        let topic = app.staticTexts["Bias–Variance Tradeoff"].exists ? app.staticTexts["Bias–Variance Tradeoff"] : app.staticTexts["MLE vs MAP"]
        topic.firstMatch.tap()
        sleep(1)
        snap("13-topic-detail")
        if app.buttons["Bookmark"].exists { app.buttons["Bookmark"].tap() }

        // Review-style practice session from the topic page.
        let practice = button(containing: "practice all")
        if practice.waitForExistence(timeout: 2) {
            practice.tap()
            sleep(1)
            snap("14-review")
            app.buttons["Close"].firstMatch.tap()
            sleep(1)
        }

        // Exam
        app.tabBars.buttons["Exam"].tap()
        sleep(1)
        snap("15-exam-setup")
        if app.buttons["10"].exists { app.buttons["10"].tap() }
        app.buttons["startExam"].tap()
        XCTAssertTrue(app.buttons["choice-0"].waitForExistence(timeout: 5))
        app.buttons["choice-1"].tap()
        usleep(500_000)
        snap("16-exam-question")
        for _ in 0..<60 {
            if app.navigationBars["Results"].exists { break }
            let c = app.buttons["choice-0"]
            if c.waitForExistence(timeout: 2) {
                c.tap()
                let cont = app.buttons["mcqContinue"]
                if cont.waitForExistence(timeout: 3) { cont.tap() }
            }
        }
        XCTAssertTrue(app.navigationBars["Results"].waitForExistence(timeout: 5))
        sleep(1)
        snap("17-exam-results")
        app.buttons["Done"].tap()
        sleep(1)

        // Progress, Saved, Settings
        app.tabBars.buttons["Progress"].tap()
        sleep(1)
        snap("18-progress")
        app.swipeUp()
        sleep(1)
        snap("19-progress-scrolled")
        app.tabBars.buttons["Saved"].tap()
        sleep(1)
        snap("20-saved")
        button(containing: "flagged").tap()
        sleep(1)
        snap("21-saved-flagged-empty")
        app.tabBars.buttons["Today"].tap()
        app.buttons["Settings"].firstMatch.tap()
        sleep(1)
        snap("22-settings")
        app.terminate()
    }
}
