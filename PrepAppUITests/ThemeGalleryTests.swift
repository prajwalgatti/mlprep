import XCTest

/// Captures the main screens in every theme (screenshots are attached to the test result).
final class ThemeGalleryTests: XCTestCase {
    override func setUp() { continueAfterFailure = false }

    func testCaptureAllThemes() {
        for theme in ["workbench", "terminal", "editorial"] {
            let app = XCUIApplication()
            // "-theme x" lands in UserDefaults' argument domain, which @AppStorage reads.
            app.launchArguments = ["-uiTesting", "-theme", theme]
            app.launch()

            func snap(_ name: String) {
                let a = XCTAttachment(screenshot: app.screenshot())
                a.name = "\(theme)-\(name)"
                a.lifetime = .keepAlways
                add(a)
            }

            XCTAssertTrue(app.buttons["startLesson"].waitForExistence(timeout: 5))
            snap("1-today")

            app.buttons["startLesson"].tap()
            let next = app.buttons["explainerNext"]
            XCTAssertTrue(next.waitForExistence(timeout: 5))
            next.tap()
            sleep(1)
            snap("2-explainer")
            for _ in 0..<10 where next.exists { next.tap() }

            let choice = app.buttons["choice-0"]
            XCTAssertTrue(choice.waitForExistence(timeout: 5))
            choice.tap()
            sleep(1)
            snap("3-mcq")
            app.buttons["Close"].firstMatch.tap()

            app.tabBars.buttons["Topics"].tap()
            snap("4-topics")
            app.tabBars.buttons["Today"].tap()
            app.buttons["Settings"].firstMatch.tap()
            sleep(1)
            snap("5-settings")
            app.terminate()
        }
    }
}
