import XCTest

/// The pixel Schmidhuber easter egg and the unlockable mascot. Screenshots are attached to the test result.
///
/// State is pinned with launch arguments (UserDefaults' argument domain wins over saved values), e.g.
/// `-schmidhuber.claims 1 -schmidhuber.unlocked NO`, so runs don't depend on what earlier runs saved.
final class SchmidhuberEasterEggUITests: XCTestCase {
    override func setUp() { continueAfterFailure = false }

    private func launch(_ theme: String, _ extra: [String]) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTesting", "-theme", theme] + extra
        app.launch()
        XCTAssertTrue(app.buttons["startLesson"].waitForExistence(timeout: 5))
        return app
    }

    private func snap(_ app: XCUIApplication, _ name: String) {
        let a = XCTAttachment(screenshot: app.screenshot())
        a.name = "egg-\(name)"
        a.lifetime = .keepAlways
        add(a)
    }

    private func summonByTitle(_ app: XCUIApplication) {
        let title = app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] 'today'")).firstMatch
        XCTAssertTrue(title.waitForExistence(timeout: 5))
        title.tap(withNumberOfTaps: 5, numberOfTouches: 1)
    }

    private func popup(_ app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label BEGINSWITH 'Jürgen Schmidhuber says'")).firstMatch
    }

    private func waitForGone(_ e: XCUIElement, timeout: TimeInterval) -> Bool {
        let gone = expectation(for: NSPredicate(format: "exists == false"), evaluatedWith: e)
        return XCTWaiter.wait(for: [gone], timeout: timeout) == .completed
    }

    private func openSettings(_ app: XCUIApplication, scrollTo id: String) -> XCUIElement {
        app.buttons["Settings"].firstMatch.tap()
        let row = app.descendants(matching: .any)[id]
        for _ in 0..<4 where !(row.exists && row.isHittable) { app.swipeUp() }
        XCTAssertTrue(row.waitForExistence(timeout: 3), "\(id) should be in Settings")
        return row
    }

    func testSecretGestureSummonsHimInEveryTheme() {
        for (i, theme) in ["workbench", "terminal", "editorial"].enumerated() {
            let app = launch(theme, ["-schmidhuber.claims", "1", "-schmidhuber.unlocked", "NO"])
            summonByTitle(app)
            let him = popup(app)
            XCTAssertTrue(him.waitForExistence(timeout: 3), "secret gesture should summon him")
            sleep(1)   // let the spring settle
            // He stands above the tab bar, never on it.
            let tabBar = app.tabBars.firstMatch
            if tabBar.exists { XCTAssertLessThanOrEqual(him.frame.maxY, tabBar.frame.minY + 1, "pop-up overlaps the tab bar") }
            snap(app, "\(theme)-today")

            if i == 0 {
                him.tap()
                XCTAssertTrue(waitForGone(him, timeout: 2), "tap should dismiss him")
                _ = openSettings(app, scrollTo: "mascotLocked")
                snap(app, "settings-locked")
            } else {
                XCTAssertTrue(waitForGone(him, timeout: 7), "he should leave on his own after ~4 s")
            }
            app.terminate()
        }
    }

    /// The 5th claim unlocks the mascot: special line, then a one-time banner, then he's on Today.
    func testFifthClaimUnlocksMascot() {
        let app = launch("editorial", ["-schmidhuber.claims", "4", "-schmidhuberResetUnlock", "YES"])
        summonByTitle(app)
        let him = popup(app)
        XCTAssertTrue(him.waitForExistence(timeout: 3))
        XCTAssertTrue(him.label.contains("supervise your lab"), "unlock line expected, got: \(him.label)")
        sleep(1)
        snap(app, "unlock-line")
        him.tap()
        let banner = app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'unlocked as mascot'")).firstMatch
        XCTAssertTrue(banner.waitForExistence(timeout: 3), "unlock banner expected")
        snap(app, "unlock-banner")
        XCTAssertTrue(app.descendants(matching: .any)["schmidhuberMascot"].waitForExistence(timeout: 3), "mascot should be on Today")
        app.terminate()
    }

    func testMascotOnTodayAndInSettings() {
        for theme in ["workbench", "terminal"] {
            let app = launch(theme, ["-schmidhuberUnlocked", "YES", "-schmidhuber.mascot", "YES", "-schmidhuber.claims", "7"])
            let mascot = app.descendants(matching: .any)["schmidhuberMascot"]
            XCTAssertTrue(mascot.waitForExistence(timeout: 3), "mascot should show on Today when unlocked")
            sleep(1)
            snap(app, "\(theme)-mascot-today")
            if theme == "workbench" {
                mascot.tap()   // tapping him plays a quote with the pop-up
                XCTAssertTrue(popup(app).waitForExistence(timeout: 3))
                sleep(1)
                snap(app, "mascot-tap-quote")
                popup(app).tap()
                _ = waitForGone(popup(app), timeout: 2)
                _ = openSettings(app, scrollTo: "mascotToggle")
                snap(app, "settings-unlocked")
            }
            app.terminate()
        }
    }
}
