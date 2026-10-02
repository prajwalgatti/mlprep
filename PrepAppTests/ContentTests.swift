import XCTest
@testable import PrepApp

/// Loads the real bundled content: catches YAML that the Python validator accepts but Swift can't decode.
final class ContentTests: XCTestCase {
    func testBundledContentLoadsCleanly() {
        let store = ContentStore(directories: [Bundle.main.resourceURL!])
        XCTAssertTrue(store.issues.isEmpty, store.issues.joined(separator: "\n"))
        XCTAssertFalse(store.areas.isEmpty)
        XCTAssertFalse(store.topics.isEmpty)
    }

    func testEveryTopicIsWellFormed() {
        let store = ContentStore(directories: [Bundle.main.resourceURL!])
        for t in store.topics {
            XCTAssertFalse(t.explainer.isEmpty, "\(t.id) has no explainer")
            for q in t.mcqs {
                XCTAssertFalse(q.wrong.contains(q.correct), "\(t.id)/\(q.id) correct answer duplicated")
            }
        }
    }

    func testRichParser() {
        let blocks = RichParser.parse("""
        Intro with $x$ math.

        $$
        a = b
        $$

        - one
        - two
          continued
        1. first
        ```python
        x = 1
        ```
        """)
        XCTAssertEqual(blocks, [
            .paragraph("Intro with $x$ math."),
            .math("a = b"),
            .bullets(["one", "two continued"]),
            .numbered(["first"]),
            .code(language: "python", code: "x = 1"),
        ])
    }
}
