import SwiftUI

/// Anything that takes over the full screen: lessons, reviews, exams.
enum ActiveSession: Identifiable {
    case lesson(Topic)
    case explainer(Topic)
    case review([StudyItem], source: ReviewSource, title: String)
    case exam(ExamConfig)

    var id: String {
        switch self {
        case .lesson(let t): "lesson-\(t.id)"
        case .explainer(let t): "explainer-\(t.id)"
        case .review(let items, let s, _): "review-\(s.rawValue)-\(items.count)-\(items.first?.id ?? "")"
        case .exam(let c): "exam-\(c.id)"
        }
    }
}

extension View {
    func sessionCover(_ session: Binding<ActiveSession?>, onDismiss: (() -> Void)? = nil) -> some View {
        fullScreenCover(item: session, onDismiss: onDismiss) { s in
            switch s {
            case .lesson(let t): LessonScreen(topic: t)
            case .explainer(let t): LessonScreen(topic: t, explainerOnly: true)
            case .review(let items, let source, let title): ReviewScreen(items: items, source: source, title: title)
            case .exam(let c): ExamScreen(config: c)
            }
        }
    }
}
