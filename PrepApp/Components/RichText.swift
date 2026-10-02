import LaTeXSwiftUI
import SwiftUI

/// Block-level structure of a content body (the inline parts are rendered by `InlineText`).
enum RichBlock: Hashable {
    case paragraph(String)
    case math(String)                 // display math, without the $$ delimiters
    case bullets([String])
    case numbered([String])
    case code(language: String, code: String)
    case figure(ref: String, caption: String)   // `![caption](topic/name)` on its own line
}

enum RichParser {
    static func parse(_ source: String) -> [RichBlock] {
        var blocks: [RichBlock] = []
        var para: [String] = []
        var list: [String] = []
        var listNumbered = false
        var lines = source.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")[...]

        func flushPara() {
            if !para.isEmpty { blocks.append(.paragraph(para.joined(separator: " "))); para = [] }
        }
        func flushList() {
            if !list.isEmpty { blocks.append(listNumbered ? .numbered(list) : .bullets(list)); list = [] }
        }

        while let raw = lines.popFirst() {
            let line = raw.trimmingCharacters(in: .whitespaces)

            if line.hasPrefix("```") {
                flushPara(); flushList()
                let lang = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                while let l = lines.popFirst(), !l.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(l)
                }
                blocks.append(.code(language: lang, code: dedent(code).joined(separator: "\n")))
                continue
            }

            if line.hasPrefix("$$") {
                flushPara(); flushList()
                var body = String(line.dropFirst(2))
                if let end = body.range(of: "$$") {
                    blocks.append(.math(String(body[..<end.lowerBound])))
                    let rest = body[end.upperBound...].trimmingCharacters(in: .whitespaces)
                    if !rest.isEmpty { para.append(rest) }
                    continue
                }
                while let l = lines.popFirst() {
                    if let end = l.range(of: "$$") {
                        body += "\n" + l[..<end.lowerBound]
                        break
                    }
                    body += "\n" + l
                }
                blocks.append(.math(body.trimmingCharacters(in: .whitespacesAndNewlines)))
                continue
            }

            if line.isEmpty {
                flushPara(); flushList()
                continue
            }

            if let fig = figure(line) {
                flushPara(); flushList()
                blocks.append(fig)
                continue
            }

            if let item = bulletItem(line) {
                flushPara()
                if !list.isEmpty && listNumbered { flushList() }
                listNumbered = false
                list.append(item)
                continue
            }
            if let item = numberedItem(line) {
                flushPara()
                if !list.isEmpty && !listNumbered { flushList() }
                listNumbered = true
                list.append(item)
                continue
            }

            // Indented continuation of a list item.
            if !list.isEmpty, raw.hasPrefix("  ") {
                list[list.count - 1] += " " + line
                continue
            }
            flushList()
            para.append(line)
        }
        flushPara(); flushList()
        return blocks
    }

    /// `![caption](ref)` occupying the whole line.
    private static func figure(_ line: String) -> RichBlock? {
        guard line.hasPrefix("!["), line.hasSuffix(")"),
              let mid = line.range(of: "]("), mid.lowerBound > line.index(line.startIndex, offsetBy: 1) else { return nil }
        let caption = String(line[line.index(line.startIndex, offsetBy: 2)..<mid.lowerBound])
        let ref = String(line[mid.upperBound..<line.index(before: line.endIndex)]).trimmingCharacters(in: .whitespaces)
        return ref.isEmpty ? nil : .figure(ref: ref, caption: caption)
    }

    private static func bulletItem(_ line: String) -> String? {
        for p in ["- ", "* ", "• "] where line.hasPrefix(p) { return String(line.dropFirst(p.count)) }
        return nil
    }

    private static func numberedItem(_ line: String) -> String? {
        guard let dot = line.firstIndex(where: { $0 == "." || $0 == ")" }),
              dot > line.startIndex, line[..<dot].allSatisfy(\.isNumber),
              line.index(after: dot) < line.endIndex, line[line.index(after: dot)] == " " else { return nil }
        return String(line[line.index(dot, offsetBy: 2)...])
    }

    private static func dedent(_ lines: [String]) -> [String] {
        let indents = lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { $0.prefix(while: { $0 == " " }).count }
        let n = indents.min() ?? 0
        return lines.map { String($0.dropFirst(min(n, $0.prefix(while: { $0 == " " }).count))) }
    }
}

/// Renders a content body: paragraphs, lists, display math and code blocks.
struct RichText: View {
    @Environment(\.theme) private var theme
    let blocks: [RichBlock]
    var role: TextRole
    var color: Color?
    var spacing: CGFloat

    init(_ source: String, role: TextRole = .body, color: Color? = nil, spacing: CGFloat = 10) {
        self.blocks = RichParser.parse(source)
        self.role = role
        self.color = color
        self.spacing = spacing
    }

    private var bullet: String { theme.id == .terminal ? ">" : "•" }

    var body: some View {
        VStack(alignment: .leading, spacing: spacing) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .paragraph(let s):
                    InlineText(s, role: role)
                case .math(let m):
                    MathBlock(m, role: role)
                case .bullets(let items):
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text(bullet).font(theme.font(role)).foregroundStyle(theme.label)
                                InlineText(item, role: role)
                            }
                        }
                    }
                case .numbered(let items):
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(items.enumerated()), id: \.offset) { i, item in
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text("\(i + 1).").font(theme.font(.mono)).foregroundStyle(theme.label)
                                InlineText(item, role: role)
                            }
                        }
                    }
                case .code(let lang, let code):
                    CodeBlock(code: code, language: lang)
                case .figure(let ref, let caption):
                    FigureView(ref: ref, caption: caption)
                }
            }
        }
        .foregroundStyle(color ?? theme.text)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// One paragraph of inline markdown, using LaTeX rendering only when it contains math.
struct InlineText: View {
    @Environment(\.theme) private var theme
    let text: String
    var role: TextRole

    init(_ text: String, role: TextRole = .body) {
        self.text = text
        self.role = role
    }

    static func containsMath(_ s: String) -> Bool {
        s.contains("$") || s.contains("\\(") || s.contains("\\[")
    }

    var body: some View {
        Group {
            if Self.containsMath(text) {
                LaTeX(text)
                    .font(theme.uiFont(role))
                    .blockMode(.blockViews)
                    .renderingStyle(.redactedOriginal)
                    .renderingAnimation(.easeIn(duration: 0.15))
            } else {
                Text(LocalizedStringKey(text)).font(theme.font(role))
            }
        }
        .lineSpacing(2)
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct MathBlock: View {
    @Environment(\.theme) private var theme
    let latex: String
    var role: TextRole

    init(_ latex: String, role: TextRole = .body) {
        self.latex = latex
        self.role = role
    }

    @State private var naturalWidth: CGFloat = 0
    @State private var available: CGFloat = 0
    @State private var height: CGFloat = 0

    /// Shrink slightly-too-wide equations to fit; scroll anything wider than this.
    private static let minScale: CGFloat = 0.78
    private static let fadeWidth: CGFloat = 36

    private var fitScale: CGFloat {
        guard naturalWidth > 0, available > 0, naturalWidth > available else { return 1 }
        return available / naturalWidth
    }
    private var scrolls: Bool { fitScale < Self.minScale }
    private var scales: Bool { fitScale < 1 && !scrolls }

    private func equation(_ mode: LaTeX.BlockMode) -> some View {
        LaTeX("$$" + latex + "$$")
            .font(theme.uiFont(role))
            .blockMode(mode)
            .renderingStyle(.redactedOriginal)
            .renderingAnimation(.easeIn(duration: 0.15))
    }

    var body: some View {
        // Fits → centered. Slightly too wide → scaled down to fit. Much too wide → LaTeXSwiftUI's own
        // horizontal scroller, with a trailing fade and chevron so it never looks clipped mid-symbol.
        Group {
            if scales {
                equation(.blockViews)
                    .frame(width: naturalWidth)
                    .background(GeometryReader { g in Color.clear.preference(key: MathHeightKey.self, value: g.size.height) })
                    .onPreferenceChange(MathHeightKey.self) { height = $0 }
                    .scaleEffect(fitScale, anchor: .center)
                    .frame(width: available, height: height > 0 ? height * fitScale : nil)
            } else {
                equation(.blockViews)
                    .contentMargins(.trailing, scrolls ? Self.fadeWidth : 0, for: .scrollContent)
                    .mask {
                        HStack(spacing: 0) {
                            Rectangle()
                            if scrolls {
                                LinearGradient(colors: [.black, .black.opacity(0)], startPoint: .leading, endPoint: .trailing)
                                    .frame(width: Self.fadeWidth)
                            }
                        }
                    }
                    .overlay(alignment: .trailing) {
                        if scrolls {
                            Image(systemName: "chevron.right")
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(theme.textMuted)
                                .allowsHitTesting(false)
                        }
                    }
            }
        }
        // Natural width, measured off-screen on a single line.
        .background(alignment: .topLeading) {
            equation(.alwaysInline)
                .fixedSize()
                .hidden()
                .background(GeometryReader { g in Color.clear.preference(key: MathSizeKey.self, value: g.size) })
                .accessibilityHidden(true)
        }
        .background(GeometryReader { g in Color.clear.preference(key: MathWidthKey.self, value: g.size.width) })
        .onPreferenceChange(MathSizeKey.self) { naturalWidth = $0.width }
        .onPreferenceChange(MathWidthKey.self) { available = $0 }
        .clipped()
        .padding(.vertical, 4)
        .frame(maxWidth: .infinity)
        .accessibilityHint(scrolls ? "Swipe sideways to see the whole equation" : "")
    }
}

private struct MathHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

private struct MathSizeKey: PreferenceKey {
    static let defaultValue: CGSize = .zero
    static func reduce(value: inout CGSize, nextValue: () -> CGSize) {
        let n = nextValue()
        if n != .zero { value = n }
    }
}

private struct MathWidthKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}
