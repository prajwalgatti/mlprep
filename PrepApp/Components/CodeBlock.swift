import SwiftUI

/// Monospaced code with light, regex-based Python-ish highlighting. Scrolls horizontally.
struct CodeBlock: View {
    @Environment(\.theme) private var theme
    let code: String
    var language: String = ""

    @State private var contentWidth: CGFloat = 0
    @State private var viewWidth: CGFloat = 0
    private var overflows: Bool { contentWidth > viewWidth + 1 && viewWidth > 0 }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: min(theme.radius, 10), style: .continuous)
        ScrollView(.horizontal, showsIndicators: false) {
            Text(Self.highlight(code, theme: theme))
                .font(theme.font(.mono))
                .foregroundStyle(theme.text)
                .textSelection(.enabled)
                .padding(12)
                .padding(.top, language.isEmpty ? 0 : 6)
                .padding(.trailing, overflows ? 20 : 0)
                .background(GeometryReader { g in
                    Color.clear.onAppear { contentWidth = g.size.width }.onChange(of: g.size.width) { contentWidth = $1 }
                })
        }
        .scrollBounceBehavior(.basedOnSize, axes: .horizontal)
        // Long lines fade out at the trailing edge instead of being cut mid-token.
        .mask {
            HStack(spacing: 0) {
                Rectangle()
                if overflows {
                    LinearGradient(colors: [.black, .black.opacity(0.05)], startPoint: .leading, endPoint: .trailing).frame(width: 32)
                }
            }
        }
        .background(GeometryReader { g in
            Color.clear.onAppear { viewWidth = g.size.width }.onChange(of: g.size.width) { viewWidth = $1 }
        })
        .background(theme.codeBg, in: shape)
        .overlay(shape.strokeBorder(theme.border, lineWidth: 1))
        .overlay(alignment: .topTrailing) {
            if !language.isEmpty {
                Text(theme.labelText(language))
                    .font(theme.font(.label))
                    .foregroundStyle(theme.textMuted)
                    .padding(6)
            }
        }
    }

    private static let keywords = [
        "def", "return", "class", "import", "from", "as", "for", "in", "if", "else", "elif", "while",
        "with", "None", "True", "False", "and", "or", "not", "lambda", "self", "super", "yield", "is",
    ]

    static func highlight(_ code: String, theme: Theme) -> AttributedString {
        var out = AttributedString(code)
        let ns = code as NSString

        func paint(_ pattern: String, _ color: Color) {
            guard let re = try? NSRegularExpression(pattern: pattern) else { return }
            for m in re.matches(in: code, range: NSRange(location: 0, length: ns.length)) {
                guard let r = Range(m.range, in: code),
                      let lo = AttributedString.Index(r.lowerBound, within: out),
                      let hi = AttributedString.Index(r.upperBound, within: out) else { continue }
                out[lo..<hi].foregroundColor = color
            }
        }

        paint("\\b(" + keywords.joined(separator: "|") + ")\\b", theme.codeKeyword)
        paint("\\b\\d+(\\.\\d+)?(e-?\\d+)?\\b", theme.codeNumber)
        paint("\\b(torch|F|nn|np|jnp|jax)\\b", theme.codeLib)
        paint("(\"[^\"\\n]*\"|'[^'\\n]*')", theme.codeString)
        paint("#[^\\n]*", theme.codeComment)   // comments last so they win
        return out
    }
}
