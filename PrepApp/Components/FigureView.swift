import SwiftUI
import UIKit

/// Content figures are vector PDFs rendered once per theme by tools/make_figures.py and bundled
/// as `<topic>--<name>.<theme>.pdf`. Content refers to them as `![caption](<topic>/<name>)`.
enum FigureStore {
    private static let cache = NSCache<NSString, UIImage>()

    static func url(ref: String, theme: ThemeID) -> URL? {
        let base = ref.replacingOccurrences(of: "/", with: "--")
        let names = ["\(base).\(theme.rawValue)", base]
        for name in names {
            let override = ContentStore.overrideDirectory.appending(path: "\(name).pdf")
            if FileManager.default.fileExists(atPath: override.path) { return override }
            if let url = Bundle.main.url(forResource: name, withExtension: "pdf") { return url }
        }
        return nil
    }

    /// Rasterizes the first PDF page at the given pixel width (cached).
    static func image(ref: String, theme: ThemeID, pixelWidth: CGFloat) -> UIImage? {
        guard let url = url(ref: ref, theme: theme) else { return nil }
        let key = "\(url.path)@\(Int(pixelWidth))" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        guard let doc = CGPDFDocument(url as CFURL), let page = doc.page(at: 1) else { return nil }
        let box = page.getBoxRect(.mediaBox)
        guard box.width > 0 else { return nil }
        let scale = pixelWidth / box.width
        let size = CGSize(width: box.width * scale, height: box.height * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        let img = UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            let c = ctx.cgContext
            c.translateBy(x: 0, y: size.height)
            c.scaleBy(x: scale, y: -scale)
            c.translateBy(x: -box.minX, y: -box.minY)
            c.drawPDFPage(page)
        }
        cache.setObject(img, forKey: key)
        return img
    }
}

/// Inline figure with caption. Tap to open a zoomable full-screen view.
struct FigureView: View {
    @Environment(\.theme) private var theme
    @Environment(\.displayScale) private var displayScale
    let ref: String
    let caption: String
    @State private var zoomed = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let img = FigureStore.image(ref: ref, theme: theme.id, pixelWidth: 420 * displayScale) {
                Button { zoomed = true } label: {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .padding(10)
                        .background(theme.surface, in: RoundedRectangle(cornerRadius: min(theme.radius, 12)))
                        .overlay(alignment: .topTrailing) {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .font(.caption2).foregroundStyle(theme.textMuted).padding(8)
                        }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(caption.isEmpty ? "Figure" : caption)
                .accessibilityHint("Opens the figure full screen")
            } else {
                Label("Missing figure \(ref)", systemImage: "photo")
                    .font(theme.font(.caption)).foregroundStyle(theme.warning)
            }
            if !caption.isEmpty {
                InlineText(caption, role: .caption)
                    .foregroundStyle(theme.textSecondary)
            }
        }
        .fullScreenCover(isPresented: $zoomed) {
            FigureZoomView(ref: ref, caption: caption)
        }
    }
}

private struct FigureZoomView: View {
    @Environment(\.theme) private var theme
    @Environment(\.dismiss) private var dismiss
    @Environment(\.displayScale) private var displayScale
    let ref: String
    let caption: String

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Spacer()
                Button { dismiss() } label: {
                    Image(systemName: "xmark").font(.headline).foregroundStyle(theme.textSecondary).padding(8)
                }
                .accessibilityLabel("Close")
            }
            if let img = FigureStore.image(ref: ref, theme: theme.id, pixelWidth: 1400 * displayScale / 2) {
                ZoomableImage(image: img)
            }
            if !caption.isEmpty {
                InlineText(caption, role: .callout).foregroundStyle(theme.textSecondary)
            }
        }
        .padding()
        .background(theme.bg.ignoresSafeArea())
    }
}

/// UIScrollView-backed pinch/double-tap zoom.
private struct ZoomableImage: UIViewRepresentable {
    let image: UIImage

    func makeUIView(context: Context) -> UIScrollView {
        let scroll = UIScrollView()
        scroll.minimumZoomScale = 1
        scroll.maximumZoomScale = 5
        scroll.showsHorizontalScrollIndicator = false
        scroll.showsVerticalScrollIndicator = false
        scroll.delegate = context.coordinator
        let iv = UIImageView(image: image)
        iv.contentMode = .scaleAspectFit
        iv.translatesAutoresizingMaskIntoConstraints = false
        scroll.addSubview(iv)
        NSLayoutConstraint.activate([
            iv.widthAnchor.constraint(equalTo: scroll.frameLayoutGuide.widthAnchor),
            iv.heightAnchor.constraint(equalTo: scroll.frameLayoutGuide.heightAnchor),
            iv.leadingAnchor.constraint(equalTo: scroll.contentLayoutGuide.leadingAnchor),
            iv.trailingAnchor.constraint(equalTo: scroll.contentLayoutGuide.trailingAnchor),
            iv.topAnchor.constraint(equalTo: scroll.contentLayoutGuide.topAnchor),
            iv.bottomAnchor.constraint(equalTo: scroll.contentLayoutGuide.bottomAnchor),
        ])
        context.coordinator.imageView = iv
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.doubleTap(_:)))
        tap.numberOfTapsRequired = 2
        scroll.addGestureRecognizer(tap)
        return scroll
    }

    func updateUIView(_ scroll: UIScrollView, context: Context) {
        context.coordinator.imageView?.image = image
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        weak var imageView: UIImageView?
        func viewForZooming(in scrollView: UIScrollView) -> UIView? { imageView }
        @objc func doubleTap(_ g: UITapGestureRecognizer) {
            guard let scroll = g.view as? UIScrollView else { return }
            if scroll.zoomScale > 1 {
                scroll.setZoomScale(1, animated: true)
            } else {
                let p = g.location(in: imageView)
                scroll.zoom(to: CGRect(x: p.x - 60, y: p.y - 60, width: 120, height: 120), animated: true)
            }
        }
    }
}
