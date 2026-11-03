import AppKit
import PDFKit
let doc = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1]))!
for i in stride(from: 2, to: CommandLine.arguments.count, by: 2) {
 let page = doc.page(at: Int(CommandLine.arguments[i])! - 1)!
 let bounds = page.bounds(for: .mediaBox)
 let width = 1200
 let height = Int(CGFloat(width) * bounds.height / bounds.width)
 let image = NSImage(size: NSSize(width: width, height: height))
 image.lockFocus()
 NSColor.white.setFill(); NSRect(x: 0, y: 0, width: width, height: height).fill()
 let context = NSGraphicsContext.current!.cgContext
 context.scaleBy(x: CGFloat(width) / bounds.width, y: CGFloat(height) / bounds.height)
 page.draw(with: .mediaBox, to: context)
 image.unlockFocus()
 let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
 try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[i + 1]))
}
