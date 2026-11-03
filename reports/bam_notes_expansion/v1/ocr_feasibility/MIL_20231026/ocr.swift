import Foundation
import Vision
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesCPUOnly = true
request.usesLanguageCorrection = false
request.recognitionLanguages = ["en-US"]
let handler = VNImageRequestHandler(url: URL(fileURLWithPath: CommandLine.arguments[1]))
do { try handler.perform([request]) } catch {
 let e = error as NSError
 FileHandle.standardError.write(("Vision error: \(e.domain) \(e.code) \(e.userInfo)\n").data(using: .utf8)!)
 exit(2)
}
let rows: [[String: Any]] = (request.results ?? []).map { r in
 let c = r.topCandidates(1).first!
 return ["text":c.string,"confidence":Double(c.confidence),"x":Double(r.boundingBox.minX),"y":Double(r.boundingBox.minY),"width":Double(r.boundingBox.width),"height":Double(r.boundingBox.height)]
}
let data = try JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys, .prettyPrinted])
FileHandle.standardOutput.write(data)
