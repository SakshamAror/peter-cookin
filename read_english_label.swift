import Foundation
import ImageIO
import Vision

// Read one base64 JPEG per input line and return recognized English lines as JSON.
while let line = readLine() {
    guard let imageData = Data(base64Encoded: line),
          let source = CGImageSourceCreateWithData(imageData as CFData, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        FileHandle.standardOutput.write(Data("[]\n".utf8))
        continue
    }

    let orientations: [CGImagePropertyOrientation] = [.up, .down, .left, .right]
    var bestLines: [String] = []
    var bestScore = 0

    for orientation in orientations {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = ["en-US"]

        do {
            let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
            try handler.perform([request])
            let lines = (request.results ?? []).compactMap { observation in
                observation.topCandidates(1).first?.string
            }
            let score = lines.joined(separator: " ").filter { $0.isLetter || $0.isNumber }.count
            if score > bestScore {
                bestScore = score
                bestLines = lines
            }
        } catch {
            continue
        }
    }

    if let output = try? JSONSerialization.data(withJSONObject: bestLines, options: [.fragmentsAllowed]) {
        FileHandle.standardOutput.write(output)
        FileHandle.standardOutput.write(Data([0x0A]))
    } else {
        FileHandle.standardOutput.write(Data("[]\n".utf8))
    }
}
