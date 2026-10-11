import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!
let src = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
do { let lib = try dev.makeLibrary(source: src, options: nil); print("derleme tamam:", lib.functionNames.sorted()) } catch { print("HATA:", error) }
