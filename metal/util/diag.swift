import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!
let lib = try dev.makeLibrary(source: try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: nil)
let pat = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : ""
for n in lib.functionNames.sorted(by: { $0.count == $1.count ? $0 < $1 : $0.count < $1.count }) where n.contains(pat) {
    let p = try dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
    print(String(format: "%-18@ maxThreads/TG=%4d  simdWidth=%d  tgMem=%d", n as NSString, p.maxTotalThreadsPerThreadgroup, p.threadExecutionWidth, p.staticThreadgroupMemoryLength))
}
