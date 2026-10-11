import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: nil)
let out = dev.makeBuffer(length: 64 << 20, options: .storageModeShared)!
func run(_ name: String, _ opsPerIter: Double, _ n: UInt32, _ threads: Int) {
    let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    var nn = n
    var ts: [Double] = []
    for r in 0..<7 {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0); e.setBytes(&nn, length: 4, index: 1)
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if r > 1 { ts.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = ts.sorted()[ts.count / 2]
    print(String(format: "%-10@ %6.2f ms  %7.1f G islem/s  (%.2f T)", name as NSString, t * 1e3, Double(threads) * Double(n) * opsPerIter / t / 1e9, Double(threads) * Double(n) * opsPerIter / t / 1e12))
}
run("fma_peak", 8, 20000, 1 << 20)
run("half_peak", 8, 20000, 1 << 20)
run("int_peak", 16, 20000, 1 << 20)
run("cvt_peak", 2, 20000, 1 << 20)
