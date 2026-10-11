import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: nil)
let out = dev.makeBuffer(length: 64 << 20, options: .storageModeShared)!
func run(_ name: String, _ flopsPerIterPerThread: Double, _ n: UInt32, _ threads: Int, _ tg: Int) {
    let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    var nn = n
    var ts: [Double] = []
    for r in 0..<7 {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0); e.setBytes(&nn, length: 4, index: 1)
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, p.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if r > 1 { ts.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = ts.sorted()[ts.count / 2]
    print(String(format: "%-10@ tg=%4d thr=%8d  %7.2f ms  %6.2f TFLOPS", name as NSString, tg, threads, t * 1e3, Double(threads) * Double(n) * flopsPerIterPerThread / t / 1e12))
}
for tg in [128, 256, 512, 1024] {
    run("fma8", 16, 20000, 1 << 20, tg)
    run("fma32", 64, 10000, 1 << 20, tg)
    run("fma32v", 8 * 4 * 2, 10000, 1 << 18, tg)
}
for tg in [128, 256, 512, 1024] {
    run("mma_f32acc", 8 * 1024.0 / 32, 20000, 1 << 20, tg)
    run("mma_hacc", 8 * 1024.0 / 32, 20000, 1 << 20, tg)
}
