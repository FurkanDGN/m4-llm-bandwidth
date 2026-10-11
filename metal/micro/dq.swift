import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: nil)
let out = dev.makeBuffer(length: 64 << 20, options: .storageModeShared)!
let xb = dev.makeBuffer(length: 64, options: .storageModeShared)!
do { let p = xb.contents().bindMemory(to: Float.self, capacity: 16); for i in 0..<16 { p[i] = Float(i + 1) * 0.0625 } }
func run(_ name: String) {
    let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    var nn: UInt32 = 4000
    let threads = 1 << 20
    var ts: [Double] = []
    for r in 0..<7 {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0); e.setBuffer(xb, offset: 0, index: 1); e.setBytes(&nn, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if r > 1 { ts.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = ts.sorted()[ts.count / 2]
    let wps = Double(threads) * Double(nn) * 32 / t / 1e9
    print(String(format: "%-10@ %7.2f ms  %7.1f G agirlik/s  (agirlik basina %.2f ns/G... = %.2f 'FMA-yuvasi' @2.76T)", name as NSString, t * 1e3, wps, 1000 / wps, 2760 / wps))
}
for n in ["dq_base", "dq_cvt", "dq_magicf", "dq_magich", "dq_halfall", "dq_lut"] { run(n) }
