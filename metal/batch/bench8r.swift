import Foundation
import Metal

struct MP { var rows: UInt32; var cols: UInt32; var nblk: UInt32; var seed: UInt32 }
let args = CommandLine.arguments
let reps = args.count >= 3 ? Int(args[2])! : 9
let iters = 3
guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no metal") }
let lib = try dev.makeLibrary(source: try String(contentsOfFile: args[1], encoding: .utf8), options: nil)
var psoCache: [String: MTLComputePipelineState] = [:]
func pso(_ n: String) -> MTLComputePipelineState {
    if let p = psoCache[n] { return p }
    guard let f = lib.makeFunction(name: n) else { fatalError("yok: \(n)") }
    let p = try! dev.makeComputePipelineState(function: f); psoCache[n] = p; return p
}
func xs32(_ s0: UInt32) -> UInt32 { var s = s0; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s }
let seed: UInt32 = 12345
let T = 1 << 29
let maxBlocks = T / 32
let maxCols = 18944
let codes = dev.makeBuffer(length: maxBlocks * 16, options: .storageModeShared)!
let scales = dev.makeBuffer(length: maxBlocks * 2, options: .storageModeShared)!
let seeds = dev.makeBuffer(length: maxBlocks * 16, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: 16 * maxCols * 4, options: .storageModeShared)!
let outs = (0..<8).map { _ in dev.makeBuffer(length: 40 << 20, options: .storageModeShared)! }
func runOnce(_ body: (MTLComputeCommandEncoder) -> Void, n: Int = 1) -> Double {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    for _ in 0..<n { body(e) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status != .completed { fatalError("cb failed: \(String(describing: cb.error))") }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(n)
}
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
_ = runOnce { e in
    var np = SIMD2<UInt32>(UInt32(maxBlocks), seed)
    e.setComputePipelineState(pso("fill_q4"))
    e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
    e.setBytes(&np, length: 8, index: 2)
    e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
_ = runOnce { e in
    var np = SIMD2<UInt32>(UInt32(maxBlocks), 777)
    e.setComputePipelineState(pso("fill_seeds"))
    e.setBuffer(seeds, offset: 0, index: 0)
    e.setBytes(&np, length: 8, index: 1)
    e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
let xp = xbuf.contents().bindMemory(to: Float.self, capacity: 16 * maxCols)
func fillX(_ cols: Int) {
    var st: UInt32 = 987654321
    for i in 0..<(16 * cols) { st = xs32(st); xp[i] = Float(Int32(bitPattern: st)) / 2147483648.0 }
}
typealias Body = (MTLComputeCommandEncoder) -> Void
func timeAll(_ bodies: [Body]) -> [Double] {
    for b in bodies { _ = runOnce(b); _ = runOnce(b) }
    var ts = [[Double]](repeating: [], count: bodies.count)
    for _ in 0..<reps { for i in 0..<bodies.count { ts[i].append(runOnce(bodies[i], n: iters)) } }
    return ts.map { median($0) }
}

// Yalniz zaman olcer (dogrulama bench15.swift icinde). kullanim: ./bench8r kernels.metal reps m0|m1p3|m2|m3 [Bmax]
let rows = 8192, cols = 8192
let Mx = max(1, T / (rows * cols)), nblk = cols / 32
fillX(cols)
let modeName = args.count >= 4 ? args[3] : "m0"
let Bmax = args.count >= 5 ? Int(args[4])! : 16
print("R taramasi (\(modeName)), 8192x8192, ms; '-' = yok")
print("  B |   R=1    R=2    R=4    R=8 | en iyi R, ms, B=1'e oran")
var b1 = 0.0
for B in 1...Bmax {
    var p = MP(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
    var bodies: [Body] = []; var rs: [Int] = []
    for R in [1, 2, 4, 8] {
        if R * B > 64 { continue }
        let ps = pso("m8_b\(B)_r\(R)_\(modeName)")
        let blkBuf = (modeName == "m1p3" || modeName == "m3") ? seeds : codes
        bodies.append({ e in
            e.setComputePipelineState(ps)
            e.setBuffer(blkBuf, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
            e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(outs[0], offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
            e.dispatchThreadgroups(MTLSize(width: rows / (4 * R), height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }); rs.append(R)
    }
    let t = timeAll(bodies)
    var line = String(format: "%3d |", B)
    for R in [1, 2, 4, 8] { if let i = rs.firstIndex(of: R) { line += String(format: " %6.2f", t[i] * 1e3) } else { line += "      -" } }
    let bi = t.indices.min(by: { t[$0] < t[$1] })!
    if B == 1 { b1 = t[bi] }
    line += String(format: " | R=%d %6.2f ms  %.2fx", rs[bi], t[bi] * 1e3, t[bi] / b1)
    print(line)
}
