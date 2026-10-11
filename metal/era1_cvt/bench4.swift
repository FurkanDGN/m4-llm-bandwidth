import Foundation
import Metal

struct Params { var rows: UInt32; var cols: UInt32; var nblk: UInt32; var seed: UInt32 }
let args = CommandLine.arguments
let reps = args.count >= 3 ? Int(args[2])! : 15
let iters = 4
guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no metal") }
let lib = try dev.makeLibrary(source: try String(contentsOfFile: args[1], encoding: .utf8), options: nil)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let seed: UInt32 = 12345
func xs32(_ s0: UInt32) -> UInt32 { var s = s0; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s }
let T = 1 << 29
let maxBlocks = T / 32
let maxCols = 18944
let codes = dev.makeBuffer(length: maxBlocks * 32, options: .storageModeShared)!
let scales = dev.makeBuffer(length: maxBlocks * 2, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: maxCols * 4, options: .storageModeShared)!
let yr = dev.makeBuffer(length: 4 << 20, options: .storageModeShared)!
let yv = dev.makeBuffer(length: 4 << 20, options: .storageModeShared)!
do {
    let xp = xbuf.contents().bindMemory(to: Float.self, capacity: maxCols)
    var st: UInt32 = 987654321
    for i in 0..<maxCols { st = xs32(st); xp[i] = Float(Int32(bitPattern: st)) / 2147483648.0 }
}
func runOnce(_ body: (MTLComputeCommandEncoder) -> Void, n: Int = 1) -> Double {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    for _ in 0..<n { body(e) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status != .completed { fatalError("cb failed") }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(n)
}
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
let fmts: [(name: String, id: Int, key: String, bytesPerBlock: Double, rpt: Int)] = [
    ("q4_0 benzeri", 0, "q4", 18, 32), ("q8_0 benzeri", 1, "q8", 34, 16)]
let fills = ["fill_q4", "fill_q8"]
let shapes: [(Int, Int)] = [(1024, 1024), (4096, 4096), (8192, 8192), (18944, 3584), (3584, 18944)]
for f in fmts {
    _ = runOnce { e in
        var np = SIMD2<UInt32>(UInt32(maxBlocks), seed)
        e.setComputePipelineState(pso(fills[f.id]))
        e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
        e.setBytes(&np, length: 8, index: 2)
        e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    print("\n=== \(f.name): satirlarin G/4'u tohumdan uretilir, gerisi bellekten okunur (sanal GB/s, ms, G=0'a gore en buyuk goreli hata) ===")
    print("boyut          |   G=0 (hepsi okuma)       |   G=1                     |   G=2                     |   G=3                     |   G=4 (hepsi uretim)")
    for (rows, cols) in shapes {
        let B = max(1, T / (rows * cols)), nblk = cols / 32
        let totalBlocks = B * rows * nblk
        var p = Params(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
        var line = String(format: "%-14@ |", "\(rows)x\(cols)" as NSString)
        let ps: [MTLComputePipelineState] = (0...4).map { pso("mv_\(f.key)_g\($0)") }
        var times = [[Double]](repeating: [], count: 5)
        func body(_ g: Int, _ out: MTLBuffer) -> (MTLComputeCommandEncoder) -> Void {
            return { e in
                e.setComputePipelineState(ps[g])
                e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
                e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(out, offset: 0, index: 3)
                e.setBytes(&p, length: MemoryLayout<Params>.size, index: 4)
                e.dispatchThreadgroups(MTLSize(width: rows / f.rpt, height: B, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        for g in 0...4 { _ = runOnce(body(g, g == 0 ? yr : yv)) }
        for _ in 0..<reps { for g in 0...4 { times[g].append(runOnce(body(g, g == 0 ? yr : yv), n: iters)) } }
        for g in 0...4 {
            _ = runOnce(body(g, g == 0 ? yr : yv))
            var ok = "ref"
            if g > 0 {
                let n = B * rows
                let a = yr.contents().bindMemory(to: Float.self, capacity: n), b = yv.contents().bindMemory(to: Float.self, capacity: n)
                var md: Float = 0, ma: Float = 0
                for i in 0..<n { md = max(md, abs(a[i] - b[i])); ma = max(ma, abs(a[i])) }
                ok = String(format: "%.0e", Double(md / ma))
            }
            let t = median(times[g])
            line += String(format: " %6.1f GB/s %5.2f ms %@ |", Double(totalBlocks) * f.bytesPerBlock / t / 1e9, t * 1e3, ok as NSString)
        }
        print(line)
    }
}
