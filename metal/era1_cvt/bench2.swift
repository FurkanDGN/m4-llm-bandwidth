import Foundation
import Metal

struct Params { var rows: UInt32; var cols: UInt32; var nblk: UInt32; var seed: UInt32 }

let args = CommandLine.arguments
guard args.count >= 2 else { fputs("usage: bench2 kernels.metal [reps] [decomp|hybrid]\n", stderr); exit(2) }
let reps = args.count >= 3 ? Int(args[2])! : 15
let iters = 3
guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no metal") }
let src = try String(contentsOfFile: args[1], encoding: .utf8)
let lib = try dev.makeLibrary(source: src, options: nil)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
func psoVar(_ mode: UInt32, _ g: UInt32) -> MTLComputePipelineState {
    let cv = MTLFunctionConstantValues()
    var m = mode, gg = g
    cv.setConstantValue(&m, type: .uint, index: 0)
    cv.setConstantValue(&gg, type: .uint, index: 1)
    return try! dev.makeComputePipelineState(function: lib.makeFunction(name: "mv_var", constantValues: cv))
}
let pFill = pso("fill_codes")
let seed: UInt32 = 12345
func xs32(_ s0: UInt32) -> UInt32 { var s = s0; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s }

let sizes: [(Int, Int)] = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (18944, 3584), (3584, 18944)]
let target = 1 << 31
func batchOf(_ r: Int, _ c: Int) -> Int { max(1, target / (r * c)) }
let maxBlocks = sizes.map { batchOf($0.0, $0.1) * $0.0 * $0.1 / 32 }.max()!
let maxCols = sizes.map { $0.1 }.max()!
let maxY = sizes.map { batchOf($0.0, $0.1) * $0.0 }.max()!
let codes = dev.makeBuffer(length: maxBlocks * 16, options: .storageModeShared)!
let scales = dev.makeBuffer(length: maxBlocks * 2, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: maxCols * 4, options: .storageModeShared)!
let yref = dev.makeBuffer(length: maxY * 4, options: .storageModeShared)!
let yv = dev.makeBuffer(length: maxY * 4, options: .storageModeShared)!
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
    if cb.status != .completed { fatalError("cb failed: \(String(describing: cb.error))") }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(n)
}
_ = runOnce { e in
    var np = SIMD2<UInt32>(UInt32(maxBlocks), seed)
    e.setComputePipelineState(pFill)
    e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
    e.setBytes(&np, length: 8, index: 2)
    e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }

func timeVar(_ ps: MTLComputePipelineState, rows: Int, cols: Int, out: MTLBuffer) -> Double {
    let B = batchOf(rows, cols), nblk = cols / 32
    var p = Params(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
    let body: (MTLComputeCommandEncoder) -> Void = { e in
        e.setComputePipelineState(ps)
        e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
        e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(out, offset: 0, index: 3)
        e.setBytes(&p, length: MemoryLayout<Params>.size, index: 4)
        e.dispatchThreadgroups(MTLSize(width: rows / 32, height: B, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    for _ in 0..<2 { _ = runOnce(body) }
    var ts: [Double] = []
    for _ in 0..<reps { ts.append(runOnce(body, n: iters)) }
    return median(ts)
}

let mode = args.count >= 4 ? args[3] : "decomp"
if mode == "decomp" {
    print("--- Parcalama: 8192x8192, B=32, 2^31 agirlik/dispatch ---")
    let variants: [(String, UInt32, UInt32)] = [
        ("okuma           (G=0)", 0, 0),
        ("okuma, x yazmacta (G=0)", 3, 0),
        ("okuma, olcek yok (G=0)", 4, 0),
        ("uretim          (G=4)", 0, 4),
        ("uretim, PRNG yok (G=4)", 1, 4),
        ("yalniz PRNG      (G=4)", 2, 4),
        ("uretim, x yazmacta (G=4)", 3, 4),
        ("uretim, olcek yok (G=4)", 4, 4),
    ]
    for (name, m, g) in variants {
        let t = timeVar(psoVar(m, g), rows: 8192, cols: 8192, out: yv)
        let w = Double(target)
        print(String(format: "%-26@ %6.2f ms  %6.1f G agirlik/s  (sanal %.1f GB/s)", name as NSString, t * 1e3, w / t / 1e9, w / 32 * 18 / t / 1e9))
    }
} else {
    print("--- Karisik: G/4 satir tohumdan uretilir, gerisi bellekten okunur; tum boyutlar ---")
    print(String(format: "%-11@ | %@", "boyut" as NSString, "G=0 (okuma) | G=1 | G=2 | G=3 | G=4 (uretim)   [sanal GB/s, 'ok' = y G=0 ile ayni mi]" as NSString))
    for (rows, cols) in sizes {
        let B = batchOf(rows, cols), nblk = cols / 32
        let totalBlocks = B * rows * nblk
        var line = String(format: "%-11@ |", "\(rows)x\(cols)" as NSString)
        for g in 0...4 {
            let ps = psoVar(0, UInt32(g))
            let out = g == 0 ? yref : yv
            let t = timeVar(ps, rows: rows, cols: cols, out: out)
            var ok = "ref"
            if g > 0 {
                let a = yref.contents().bindMemory(to: Float.self, capacity: B * rows)
                let b = yv.contents().bindMemory(to: Float.self, capacity: B * rows)
                var md: Float = 0
                for i in 0..<(B * rows) { md = max(md, abs(a[i] - b[i])) }
                ok = md < 1e-4 ? "ok" : String(format: "FARK %.1e", md)
            }
            line += String(format: " %6.1f (%5.2f ms, %@) |", Double(totalBlocks) * 18 / t / 1e9, t * 1e3, ok as NSString)
        }
        print(line)
    }
}
