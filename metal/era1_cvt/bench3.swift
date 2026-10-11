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
func genWords(_ g: UInt32, _ nw: Int, _ fmt: Int) -> [UInt32] {
    var s = (g &+ seed) &* 0x9E3779B1; s ^= s >> 15; s |= 1
    var w = [UInt32](repeating: 0, count: nw)
    for i in 0..<nw { s = xs32(s); w[i] = fmt == 2 ? ((s & 0x83FF83FF) | 0x38003800) : s }
    return w
}

let T = 1 << (args.count >= 4 ? Int(args[3])! : 29)
let maxBlocks = T / 32
let maxCols = 18944
let codes = dev.makeBuffer(length: maxBlocks * (args.count >= 5 ? Int(args[4])! * 4 : 64), options: .storageModeShared)!
let scales = dev.makeBuffer(length: maxBlocks * 2, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: maxCols * 4, options: .storageModeShared)!
let y1 = dev.makeBuffer(length: 4 << 20, options: .storageModeShared)!
let y2 = dev.makeBuffer(length: 4 << 20, options: .storageModeShared)!
let sout = dev.makeBuffer(length: (1 << 20) * 4, options: .storageModeShared)!
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
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }

let fmts: [(name: String, id: Int, nw: Int, bytesPerBlock: Double, rpt: Int)] = [
    ("q4_0 benzeri (0.5625 B/agirlik)", 0, 4, 18, 32),
    ("q8_0 benzeri (1.0625 B/agirlik)", 1, 8, 34, 16),
    ("fp16 benzeri (2 B/agirlik)", 2, 16, 64, 16),
]
let fillNames = ["fill_q4", "fill_q8", "fill_f16"]
let shapes: [(Int, Int)] = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (18944, 3584), (3584, 18944)]

for f in fmts {
    _ = runOnce { e in
        var np = SIMD2<UInt32>(UInt32(maxBlocks), seed)
        e.setComputePipelineState(pso(fillNames[f.id]))
        e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
        e.setBytes(&np, length: 8, index: 2)
        e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    let usedBytes = maxBlocks * f.nw * 4
    var nU4 = UInt32(usedBytes / 16 / (4 << 20) * (4 << 20))
    do {
        let nt = 1 << 20
        let ps = pso("stream_read")
        let body: (MTLComputeCommandEncoder) -> Void = { e in
            e.setComputePipelineState(ps)
            e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(sout, offset: 0, index: 1)
            e.setBytes(&nU4, length: 4, index: 2)
            e.dispatchThreads(MTLSize(width: nt, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        for _ in 0..<2 { _ = runOnce(body) }
        var ts: [Double] = []
        for _ in 0..<reps { ts.append(runOnce(body, n: iters)) }
        let gbs = Double(nU4) * 16 / median(ts) / 1e9
        print("\n=== \(f.name) ===")
        print(String(format: "duz okuma tavani (%.0f MB): %.1f GB/s  -> bellekten en fazla %.0f G agirlik/s", Double(nU4) * 16 / 1e6, gbs, gbs / f.bytesPerBlock * 32))
    }
    for (rows, cols) in shapes {
        let B = max(1, T / (rows * cols)), nblk = cols / 32
        let totalBlocks = B * rows * nblk
        var p = Params(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
        let grid = MTLSize(width: rows / f.rpt, height: B, depth: 1)
        let tgs = MTLSize(width: 128, height: 1, depth: 1)
        let names = ["q4", "q8", "f16"]
        let pr = pso("mv_\(names[f.id])_g0"), pg = pso("mv_\(names[f.id])_g4")
        let b1: (MTLComputeCommandEncoder) -> Void = { e in
            e.setComputePipelineState(pr)
            e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
            e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(y1, offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<Params>.size, index: 4)
            e.dispatchThreadgroups(grid, threadsPerThreadgroup: tgs)
        }
        let b2: (MTLComputeCommandEncoder) -> Void = { e in
            e.setComputePipelineState(pg)
            e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
            e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(y2, offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<Params>.size, index: 4)
            e.dispatchThreadgroups(grid, threadsPerThreadgroup: tgs)
        }
        for _ in 0..<2 { _ = runOnce(b1); _ = runOnce(b2) }
        var t1: [Double] = [], t2: [Double] = []
        for _ in 0..<reps { t1.append(runOnce(b1, n: iters)); t2.append(runOnce(b2, n: iters)) }
        let n = B * rows
        let a = y1.contents().bindMemory(to: Float.self, capacity: n)
        let b = y2.contents().bindMemory(to: Float.self, capacity: n)
        var maxd: Float = 0, maxa: Float = 0
        for i in 0..<n { maxd = max(maxd, abs(a[i] - b[i])); maxa = max(maxa, abs(a[i])) }
        let xp = xbuf.contents().bindMemory(to: Float.self, capacity: maxCols)
        let sp = scales.contents().bindMemory(to: Float16.self, capacity: maxBlocks)
        var maxRef = 0.0
        for g in [0, n / 2 + 3, n - 1] {
            var tot = 0.0
            for bl in 0..<nblk {
                let gb = g * nblk + bl
                let w = genWords(UInt32(gb), f.nw, f.id)
                var acc = 0.0, xsum = 0.0
                if f.id == 0 {
                    for k in 0..<4 { for j in 0..<4 {
                        let lo = Double((w[k] >> UInt32(8 * j)) & 0xF), hi = Double((w[k] >> UInt32(8 * j + 4)) & 0xF)
                        let xl = Double(xp[bl * 32 + 8 * k + j]), xh = Double(xp[bl * 32 + 8 * k + 4 + j])
                        acc += lo * xl + hi * xh; xsum += xl + xh
                    } }
                    tot += Double(sp[gb]) * (acc - 8 * xsum)
                } else if f.id == 1 {
                    for k in 0..<8 { for j in 0..<4 {
                        let q = Double(UInt8(truncatingIfNeeded: w[k] >> UInt32(8 * j))) - 128
                        acc += q * Double(xp[bl * 32 + 4 * k + j])
                    } }
                    tot += Double(sp[gb]) * acc
                } else {
                    for k in 0..<16 {
                        let h0 = Float16(bitPattern: UInt16(w[k] & 0xFFFF)), h1 = Float16(bitPattern: UInt16(w[k] >> 16))
                        acc += Double(h0) * Double(xp[bl * 32 + 2 * k]) + Double(h1) * Double(xp[bl * 32 + 2 * k + 1])
                    }
                    tot += acc
                }
            }
            maxRef = max(maxRef, abs(tot - Double(a[g])), abs(tot - Double(b[g])))
        }
        let vbytes = Double(totalBlocks) * f.bytesPerBlock
        let m1 = median(t1), m2 = median(t2)
        print(String(format: "%-11@ B=%3d | okuyan %6.2f ms %6.1f GB/s %4.0f Gw/s | ureten %6.2f ms %6.1f GB/s %4.0f Gw/s | oran %.2fx | y1-y2 %.1e CPU %.1e (|y| %.0f)",
                     "\(rows)x\(cols)" as NSString, B, m1 * 1e3, vbytes / m1 / 1e9, Double(totalBlocks) * 32 / m1 / 1e9,
                     m2 * 1e3, vbytes / m2 / 1e9, Double(totalBlocks) * 32 / m2 / 1e9, m1 / m2, Double(maxd), maxRef, Double(maxa)))
    }
}
