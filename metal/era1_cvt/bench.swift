import Foundation
import Metal

struct Params { var rows: UInt32; var cols: UInt32; var nblk: UInt32; var seed: UInt32 }

let args = CommandLine.arguments
guard args.count >= 2 else { fputs("usage: bench kernels.metal [reps]\n", stderr); exit(2) }
let reps = args.count >= 3 ? Int(args[2])! : 15
let iters = 3

guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no metal") }
print("GPU: \(dev.name)  workingSet: \(dev.recommendedMaxWorkingSetSize / 1_000_000) MB")
let src = try String(contentsOfFile: args[1], encoding: .utf8)
let lib = try dev.makeLibrary(source: src, options: nil)
func pso(_ n: String) -> MTLComputePipelineState {
    try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
}
let pFill = pso("fill_codes"), pRead = pso("mv_read"), pGen = pso("mv_gen"), pStream = pso("stream_read")

let seed: UInt32 = 12345
func xs32(_ s0: UInt32) -> UInt32 { var s = s0; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s }
func genBlock(_ gidx: UInt32) -> [UInt32] {
    var s = (gidx &+ seed) &* 0x9E3779B1
    s ^= s >> 15
    s |= 1
    var w = [UInt32](repeating: 0, count: 4)
    for k in 0..<4 { s = xs32(s); w[k] = s }
    return w
}

let sizes: [(Int, Int)] = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (18944, 3584), (3584, 18944)]
let target = 1 << 31
func batchOf(_ r: Int, _ c: Int) -> Int { max(1, target / (r * c)) }
let maxBlocks = sizes.map { batchOf($0.0, $0.1) * $0.0 * $0.1 / 32 }.max()!
let maxCols = sizes.map { $0.1 }.max()!
let maxY = sizes.map { batchOf($0.0, $0.1) * $0.0 }.max()!

let codes = dev.makeBuffer(length: maxBlocks * 16, options: .storageModeShared)!
let scales = dev.makeBuffer(length: maxBlocks * 2, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: maxCols * 4, options: .storageModeShared)!
let y1 = dev.makeBuffer(length: maxY * 4, options: .storageModeShared)!
let y2 = dev.makeBuffer(length: maxY * 4, options: .storageModeShared)!
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
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if cb.status != .completed { fatalError("cb failed: \(String(describing: cb.error))") }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(n)
}

_ = runOnce { e in
    var np = SIMD2<UInt32>(UInt32(maxBlocks), seed)
    e.setComputePipelineState(pFill)
    e.setBuffer(codes, offset: 0, index: 0)
    e.setBuffer(scales, offset: 0, index: 1)
    e.setBytes(&np, length: 8, index: 2)
    e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }

// Bellek bandi tavani
do {
    let nt = 1 << 20
    var n = UInt32(60 * nt)
    let body: (MTLComputeCommandEncoder) -> Void = { e in
        e.setComputePipelineState(pStream)
        e.setBuffer(codes, offset: 0, index: 0)
        e.setBuffer(sout, offset: 0, index: 1)
        e.setBytes(&n, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: nt, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    for _ in 0..<3 { _ = runOnce(body) }
    var ts: [Double] = []
    for _ in 0..<reps { ts.append(runOnce(body, n: iters)) }
    let bytes = Double(n) * 16
    print(String(format: "Duz okuma (bant tavani): medyan %.2f ms = %.1f GB/s, en iyi %.1f GB/s", median(ts) * 1e3, bytes / median(ts) / 1e9, bytes / ts.min()! / 1e9))
}

print("sanal bayt = kod (16 B) + olcek (2 B) / 32 agirlik blogu, yani q4_0 ile ayni 0.5625 B/agirlik")
print(String(format: "%-12@ %4@ | %-26@ | %-34@ | %@", "boyut" as NSString, "B" as NSString, "OKUYAN (medyan)" as NSString, "URETEN (medyan)" as NSString, "oran / dogrulama" as NSString))

for (rows, cols) in sizes {
    let B = batchOf(rows, cols)
    let nblk = cols / 32
    let totalBlocks = B * rows * nblk
    var p = Params(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
    let grid = MTLSize(width: rows / 32, height: B, depth: 1)
    let tgs = MTLSize(width: 128, height: 1, depth: 1)
    let b1: (MTLComputeCommandEncoder) -> Void = { e in
        e.setComputePipelineState(pRead)
        e.setBuffer(codes, offset: 0, index: 0)
        e.setBuffer(scales, offset: 0, index: 1)
        e.setBuffer(xbuf, offset: 0, index: 2)
        e.setBuffer(y1, offset: 0, index: 3)
        e.setBytes(&p, length: MemoryLayout<Params>.size, index: 4)
        e.dispatchThreadgroups(grid, threadsPerThreadgroup: tgs)
    }
    let b2: (MTLComputeCommandEncoder) -> Void = { e in
        e.setComputePipelineState(pGen)
        e.setBuffer(scales, offset: 0, index: 0)
        e.setBuffer(xbuf, offset: 0, index: 1)
        e.setBuffer(y2, offset: 0, index: 2)
        e.setBytes(&p, length: MemoryLayout<Params>.size, index: 3)
        e.dispatchThreadgroups(grid, threadsPerThreadgroup: tgs)
    }
    for _ in 0..<2 { _ = runOnce(b1); _ = runOnce(b2) }
    var t1: [Double] = [], t2: [Double] = []
    for _ in 0..<reps { t1.append(runOnce(b1, n: iters)); t2.append(runOnce(b2, n: iters)) }

    // dogrulama: tum y1 ile y2, ve ornek satirlar CPU referansi
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
            let w = genBlock(UInt32(gb))
            var acc = 0.0, xsum = 0.0
            for k in 0..<4 {
                for j in 0..<4 {
                    let lo = Double((w[k] >> UInt32(8 * j)) & 0xF), hi = Double((w[k] >> UInt32(8 * j + 4)) & 0xF)
                    let xl = Double(xp[bl * 32 + 8 * k + j]), xh = Double(xp[bl * 32 + 8 * k + 4 + j])
                    acc += lo * xl + hi * xh
                    xsum += xl + xh
                }
            }
            tot += Double(sp[gb]) * (acc - 8 * xsum)
        }
        maxRef = max(maxRef, abs(tot - Double(a[g])), abs(tot - Double(b[g])))
    }

    let vbytes = Double(totalBlocks) * 18
    let m1 = median(t1), m2 = median(t2)
    let s1 = String(format: "%7.2f ms %6.1f GB/s", m1 * 1e3, vbytes / m1 / 1e9)
    let s2 = String(format: "%7.2f ms %6.1f GB/s (gercek %5.1f)", m2 * 1e3, vbytes / m2 / 1e9, Double(totalBlocks) * 2 / m2 / 1e9)
    let s3 = String(format: "%.2fx  y1-y2 %.1e, CPU %.1e (|y|max %.1f)", m1 / m2, Double(maxd), maxRef, Double(maxa))
    print(String(format: "%-12@ %4d | %@ | %@ | %@", "\(rows)x\(cols)" as NSString, B, s1 as NSString, s2 as NSString, s3 as NSString))
    let g1 = Double(totalBlocks) * 32 / m1 / 1e9, g2 = Double(totalBlocks) * 32 / m2 / 1e9
    print(String(format: "             en iyi tekrar: okuyan %.1f GB/s, ureten %.1f GB/s | G agirlik/s: okuyan %.0f, ureten %.0f", vbytes / t1.min()! / 1e9, vbytes / t2.min()! / 1e9, g1, g2))
}
