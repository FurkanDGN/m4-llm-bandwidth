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
let cw = codes.contents().bindMemory(to: UInt32.self, capacity: maxBlocks * 4)
let sw = seeds.contents().bindMemory(to: UInt32.self, capacity: maxBlocks * 4)
let sp = scales.contents().bindMemory(to: Float16.self, capacity: maxBlocks)
let xp = xbuf.contents().bindMemory(to: Float.self, capacity: 16 * maxCols)
func fillX(_ cols: Int) {
    var st: UInt32 = 987654321
    for i in 0..<(16 * cols) { st = xs32(st); xp[i] = Float(Int32(bitPattern: st)) / 2147483648.0 }
}
func nib(_ w: UInt32, _ j: Int) -> Double { // cift duzeni: agirlik 2s -> bit 4s, agirlik 2s+1 -> bit 16+4s
    let s = j / 2
    return Double((w >> UInt32(j % 2 == 0 ? 4 * s : 16 + 4 * s)) & 15)
}
// mode 0 oku, 1 turet(P), 2 rastgele
func ref(_ mode: Int, _ P: Int, _ c: Int, _ g: Int, _ nblk: Int, _ cols: Int) -> Double {
    var tot = 0.0
    for bl in 0..<nblk {
        let gb = g * nblk + bl
        var bs = 0.0
        var gen: [UInt32] = []
        if mode == 2 {
            var s = (UInt32(gb) &+ seed) &* 0x9E3779B1; s ^= s >> 15; s |= 1
            for _ in 0..<4 { s = xs32(s); gen.append(s) }
        }
        for k in 0..<4 {
            for j in 0..<8 {
                var wv = 0.0
                if mode == 0 { wv = nib(cw[gb * 4 + k], j) - 8 }
                else if mode == 2 { wv = nib(gen[k], j) - 8 }
                else {
                    let w = sw[gb * 4 + k]
                    var s = ((w & 0xFFFF) &+ 1) &* 0x9E3779B1; s ^= s >> 15; s |= 1
                    for pp in 0..<P {
                        s = xs32(s)
                        wv += (Double((w >> UInt32(16 + 4 * pp)) & 15) - 8) * (nib(s, j) - 8)
                    }
                }
                bs += wv * Double(xp[c * cols + bl * 32 + 8 * k + j])
            }
        }
        tot += Double(sp[gb]) * bs
    }
    return tot
}
func relErr(_ mode: Int, _ P: Int, _ B: Int, _ n: Int, _ nblk: Int, _ cols: Int, _ value: (Int, Int) -> Float) -> Double {
    var md = 0.0, ma = 1e-30
    for g in [0, 1, 7, n / 2 + 3, n - 1] {
        for c in 0..<B {
            let r = ref(mode, P, c, g, nblk, cols)
            md = max(md, abs(r - Double(value(c, g)))); ma = max(ma, abs(r))
        }
    }
    return md / ma
}
typealias Body = (MTLComputeCommandEncoder) -> Void
func timeAll(_ bodies: [Body]) -> [Double] {
    for b in bodies { _ = runOnce(b); _ = runOnce(b) }
    var ts = [[Double]](repeating: [], count: bodies.count)
    for _ in 0..<reps { for i in 0..<bodies.count { ts[i].append(runOnce(bodies[i], n: iters)) } }
    return ts.map { median($0) }
}
func Rof(_ B: Int) -> Int { B <= 8 ? 8 : (B <= 12 ? 4 : 2) }

let sizes: [(Int, Int)] = [(4096, 4096), (8192, 8192), (18944, 3584), (3584, 18944)]
let sel = args.count >= 4 ? Int(args[3])! : -1
let Bmax = args.count >= 5 ? Int(args[4])! : 16
for (si, (rows, cols)) in sizes.enumerated() {
    if sel >= 0 && si != sel { continue }
    let Mx = max(1, T / (rows * cols)), nblk = cols / 32
    let n = Mx * rows
    fillX(cols)
    print("\n=== \(rows)x\(cols), \(Mx) matris, 2^29 agirlik; ms ; 7B-esdegeri = ms x 14.2 ===")
    print("  B | a: B kez matvec | b: matmat q4 OKU | c: matmat TUREV P=3 | d: matmat RASTGELE | b/b1  a/a1  c/b  d/b | 7B-es.b ms | hata a b c d")
    var base: [Double] = [0, 0]
    for B in 1...Bmax {
        let R = Rof(B)
        var p = MP(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
        let pA = pso("m7_b1_m0"), pB = pso("m7_b\(B)_m0"), pC = pso("m7_b\(B)_m1p3"), pD = pso("m7_b\(B)_m2")
        let a: Body = { e in
            e.setComputePipelineState(pA)
            for c in 0..<B {
                e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
                e.setBuffer(xbuf, offset: c * cols * 4, index: 2); e.setBuffer(outs[0], offset: c * n * 4, index: 3)
                e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
                e.dispatchThreadgroups(MTLSize(width: rows / (4 * 8), height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        func mk(_ ps: MTLComputePipelineState, _ blkBuf: MTLBuffer, _ o: MTLBuffer) -> Body {
            return { e in
                e.setComputePipelineState(ps)
                e.setBuffer(blkBuf, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
                e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(o, offset: 0, index: 3)
                e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
                e.dispatchThreadgroups(MTLSize(width: rows / (4 * R), height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        let t = timeAll([a, mk(pB, codes, outs[2]), mk(pC, seeds, outs[3]), mk(pD, codes, outs[4])])
        if B == 1 { base = [t[0], t[1]] }
        let ya = outs[0].contents().bindMemory(to: Float.self, capacity: 16 * n)
        let yb = outs[2].contents().bindMemory(to: Float.self, capacity: 16 * n)
        let yc = outs[3].contents().bindMemory(to: Float.self, capacity: 16 * n)
        let yd = outs[4].contents().bindMemory(to: Float.self, capacity: 16 * n)
        let e0 = relErr(0, 0, B, n, nblk, cols) { c, g in ya[c * n + g] }
        let e1 = relErr(0, 0, B, n, nblk, cols) { c, g in yb[g * B + c] }
        let e2 = relErr(1, 3, B, n, nblk, cols) { c, g in yc[g * B + c] }
        let e3 = relErr(2, 0, B, n, nblk, cols) { c, g in yd[g * B + c] }
        print(String(format: "%3d | %9.2f | %9.2f | %9.2f | %9.2f | %5.2f %5.2f %5.2f %5.2f | %7.1f | %.0e %.0e %.0e %.0e",
                     B, t[0] * 1e3, t[1] * 1e3, t[2] * 1e3, t[3] * 1e3, t[1] / base[1], t[0] / base[0], t[2] / t[1], t[3] / t[1], t[1] * 1e3 * 14.2, e0, e1, e2, e3))
    }
}
