import Foundation
import Metal

// kullanim: ./bench15 kernels14.metal kernels11.metal [reps] [rows] [cols] [verify]
struct MP { var rows: UInt32; var cols: UInt32; var nblk: UInt32; var seed: UInt32 }
let args = CommandLine.arguments
let reps = args.count >= 4 ? Int(args[3])! : 7
let iters = 3
guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no metal") }
let lib = try dev.makeLibrary(source: try String(contentsOfFile: args[1], encoding: .utf8), options: nil)
let lib2 = try dev.makeLibrary(source: try String(contentsOfFile: args[2], encoding: .utf8), options: nil)
var psoCache: [String: MTLComputePipelineState] = [:]
func pso(_ n: String) -> MTLComputePipelineState {
    if let p = psoCache[n] { return p }
    guard let f = lib.makeFunction(name: n) ?? lib2.makeFunction(name: n) else { fatalError("yok: \(n)") }
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
let outs = (0..<12).map { _ in dev.makeBuffer(length: 40 << 20, options: .storageModeShared)! }
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
var halfX = false
func xv(_ i: Int) -> Double { halfX ? Double(Float(Float16(xp[i]))) : Double(xp[i]) }
// mode 0 oku, 1 turet(xorshift,P), 2 rastgele, 3 tek carpimli hash +-1 secim
func ref(_ mode: Int, _ P: Int, _ c: Int, _ g: Int, _ nblk: Int, _ cols: Int) -> Double {
    var tot = 0.0
    for bl in 0..<nblk {
        let gb = g * nblk + bl
        var bs = 0.0
        for k in 0..<4 {
            for j in 0..<8 {
                var wv = 0.0
                if mode == 0 { wv = nib(cw[gb * 4 + k], j) - 8 }
                else if mode == 3 {
                    let wd = sw[gb * 4 + k]; let gw = UInt32(gb * 4 + k)
                    let h = (gw &+ 1) &* (((wd & 0xFFFF) << 1) | 1)
                    let tm0 = Double((wd >> 16) & 15) - 8, tm1 = Double((wd >> 20) & 15) - 8
                    let s0 = ((h >> UInt32(16 + j)) & 1) == 1 ? 1.0 : -1.0, s1 = ((h >> UInt32(j)) & 1) == 1 ? 1.0 : -1.0
                    wv = s0 * tm0 + s1 * tm1
                }
                else if mode == 2 { var s = (UInt32(gb * 4 + k) &+ seed) &* 0x9E3779B1; s ^= s >> 15; s |= 1; s = xs32(s); s = xs32(s); wv = nib(s, j) - 8 }
                else {
                    let w = sw[gb * 4 + k]
                    var s = ((w & 0xFFFF) &+ 1) &* 0x9E3779B1; s ^= s >> 15; s |= 1
                    for pp in 0..<P {
                        s = xs32(s)
                        wv += (Double((w >> UInt32(16 + 4 * pp)) & 15) - 8) * (nib(s, j) - 8)
                    }
                }
                bs += wv * xv(c * cols + bl * 32 + 8 * k + j)
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

let xtb = dev.makeBuffer(length: 18944 * 16 * 2, options: .storageModeShared)!
func fillXt(_ cols: Int, _ ldx: Int) {
    let q = xtb.contents().bindMemory(to: Float16.self, capacity: cols * ldx)
    for k in 0..<cols { for c in 0..<ldx { q[k * ldx + c] = Float16(xp[c * cols + k]) } }
}
let rows = args.count >= 5 ? Int(args[4])! : 8192, cols = args.count >= 6 ? Int(args[5])! : 8192
let Mx = max(1, T / (rows * cols)), nblk = cols / 32
let n = Mx * rows
fillX(cols)

if args.count >= 7 && args[6] == "verify" {
    var worst = 0.0, cnt = 0
    for B in 1...8 {
        for R in [1, 2, 4, 8] where R * B <= 64 {
            for (nm, mode, buf) in [("m0", 0, codes), ("m3", 3, seeds), ("m2", 2, codes)] {
                var p = MP(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
                let ps = pso("m8_b\(B)_r\(R)_\(nm)")
                _ = runOnce { e in
                    e.setComputePipelineState(ps)
                    e.setBuffer(buf, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
                    e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(outs[0], offset: 0, index: 3)
                    e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
                    e.dispatchThreadgroups(MTLSize(width: rows / (4 * R), height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                }
                let y = outs[0].contents().bindMemory(to: Float.self, capacity: 16 * n)
                let e = relErr(mode, 0, B, n, nblk, cols) { c, g in y[g * B + c] }
                worst = max(worst, e); cnt += 1
                if e > 1e-5 { print("SORUN B=\(B) R=\(R) \(nm) hata \(e)") }
            }
        }
    }
    print("skaler dogrulama: \(cnt) cekirdek, en buyuk goreli hata \(String(format: "%.1e", worst))")
    exit(0)
}

print("=== \(rows)x\(cols) (\(Mx) matris, 2^29 agirlik) ; ms ===")
print("  B | a:B kez | skaler OKU (R) | MMA OKU tm4 | tm2 | tm1 | skaler TUREV-E | MMA TUREV-E (en iyi) | skaler RASTGELE | MMA RASTGELE | hata mma0/1/2")
var base = 0.0
for B in 1...16 {
    let NCT = B <= 8 ? 1 : 2, ldx = NCT * 8
    fillXt(cols, ldx)
    var p = MP(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
    var bodies: [Body] = []
    func scalarBodies(_ nm: String, _ blkBuf: MTLBuffer) -> [(Body, Int)] {
        var r: [(Body, Int)] = []
        if B > 8 { return r }
        for R in [1, 2, 4, 8] where R * B <= 64 {
            let ps = pso("m8_b\(B)_r\(R)_\(nm)")
            r.append(({ e in
                e.setComputePipelineState(ps)
                e.setBuffer(blkBuf, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
                e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(outs[0], offset: 0, index: 3)
                e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
                e.dispatchThreadgroups(MTLSize(width: rows / (4 * R), height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }, R))
        }
        return r
    }
    let pA = pso("m8_b1_r8_m0")
    let a: Body = { e in
        e.setComputePipelineState(pA)
        for c in 0..<B {
            e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
            e.setBuffer(xbuf, offset: c * cols * 4, index: 2); e.setBuffer(outs[1], offset: c * n * 4, index: 3)
            e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
            e.dispatchThreadgroups(MTLSize(width: rows / 32, height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }
    func mma(_ nm: String, _ tm: Int, _ blkBuf: MTLBuffer, _ o: MTLBuffer) -> Body {
        let ps = pso(nm)
        return { e in
            e.setComputePipelineState(ps)
            e.setBuffer(blkBuf, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
            e.setBuffer(xtb, offset: 0, index: 2); e.setBuffer(o, offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
            e.dispatchThreadgroups(MTLSize(width: rows / (4 * 8 * tm), height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }
    let sR = scalarBodies("m0", codes), sD = scalarBodies("m3", seeds), sG = scalarBodies("m2", codes)
    bodies.append(a)
    for x in sR { bodies.append(x.0) }; for x in sD { bodies.append(x.0) }; for x in sG { bodies.append(x.0) }
    bodies.append(mma("mma_tm4_n\(NCT)_m0", 4, codes, outs[2]))
    bodies.append(mma("mma_tm2_n\(NCT)_m0", 2, codes, outs[5]))
    bodies.append(mma("mma_tm1_n\(NCT)_m0", 1, codes, outs[6]))
    bodies.append(mma("mma_tm4_n\(NCT)_m3", 4, seeds, outs[3]))
    bodies.append(mma("mma_tm2_n\(NCT)_m3", 2, seeds, outs[7]))
    bodies.append(mma("mma_tm1_n\(NCT)_m3", 1, seeds, outs[8]))
    bodies.append(mma("mma_tm4_n\(NCT)_m2", 4, codes, outs[4]))
    let t = timeAll(bodies)
    var idx = 1
    func best(_ cnt: Int) -> (Double, Int) { if cnt == 0 { return (Double.nan, 0) }; var bi = 0; for i in 0..<cnt where t[idx + i] < t[idx + bi] { bi = i }; return (t[idx + bi], bi) }
    let rs = [1, 2, 4, 8].filter { $0 * B <= 64 && B <= 8 }
    let bR = best(sR.count); idx += sR.count
    let bD = best(sD.count); idx += sD.count
    let bG = best(sG.count); idx += sG.count
    let tm4 = t[idx], tm2 = t[idx + 1], tm1 = t[idx + 2], mD = min(t[idx + 3], t[idx + 4], t[idx + 5]), mG = t[idx + 6]
    if B == 1 { base = min(bR.0, tm4) }
    halfX = true
    let ym = outs[2].contents().bindMemory(to: Float.self, capacity: n * 16)
    let yd = outs[3].contents().bindMemory(to: Float.self, capacity: n * 16)
    let yg = outs[4].contents().bindMemory(to: Float.self, capacity: n * 16)
    let e0 = relErr(0, 0, B, n, nblk, cols) { c, g in ym[g * ldx + c] }
    let yd2 = outs[8].contents().bindMemory(to: Float.self, capacity: n * 16)
    let e1 = max(relErr(3, 0, B, n, nblk, cols) { c, g in yd[g * ldx + c] }, relErr(3, 0, B, n, nblk, cols) { c, g in yd2[g * ldx + c] })
    let e2 = relErr(2, 0, B, n, nblk, cols) { c, g in yg[g * ldx + c] }
    halfX = false
    func f(_ v: Double) -> String { v.isNaN ? "     -" : String(format: "%6.2f", v * 1e3) }
    print(String(format: "%3d | %6.2f | %@ R=%d | %@ | %@ | %@ | %@ | %@ | %@ | %@ | %.0e %.0e %.0e",
                 B, t[0] * 1e3, f(bR.0) as NSString, bR.1 < rs.count ? rs[bR.1] : 0, f(tm4) as NSString, f(tm2) as NSString, f(tm1) as NSString,
                 f(bD.0) as NSString, f(mD) as NSString, f(bG.0) as NSString, f(mG) as NSString, e0, e1, e2))
}
