import Foundation
import Metal

struct Params { var rows: UInt32; var cols: UInt32; var nblk: UInt32; var seed: UInt32 }
let args = CommandLine.arguments
let reps = args.count >= 3 ? Int(args[2])! : 15
let part = args.count >= 4 ? args[3] : "all"
let iters = 4
guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no metal") }
let lib = try dev.makeLibrary(source: try String(contentsOfFile: args[1], encoding: .utf8), options: nil)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let seed: UInt32 = 12345
func xs32(_ s0: UInt32) -> UInt32 { var s = s0; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s }

let T = 1 << 29
let maxBlocks = T / 32
let maxCols = 18944
let codes = dev.makeBuffer(length: maxBlocks * 64, options: .storageModeShared)!
let scales = dev.makeBuffer(length: maxBlocks * 2, options: .storageModeShared)!
let seeds = dev.makeBuffer(length: maxBlocks * 16, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: maxCols * 4, options: .storageModeShared)!
let outs = (0..<20).map { _ in dev.makeBuffer(length: 4 << 20, options: .storageModeShared)! }
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
func fillCodes(_ name: String) {
    _ = runOnce { e in
        var np = SIMD2<UInt32>(UInt32(maxBlocks), seed)
        e.setComputePipelineState(pso(name))
        e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
        e.setBytes(&np, length: 8, index: 2)
        e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
_ = runOnce { e in
    var np = SIMD2<UInt32>(UInt32(maxBlocks), 777)
    e.setComputePipelineState(pso("fill_seeds"))
    e.setBuffer(seeds, offset: 0, index: 0)
    e.setBytes(&np, length: 8, index: 1)
    e.dispatchThreads(MTLSize(width: maxBlocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

let cw = codes.contents().bindMemory(to: UInt32.self, capacity: maxBlocks * 16)
let sw = seeds.contents().bindMemory(to: UInt32.self, capacity: maxBlocks * 4)
let sp = scales.contents().bindMemory(to: Float16.self, capacity: maxBlocks)
let xp = xbuf.contents().bindMemory(to: Float.self, capacity: maxCols)

func cpuRow(_ g: Int, _ nblk: Int, _ fmt: Int, _ gf: Int, _ P: Int) -> Double {
    var tot = 0.0
    let nw = fmt == 0 ? 4 : (fmt == 1 ? 8 : 16)
    let derived = (g & 3) < gf
    for bl in 0..<nblk {
        let gb = g * nblk + bl
        if derived {
            var bs = 0.0
            for sb in 0..<4 {
                let w = sw[gb * 4 + sb]
                var s = ((w & 0xFFFF) &+ 1) &* 0x9E3779B1
                s ^= s >> 15; s |= 1
                for pp in 0..<P {
                    s = xs32(s)
                    let t = Double((w >> UInt32(16 + 4 * pp)) & 15) - 8
                    var d = 0.0
                    for j in 0..<8 {
                        let u = j < 4 ? Double((s >> UInt32(8 * j)) & 15) : Double((s >> UInt32(8 * (j - 4) + 4)) & 15)
                        d += (u - 8) * Double(xp[bl * 32 + 8 * sb + j])
                    }
                    bs += t * d
                }
            }
            tot += Double(sp[gb]) * bs
        } else if fmt == 0 {
            var acc = 0.0
            for k in 0..<4 { for j in 0..<4 {
                let w = cw[gb * nw + k]
                let lo = Double((w >> UInt32(8 * j)) & 0xF), hi = Double((w >> UInt32(8 * j + 4)) & 0xF)
                acc += (lo - 8) * Double(xp[bl * 32 + 8 * k + j]) + (hi - 8) * Double(xp[bl * 32 + 8 * k + 4 + j])
            } }
            tot += Double(sp[gb]) * acc
        } else if fmt == 1 {
            var acc = 0.0
            for k in 0..<8 { for j in 0..<4 {
                let q = Double(UInt8(truncatingIfNeeded: cw[gb * nw + k] >> UInt32(8 * j))) - 128
                acc += q * Double(xp[bl * 32 + 4 * k + j])
            } }
            tot += Double(sp[gb]) * acc
        } else {
            var acc = 0.0
            for k in 0..<16 {
                let w = cw[gb * nw + k]
                let h0 = Float16(bitPattern: UInt16(w & 0xFFFF)), h1 = Float16(bitPattern: UInt16(w >> 16))
                acc += Double(h0) * Double(xp[bl * 32 + 2 * k]) + Double(h1) * Double(xp[bl * 32 + 2 * k + 1])
            }
            tot += acc
        }
    }
    return tot
}
func relErr(_ out: MTLBuffer, _ rows: Int, _ B: Int, _ nblk: Int, _ fmt: Int, _ gf: Int, _ P: Int) -> Double {
    let n = B * rows
    let y = out.contents().bindMemory(to: Float.self, capacity: n)
    var md = 0.0, ma = 0.0
    for g in [0, 1, 2, 3, n / 2, n / 2 + 1, n - 4, n - 3, n - 2, n - 1] {
        let r = cpuRow(g, nblk, fmt, gf, P)
        md = max(md, abs(r - Double(y[g]))); ma = max(ma, abs(r))
    }
    return md / ma
}

struct K { let name: String; let rpt: Int; let useSeeds: Bool }
func timeAll(_ ks: [K], _ rows: Int, _ cols: Int) -> [Double] {
    let B = max(1, T / (rows * cols)), nblk = cols / 32
    var p = Params(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
    let pss = ks.map { pso($0.name) }
    func body(_ i: Int) -> (MTLComputeCommandEncoder) -> Void {
        return { e in
            e.setComputePipelineState(pss[i])
            e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
            e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(outs[i], offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<Params>.size, index: 4)
            if ks[i].useSeeds { e.setBuffer(seeds, offset: 0, index: 5) }
            e.dispatchThreadgroups(MTLSize(width: rows / ks[i].rpt, height: B, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }
    for i in 0..<ks.count { _ = runOnce(body(i)); _ = runOnce(body(i)) }
    var ts = [[Double]](repeating: [], count: ks.count)
    for _ in 0..<reps { for i in 0..<ks.count { ts[i].append(runOnce(body(i), n: iters)) } }
    return ts.map { median($0) }
}

let sizes: [(Int, Int)] = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (18944, 3584), (3584, 18944)]

if part == "all" || part == "A" {
    fillCodes("fill_q4")
    print("=== A) Saf turetme: 8 agirliklik alt blok = 16 bit tohum + P adet 4 bit katsayi; bellek 18 B/32 agirlik (q4_0 ile ayni) ===")
    print("G agirlik/s (ms) ; 'hata' = CPU'ya gore goreli hata")
    let ks = [K(name: "mv_q4_g0", rpt: 32, useSeeds: false), K(name: "mv_q4_g4", rpt: 32, useSeeds: false),
              K(name: "mv_seed_p1", rpt: 32, useSeeds: true), K(name: "mv_seed_p2", rpt: 32, useSeeds: true),
              K(name: "mv_seed_p3", rpt: 32, useSeeds: true), K(name: "mv_seed_p4", rpt: 32, useSeeds: true)]
    print(String(format: "%-11@ | %-15@ | %-15@ | %-22@ | %-22@ | %-22@ | %-22@", "boyut" as NSString, "q4 OKU" as NSString, "q4 RASTGELE" as NSString, "TURET P=1" as NSString, "TURET P=2" as NSString, "TURET P=3" as NSString, "TURET P=4" as NSString))
    for (rows, cols) in sizes {
        let B = max(1, T / (rows * cols)), nblk = cols / 32
        let w = Double(B * rows * cols)
        let t = timeAll(ks, rows, cols)
        var line = String(format: "%-11@ |", "\(rows)x\(cols)" as NSString)
        for i in 0..<ks.count {
            var extra = ""
            if i >= 2 { extra = String(format: " hata %.0e", relErr(outs[i], rows, B, nblk, 0, 4, i - 1)) }
            line += String(format: " %5.0f (%5.2f ms)%@ |", w / t[i] / 1e9, t[i] * 1e3, extra as NSString)
        }
        print(line)
    }
}
if part == "all" || part == "B" {
    print("\n=== B) Karisik: satirlarin G/4'u tohumdan TURETILIR (P=3), gerisi bellekten okunur. sanal GB/s = tum agirliklar o formatta saklansaydi ===")
    let fm: [(String, Int, String, String, Double, Int)] = [("q8_0", 1, "fill_q8", "q8", 34, 16), ("fp16", 2, "fill_f16", "f16", 64, 16), ("q4_0", 0, "fill_q4", "q4", 18, 32)]
    for (name, id, fill, key, bpb, rpt) in fm {
        fillCodes(fill)
        print("\n--- \(name) ---   (sol: okuma+RASTGELE uretim, sag: okuma+TURETME)")
        for (rows, cols) in [(1024, 1024), (4096, 4096), (8192, 8192), (18944, 3584)] {
            let B = max(1, T / (rows * cols)), nblk = cols / 32
            let nb = Double(B * rows * nblk)
            var ks: [K] = []
            for g in 0...4 { ks.append(K(name: "mv_\(key)_g\(g)", rpt: rpt, useSeeds: false)) }
            for g in 0...4 { ks.append(K(name: "mv_m\(key)_g\(g)", rpt: rpt, useSeeds: true)) }
            if key == "q4" { ks[9] = K(name: "mv_seed_p3", rpt: 32, useSeeds: true) }
            let t = timeAll(ks, rows, cols)
            var l1 = String(format: "%-11@ rastgele |", "\(rows)x\(cols)" as NSString)
            var l2 = String(format: "%-11@ turetme  |", "" as NSString)
            for g in 0...4 {
                l1 += String(format: " %6.1f |", nb * bpb / t[g] / 1e9)
                l2 += String(format: " %6.1f (%.0e) |", nb * bpb / t[5 + g] / 1e9, relErr(outs[5 + g], rows, B, nblk, id, g, 3))
            }
            print(l1); print(l2)
        }
    }
}

if part == "all" || part == "C" {
    print("\n=== C) Katsayi sayisi P etkisi (8192x8192): sanal GB/s, GF=0..4 (GF satirlarin G/4'u turetilir) ===")
    let cfgs: [(String, Int, String, Double, [(Int, String)])] = [
        ("q8_0", 1, "fill_q8", 34, [(2, "mv_mq8p2"), (3, "mv_mq8")]),
        ("fp16", 2, "fill_f16", 64, [(2, "mv_mf16p2"), (3, "mv_mf16"), (4, "mv_mf16p4")]),
    ]
    let rows = 8192, cols = 8192
    let B = max(1, T / (rows * cols)), nblk = cols / 32
    let nb = Double(B * rows * nblk)
    for (name, id, fill, bpb, ps) in cfgs {
        fillCodes(fill)
        var ks: [K] = []
        for (_, pre) in ps { for g in 0...4 { ks.append(K(name: g == 0 ? "\(name == "q8_0" ? "mv_mq8" : "mv_mf16")_g0" : "\(pre)_g\(g)", rpt: 16, useSeeds: true)) } }
        let t = timeAll(ks, rows, cols)
        for (pi, (P, _)) in ps.enumerated() {
            var line = String(format: "%-6@ P=%d |", name as NSString, P)
            for g in 0...4 {
                let idx = pi * 5 + g
                line += String(format: " %6.1f (%.0e) |", nb * bpb / t[idx] / 1e9, relErr(outs[idx], rows, B, nblk, id, g, P))
            }
            print(line)
        }
    }
}
