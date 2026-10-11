import Foundation
import Metal

struct MP { var rows: UInt32; var cols: UInt32; var nblk: UInt32; var seed: UInt32 }
let args = CommandLine.arguments
let reps = args.count >= 3 ? Int(args[2])! : 9
let part = args.count >= 4 ? args[3] : "all"
let iters = 4
guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else { fatalError("no metal") }
let lib = try dev.makeLibrary(source: try String(contentsOfFile: args[1], encoding: .utf8), options: nil)
var psoCache: [String: MTLComputePipelineState] = [:]
func pso(_ n: String) -> MTLComputePipelineState {
    if let p = psoCache[n] { return p }
    guard let f = lib.makeFunction(name: n) else { fatalError("yok: \(n)") }
    let p = try! dev.makeComputePipelineState(function: f); psoCache[n] = p; return p
}
func xs32(_ s0: UInt32) -> UInt32 { var s = s0; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s }
func hashw(_ i: UInt32, _ seed: UInt32) -> UInt32 { var s = (i &+ seed) &* 0x9E3779B1; s ^= s >> 15; s |= 1; return xs32(xs32(s)) }
let seed: UInt32 = 12345
let T = 1 << 29
let nchMax = T / 8
let codes = dev.makeBuffer(length: nchMax * 16, options: .storageModeShared)!
let scales = dev.makeBuffer(length: nchMax / 4 * 2, options: .storageModeShared)!
let seeds = dev.makeBuffer(length: nchMax * 4, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: 18944 * 4, options: .storageModeShared)!
let sout = dev.makeBuffer(length: (1 << 20) * 4, options: .storageModeShared)!
let outs = (0..<20).map { _ in dev.makeBuffer(length: 4 << 20, options: .storageModeShared)! }
func runOnce(_ body: (MTLComputeCommandEncoder) -> Void, n: Int = 1) -> Double {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    for _ in 0..<n { body(e) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status != .completed { fatalError("cb failed: \(String(describing: cb.error))") }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(n)
}
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
func dispatch1(_ name: String, _ bufs: [MTLBuffer], _ bytes: [UInt32], _ n: Int) {
    _ = runOnce { e in
        e.setComputePipelineState(pso(name))
        for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
        var bb = bytes
        e.setBytes(&bb, length: bb.count * 4, index: bufs.count)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
dispatch1("fill_scales", [scales], [UInt32(nchMax / 4)], nchMax / 4)
dispatch1("fill_seeds", [seeds], [UInt32(nchMax), 777], nchMax)
let sp = scales.contents().bindMemory(to: Float16.self, capacity: nchMax / 4)
let sw = seeds.contents().bindMemory(to: UInt32.self, capacity: nchMax)
let cw = codes.contents().bindMemory(to: UInt32.self, capacity: nchMax * 4)
let xp = xbuf.contents().bindMemory(to: Float.self, capacity: 18944)
do { var st: UInt32 = 987654321; for i in 0..<18944 { st = xs32(st); xp[i] = Float(Int32(bitPattern: st)) / 2147483648.0 } }

func nib(_ w: UInt32, _ j: Int) -> Double { let s = j / 2; return Double((w >> UInt32(j % 2 == 0 ? 4 * s : 16 + 4 * s)) & 15) }
func byt(_ v0: UInt32, _ v1: UInt32, _ j: Int) -> Double { let p = j / 2; let w = p < 2 ? v0 : v1; let s = UInt32(8 * (p & 1)); return Double((w >> (j % 2 == 0 ? s : 16 + s)) & 255) }
func refRow(_ g: Int, _ cols: Int, _ fmt: Int, _ gf: Int, _ gmode: Int, _ P: Int) -> Double {
    let nch = cols / 8
    let derived = (g & 3) < gf
    var tot = 0.0
    for ch in 0..<nch {
        let gw = g * nch + ch
        let sc = Double(sp[gw >> 2])
        var w8 = [Double](repeating: 0, count: 8)
        var factor = sc
        if derived {
            if gmode == 2 { let wd = hashw(UInt32(gw), seed); for j in 0..<8 { w8[j] = nib(wd, j) - 8 } }
            else if gmode == 1 || gmode == 3 || gmode == 4 {
                let wd = sw[gw]
                var s = ((wd & 0xFFFF) &+ 1) &* 0x9E3779B1; s ^= s >> 15; s |= 1
                if gmode == 4 { s = xs32(s) }
                for pp in 0..<P {
                    var u: UInt32
                    if gmode == 4 { u = pp == 0 ? s : ((s >> UInt32(8 * pp)) | (s << UInt32(32 - 8 * pp))) } else { s = xs32(s); u = s }
                    let tm = Double((wd >> UInt32(16 + 4 * pp)) & 15) - 8
                    for j in 0..<8 { w8[j] += tm * (nib(u, j) - 8) }
                }
            } else {
                let wd = sw[gw]
                let h = (UInt32(gw) &+ 1) &* (((wd & 0xFFFF) << 1) | 1)
                let tm0 = Double((wd >> 16) & 15) - 8, tm1 = Double((wd >> 20) & 15) - 8
                for j in 0..<8 {
                    if gmode == 5 {
                        let e0 = Double((h >> UInt32(16 + 2 * j)) & 3), e1 = Double((h >> UInt32(2 * j)) & 3)
                        w8[j] = tm0 * (e0 - 1.5) + tm1 * (e1 - 1.5)
                    } else {
                        let s0 = ((h >> UInt32(16 + j)) & 1) == 1 ? 1.0 : -1.0, s1 = ((h >> UInt32(j)) & 1) == 1 ? 1.0 : -1.0
                        w8[j] = s0 * tm0 + s1 * tm1
                    }
                }
            }
        } else if fmt == 0 { let wd = cw[gw]; for j in 0..<8 { w8[j] = nib(wd, j) - 8 } }
        else if fmt == 1 { let v0 = cw[gw * 2], v1 = cw[gw * 2 + 1]; for j in 0..<8 { w8[j] = byt(v0, v1, j) - 128 } }
        else {
            factor = 1
            for j in 0..<8 {
                let w = cw[gw * 4 + j / 2]
                let bits = UInt16((j % 2 == 0) ? (w & 0xFFFF) : (w >> 16))
                w8[j] = Double(Float16(bitPattern: bits))
            }
        }
        var s = 0.0
        for j in 0..<8 { s += w8[j] * Double(xp[ch * 8 + j]) }
        tot += factor * s
    }
    return tot
}
func relErr(_ o: MTLBuffer, _ n: Int, _ cols: Int, _ fmt: Int, _ gf: Int, _ gmode: Int, _ P: Int) -> Double {
    let y = o.contents().bindMemory(to: Float.self, capacity: n)
    var md = 0.0, ma = 1e-30
    for g in [0, 1, 2, 3, n / 2, n / 2 + 1, n - 4, n - 3, n - 2, n - 1] {
        let r = refRow(g, cols, fmt, gf, gmode, P)
        md = max(md, abs(r - Double(y[g]))); ma = max(ma, abs(r))
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

let fnames = ["q4_0 benzeri", "q8_0 benzeri", "fp16 benzeri"]
let bpb: [Double] = [18, 34, 64]
let sizes: [(Int, Int)] = [(8192, 8192), (18944, 3584)]
// (ad, gmode, P, ek)
let variants: [(String, Int, Int, String)] = [
    ("rastgele (taban)", 2, 3, "rnd"),
    ("A xorshift P=2", 1, 2, "m1p2"), ("B ertelenmis ofset P=2", 3, 2, "m3p2"), ("C tek-adim rotasyon P=2", 4, 2, "m4p2"),
    ("D tek-carpim hash 2bit P=2", 5, 2, "m5p2"), ("E tek-carpim +-1 secim P=2", 6, 2, "m6p2"),
    ("A xorshift P=3", 1, 3, "m1p3"), ("B ertelenmis ofset P=3", 3, 3, "m3p3"), ("C tek-adim rotasyon P=3", 4, 3, "m4p3"),
]
for F in 0..<3 {
    dispatch1("fill_c\(F)", [codes], [UInt32(nchMax), seed], nchMax)
    print("\n=== \(fnames[F]) ===  (sanal GB/s ; oran = okumaya gore ; GF=4 tum satir, GF=3, GF=2 karisik)")
    for (rows, cols) in sizes {
        let Mx = max(1, T / (rows * cols)), nblk = cols / 32, n = Mx * rows
        var p = MP(rows: UInt32(rows), cols: UInt32(cols), nblk: UInt32(nblk), seed: seed)
        func mk(_ name: String, _ o: MTLBuffer) -> Body {
            let ps = pso(name)
            return { e in
                e.setComputePipelineState(ps)
                e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1)
                e.setBuffer(xbuf, offset: 0, index: 2); e.setBuffer(o, offset: 0, index: 3)
                e.setBytes(&p, length: MemoryLayout<MP>.size, index: 4)
                e.setBuffer(seeds, offset: 0, index: 5)
                e.dispatchThreadgroups(MTLSize(width: rows / 32, height: Mx, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        var names = ["w_f\(F)_g0"]
        for (_, _, _, tag) in variants { for G in [4, 3, 2] { names.append("w_f\(F)_g\(G)_\(tag)") } }
        let bodies = names.enumerated().map { mk($0.element, outs[$0.offset % 20]) }
        var t = [Double]()
        t += timeAll(Array(bodies[0..<14]))
        t += timeAll(Array(bodies[14...]))
        let nb = Double(Mx * rows * nblk)
        func gbs(_ i: Int) -> Double { nb * bpb[F] / t[i] / 1e9 }
        print(String(format: "%-11@ okuma %6.1f GB/s (%.0f Gw/s)", "\(rows)x\(cols)" as NSString, gbs(0), nb * 32 / t[0] / 1e9))
        var errMax = 0.0
        for (vi, (name, gm, P, _)) in variants.enumerated() {
            var line = String(format: "   %-28@", name as NSString)
            for (gi, G) in [4, 3, 2].enumerated() {
                let idx = 1 + vi * 3 + gi
                _ = runOnce(mk(names[idx], outs[0]))
                let e = relErr(outs[0], n, cols, F, G, gm, P)
                errMax = max(errMax, e)
                line += String(format: " GF=%d %6.1f (%.2fx)", G, gbs(idx), t[0] / t[idx])
            }
            print(line)
        }
        print(String(format: "   en buyuk hata %.0e", errMax))
    }
}
