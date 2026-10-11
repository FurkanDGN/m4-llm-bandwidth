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
            else {
                let wd = sw[gw]
                var s = ((wd & 0xFFFF) &+ 1) &* 0x9E3779B1; s ^= s >> 15; s |= 1
                for pp in 0..<P {
                    s = xs32(s)
                    let tm = Double((wd >> UInt32(16 + 4 * pp)) & 15) - 8
                    for j in 0..<8 { w8[j] += tm * (nib(s, j) - 8) }
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
let sizes: [(Int, Int)] = [(4096, 4096), (8192, 8192), (18944, 3584), (3584, 18944)]
let fnames = ["q4_0 benzeri", "q8_0 benzeri", "fp16 benzeri"]
let bpb: [Double] = [18, 34, 64]
var ceiling = 0.0

for F in 0..<3 {
    dispatch1("fill_c\(F)", [codes], [UInt32(nchMax), seed], nchMax)
    do {
        let nU4 = UInt32((nchMax * [4, 8, 16][F] / 16) / (4 << 20) * (4 << 20))
        var n = nU4
        let ps = pso("stream_read")
        let body: Body = { e in
            e.setComputePipelineState(ps); e.setBuffer(codes, offset: 0, index: 0); e.setBuffer(sout, offset: 0, index: 1)
            e.setBytes(&n, length: 4, index: 2)
            e.dispatchThreads(MTLSize(width: 1 << 20, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        for _ in 0..<2 { _ = runOnce(body) }
        var ts: [Double] = []; for _ in 0..<reps { ts.append(runOnce(body, n: iters)) }
        ceiling = Double(nU4) * 16 / median(ts) / 1e9
        print(String(format: "\n=== %@ ===  duz okuma tavani %.1f GB/s -> bellekten en fazla %.0f G agirlik/s", fnames[F] as NSString, ceiling, ceiling / bpb[F] * 32))
    }
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
        var names = ["v9_f\(F)_g0"]
        var meta: [(Int, Int, Int)] = [(0, 2, 3)]
        for G in 1...4 { for (suffix, gm, P) in [("rnd", 2, 3), ("p2", 1, 2), ("p3", 1, 3)] { names.append("v9_f\(F)_g\(G)_\(suffix)"); meta.append((G, gm, P)) } }
        let t = timeAll(names.enumerated().map { mk($0.element, outs[$0.offset]) })
        let nb = Double(Mx * rows * nblk)
        func gbs(_ i: Int) -> Double { nb * bpb[F] / t[i] / 1e9 }
        var errs = 0.0
        for i in 0..<names.count { errs = max(errs, relErr(outs[i], n, cols, F, meta[i].0, meta[i].1, meta[i].2)) }
        print(String(format: "%-11@ okuma %6.1f GB/s (%4.0f Gw/s) | tum satir URETIM: rastgele %6.1f (%4.0f Gw/s, %.2fx) turet P=2 %6.1f (%.2fx) P=3 %6.1f (%.2fx) | en buyuk hata %.0e",
                     "\(rows)x\(cols)" as NSString, gbs(0), nb * 32 / t[0] / 1e9, gbs(10), nb * 32 / t[10] / 1e9, t[0] / t[10], gbs(11), t[0] / t[11], gbs(12), t[0] / t[12], errs))
        if (rows == 8192 || rows == 18944) {
            for (suffix, off) in [("rastgele", 1), ("turet P=2", 2), ("turet P=3", 3)] {
                var line = String(format: "    karisik %-10@ GF=0..4: %6.1f", suffix as NSString, gbs(0))
                for G in 1...4 { line += String(format: " %6.1f", gbs(1 + (G - 1) * 3 + (off - 1))) }
                print(line)
            }
        }
    }
}
