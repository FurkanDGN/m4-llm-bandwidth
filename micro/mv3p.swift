import Foundation
import Metal
// E21: q3p (3.5 bpw) vs q4m (4.5 bpw) mat-vec mikro. kullanim: ./mv3p mv3p.metal [warmMs]
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: nil)
let warmMs = CommandLine.arguments.count >= 3 ? Int(CommandLine.arguments[2])! : 600

let R = 16384, K = 16384
let nch = K/32, nq2r = K/16, nq4r = K/8
var Ku = UInt32(K), Ru = UInt32(R)

func buf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: bytes, options: .storageModeShared)! }
let bq2 = buf(R*nq2r*4), bhm = buf(R*nch*4), bdd = buf(R*nch*2), bq4 = buf(R*nq4r*4)
let by = buf(K*4), bdst = buf(R*4)

// doldurma: her sayfaya dokun, deterministik LCG
func fillU32(_ b: MTLBuffer, _ n: Int, _ seed: UInt32) {
    let p = b.contents().bindMemory(to: UInt32.self, capacity: n)
    var s = seed
    for i in 0..<n { s = s &* 1664525 &+ 1013904223; p[i] = s }
}
fillU32(bq2, R*nq2r, 11); fillU32(bhm, R*nch, 22); fillU32(bq4, R*nq4r, 33)
do { let p = bdd.contents().bindMemory(to: Float16.self, capacity: R*nch)
     var s: UInt32 = 44
     for i in 0..<(R*nch) { s = s &* 1664525 &+ 1013904223; p[i] = Float16(0.5 + Double(s % 1000)/1000.0) } }
do { let p = by.contents().bindMemory(to: Float.self, capacity: K)
     var s: UInt32 = 55
     for i in 0..<K { s = s &* 1664525 &+ 1013904223; p[i] = Float(Double(s % 2001)/1000.0 - 1.0) } }

func once(_ name: String) -> Double {
    let pl = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pl)
    e.setBuffer(name == "mv3p" ? bq2 : bq4, offset: 0, index: 0)
    e.setBuffer(bhm, offset: 0, index: 1); e.setBuffer(bdd, offset: 0, index: 2)
    e.setBuffer(by, offset: 0, index: 3);  e.setBuffer(bdst, offset: 0, index: 4)
    e.setBytes(&Ku, length: 4, index: 5);  e.setBytes(&Ru, length: 4, index: 6)
    e.dispatchThreadgroups(MTLSize(width: R/16, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}

// CPU dogrulama (cift duyarlik, 2 satir)
func cpuRow3p(_ row: Int) -> Double {
    let q2 = bq2.contents().bindMemory(to: UInt32.self, capacity: R*nq2r)
    let hm = bhm.contents().bindMemory(to: UInt32.self, capacity: R*nch)
    let dd = bdd.contents().bindMemory(to: Float16.self, capacity: R*nch)
    let y  = by.contents().bindMemory(to: Float.self, capacity: K)
    var s = 0.0
    for ci in 0..<nch {
        let a = q2[row*nq2r + 2*ci], b = q2[row*nq2r + 2*ci + 1], h = hm[row*nch + ci]
        let d = Double(dd[row*nch + ci]); var acc = 0.0
        for k in 0..<32 {
            var q2v: UInt32; var hb: UInt32
            switch k {
            case 0..<8:    q2v = (a >> UInt32(2*k)) & 3;        hb = (h >> UInt32(k)) & 1
            case 8..<16:   q2v = (b >> UInt32(2*(k-8))) & 3;    hb = (h >> UInt32(k)) & 1
            case 16..<24:  q2v = (a >> UInt32(16+2*(k-16))) & 3; hb = (h >> UInt32(k)) & 1
            default:       q2v = (b >> UInt32(16+2*(k-24))) & 3; hb = (h >> UInt32(k)) & 1
            }
            acc += Double(Int(q2v | (hb << 2)) - 4) * Double(y[ci*32 + k])
        }
        s += d * acc
    }
    return s
}
func cpuRow4m(_ row: Int) -> Double {
    let q4 = bq4.contents().bindMemory(to: UInt32.self, capacity: R*nq4r)
    let dd = bdd.contents().bindMemory(to: Float16.self, capacity: R*nch)
    let y  = by.contents().bindMemory(to: Float.self, capacity: K)
    var s = 0.0
    for ci in 0..<nch {
        let d = Double(dd[row*nch + ci]); var acc = 0.0
        for u in 0..<4 {
            let w = q4[row*nq4r + 4*ci + u]
            let yb = ci*32 + 8*u
            let ks: [(Int, Int)] = [(0,0),(4,1),(8,4),(12,5),(16,2),(20,3),(24,6),(28,7)]
            for (sh, yo) in ks {
                acc += Double(Int((w >> UInt32(sh)) & 15) - 8) * Double(y[yb + yo])
            }
        }
        s += d * acc
    }
    return s
}

for name in ["mv3p", "mv4m"] {
    _ = once(name)
    let dst = bdst.contents().bindMemory(to: Float.self, capacity: R)
    for row in [1, R/2 + 1] {
        let ref = name == "mv3p" ? cpuRow3p(row) : cpuRow4m(row)
        let got = Double(dst[row])
        let rel = abs(got - ref) / max(1e-6, abs(ref))
        print(String(format: "%@ dogruluk satir %d: gpu %.4f cpu %.4f gorelifark %.2e %@",
                     name, row, got, ref, rel, rel < 1e-3 ? "OK" : "HATA"))
    }
}

let t0 = Date()
while Date().timeIntervalSince(t0)*1000 < Double(warmMs) { _ = once("mv3p") }

for name in ["mv3p", "mv4m"] {
    _ = once(name); _ = once(name)
    var ts: [Double] = []
    for _ in 0..<7 { ts.append(once(name)) }
    let med = ts.sorted()[3]
    let wbytes = name == "mv3p" ? Double(R*nq2r*4 + R*nch*4 + R*nch*2) : Double(R*nq4r*4 + R*nch*2)
    let gw = Double(R)*Double(K)/med/1e9
    print(String(format: "%@  medyan %.3f ms  etkin %6.1f GB/s  %6.1f Gw/s", name, med*1000, wbytes/med/1e9, gw))
}
