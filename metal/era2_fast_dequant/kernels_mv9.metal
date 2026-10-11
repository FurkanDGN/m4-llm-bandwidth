#include <metal_stdlib>
using namespace metal;

struct MP { uint rows; uint cols; uint nblk; uint seed; };
constant constexpr uint SGS = 4;
inline uint xs32(uint s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
inline half gen_scale(uint gidx) {
    return half(0.002f + 0.0005f * float(((gidx * 0x85EBCA6Bu) >> 16) & 15u));
}
inline uint hashw(uint i, uint seed) {
    uint s = (i + seed) * 0x9E3779B1u;
    s ^= s >> 15;
    s |= 1u;
    return xs32(xs32(s));
}
template<uint FMT>
kernel void fill_codes(device uint *codes [[buffer(0)]], constant uint2 &np [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= np.x) return;
    constexpr uint NW = FMT == 0 ? 1 : (FMT == 1 ? 2 : 4);
    uint s = (gid + np.y) * 0x9E3779B1u;
    s ^= s >> 15;
    s |= 1u;
    for (uint i = 0; i < NW; i++) {
        s = xs32(s);
        codes[gid * NW + i] = (FMT == 2) ? ((s & 0x83FF83FFu) | 0x38003800u) : s;
    }
}
kernel void fill_scales(device half *scales [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) scales[gid] = gen_scale(gid);
}
kernel void fill_seeds(device uint *seeds [[buffer(0)]], constant uint2 &np [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < np.x) seeds[gid] = hashw(gid, np.y);
}
inline float2 deq2(uint w, uint s) {
    half2 h = as_type<half2>(((w >> s) & 0x000F000Fu) | 0x64006400u);
    return float2(h - half2(1032.0h));
}
inline float2 deq8(uint w, uint s) {
    half2 h = as_type<half2>(((w >> s) & 0x00FF00FFu) | 0x64006400u);
    return float2(h - half2(1152.0h));
}
inline float2 raw2(uint w, uint s) { return float2(as_type<half2>(((w >> s) & 0x000F000Fu) | 0x64006400u)); }

// FMT: okunan format (0 q4, 1 q8, 2 fp16). GF: her 4 satirdan kacinin uretilecegi.
// GMODE: 1 xorshift turet(P), 2 rastgele, 3 ertelenmis ofset, 4 tek-adim rotasyon, 5 tek carpimli hash 2 bit, 6 tek carpimli hash +-1 secim
template<uint FMT, uint R, uint GF, uint GMODE, uint P>
kernel void mv9(device const uint *codes [[buffer(0)]],
                device const half *scales [[buffer(1)]],
                device const float4 *X [[buffer(2)]],
                device float *y [[buffer(3)]],
                constant MP &p [[buffer(4)]],
                device const uint *seeds [[buffer(5)]],
                uint3 tg [[threadgroup_position_in_grid]],
                uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    uint grow0 = tg.y * p.rows + (tg.x * SGS + sg) * R;
    uint nch = p.cols >> 3;
    float acc[R];
    for (uint r = 0; r < R; r++) acc[r] = 0.0f;
    for (uint ci = lane; ci < nch; ci += 32) {
        float4 xa = X[ci * 2], xb = X[ci * 2 + 1];
        float2 v0, v1, v2, v3;
        float2 ds[R][4];
        float sc[R];
        uint4 raw[R];
        for (uint r = 0; r < R; r++) {
            uint gw = (grow0 + r) * nch + ci;
            if ((r & 3u) < GF) {
                sc[r] = float(scales[gw >> 2]);
                if (GMODE == 2) raw[r] = uint4(hashw(gw, p.seed), 0, 0, 0); else raw[r] = uint4(seeds[gw], 0, 0, 0);
            } else if (FMT == 0) {
                raw[r] = uint4(codes[gw], 0, 0, 0); sc[r] = float(scales[gw >> 2]);
            } else if (FMT == 1) {
                raw[r] = uint4(*(device const uint2 *)(codes + gw * 2), 0, 0); sc[r] = float(scales[gw >> 2]);
            } else {
                raw[r] = *(device const uint4 *)(codes + gw * 4); sc[r] = 1.0f;
            }
        }
        for (uint r = 0; r < R; r++) {
            float2 f0, f1, f2, f3;
            if ((r & 3u) < GF) {
                uint wd = raw[r].x;
                uint gw = (grow0 + r) * nch + ci;
                if (GMODE == 2) {
                    f0 = deq2(wd, 0); f1 = deq2(wd, 4); f2 = deq2(wd, 8); f3 = deq2(wd, 12);
                } else if (GMODE == 1) {
                    uint s = ((wd & 0xFFFFu) + 1u) * 0x9E3779B1u;
                    s ^= s >> 15;
                    s |= 1u;
                    f0 = f1 = f2 = f3 = float2(0.0f);
                    for (uint pp = 0; pp < P; pp++) {
                        s = xs32(s);
                        float tm = float((wd >> (16 + 4 * pp)) & 15u) - 8.0f;
                        f0 = fma(deq2(s, 0), float2(tm), f0);
                        f1 = fma(deq2(s, 4), float2(tm), f1);
                        f2 = fma(deq2(s, 8), float2(tm), f2);
                        f3 = fma(deq2(s, 12), float2(tm), f3);
                    }
                } else if (GMODE == 3) {
                    uint s = ((wd & 0xFFFFu) + 1u) * 0x9E3779B1u;
                    s ^= s >> 15;
                    s |= 1u;
                    f0 = f1 = f2 = f3 = float2(0.0f);
                    float ts = 0.0f;
                    for (uint pp = 0; pp < P; pp++) {
                        s = xs32(s);
                        float tm = float((wd >> (16 + 4 * pp)) & 15u) - 8.0f;
                        ts += tm;
                        f0 = fma(raw2(s, 0), float2(tm), f0);
                        f1 = fma(raw2(s, 4), float2(tm), f1);
                        f2 = fma(raw2(s, 8), float2(tm), f2);
                        f3 = fma(raw2(s, 12), float2(tm), f3);
                    }
                    float2 off = float2(1032.0f * ts);
                    f0 -= off; f1 -= off; f2 -= off; f3 -= off;
                } else if (GMODE == 4) {
                    uint s = ((wd & 0xFFFFu) + 1u) * 0x9E3779B1u;
                    s ^= s >> 15;
                    s |= 1u;
                    s = xs32(s);
                    f0 = f1 = f2 = f3 = float2(0.0f);
                    for (uint pp = 0; pp < P; pp++) {
                        uint u = pp == 0 ? s : ((s >> (8 * pp)) | (s << (32 - 8 * pp)));
                        float tm = float((wd >> (16 + 4 * pp)) & 15u) - 8.0f;
                        f0 = fma(deq2(u, 0), float2(tm), f0);
                        f1 = fma(deq2(u, 4), float2(tm), f1);
                        f2 = fma(deq2(u, 8), float2(tm), f2);
                        f3 = fma(deq2(u, 12), float2(tm), f3);
                    }
                } else if (GMODE == 5) {
                    uint h = (gw + 1u) * (((wd & 0xFFFFu) << 1) | 1u);
                    float tm0 = float((wd >> 16) & 15u) - 8.0f, tm1 = float((wd >> 20) & 15u) - 8.0f;
                    float c0 = -1.5f * (tm0 + tm1);
                    float wj[8];
                    for (uint j = 0; j < 8; j++) {
                        half2 hh = as_type<half2>(((h >> (2 * j)) & 0x00030003u) | 0x64006400u) - half2(1024.0h);
                        float2 e = float2(hh);
                        wj[j] = fma(e.x, tm1, fma(e.y, tm0, c0));
                    }
                    f0 = float2(wj[0], wj[1]); f1 = float2(wj[2], wj[3]); f2 = float2(wj[4], wj[5]); f3 = float2(wj[6], wj[7]);
                } else {
                    uint h = (gw + 1u) * (((wd & 0xFFFFu) << 1) | 1u);
                    float tm0 = float((wd >> 16) & 15u) - 8.0f, tm1 = float((wd >> 20) & 15u) - 8.0f;
                    float A = tm0 + tm1, Bv = tm0 - tm1;
                    uint m = (h >> 16) ^ h;
                    float wj[8];
                    for (uint j = 0; j < 8; j++) {
                        float v = ((m >> j) & 1u) ? Bv : A;
                        wj[j] = ((h >> (16 + j)) & 1u) ? v : -v;
                    }
                    f0 = float2(wj[0], wj[1]); f1 = float2(wj[2], wj[3]); f2 = float2(wj[4], wj[5]); f3 = float2(wj[6], wj[7]);
                }
            } else if (FMT == 0) {
                uint wd = raw[r].x;
                f0 = deq2(wd, 0); f1 = deq2(wd, 4); f2 = deq2(wd, 8); f3 = deq2(wd, 12);
            } else if (FMT == 1) {
                f0 = deq8(raw[r].x, 0); f1 = deq8(raw[r].x, 8); f2 = deq8(raw[r].y, 0); f3 = deq8(raw[r].y, 8);
            } else {
                f0 = float2(as_type<half2>(raw[r].x)); f1 = float2(as_type<half2>(raw[r].y));
                f2 = float2(as_type<half2>(raw[r].z)); f3 = float2(as_type<half2>(raw[r].w));
            }
            float a = f0.x * xa.x;
            a = fma(f0.y, xa.y, a); a = fma(f1.x, xa.z, a); a = fma(f1.y, xa.w, a);
            a = fma(f2.x, xb.x, a); a = fma(f2.y, xb.y, a); a = fma(f3.x, xb.z, a); a = fma(f3.y, xb.w, a);
            if (FMT == 2 && (r & 3u) >= GF) acc[r] += a; else acc[r] = fma(a, sc[r], acc[r]);
        }
    }
    for (uint r = 0; r < R; r++) {
        float t = simd_sum(acc[r]);
        if (lane == 0) y[grow0 + r] = t;
    }
}

template [[host_name("fill_c0")]] kernel decltype(fill_codes<0>) fill_codes<0>;
template [[host_name("fill_c1")]] kernel decltype(fill_codes<1>) fill_codes<1>;
template [[host_name("fill_c2")]] kernel decltype(fill_codes<2>) fill_codes<2>;

kernel void stream_read(device const uint4 *buf [[buffer(0)]], device uint *out [[buffer(1)]],
                        constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) {
    uint4 a = uint4(0);
    for (uint i = gid; i + 3 * nt < n; i += 4 * nt) { a ^= buf[i] ^ buf[i + nt] ^ buf[i + 2 * nt] ^ buf[i + 3 * nt]; }
    out[gid] = a.x ^ a.y ^ a.z ^ a.w;
}
