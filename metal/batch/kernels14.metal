#include <metal_stdlib>
using namespace metal;

struct MP { uint rows; uint cols; uint nblk; uint seed; };
constant constexpr uint SGS = 4;

inline uint xs32(uint s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

inline uint4 gen_block(uint gidx, uint seed) {
    uint s = (gidx + seed) * 0x9E3779B1u;
    s ^= s >> 15;
    s |= 1u;
    uint4 w;
    s = xs32(s); w.x = s;
    s = xs32(s); w.y = s;
    s = xs32(s); w.z = s;
    s = xs32(s); w.w = s;
    return w;
}
inline half gen_scale(uint gidx) {
    return half(0.002f + 0.0005f * float(((gidx * 0x85EBCA6Bu) >> 16) & 15u));
}
kernel void fill_q4(device uint4 *codes [[buffer(0)]], device half *scales [[buffer(1)]],
                    constant uint2 &np [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= np.x) return;
    codes[gid] = gen_block(gid, np.y);
    scales[gid] = gen_scale(gid);
}
kernel void fill_seeds(device uint4 *seeds [[buffer(0)]], constant uint2 &np [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= np.x) return;
    uint s = (gid + np.y) * 0x85EBCA6Bu;
    s ^= s >> 13;
    s |= 1u;
    uint4 w;
    s = xs32(s); w.x = s;
    s = xs32(s); w.y = s;
    s = xs32(s); w.z = s;
    s = xs32(s); w.w = s;
    seeds[gid] = w;
}

inline float2 deq2(uint w, uint s) {
    half2 h = as_type<half2>(((w >> s) & 0x000F000Fu) | 0x64006400u);
    return float2(h - half2(1032.0h));
}

// MODE 0: q4 oku, MODE 1: tohum+katsayi turet (P taban, xorshift), MODE 2: rastgele uret,
// MODE 3: tek carpimli hash, +-1 giris, P=2 (4 degerli secim)
// Esleme: lane = bir 32-bit sozcuk (8 agirlik); bir adimda simdgroup 32 bitisik sozcuk = 256 agirlik.
// Satir basina R, sutun sayisi B sabit (sablon). Olcek agirlik duzeyinde uygulanir.
template<uint B, uint R, uint MODE, uint P>
kernel void mm8(device const uint *blk [[buffer(0)]],
                device const half *scales [[buffer(1)]],
                device const float4 *X [[buffer(2)]],
                device float *y [[buffer(3)]],
                constant MP &p [[buffer(4)]],
                uint3 tg [[threadgroup_position_in_grid]],
                uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    uint grow0 = tg.y * p.rows + (tg.x * SGS + sg) * R;
    uint cols4 = p.cols >> 2;
    uint nw = p.nblk * 4;
    float acc[R][B];
    for (uint r = 0; r < R; r++) for (uint c = 0; c < B; c++) acc[r][c] = 0.0f;
    for (uint wi = lane; wi < nw; wi += 32) {
        float4 xl[B], xh[B];
        for (uint c = 0; c < B; c++) {
            xl[c] = X[c * cols4 + wi * 2];
            xh[c] = X[c * cols4 + wi * 2 + 1];
        }
        uint word[R];
        float sc[R];
        for (uint r = 0; r < R; r++) {
            uint gw = (grow0 + r) * nw + wi;
            if (MODE == 2) {
                uint s = (gw + p.seed) * 0x9E3779B1u;
                s ^= s >> 15;
                s |= 1u;
                s = xs32(s);
                word[r] = xs32(s);
            } else {
                word[r] = blk[gw];
            }
            sc[r] = float(scales[gw >> 2]);
        }
        for (uint r = 0; r < R; r++) {
            float2 f0, f1, f2, f3;
            if (MODE == 3) {
                uint w0 = word[r];
                uint gwv = (grow0 + r) * nw + wi;
                uint h = (gwv + 1u) * (((w0 & 0xFFFFu) << 1) | 1u);
                float tm0 = float((w0 >> 16) & 15u) - 8.0f, tm1 = float((w0 >> 20) & 15u) - 8.0f;
                float A = tm0 + tm1, Bv = tm0 - tm1;
                uint m = (h >> 16) ^ h;
                float wj[8];
                for (uint j = 0; j < 8; j++) {
                    float v = ((m >> j) & 1u) ? Bv : A;
                    wj[j] = ((h >> (16 + j)) & 1u) ? v : -v;
                }
                f0 = float2(wj[0], wj[1]); f1 = float2(wj[2], wj[3]); f2 = float2(wj[4], wj[5]); f3 = float2(wj[6], wj[7]);
            } else if (MODE == 1) {
                uint w0 = word[r];
                uint s = ((w0 & 0xFFFFu) + 1u) * 0x9E3779B1u;
                s ^= s >> 15;
                s |= 1u;
                f0 = f1 = f2 = f3 = float2(0.0f);
                for (uint pp = 0; pp < P; pp++) {
                    s = xs32(s);
                    float tm = float((w0 >> (16 + 4 * pp)) & 15u) - 8.0f;
                    f0 = fma(deq2(s, 0), float2(tm), f0);
                    f1 = fma(deq2(s, 4), float2(tm), f1);
                    f2 = fma(deq2(s, 8), float2(tm), f2);
                    f3 = fma(deq2(s, 12), float2(tm), f3);
                }
            } else {
                f0 = deq2(word[r], 0); f1 = deq2(word[r], 4); f2 = deq2(word[r], 8); f3 = deq2(word[r], 12);
            }
            f0 *= sc[r]; f1 *= sc[r]; f2 *= sc[r]; f3 *= sc[r];
            for (uint c = 0; c < B; c++) {
                float a = acc[r][c];
                a = fma(f0.x, xl[c].x, a); a = fma(f0.y, xl[c].y, a);
                a = fma(f1.x, xl[c].z, a); a = fma(f1.y, xl[c].w, a);
                a = fma(f2.x, xh[c].x, a); a = fma(f2.y, xh[c].y, a);
                a = fma(f3.x, xh[c].z, a); a = fma(f3.y, xh[c].w, a);
                acc[r][c] = a;
            }
        }
    }
    for (uint r = 0; r < R; r++) {
        for (uint c = 0; c < B; c++) {
            float t = simd_sum(acc[r][c]);
            if (lane == 0) y[(grow0 + r) * B + c] = t;
        }
    }
}
