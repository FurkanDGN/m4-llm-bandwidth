#include <metal_stdlib>
using namespace metal;

struct MP { uint rows; uint cols; uint nblk; uint ystride; };

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

template<uint B, uint R, uint MODE, uint COLG>
kernel void mm_s(device const uint4 *blk [[buffer(0)]],
                 device const half *scales [[buffer(1)]],
                 device const float4 *X [[buffer(2)]],
                 device float *y [[buffer(3)]],
                 constant MP &p [[buffer(4)]],
                 uint3 tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    uint rt = COLG ? tg.y : tg.x;
    uint mat = COLG ? tg.z : tg.y;
    uint col0 = COLG ? tg.x : 0;
    uint grow0 = mat * p.rows + (rt * SGS + sg) * R;
    uint cols4 = p.cols >> 2;
    float acc[R][B];
    for (uint r = 0; r < R; r++) for (uint c = 0; c < B; c++) acc[r][c] = 0.0f;
    for (uint b = lane; b < p.nblk; b += 32) {
        uint4 w[R];
        float sc[R];
        float sc8[R];
        for (uint r = 0; r < R; r++) {
            uint gb = (grow0 + r) * p.nblk + b;
            w[r] = blk[gb];
            sc[r] = float(scales[gb]);
            sc8[r] = -8.0f * sc[r];
        }
        for (uint k = 0; k < 4; k++) {
            float4 xl[B], xh[B];
            for (uint c = 0; c < B; c++) {
                uint cc = COLG ? col0 : c;
                xl[c] = X[cc * cols4 + b * 8 + 2 * k];
                xh[c] = X[cc * cols4 + b * 8 + 2 * k + 1];
            }
            for (uint r = 0; r < R; r++) {
                float4 wl, wh;
                uint word = w[r][k];
                if (MODE == 0) {
                    float4 lo = float4(as_type<uchar4>(word & 0x0F0F0F0Fu));
                    float4 hi = float4(as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu));
                    wl = fma(lo, float4(sc[r]), float4(sc8[r]));
                    wh = fma(hi, float4(sc[r]), float4(sc8[r]));
                } else {
                    uint s = ((word & 0xFFFFu) + 1u) * 0x9E3779B1u;
                    s ^= s >> 15;
                    s |= 1u;
                    float4 al = float4(0.0f), ah = float4(0.0f);
                    float ts = 0.0f;
                    for (uint pp = 0; pp < 3; pp++) {
                        s = xs32(s);
                        float tm = float((word >> (16 + 4 * pp)) & 15u) - 8.0f;
                        ts += tm;
                        al = fma(float4(as_type<uchar4>(s & 0x0F0F0F0Fu)), float4(tm), al);
                        ah = fma(float4(as_type<uchar4>((s >> 4) & 0x0F0F0F0Fu)), float4(tm), ah);
                    }
                    float off = 8.0f * ts;
                    wl = (al - off) * sc[r];
                    wh = (ah - off) * sc[r];
                }
                for (uint c = 0; c < B; c++) {
                    acc[r][c] += dot(wl, xl[c]) + dot(wh, xh[c]);
                }
            }
        }
    }
    for (uint r = 0; r < R; r++) {
        for (uint c = 0; c < B; c++) {
            float t = simd_sum(acc[r][c]);
            if (lane == 0) {
                if (COLG) y[col0 * p.ystride + grow0 + r] = t;
                else y[(grow0 + r) * B + c] = t;
            }
        }
    }
}
