#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint nblk;
    uint seed;
};

constant constexpr uint R = 8;
constant constexpr uint SGS = 4;

inline uint xs32(uint s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

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

inline float block_acc(uint4 w, thread const float4 *xv) {
    float acc = 0.0f;
    for (uint k = 0; k < 4; k++) {
        uint word = w[k];
        uchar4 lo = as_type<uchar4>(word & 0x0F0F0F0Fu);
        uchar4 hi = as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu);
        acc += dot(float4(lo), xv[2 * k]) + dot(float4(hi), xv[2 * k + 1]);
    }
    return acc;
}

kernel void fill_codes(device uint4 *codes [[buffer(0)]],
                       device half *scales [[buffer(1)]],
                       constant uint2 &np [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= np.x) return;
    codes[gid] = gen_block(gid, np.y);
    scales[gid] = gen_scale(gid);
}

kernel void mv_read(device const uint4 *codes [[buffer(0)]],
                    device const half *scales [[buffer(1)]],
                    device const float4 *x [[buffer(2)]],
                    device float *y [[buffer(3)]],
                    constant Params &p [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    uint grow0 = tg.y * p.rows + (tg.x * SGS + sg) * R;
    float sum[R];
    for (uint r = 0; r < R; r++) sum[r] = 0.0f;
    for (uint b = lane; b < p.nblk; b += 32) {
        float4 xv[8];
        float xsum = 0.0f;
        for (uint i = 0; i < 8; i++) {
            xv[i] = x[b * 8 + i];
            xsum += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
        }
        uint4 w[R];
        half sc[R];
        for (uint r = 0; r < R; r++) {
            uint gb = (grow0 + r) * p.nblk + b;
            w[r] = codes[gb];
            sc[r] = scales[gb];
        }
        for (uint r = 0; r < R; r++) {
            sum[r] += float(sc[r]) * (block_acc(w[r], xv) - 8.0f * xsum);
        }
    }
    for (uint r = 0; r < R; r++) {
        float t = simd_sum(sum[r]);
        if (lane == 0) y[grow0 + r] = t;
    }
}

kernel void mv_gen(device const half *scales [[buffer(0)]],
                   device const float4 *x [[buffer(1)]],
                   device float *y [[buffer(2)]],
                   constant Params &p [[buffer(3)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    uint grow0 = tg.y * p.rows + (tg.x * SGS + sg) * R;
    float sum[R];
    for (uint r = 0; r < R; r++) sum[r] = 0.0f;
    for (uint b = lane; b < p.nblk; b += 32) {
        float4 xv[8];
        float xsum = 0.0f;
        for (uint i = 0; i < 8; i++) {
            xv[i] = x[b * 8 + i];
            xsum += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
        }
        half sc[R];
        for (uint r = 0; r < R; r++) {
            sc[r] = scales[(grow0 + r) * p.nblk + b];
        }
        for (uint r = 0; r < R; r++) {
            uint4 w = gen_block((grow0 + r) * p.nblk + b, p.seed);
            sum[r] += float(sc[r]) * (block_acc(w, xv) - 8.0f * xsum);
        }
    }
    for (uint r = 0; r < R; r++) {
        float t = simd_sum(sum[r]);
        if (lane == 0) y[grow0 + r] = t;
    }
}

kernel void stream_read(device const uint4 *buf [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        constant uint &n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint nt [[threads_per_grid]]) {
    uint4 a = uint4(0);
    for (uint i = gid; i + 3 * nt < n; i += 4 * nt) {
        uint4 v0 = buf[i];
        uint4 v1 = buf[i + nt];
        uint4 v2 = buf[i + 2 * nt];
        uint4 v3 = buf[i + 3 * nt];
        a ^= v0 ^ v1 ^ v2 ^ v3;
    }
    out[gid] = a.x ^ a.y ^ a.z ^ a.w;
}

constant uint MODE [[function_constant(0)]];
constant uint GFRAC [[function_constant(1)]];

kernel void mv_var(device const uint4 *codes [[buffer(0)]],
                   device const half *scales [[buffer(1)]],
                   device const float4 *x [[buffer(2)]],
                   device float *y [[buffer(3)]],
                   constant Params &p [[buffer(4)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    uint grow0 = tg.y * p.rows + (tg.x * SGS + sg) * R;
    float sum[R];
    uint ux[R];
    for (uint r = 0; r < R; r++) { sum[r] = 0.0f; ux[r] = 0u; }
    float4 xv[8];
    float xsum = 0.0f;
    if (MODE == 3) {
        for (uint i = 0; i < 8; i++) {
            xv[i] = x[lane * 8 + i];
            xsum += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
        }
    }
    for (uint b = lane; b < p.nblk; b += 32) {
        if (MODE != 3) {
            xsum = 0.0f;
            for (uint i = 0; i < 8; i++) {
                xv[i] = x[b * 8 + i];
                xsum += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
            }
        }
        uint4 w[R];
        half sc[R];
        for (uint r = 0; r < R; r++) {
            uint gb = (grow0 + r) * p.nblk + b;
            if ((r & 3u) >= GFRAC) w[r] = codes[gb];
            if (MODE == 4) sc[r] = half(0.003f); else sc[r] = scales[gb];
        }
        for (uint r = 0; r < R; r++) {
            uint gb = (grow0 + r) * p.nblk + b;
            if ((r & 3u) < GFRAC) {
                if (MODE == 1) {
                    w[r] = uint4(gb, gb ^ 0x5A5A5A5Au, gb ^ 0x0F0F0F0Fu, gb ^ 0x3C3C3C3Cu);
                } else {
                    w[r] = gen_block(gb, p.seed);
                }
            }
            if (MODE == 2) {
                ux[r] ^= w[r].x ^ w[r].y ^ w[r].z ^ w[r].w;
            } else {
                sum[r] += float(sc[r]) * (block_acc(w[r], xv) - 8.0f * xsum);
            }
        }
    }
    for (uint r = 0; r < R; r++) {
        float t = (MODE == 2) ? float(ux[r] & 0xFFFFu) : sum[r];
        t = simd_sum(t);
        if (lane == 0) y[grow0 + r] = t;
    }
}
