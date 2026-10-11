#include <metal_stdlib>
using namespace metal;
kernel void fma_peak(device float *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a0 = gid * 1e-9f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    float m = 0.999999f, c = 1e-7f;
    for (uint i = 0; i < n; i++) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
kernel void half_peak(device half *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    half a0 = half(gid) * 1e-4h, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    half m = 0.999h, c = 1e-3h;
    for (uint i = 0; i < n; i++) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
kernel void int_peak(device uint *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    uint a0 = gid, a1 = gid + 1, a2 = gid + 2, a3 = gid + 3, a4 = gid + 4, a5 = gid + 5, a6 = gid + 6, a7 = gid + 7;
    for (uint i = 0; i < n; i++) {
        a0 ^= a0 << 13; a1 ^= a1 << 13; a2 ^= a2 << 13; a3 ^= a3 << 13; a4 ^= a4 << 13; a5 ^= a5 << 13; a6 ^= a6 << 13; a7 ^= a7 << 13;
        a0 ^= a0 >> 17; a1 ^= a1 >> 17; a2 ^= a2 >> 17; a3 ^= a3 >> 17; a4 ^= a4 >> 17; a5 ^= a5 >> 17; a6 ^= a6 >> 17; a7 ^= a7 >> 17;
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
kernel void cvt_peak(device float *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    uint w = gid * 0x9E3779B1u;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < n; i++) {
        w = w * 1664525u + 1013904223u;
        acc += float4(as_type<uchar4>(w & 0x0F0F0F0Fu));
        acc += float4(as_type<uchar4>((w >> 4) & 0x0F0F0F0Fu));
    }
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
