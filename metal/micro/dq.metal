#include <metal_stdlib>
using namespace metal;

#define PREAMBLE \
    uint w = gid * 0x9E3779B1u + 12345u; \
    float4 xa = x[0], xb = x[1]; \
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

#define NEXTW \
    w = w * 1664525u + 1013904223u; \
    uint w0 = w, w1 = w ^ 0x5A5A5A5Au, w2 = w ^ 0x0F0F0F0Fu, w3 = w ^ 0x3C3C3C3Cu;

kernel void dq_cvt(device float *out [[buffer(0)]], device const float4 *x [[buffer(1)]], constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    PREAMBLE
    for (uint i = 0; i < n; i++) {
        NEXTW
        acc0 += dot(float4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), xa) + dot(float4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), xb);
        acc1 += dot(float4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), xa) + dot(float4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), xb);
        acc2 += dot(float4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), xa) + dot(float4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), xb);
        acc3 += dot(float4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), xa) + dot(float4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), xb);
    }
    out[gid] = acc0 + acc1 + acc2 + acc3;
}

inline float4 magic4(uint m) {
    float4 r;
    r.x = as_type<float>((m & 0xFFu) | 0x4B000000u);
    r.y = as_type<float>(((m >> 8) & 0xFFu) | 0x4B000000u);
    r.z = as_type<float>(((m >> 16) & 0xFFu) | 0x4B000000u);
    r.w = as_type<float>((m >> 24) | 0x4B000000u);
    return r - 8388608.0f;
}
kernel void dq_magicf(device float *out [[buffer(0)]], device const float4 *x [[buffer(1)]], constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    PREAMBLE
    for (uint i = 0; i < n; i++) {
        NEXTW
        acc0 += dot(magic4(w0 & 0x0F0F0F0Fu), xa) + dot(magic4((w0 >> 4) & 0x0F0F0F0Fu), xb);
        acc1 += dot(magic4(w1 & 0x0F0F0F0Fu), xa) + dot(magic4((w1 >> 4) & 0x0F0F0F0Fu), xb);
        acc2 += dot(magic4(w2 & 0x0F0F0F0Fu), xa) + dot(magic4((w2 >> 4) & 0x0F0F0F0Fu), xb);
        acc3 += dot(magic4(w3 & 0x0F0F0F0Fu), xa) + dot(magic4((w3 >> 4) & 0x0F0F0F0Fu), xb);
    }
    out[gid] = acc0 + acc1 + acc2 + acc3;
}

inline float2 magich2(uint m) {
    half2 h = as_type<half2>((m & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
    return float2(h);
}
kernel void dq_magich(device float *out [[buffer(0)]], device const float4 *x [[buffer(1)]], constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    PREAMBLE
    float2 x04 = float2(xa.x, xb.x), x15 = float2(xa.y, xb.y), x26 = float2(xa.z, xb.z), x37 = float2(xa.w, xb.w);
    for (uint i = 0; i < n; i++) {
        NEXTW
        #define H(W, a) \
            { float2 t0 = magich2(W), t1 = magich2((W) >> 4), t2 = magich2((W) >> 8), t3 = magich2((W) >> 12); \
              a += dot(t0, x04) + dot(t1, x15) + dot(t2, x26) + dot(t3, x37); }
        H(w0, acc0) H(w1, acc1) H(w2, acc2) H(w3, acc3)
    }
    out[gid] = acc0 + acc1 + acc2 + acc3;
}

kernel void dq_halfall(device float *out [[buffer(0)]], device const float4 *x [[buffer(1)]], constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint w = gid * 0x9E3779B1u + 12345u;
    half2 x04 = half2(half(x[0].x), half(x[1].x)), x15 = half2(half(x[0].y), half(x[1].y)), x26 = half2(half(x[0].z), half(x[1].z)), x37 = half2(half(x[0].w), half(x[1].w));
    half2 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (uint i = 0; i < n; i++) {
        NEXTW
        #define HH(W, a) \
            { half2 t0 = as_type<half2>(((W) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h); \
              half2 t1 = as_type<half2>((((W) >> 4) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h); \
              half2 t2 = as_type<half2>((((W) >> 8) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h); \
              half2 t3 = as_type<half2>((((W) >> 12) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h); \
              a = fma(t0, x04, a); a = fma(t1, x15, a); a = fma(t2, x26, a); a = fma(t3, x37, a); }
        HH(w0, a0) HH(w1, a1) HH(w2, a2) HH(w3, a3)
    }
    half2 s = a0 + a1 + a2 + a3;
    out[gid] = float(s.x) + float(s.y);
}

kernel void dq_lut(device float *out [[buffer(0)]], device const float4 *x [[buffer(1)]], constant uint &n [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
    threadgroup half2 lut[256];
    for (uint i = lid; i < 256; i += 256) lut[i] = half2(half(float(i & 15) - 8.0f), half(float(i >> 4) - 8.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w = gid * 0x9E3779B1u + 12345u;
    half2 x01 = half2(half(x[0].x), half(x[0].y)), x23 = half2(half(x[0].z), half(x[0].w)), x45 = half2(half(x[1].x), half(x[1].y)), x67 = half2(half(x[1].z), half(x[1].w));
    half2 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (uint i = 0; i < n; i++) {
        NEXTW
        #define L(W, a) \
            { a = fma(lut[(W) & 0xFFu], x01, a); a = fma(lut[((W) >> 8) & 0xFFu], x23, a); \
              a = fma(lut[((W) >> 16) & 0xFFu], x45, a); a = fma(lut[(W) >> 24], x67, a); }
        L(w0, a0) L(w1, a1) L(w2, a2) L(w3, a3)
    }
    half2 s = a0 + a1 + a2 + a3;
    out[gid] = float(s.x) + float(s.y);
}

kernel void dq_base(device float *out [[buffer(0)]], device const float4 *x [[buffer(1)]], constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    PREAMBLE
    uint t = 0;
    for (uint i = 0; i < n; i++) {
        NEXTW
        t += (w0 & 0x0F0F0F0Fu) + ((w0 >> 4) & 0x0F0F0F0Fu) + (w1 & 0x0F0F0F0Fu) + ((w1 >> 4) & 0x0F0F0F0Fu)
           + (w2 & 0x0F0F0F0Fu) + ((w2 >> 4) & 0x0F0F0F0Fu) + (w3 & 0x0F0F0F0Fu) + ((w3 >> 4) & 0x0F0F0F0Fu);
    }
    out[gid] = float(t) + xa.x + xb.x;
}
