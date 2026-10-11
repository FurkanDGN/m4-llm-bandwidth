#include <metal_stdlib>
using namespace metal;
#define CH8 a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c); a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
kernel void fma32(device float *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a0 = gid * 1e-9f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    float b0 = a0 + 8, b1 = a0 + 9, b2 = a0 + 10, b3 = a0 + 11, b4 = a0 + 12, b5 = a0 + 13, b6 = a0 + 14, b7 = a0 + 15;
    float c0 = a0 + 16, c1 = a0 + 17, c2 = a0 + 18, c3 = a0 + 19, c4 = a0 + 20, c5 = a0 + 21, c6 = a0 + 22, c7 = a0 + 23;
    float d0 = a0 + 24, d1 = a0 + 25, d2 = a0 + 26, d3 = a0 + 27, d4 = a0 + 28, d5 = a0 + 29, d6 = a0 + 30, d7 = a0 + 31;
    float m = 0.999999f, c = 1e-7f;
    for (uint i = 0; i < n; i++) {
        CH8
        b0 = fma(b0, m, c); b1 = fma(b1, m, c); b2 = fma(b2, m, c); b3 = fma(b3, m, c); b4 = fma(b4, m, c); b5 = fma(b5, m, c); b6 = fma(b6, m, c); b7 = fma(b7, m, c);
        c0 = fma(c0, m, c); c1 = fma(c1, m, c); c2 = fma(c2, m, c); c3 = fma(c3, m, c); c4 = fma(c4, m, c); c5 = fma(c5, m, c); c6 = fma(c6, m, c); c7 = fma(c7, m, c);
        d0 = fma(d0, m, c); d1 = fma(d1, m, c); d2 = fma(d2, m, c); d3 = fma(d3, m, c); d4 = fma(d4, m, c); d5 = fma(d5, m, c); d6 = fma(d6, m, c); d7 = fma(d7, m, c);
    }
    out[gid] = a0+a1+a2+a3+a4+a5+a6+a7+b0+b1+b2+b3+b4+b5+b6+b7+c0+c1+c2+c3+c4+c5+c6+c7+d0+d1+d2+d3+d4+d5+d6+d7;
}
kernel void fma8(device float *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a0 = gid * 1e-9f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    float m = 0.999999f, c = 1e-7f;
    for (uint i = 0; i < n; i++) { CH8 }
    out[gid] = a0+a1+a2+a3+a4+a5+a6+a7;
}
kernel void fma32v(device float4 *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float4 a0 = float4(gid * 1e-9f), a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    float4 m = float4(0.999999f), c = float4(1e-7f);
    for (uint i = 0; i < n; i++) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c); a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    out[gid] = a0+a1+a2+a3+a4+a5+a6+a7;
}
kernel void mma_f32acc(device float *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    simdgroup_half8x8 A = simdgroup_half8x8(half(0.5)), B = simdgroup_half8x8(half(0.25));
    simdgroup_float8x8 c0 = simdgroup_float8x8(0.0f), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (uint i = 0; i < n; i++) {
        simdgroup_multiply_accumulate(c0, A, B, c0); simdgroup_multiply_accumulate(c1, A, B, c1);
        simdgroup_multiply_accumulate(c2, A, B, c2); simdgroup_multiply_accumulate(c3, A, B, c3);
        simdgroup_multiply_accumulate(c4, A, B, c4); simdgroup_multiply_accumulate(c5, A, B, c5);
        simdgroup_multiply_accumulate(c6, A, B, c6); simdgroup_multiply_accumulate(c7, A, B, c7);
    }
    out[gid] = c0.thread_elements()[0] + c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0] + c4.thread_elements()[0] + c5.thread_elements()[0] + c6.thread_elements()[0] + c7.thread_elements()[0];
}
kernel void mma_hacc(device half *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    simdgroup_half8x8 A = simdgroup_half8x8(half(0.5)), B = simdgroup_half8x8(half(0.25));
    simdgroup_half8x8 c0 = simdgroup_half8x8(0.0h), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (uint i = 0; i < n; i++) {
        simdgroup_multiply_accumulate(c0, A, B, c0); simdgroup_multiply_accumulate(c1, A, B, c1);
        simdgroup_multiply_accumulate(c2, A, B, c2); simdgroup_multiply_accumulate(c3, A, B, c3);
        simdgroup_multiply_accumulate(c4, A, B, c4); simdgroup_multiply_accumulate(c5, A, B, c5);
        simdgroup_multiply_accumulate(c6, A, B, c6); simdgroup_multiply_accumulate(c7, A, B, c7);
    }
    out[gid] = c0.thread_elements()[0] + c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0] + c4.thread_elements()[0] + c5.thread_elements()[0] + c6.thread_elements()[0] + c7.thread_elements()[0];
}
