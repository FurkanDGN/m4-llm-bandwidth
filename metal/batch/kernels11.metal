#include <metal_stdlib>
using namespace metal;

struct MP { uint rows; uint cols; uint nblk; uint seed; };
constant constexpr uint SGS = 4;
inline uint xs32(uint s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

// MODE 0: q4 oku, MODE 1: tohum+katsayi turet (P=3, xorshift), MODE 2: rastgele uret, MODE 3: tek carpimli hash +-1 secim (P=2)
// Esleme: simdgroup 8*TM satir; her lane matris elemani (fm satiri, fn,fn+1 k-cifti); K blok blok (32 k = 4 adim)
template<uint TM, uint NCT, uint MODE>
kernel void mma_q4(device const uint4 *blk [[buffer(0)]],
                   device const half *scales [[buffer(1)]],
                   device const half *Xt [[buffer(2)]],      // [k][NCT*8] half
                   device float *y [[buffer(3)]],            // [satir][NCT*8]
                   constant MP &p [[buffer(4)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint LD = NCT * 8;
    uint row0 = (tg.x * SGS + sg) * (8 * TM);
    uint grow0 = tg.y * p.rows + row0;
    uint qid = lane >> 2;
    uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn = ((qid & 2u) << 1) + ((lane & 1u) << 1);
    simdgroup_float8x8 C[TM][NCT];
    for (uint t = 0; t < TM; t++) for (uint c = 0; c < NCT; c++) C[t][c] = simdgroup_float8x8(0.0f);
    for (uint b = 0; b < p.nblk; b++) {
        uint4 w[TM];
        float sc[TM];
        for (uint t = 0; t < TM; t++) {
            uint gb = (grow0 + 8 * t + fm) * p.nblk + b;
            if (MODE == 2) w[t] = uint4(0); else w[t] = blk[gb];
            sc[t] = float(scales[gb]);
        }
        simdgroup_float8x8 T2[TM][NCT];
        for (uint t = 0; t < TM; t++) for (uint c = 0; c < NCT; c++) T2[t][c] = simdgroup_float8x8(0.0f);
        for (uint j = 0; j < 4; j++) {
            simdgroup_half8x8 Bt[NCT];
            for (uint c = 0; c < NCT; c++) simdgroup_load(Bt[c], Xt + (b * 32 + j * 8) * LD + c * 8, LD);
            for (uint t = 0; t < TM; t++) {
                half2 h;
                if (MODE == 0) {
                    h = as_type<half2>(((w[t][j] >> (2 * fn)) & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);
                } else if (MODE == 2) {
                    uint gw = ((grow0 + 8 * t + fm) * p.nblk + b) * 4 + j;
                    uint s = (gw + p.seed) * 0x9E3779B1u;
                    s ^= s >> 15;
                    s |= 1u;
                    s = xs32(s);
                    s = xs32(s);
                    h = as_type<half2>(((s >> (2 * fn)) & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);
                } else if (MODE == 3) {
                    uint wd = w[t][j];
                    uint gw = ((grow0 + 8 * t + fm) * p.nblk + b) * 4 + j;
                    uint hh = (gw + 1u) * (((wd & 0xFFFFu) << 1) | 1u);
                    float tm0 = float((wd >> 16) & 15u) - 8.0f, tm1 = float((wd >> 20) & 15u) - 8.0f;
                    float A = tm0 + tm1, Bv = tm0 - tm1;
                    uint m = (hh >> 16) ^ hh;
                    float v0 = ((m >> fn) & 1u) ? Bv : A;
                    float v1 = ((m >> (fn + 1)) & 1u) ? Bv : A;
                    float w0 = ((hh >> (16 + fn)) & 1u) ? v0 : -v0;
                    float w1 = ((hh >> (17 + fn)) & 1u) ? v1 : -v1;
                    h = half2(half(w0), half(w1));
                } else {
                    uint wd = w[t][j];
                    uint s = ((wd & 0xFFFFu) + 1u) * 0x9E3779B1u;
                    s ^= s >> 15;
                    s |= 1u;
                    float2 f = float2(0.0f);
                    for (uint pp = 0; pp < 3; pp++) {
                        s = xs32(s);
                        float tm = float((wd >> (16 + 4 * pp)) & 15u) - 8.0f;
                        half2 u = as_type<half2>(((s >> (2 * fn)) & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);
                        f = fma(float2(u), float2(tm), f);
                    }
                    h = half2(f);
                }
                simdgroup_half8x8 A;
                A.thread_elements()[0] = h.x;
                A.thread_elements()[1] = h.y;
                for (uint c = 0; c < NCT; c++) simdgroup_multiply_accumulate(T2[t][c], A, Bt[c], T2[t][c]);
            }
        }
        for (uint t = 0; t < TM; t++) for (uint c = 0; c < NCT; c++) {
            C[t][c].thread_elements()[0] = fma(T2[t][c].thread_elements()[0], sc[t], C[t][c].thread_elements()[0]);
            C[t][c].thread_elements()[1] = fma(T2[t][c].thread_elements()[1], sc[t], C[t][c].thread_elements()[1]);
        }
    }
    for (uint t = 0; t < TM; t++) for (uint c = 0; c < NCT; c++)
        simdgroup_store(C[t][c], y + (grow0 + 8 * t) * LD + c * 8, LD);
}
template [[host_name("mma_tm4_n1_m0")]] kernel decltype(mma_q4<4, 1, 0>) mma_q4<4, 1, 0>;
template [[host_name("mma_tm4_n1_m1")]] kernel decltype(mma_q4<4, 1, 1>) mma_q4<4, 1, 1>;
template [[host_name("mma_tm4_n1_m2")]] kernel decltype(mma_q4<4, 1, 2>) mma_q4<4, 1, 2>;
template [[host_name("mma_tm4_n2_m0")]] kernel decltype(mma_q4<4, 2, 0>) mma_q4<4, 2, 0>;
template [[host_name("mma_tm4_n2_m1")]] kernel decltype(mma_q4<4, 2, 1>) mma_q4<4, 2, 1>;
template [[host_name("mma_tm4_n2_m2")]] kernel decltype(mma_q4<4, 2, 2>) mma_q4<4, 2, 2>;
template [[host_name("mma_tm2_n1_m0")]] kernel decltype(mma_q4<2, 1, 0>) mma_q4<2, 1, 0>;
template [[host_name("mma_tm2_n2_m0")]] kernel decltype(mma_q4<2, 2, 0>) mma_q4<2, 2, 0>;
template [[host_name("mma_tm1_n1_m0")]] kernel decltype(mma_q4<1, 1, 0>) mma_q4<1, 1, 0>;
template [[host_name("mma_tm1_n2_m0")]] kernel decltype(mma_q4<1, 2, 0>) mma_q4<1, 2, 0>;
template [[host_name("mma_tm1_n1_m3")]] kernel decltype(mma_q4<1, 1, 3>) mma_q4<1, 1, 3>;
template [[host_name("mma_tm1_n2_m3")]] kernel decltype(mma_q4<1, 2, 3>) mma_q4<1, 2, 3>;
template [[host_name("mma_tm2_n1_m3")]] kernel decltype(mma_q4<2, 1, 3>) mma_q4<2, 1, 3>;
template [[host_name("mma_tm2_n2_m3")]] kernel decltype(mma_q4<2, 2, 3>) mma_q4<2, 2, 3>;
template [[host_name("mma_tm4_n1_m3")]] kernel decltype(mma_q4<4, 1, 3>) mma_q4<4, 1, 3>;
template [[host_name("mma_tm4_n2_m3")]] kernel decltype(mma_q4<4, 2, 3>) mma_q4<4, 2, 3>;
