#include <metal_stdlib>
using namespace metal;

struct Params { uint rows; uint cols; uint nblk; uint seed; };

constant constexpr uint SGS = 4;

inline uint xs32(uint s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

template<uint FMT> constexpr uint NW_OF() { return FMT == 0 ? 4 : (FMT == 1 ? 8 : 16); }
template<uint FMT> constexpr uint R_OF() { return FMT == 0 ? 8 : 4; }

template<uint FMT>
inline void gen_words(uint gidx, uint seed, thread uint *w) {
    uint s = (gidx + seed) * 0x9E3779B1u;
    s ^= s >> 15;
    s |= 1u;
    for (uint i = 0; i < NW_OF<FMT>(); i++) {
        s = xs32(s);
        w[i] = (FMT == 2) ? ((s & 0x83FF83FFu) | 0x38003800u) : s;
    }
}

inline half gen_scale(uint gidx) {
    return half(0.002f + 0.0005f * float(((gidx * 0x85EBCA6Bu) >> 16) & 15u));
}

template<uint FMT>
inline float dec(thread const uint *w, thread const float4 *xv, float xsum) {
    float acc = 0.0f;
    if (FMT == 0) {
        for (uint k = 0; k < 4; k++) {
            uchar4 lo = as_type<uchar4>(w[k] & 0x0F0F0F0Fu);
            uchar4 hi = as_type<uchar4>((w[k] >> 4) & 0x0F0F0F0Fu);
            acc += dot(float4(lo), xv[2 * k]) + dot(float4(hi), xv[2 * k + 1]);
        }
        return acc - 8.0f * xsum;
    } else if (FMT == 1) {
        for (uint k = 0; k < 8; k++) {
            acc += dot(float4(as_type<uchar4>(w[k])), xv[k]);
        }
        return acc - 128.0f * xsum;
    } else {
        for (uint k = 0; k < 16; k++) {
            float2 f = float2(as_type<half2>(w[k]));
            float4 xx = xv[k >> 1];
            acc += (k & 1) ? (f.x * xx.z + f.y * xx.w) : (f.x * xx.x + f.y * xx.y);
        }
        return acc;
    }
}

template<uint FMT>
kernel void fill_fmt(device uint *codes [[buffer(0)]],
                     device half *scales [[buffer(1)]],
                     constant uint2 &np [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= np.x) return;
    uint w[NW_OF<FMT>()];
    gen_words<FMT>(gid, np.y, w);
    for (uint i = 0; i < NW_OF<FMT>(); i++) codes[gid * NW_OF<FMT>() + i] = w[i];
    scales[gid] = gen_scale(gid);
}

template<uint FMT, uint GF>
kernel void mv_fmt(device const uint *codes [[buffer(0)]],
                   device const half *scales [[buffer(1)]],
                   device const float4 *x [[buffer(2)]],
                   device float *y [[buffer(3)]],
                   constant Params &p [[buffer(4)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint R = R_OF<FMT>();
    constexpr uint NW = NW_OF<FMT>();
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
        uint w[R][NW];
        half sc[R];
        for (uint r = 0; r < R; r++) {
            uint gb = (grow0 + r) * p.nblk + b;
            if (FMT != 2) sc[r] = scales[gb];
            if ((r & 3u) >= GF) {
                device const uint *cp = codes + (ulong)gb * NW;
                for (uint i = 0; i < NW; i += 4) {
                    uint4 v = *(device const uint4 *)(cp + i);
                    w[r][i] = v.x; w[r][i + 1] = v.y; w[r][i + 2] = v.z; w[r][i + 3] = v.w;
                }
            }
        }
        for (uint r = 0; r < R; r++) {
            if ((r & 3u) < GF) gen_words<FMT>((grow0 + r) * p.nblk + b, p.seed, w[r]);
            float d = dec<FMT>(w[r], xv, xsum);
            sum[r] += (FMT == 2) ? d : float(sc[r]) * d;
        }
    }
    for (uint r = 0; r < R; r++) {
        float t = simd_sum(sum[r]);
        if (lane == 0) y[grow0 + r] = t;
    }
}

template [[host_name("fill_q4")]] kernel decltype(fill_fmt<0>) fill_fmt<0>;
template [[host_name("fill_q8")]] kernel decltype(fill_fmt<1>) fill_fmt<1>;
template [[host_name("fill_f16")]] kernel decltype(fill_fmt<2>) fill_fmt<2>;
template [[host_name("mv_q4_g0")]] kernel decltype(mv_fmt<0, 0>) mv_fmt<0, 0>;
template [[host_name("mv_q4_g1")]] kernel decltype(mv_fmt<0, 1>) mv_fmt<0, 1>;
template [[host_name("mv_q4_g2")]] kernel decltype(mv_fmt<0, 2>) mv_fmt<0, 2>;
template [[host_name("mv_q4_g3")]] kernel decltype(mv_fmt<0, 3>) mv_fmt<0, 3>;
template [[host_name("mv_q4_g4")]] kernel decltype(mv_fmt<0, 4>) mv_fmt<0, 4>;
template [[host_name("mv_q8_g0")]] kernel decltype(mv_fmt<1, 0>) mv_fmt<1, 0>;
template [[host_name("mv_q8_g1")]] kernel decltype(mv_fmt<1, 1>) mv_fmt<1, 1>;
template [[host_name("mv_q8_g2")]] kernel decltype(mv_fmt<1, 2>) mv_fmt<1, 2>;
template [[host_name("mv_q8_g3")]] kernel decltype(mv_fmt<1, 3>) mv_fmt<1, 3>;
template [[host_name("mv_q8_g4")]] kernel decltype(mv_fmt<1, 4>) mv_fmt<1, 4>;
template [[host_name("mv_f16_g0")]] kernel decltype(mv_fmt<2, 0>) mv_fmt<2, 0>;
template [[host_name("mv_f16_g1")]] kernel decltype(mv_fmt<2, 1>) mv_fmt<2, 1>;
template [[host_name("mv_f16_g2")]] kernel decltype(mv_fmt<2, 2>) mv_fmt<2, 2>;
template [[host_name("mv_f16_g3")]] kernel decltype(mv_fmt<2, 3>) mv_fmt<2, 3>;
template [[host_name("mv_f16_g4")]] kernel decltype(mv_fmt<2, 4>) mv_fmt<2, 4>;

kernel void stream_read(device const uint4 *buf [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        constant uint &n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint nt [[threads_per_grid]]) {
    uint4 a = uint4(0);
    for (uint i = gid; i + 3 * nt < n; i += 4 * nt) {
        a ^= buf[i] ^ buf[i + nt] ^ buf[i + 2 * nt] ^ buf[i + 3 * nt];
    }
    out[gid] = a.x ^ a.y ^ a.z ^ a.w;
}
