#include <metal_stdlib>
using namespace metal;
// E21: 3.5 bpw duzlem formati (q3p) mat-vec fizibilite mikrosu.
// q3p SoA: q2[K/16 uint/satir] (2-bit duzlem, uint = 16 agirlik: bit 2k ve 16+2k),
//          hm[K/32 uint/satir] (1-bit duzlem, uint = 32 agirlik: bit k ve 16+k ciftleri),
//          dd[K/32 half/satir] (blok olcegi). Deger = ((2bit | hbit<<2) - 4) * d.
// mv4m: q4_0 esdegeri referans (SoA: q4[K/8 uint] + dd[K/32 half]), sihirli half2 acma.

constant short NR0 = 8;
constant short NSG = 2;

kernel void mv3p(device const uint  * q2 [[buffer(0)]],
                 device const uint  * hm [[buffer(1)]],
                 device const half  * dd [[buffer(2)]],
                 device const float * y  [[buffer(3)]],
                 device       float * dst[[buffer(4)]],
                 constant     uint  & K  [[buffer(5)]],
                 constant     uint  & R  [[buffer(6)]],
                 uint3  tgpig[[threadgroup_position_in_grid]],
                 ushort tiisg[[thread_index_in_simdgroup]],
                 ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint first_row = (tgpig.x*NSG + sgitg)*NR0;
    const uint nch = K/32;          // 32'lik parca sayisi
    const uint nq2r = K/16;         // satir basina q2 uint sayisi
    float sumf[NR0] = {0.f};

    for (uint ci = tiisg; ci < nch; ci += 32) {
        // y: 32 eleman, NR0 satira amortize
        device const float4 * y4 = (device const float4 *)(y + ci*32);
        float4 yl[8];
        float sumx = 0.f;
        for (short i = 0; i < 8; ++i) {
            yl[i] = y4[i];
            sumx += yl[i].x + yl[i].y + yl[i].z + yl[i].w;
        }
        // T3 esleme duzeni: uint q2 içindeki (s, s+16) cifti agirlik (2t, 16+2t)'ye denk gelir;
        // y'yi ayni duzenle ciftle: pair(t) = (y[2t], y[16+2t]) ve (y[2t+1], y[16+2t+1])
        for (short r = 0; r < NR0; ++r) {
            const uint row = first_row + r;
            const uint q2a = q2[row*nq2r + 2*ci    ];
            const uint q2b = q2[row*nq2r + 2*ci + 1];
            const uint hmw = hm[row*nch + ci];
            const half d   = dd[row*nch + ci];

            float acc = 0.f;
            #define P3(Q2W, HMW, s, t, xlo, xhi) \
                { half2 h = as_type<half2>((((Q2W) >> (s)) & 0x00030003u) | ((((HMW) >> (t)) & 0x00010001u) << 2) | 0x64006400u) - half2(1024.0h); \
                  acc += h.x*(xlo) + h.y*(xhi); }
            // q2a: agirlik 0..7 (bit 2t) ve 16..23 (bit 16+2t); hm bitleri 0..7 ve 16..23
            P3(q2a, hmw, 0, 0, yl[0].x, yl[4].x) P3(q2a, hmw,  2, 1, yl[0].y, yl[4].y)
            P3(q2a, hmw, 4, 2, yl[0].z, yl[4].z) P3(q2a, hmw,  6, 3, yl[0].w, yl[4].w)
            P3(q2a, hmw, 8, 4, yl[1].x, yl[5].x) P3(q2a, hmw, 10, 5, yl[1].y, yl[5].y)
            P3(q2a, hmw,12, 6, yl[1].z, yl[5].z) P3(q2a, hmw, 14, 7, yl[1].w, yl[5].w)
            // q2b: agirlik 8..15 ve 24..31; hm bitleri 8..15 ve 24..31
            P3(q2b, hmw>>8, 0, 0, yl[2].x, yl[6].x) P3(q2b, hmw>>8,  2, 1, yl[2].y, yl[6].y)
            P3(q2b, hmw>>8, 4, 2, yl[2].z, yl[6].z) P3(q2b, hmw>>8,  6, 3, yl[2].w, yl[6].w)
            P3(q2b, hmw>>8, 8, 4, yl[3].x, yl[7].x) P3(q2b, hmw>>8, 10, 5, yl[3].y, yl[7].y)
            P3(q2b, hmw>>8,12, 6, yl[3].z, yl[7].z) P3(q2b, hmw>>8, 14, 7, yl[3].w, yl[7].w)

            sumf[r] = fma((float)d, acc - 4.f*sumx, sumf[r]);
        }
    }

    for (short r = 0; r < NR0; ++r) {
        const float s = simd_sum(sumf[r]);
        if (tiisg == 0 && first_row + r < R) dst[first_row + r] = s;
    }
}

kernel void mv4m(device const uint  * q4 [[buffer(0)]],
                 device const uint  * hm [[buffer(1)]],   // kullanilmiyor (ayni imza)
                 device const half  * dd [[buffer(2)]],
                 device const float * y  [[buffer(3)]],
                 device       float * dst[[buffer(4)]],
                 constant     uint  & K  [[buffer(5)]],
                 constant     uint  & R  [[buffer(6)]],
                 uint3  tgpig[[threadgroup_position_in_grid]],
                 ushort tiisg[[thread_index_in_simdgroup]],
                 ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint first_row = (tgpig.x*NSG + sgitg)*NR0;
    const uint nch = K/32;
    const uint nq4r = K/8;          // satir basina q4 uint sayisi (uint = 8 agirlik, nibble'lar)
    float sumf[NR0] = {0.f};

    for (uint ci = tiisg; ci < nch; ci += 32) {
        device const float4 * y4 = (device const float4 *)(y + ci*32);
        float4 yl[8];
        float sumx = 0.f;
        for (short i = 0; i < 8; ++i) {
            yl[i] = y4[i];
            sumx += yl[i].x + yl[i].y + yl[i].z + yl[i].w;
        }
        for (short r = 0; r < NR0; ++r) {
            const uint row = first_row + r;
            device const uint * qp = q4 + row*nq4r + 4*ci;
            const half d = dd[row*nch + ci];

            float acc = 0.f;
            #define P4(W, s, xlo, xhi) \
                { half2 h = as_type<half2>((((W) >> (s)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h); \
                  acc += h.x*(xlo) + h.y*(xhi); }
            // uint = 8 agirlik: nibble j'de agirlik (2j?) — duzen: bit 4k ve 16+4k cifti = agirlik (k, 4+k), k=0..3
            for (short u = 0; u < 4; ++u) {
                const uint w = qp[u];
                P4(w,  0, yl[2*u].x,   yl[2*u].z)
                P4(w,  4, yl[2*u].y,   yl[2*u].w)
                P4(w,  8, yl[2*u+1].x, yl[2*u+1].z)
                P4(w, 12, yl[2*u+1].y, yl[2*u+1].w)
            }
            sumf[r] = fma((float)d, acc - 8.f*sumx, sumf[r]);
        }
    }

    for (short r = 0; r < NR0; ++r) {
        const float s = simd_sum(sumf[r]);
        if (tiisg == 0 && first_row + r < R) dst[first_row + r] = s;
    }
}
