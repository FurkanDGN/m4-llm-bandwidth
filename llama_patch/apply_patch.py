import sys

# kullanim: python3 apply_patch.py <llama.cpp-kok-dizini>   (kaynak: github.com/ggml-org/llama.cpp, commit e475fa2b5)
# Her degisiklik oncesinde capa metni dogrulanir; capa bulunamazsa durur.

root = sys.argv[1] if len(sys.argv) > 1 else "."
d = root + "/ggml/src/ggml-metal/"


def rd(name):
    return open(d + name).read()


def wr(name, text):
    open(d + name, "w").write(text)


METAL_BLOCK_HEAD = r'''// small-batch q4_0 kernels (BS 2..8): the weights are read once and applied to all columns
//   sb_deq2: two nibbles held in 16-bit lanes -> float2 (nibble - 8), via the half exponent trick
inline float2 sb_deq2(uint w) {
    return float2(as_type<half2>((w & 0x000F000Fu) | 0x64006400u) - half2(1032.0h));
}

// one thread handles 8 consecutive weights of NR rows and shares the src1 loads across the rows
template<short NB, short NR>
kernel void kernel_mul_mv_sb_q4_0_f32(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int row0 = (tgpig.x*4 + sgitg)*NR;
    const int nch  = args.ne00/8;

    device const char * rp[NR];
    for (short r = 0; r < NR; ++r) {
        rp[r] = src0 + (uint64_t) min(row0 + r, args.ne01 - 1)*args.nb01;
    }

    float acc[NR][NB];
    for (short r = 0; r < NR; ++r) {
        for (short c = 0; c < NB; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    for (int ci = tiisg; ci < nch; ci += 32) {
        const int   blk = ci >> 2;
        const short cc  = ci & 3;
        const short sh  = cc < 2 ? 0 : 4;
        const short bo  = (cc & 1)*4;

        float4 xl[NB];
        float4 xh[NB];
        for (short c = 0; c < NB; ++c) {
            device const float4 * xp = (device const float4 *) (src1 + (uint64_t) c*args.nb11) + 2*ci;
            xl[c] = xp[0];
            xh[c] = xp[1];
        }

        for (short r = 0; r < NR; ++r) {
            device const block_q4_0 * bq = (device const block_q4_0 *) rp[r] + blk;
            device const ushort * qs = (device const ushort *) bq->qs + bo;

            const uint wa = uint(qs[0]) | (uint(qs[1]) << 16);
            const uint wb = uint(qs[2]) | (uint(qs[3]) << 16);

            const float2 p02 = sb_deq2(wa >> sh);
            const float2 p13 = sb_deq2(wa >> (sh + 8));
            const float2 p46 = sb_deq2(wb >> sh);
            const float2 p57 = sb_deq2(wb >> (sh + 8));
            const float  d   = bq->d;

            for (short c = 0; c < NB; ++c) {
                float a = p02.x*xl[c].x;
                a = fma(p02.y, xl[c].z, a);
                a = fma(p13.x, xl[c].y, a);
                a = fma(p13.y, xl[c].w, a);
                a = fma(p46.x, xh[c].x, a);
                a = fma(p46.y, xh[c].z, a);
                a = fma(p57.x, xh[c].y, a);
                a = fma(p57.y, xh[c].w, a);
                acc[r][c] = fma(a, d, acc[r][c]);
            }
        }
    }

    device float * dst_f32 = (device float *) dst;
    for (short r = 0; r < NR; ++r) {
        for (short c = 0; c < NB; ++c) {
            const float sum = simd_sum(acc[r][c]);
            if (tiisg == 0 && row0 + r < args.ne01) {
                dst_f32[(uint64_t) c*args.ne0 + row0 + r] = sum;
            }
        }
    }
}

typedef decltype(kernel_mul_mv_sb_q4_0_f32<2, 1>) mul_mv_sb_q4_0_f32_t;

'''

METAL_BLOCK_TAIL = r'''
// a simdgroup handles 8*TM rows; the 8 columns of the matrix operand are src1 rows (zero-padded by clamping)
template<short TM>
kernel void kernel_mul_mm_sb_q4_0_f32(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int row0 = (tgpig.x*4 + sgitg)*8*TM;

    const short qid = tiisg >> 2;
    const short fm  = (qid & 4) + ((tiisg >> 1) & 3);
    const short fn  = ((qid & 2) << 1) + ((tiisg & 1) << 1);

    const int nb = args.ne00/32;

    device const char * rp[TM];
    for (short t = 0; t < TM; ++t) {
        rp[t] = src0 + (uint64_t) min(row0 + 8*t + fm, args.ne01 - 1)*args.nb01;
    }

    device const float * xr0 = (device const float *) (src1 + (uint64_t) min((int) fn,     args.ne11 - 1)*args.nb11) + fm;
    device const float * xr1 = (device const float *) (src1 + (uint64_t) min((int) fn + 1, args.ne11 - 1)*args.nb11) + fm;

    simdgroup_float8x8 C[TM];
    for (short t = 0; t < TM; ++t) {
        C[t] = simdgroup_float8x8(0.0f);
    }

    for (int b = 0; b < nb; ++b) {
        ushort ua[TM];
        ushort ub[TM];
        float  dd[TM];
        for (short t = 0; t < TM; ++t) {
            device const block_q4_0 * bq = (device const block_q4_0 *) rp[t] + b;
            device const ushort * qs = (device const ushort *) bq->qs;
            ua[t] = qs[fn/2];
            ub[t] = qs[4 + fn/2];
            dd[t] = bq->d;
        }

        simdgroup_float8x8 T2[TM];
        for (short t = 0; t < TM; ++t) {
            T2[t] = simdgroup_float8x8(0.0f);
        }

        for (short j = 0; j < 4; ++j) {
            simdgroup_half8x8 Bt;
            Bt.thread_elements()[0] = half(xr0[b*32 + j*8]);
            Bt.thread_elements()[1] = half(xr1[b*32 + j*8]);

            for (short t = 0; t < TM; ++t) {
                const ushort u = (j & 1) ? ub[t] : ua[t];
                const uint   w = (uint(u & 0xFFu) | (uint(u >> 8) << 16)) >> (j < 2 ? 0 : 4);
                const half2  h = as_type<half2>((w & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);

                simdgroup_half8x8 A;
                A.thread_elements()[0] = h.x;
                A.thread_elements()[1] = h.y;

                simdgroup_multiply_accumulate(T2[t], A, Bt, T2[t]);
            }
        }

        for (short t = 0; t < TM; ++t) {
            C[t].thread_elements()[0] = fma(T2[t].thread_elements()[0], dd[t], C[t].thread_elements()[0]);
            C[t].thread_elements()[1] = fma(T2[t].thread_elements()[1], dd[t], C[t].thread_elements()[1]);
        }
    }

    device float * dst_f32 = (device float *) dst;
    for (short t = 0; t < TM; ++t) {
        const int row = row0 + 8*t + fm;
        for (short e = 0; e < 2; ++e) {
            const int col = fn + e;
            if (col < args.ne11 && row < args.ne01) {
                dst_f32[(uint64_t) col*args.ne0 + row] = C[t].thread_elements()[e];
            }
        }
    }
}

typedef decltype(kernel_mul_mm_sb_q4_0_f32<1>) mul_mm_sb_q4_0_f32_t;

template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm1")]] kernel mul_mm_sb_q4_0_f32_t kernel_mul_mm_sb_q4_0_f32<1>;
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2")]] kernel mul_mm_sb_q4_0_f32_t kernel_mul_mm_sb_q4_0_f32<2>;

'''


def metal_block():
    inst = ""
    for nb in (2, 3):
        for nr in (1, 2, 3, 4, 6, 8):
            inst += f'template [[host_name("kernel_mul_mv_sb_q4_0_f32_n{nb}_r{nr}")]] kernel mul_mv_sb_q4_0_f32_t kernel_mul_mv_sb_q4_0_f32<{nb}, {nr}>;\n'
    return METAL_BLOCK_HEAD + inst + METAL_BLOCK_TAIL


# 1) ggml-metal.metal
m = rd("ggml-metal.metal")
anchor = "typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;"
assert m.count(anchor) == 1
m = m.replace(anchor, metal_block() + anchor)
wr("ggml-metal.metal", m)

# 2) ggml-metal-device.cpp
dev = rd("ggml-metal-device.cpp")
anchor2 = "ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, const ggml_tensor * op, int nsg, int nxpsg, int r1ptg) {"
assert dev.count(anchor2) == 1
getter = '''ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_sb(ggml_metal_library_t lib, int ne11, bool mma, int tm, int nr) {
    char name[256];

    if (mma) {
        snprintf(name, 256, "kernel_mul_mm_sb_q4_0_f32_tm%d", tm);
    } else {
        snprintf(name, 256, "kernel_mul_mv_sb_q4_0_f32_n%d_r%d", ne11, nr);
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

'''
dev = dev.replace(anchor2, getter + anchor2)
wr("ggml-metal-device.cpp", dev)

# 3) ggml-metal-device.h
h = rd("ggml-metal-device.h")
a3 = "struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, const struct ggml_tensor * op, int nsg, int nxpsg, int r1ptg);"
assert h.count(a3) == 1
h = h.replace(a3, a3 + "\nstruct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_sb         (ggml_metal_library_t lib, int ne11, bool mma, int tm, int nr);")
wr("ggml-metal-device.h", h)

# 4) ggml-metal-ops.cpp
o = rd("ggml-metal-ops.cpp")
a4 = '''    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;
'''
assert o.count(a4) == 1
new4 = '''    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    static const int ne11_mm_min = []() {
        const char * s = getenv("GGML_METAL_MM_MIN");
        return s != nullptr ? atoi(s) : 8;
    }();

    // q4_0 small batches (BS 2..8): read the weights once for all columns
    // GGML_METAL_SB: unset/1 = scalar for BS 2..3 and simdgroup matrix for BS 4..8, 0 = off, s = scalar only, m = matrix for BS 2..8
    {
        static const int sb_mode = []() {
            const char * s = getenv("GGML_METAL_SB");
            if (s == nullptr) {
                return 1;
            }
            return strcmp(s, "0") == 0 ? 0 : (strcmp(s, "s") == 0 ? 2 : (strcmp(s, "m") == 0 ? 3 : 1));
        }();
        static const int sb_tm = []() {
            const char * s = getenv("GGML_METAL_SB_TM");
            return (s != nullptr && atoi(s) == 1) ? 1 : 2;
        }();
        static const int sb_nr_env = []() {
            const char * s = getenv("GGML_METAL_SB_R");
            return s != nullptr ? atoi(s) : 0;
        }();
        const int sb_nr = sb_nr_env > 0 ? sb_nr_env : (ne11 == 2 ? 4 : 3);

        const bool sb_mma = sb_mode == 3 || (sb_mode == 1 && ne11 >= 4);

        if (sb_mode != 0 &&
            op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1]->type == GGML_TYPE_F32 &&
            ne00 % 256 == 0 && ne11 >= 2 && ne11 <= 8 && (sb_mma || ne11 <= 3) &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            nb10 == sizeof(float) && nb1 == (uint64_t) ne0*sizeof(float)) {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mv_sb(lib, ne11, sb_mma, sb_tm, sb_nr);

            ggml_metal_kargs_mul_mv_ext args = {
                /*.ne00  =*/ ne00,
                /*.ne01  =*/ ne01,
                /*.ne02  =*/ ne02,
                /*.nb00  =*/ nb00,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne10  =*/ ne10,
                /*.ne11  =*/ ne11,
                /*.ne12  =*/ ne12,
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

            const int rows_per_tg = sb_mma ? 4*8*sb_tm : 4*sb_nr;

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + rows_per_tg - 1)/rows_per_tg, 1, 1, 32, 4, 1);

            return 1;
        }
    }
'''
o = o.replace(a4, new4)
assert o.count("(ne11 >= 2 && ne11 <= 8)") == 1 and o.count("(ne11 >= 4 && ne11 <= 8)") == 1
o = o.replace("(ne11 >= 2 && ne11 <= 8)", "(ne11 >= 2 && ne11 <= ne11_mm_min)")
o = o.replace("(ne11 >= 4 && ne11 <= 8)", "(ne11 >= 4 && ne11 <= ne11_mm_min)")
wr("ggml-metal-ops.cpp", o)
print("yama uygulandi")
