import re, sys
d = "~/claude-scratch/llamacpp-ablasyon-sb/ggml/src/ggml-metal/"

def edit(fn, f):
    p = d + fn
    s = open(p).read()
    s2 = f(s)
    assert s2 != s, fn
    open(p, "w").write(s2)

NM_T2 = """if (NM) {
                    T2[t].thread_elements()[0] += float(A.thread_elements()[0])*float(Bt.thread_elements()[0]);
                    T2[t].thread_elements()[1] += float(A.thread_elements()[1])*float(Bt.thread_elements()[1]);
                } else {
                    simdgroup_multiply_accumulate(T2[t], A, Bt, T2[t]);
                }"""
NM_T3 = """if (NM) {
                        T3[t].thread_elements()[0] += float(A2.thread_elements()[0])*float(Bt.thread_elements()[0]);
                        T3[t].thread_elements()[1] += float(A2.thread_elements()[1])*float(Bt.thread_elements()[1]);
                    } else {
                        simdgroup_multiply_accumulate(T3[t], A2, Bt, T3[t]);
                    }"""

def metal(s):
    s = s.replace("simdgroup_multiply_accumulate(T2[t], A, Bt, T2[t]);", NM_T2)
    s = s.replace("simdgroup_multiply_accumulate(T3[t], A2, Bt, T3[t]);", NM_T3)
    s = s.replace("template<short TM, short MODE>\ninline void mm_sb_core(", "template<short TM, short MODE, short NM = 0>\ninline void mm_sb_core(")
    s = s.replace("template<short TM>\nkernel void kernel_mul_mm_sb_q4_0_f32(", "template<short TM, short NM = 0>\nkernel void kernel_mul_mm_sb_q4_0_f32(")
    for k in ("add", "glu", "qkv"):
        s = s.replace("template<short TM>\nkernel void kernel_mul_mm_sb_q4_0_f32_%s(" % k,
                      "template<short TM, short NM = 0>\nkernel void kernel_mul_mm_sb_q4_0_f32_%s(" % k)
    s = s.replace("mm_sb_core<TM, 1>(", "mm_sb_core<TM, 1, NM>(").replace("mm_sb_core<TM, 2>(", "mm_sb_core<TM, 2, NM>(")
    extra_mm = """
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2_nm")]] kernel mul_mm_sb_q4_0_f32_t kernel_mul_mm_sb_q4_0_f32<2, 1>;
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2_add_nm")]] kernel mul_mm_sb_q4_0_f32_add_t kernel_mul_mm_sb_q4_0_f32_add<2, 1>;
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2_qkv_nm")]] kernel mul_mm_sb_q4_0_f32_qkv_t kernel_mul_mm_sb_q4_0_f32_qkv<2, 1>;
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm1_glu_nm")]] kernel mul_mm_sb_q4_0_f32_glu_t kernel_mul_mm_sb_q4_0_f32_glu<1, 1>;
"""
    anchor = 'template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2")]] kernel mul_mm_sb_q4_0_f32_t kernel_mul_mm_sb_q4_0_f32<2>;\n'
    assert anchor in s
    s = s.replace(anchor, anchor + extra_mm)
    plain = ""
    for n in (4, 5):
        for r in (1, 2, 3):
            plain += 'template [[host_name("kernel_mul_mv_sb_q4_0_f32_n%d_r%d")]] kernel mul_mv_sb_q4_0_f32_t kernel_mul_mv_sb_q4_0_f32<%d, %d>;\n' % (n, r, n, r)
    a1 = 'template [[host_name("kernel_mul_mv_sb_q4_0_f32_n3_r8")]] kernel mul_mv_sb_q4_0_f32_t kernel_mul_mv_sb_q4_0_f32<3, 8>;\n'
    assert a1 in s
    s = s.replace(a1, a1 + plain)
    add = ""
    for n in (4, 5):
        for r in (2, 3):
            add += 'template [[host_name("kernel_mul_mv_sb_q4_0_f32_n%d_r%d_add")]] kernel mul_mv_sb_q4_0_f32_add_t kernel_mul_mv_sb_q4_0_f32_add<%d, %d>;\n' % (n, r, n, r)
    a2 = 'template [[host_name("kernel_mul_mv_sb_q4_0_f32_n3_r4_add")]] kernel mul_mv_sb_q4_0_f32_add_t kernel_mul_mv_sb_q4_0_f32_add<3, 4>;\n'
    assert a2 in s
    s = s.replace(a2, a2 + add)
    return s

def common(s):
    old = "ne11 >= 2 && ne11 <= 8 && (ggml_metal_sb_matrix(ne11) || ne11 <= 3) &&"
    assert old in s
    new = "ne11 >= 2 && ne11 <= 8 && (ggml_metal_sb_matrix(ne11) || ne11 <= sb_smax) &&"
    s = s.replace(old, new)
    old2 = "bool ggml_metal_sb_eligible(const ggml_tensor * mm) {\n"
    s = s.replace(old2, old2 + "    static const int sb_smax = []() { const char * e = getenv(\"GGML_METAL_SB_SMAX\"); return e ? atoi(e) : 3; }();\n\n")
    return s

def device(s):
    old = 'snprintf(name, 256, "kernel_mul_mm_sb_q4_0_f32_tm%d%s", tm, suffix[fuse]);'
    assert old in s
    new = ('static const bool nm = getenv("GGML_METAL_SB_NM") != nullptr;\n'
           '        snprintf(name, 256, "kernel_mul_mm_sb_q4_0_f32_tm%d%s%s", tm, suffix[fuse], nm ? "_nm" : "");')
    return s.replace(old, new)

edit("ggml-metal.metal", metal)
edit("ggml-metal-common.cpp", common)
edit("ggml-metal-device.cpp", device)
print("tamam")
