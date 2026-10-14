import re
d="~/claude-scratch/llamacpp-ablasyon-sb/ggml/src/ggml-metal/"
p=d+"ggml-metal.metal"; s=open(p).read()
s=s.replace("Bt.thread_elements()[0] = half(xr0[b*32 + j*8]);","Bt.thread_elements()[0] = (NM == 2) ? half(j + 1) : half(xr0[b*32 + j*8]);")
s=s.replace("Bt.thread_elements()[1] = half(xr1[b*32 + j*8]);","Bt.thread_elements()[1] = (NM == 2) ? half(j + 2) : half(xr1[b*32 + j*8]);")
s=s.replace("const half2  h = as_type<half2>((w & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);","const half2  h = (NM == 3) ? as_type<half2>(uint(u) * 0x00010001u) : as_type<half2>((w & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);")
s=s.replace("const half2  h2 = as_type<half2>((w2 & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);","const half2  h2 = (NM == 3) ? as_type<half2>(uint(u2) * 0x00010001u) : as_type<half2>((w2 & 0x000F000Fu) | 0x64006400u) - half2(1032.0h);")
ex=""
for n in (2,3):
    ex+='''template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2_nm%d")]] kernel mul_mm_sb_q4_0_f32_t kernel_mul_mm_sb_q4_0_f32<2, %d>;
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2_add_nm%d")]] kernel mul_mm_sb_q4_0_f32_add_t kernel_mul_mm_sb_q4_0_f32_add<2, %d>;
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm2_qkv_nm%d")]] kernel mul_mm_sb_q4_0_f32_qkv_t kernel_mul_mm_sb_q4_0_f32_qkv<2, %d>;
template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm1_glu_nm%d")]] kernel mul_mm_sb_q4_0_f32_glu_t kernel_mul_mm_sb_q4_0_f32_glu<1, %d>;
'''%((n,)*8)
a='template [[host_name("kernel_mul_mm_sb_q4_0_f32_tm1_glu_nm")]] kernel mul_mm_sb_q4_0_f32_glu_t kernel_mul_mm_sb_q4_0_f32_glu<1, 1>;\n'
assert a in s
s=s.replace(a,a+ex); open(p,"w").write(s)
p=d+"ggml-metal-device.cpp"; s=open(p).read()
o='static const bool nm = getenv("GGML_METAL_SB_NM") != nullptr;'
assert o in s
s=s.replace(o,'static const int nm = []() { const char * e = getenv("GGML_METAL_SB_NM"); return e ? atoi(e) : 0; }();\n        static const char * nm_s[4] = { "", "_nm", "_nm2", "_nm3" };')
s=s.replace('suffix[fuse], nm ? "_nm" : "");','suffix[fuse], nm_s[nm]);')
open(p,"w").write(s)
