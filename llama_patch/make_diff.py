import difflib, os, re, runpy, sys, tempfile

# apply_patch.py'yi, gercek dosyalardan bilinen baglam satirlarini iceren kucuk bir sahte agacta calistirip
# sb_q4_0.diff uretir. Satir numaralari gercek dosyalarin (commit e475fa2b5) numaralarina cevrilir.
# Bagli bilgi: ops.cpp 2055-2091, device.h 131-136, device.cpp 667-, metal 3965-3967 satirlari gercek metinlerdir.

here = os.path.dirname(os.path.abspath(__file__))

OPS = '''    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
'''
HDR = '''struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, const struct ggml_tensor * op, int nsg, int nxpsg, int r1ptg);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, const struct ggml_tensor * op);
'''
DEV = '''ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, const ggml_tensor * op, int nsg, int nxpsg, int r1ptg) {
    char base[256];
    char name[256];
'''
MET = '''typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;
typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>)    mul_mv_ext_q4x4_f32_t;

'''
# dosya -> (stub metin, stub satir 1'in gercek satir numarasi)
files = {
    "ggml-metal-ops.cpp": (OPS, 2055),
    "ggml-metal-device.h": (HDR, 131),
    "ggml-metal-device.cpp": (DEV, 667),
    "ggml-metal.metal": (MET, 3965),
}

tmp = tempfile.mkdtemp()
d = tmp + "/ggml/src/ggml-metal/"
os.makedirs(d)
for name, (text, _) in files.items():
    open(d + name, "w").write(text)
before = {n: open(d + n).read() for n in files}
sys.argv = ["apply_patch.py", tmp]
runpy.run_path(os.path.join(here, "apply_patch.py"))

out = []
for name, (text, base) in files.items():
    a = before[name].splitlines()
    b = open(d + name).read().splitlines()
    delta = 0
    hunks = list(difflib.unified_diff(a, b, "a/ggml/src/ggml-metal/" + name, "b/ggml/src/ggml-metal/" + name, n=3, lineterm=""))
    for line in hunks:
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            oa, oc = int(m[1]), int(m[2] or 1)
            nc = int(m[4] or 1)
            line = "@@ -%d,%d +%d,%d @@" % (oa + base - 1, oc, oa + base - 1 + delta, nc)
            delta += nc - oc
        out.append(line)
open(os.path.join(here, "sb_q4_0.diff"), "w").write("\n".join(out) + "\n")
print("diff yazildi:", len(out), "satir")
