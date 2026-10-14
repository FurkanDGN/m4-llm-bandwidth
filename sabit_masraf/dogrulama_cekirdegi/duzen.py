p="~/claude-scratch/llamacpp-ablasyon-sb/ggml/src/ggml-metal/ggml-metal.metal"
s=open(p).read()
def rep(a,b,n):
    global s
    assert s.count(a)==n,(a,s.count(a))
    s=s.replace(a,b)
rep("ua[t] = qs[fn/2];","ua[t] = qs[fn];",2)
rep("ub[t] = qs[4 + fn/2];","ub[t] = qs[fn + 1];",2)
rep("ua2[t] = qs2[fn/2];","ua2[t] = qs2[fn];",1)
rep("ub2[t] = qs2[4 + fn/2];","ub2[t] = qs2[fn + 1];",1)
rep("const ushort u = (j & 1) ? ub[t] : ua[t];\n                const uint   w = (uint(u & 0xFFu) | (uint(u >> 8) << 16)) >> (j < 2 ? 0 : 4);",
    "const uint   w = (uint(ua[t]) | (uint(ub[t]) << 16)) >> ((j & 1)*8 + (j >> 1)*4);",2)
rep("const ushort u2 = (j & 1) ? ub2[t] : ua2[t];\n                    const uint   w2 = (uint(u2 & 0xFFu) | (uint(u2 >> 8) << 16)) >> (j < 2 ? 0 : 4);",
    "const uint   w2 = (uint(ua2[t]) | (uint(ub2[t]) << 16)) >> ((j & 1)*8 + (j >> 1)*4);",1)
rep("as_type<half2>(uint(u) * 0x00010001u)","as_type<half2>(w)",2)
rep("as_type<half2>(uint(u2) * 0x00010001u)","as_type<half2>(w2)",1)
rep("args.nb11) + fm;","args.nb11) + 2*fm;",4)
rep("        for (short j = 0; j < 4; ++j) {\n            simdgroup_half8x8 Bt;",
    "        float2 xa[2];\n        float2 xb[2];\n        for (short h = 0; h < 2; ++h) {\n            xa[h] = *(device const float2 *) (xr0 + b*32 + 16*h);\n            xb[h] = *(device const float2 *) (xr1 + b*32 + 16*h);\n        }\n\n        for (short j = 0; j < 4; ++j) {\n            simdgroup_half8x8 Bt;",2)
rep("half(xr0[b*32 + j*8])","half(xa[j >> 1][j & 1])",2)
rep("half(xr1[b*32 + j*8])","half(xb[j >> 1][j & 1])",2)
open(p,"w").write(s); print("ok")
