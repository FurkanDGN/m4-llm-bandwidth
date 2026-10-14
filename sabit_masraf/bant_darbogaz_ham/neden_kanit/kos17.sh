#!/bin/bash
# on_ub512 yerlesimini sabit tutup tum tensorleri SHIFT bayt kaydir; 3 sutunluk grafin GPU suresi
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
for sh in "$@"; do
  f=$D/x_sh$sh
  env $ON GGML_ALLOC_ALIGN=256 GGML_ALLOC_SHIFT=$sh GGML_METAL_GPU_TIME=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "The history of computing" </dev/null > $f.txt 2> $f.err
  python3 -I - "$f.err" "$sh" <<'PY'
import sys, re, statistics as st
v = [float(x) for x in re.findall(r"GPU_TIME: dugum=957 sutun=3 gpu_us=([0-9.]+)", open(sys.argv[1], errors="ignore").read())]
print("shift=%6s n=%d medyan_ms=%.3f" % (sys.argv[2], len(v), st.median(v)/1000 if v else -1))
PY
done
