#!/bin/bash
# Hizalama 1024 / 4096 / 256 ve ub3 varsayilan: GPU suresi, 3 tur donusumlu
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
CF=("off512|$OFF|512" "on_a32|$ON|512" "on_a256|$ON GGML_ALLOC_ALIGN=256|512" "on_a1024|$ON GGML_ALLOC_ALIGN=1024|512" "on_a4096|$ON GGML_ALLOC_ALIGN=4096|512" "ub3_a32|$OFF|3" "ub3_a1024|$OFF GGML_ALLOC_ALIGN=1024|3")
for tur in 1 2 3; do
  for ((j=0;j<${#CF[@]};j++)); do
    c=${CF[$(( (j + tur - 1) % ${#CF[@]} ))]}
    IFS='|' read ad envs ub <<< "$c"
    f=$D/y_${ad}_r$tur
    env $envs GGML_METAL_GPU_TIME=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -p "The history of computing" </dev/null > $f.txt 2> $f.err
  done
done
python3 -I - "$D" <<'PY'
import sys, re, statistics as st, glob
D = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{D}/y_*_r*.err")):
    m = re.match(r".*/y_(.+)_r(\d+)\.err", f)
    v = [float(x) for x in re.findall(r"GPU_TIME: dugum=957 sutun=3 gpu_us=([0-9.]+)", open(f, errors="ignore").read())]
    if v:
        rows.setdefault(m.group(1), []).append(st.median(v)/1000)
for ad, r in rows.items():
    print(f"{ad:11s} tur_medyan_ms: {[round(x,3) for x in r]}  ortalama: {st.mean(r):.3f}")
PY
