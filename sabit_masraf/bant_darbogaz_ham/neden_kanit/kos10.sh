#!/bin/bash
# Hizalama denemesi: ayirici hizalamasini zorla (GGML_ALLOC_ALIGN), 3 sutunluk grafin GPU suresi. 3 tur donusumlu.
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
P="The history of computing"
CF=("ub512|512|0" "ub4|4|0" "ub4_a256|4|256" "ub4_a4096|4|4096" "ub3|3|0" "ub3_a256|3|256" "ub3_a4096|3|4096")
for tur in 1 2 3; do
  for ((j=0;j<${#CF[@]};j++)); do
    c=${CF[$(( (j + tur - 1) % ${#CF[@]} ))]}
    IFS='|' read ad ub al <<< "$c"
    f=$D/h_${ad}_r$tur
    env $OFF GGML_ALLOC_ALIGN=$al GGML_METAL_GPU_TIME=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -p "$P" </dev/null > $f.txt 2> $f.err
  done
done
python3 -I - "$D" <<'PY'
import sys, re, statistics as st, glob
D = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{D}/h_*_r*.err")):
    m = re.match(r".*/h_(.+)_r(\d+)\.err", f)
    v = [float(x) for x in re.findall(r"GPU_TIME: dugum=957 sutun=3 gpu_us=([0-9.]+)", open(f, errors="ignore").read())]
    if v:
        rows.setdefault(m.group(1), []).append(st.median(v)/1000)
for ad, r in rows.items():
    print(f"{ad:10s} tur_medyan_ms: {[round(x,3) for x in r]}  ortalama: {st.mean(r):.3f}")
PY
