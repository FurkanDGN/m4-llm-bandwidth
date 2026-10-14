#!/bin/bash
# on_ub512 yerlesimi uzerinde aralik (PAD) ve kaydirma (SHIFT) taramasi, GPU suresi. 3 tur donusumlu.
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
CF=("off512|$OFF" "on_a256|$ON GGML_ALLOC_ALIGN=256" "on_pad256|$ON GGML_ALLOC_ALIGN=256 GGML_ALLOC_PAD=256" "on_pad4096|$ON GGML_ALLOC_ALIGN=256 GGML_ALLOC_PAD=4096" "on_sh256|$ON GGML_ALLOC_ALIGN=256 GGML_ALLOC_SHIFT=256" "on_sh4096|$ON GGML_ALLOC_ALIGN=256 GGML_ALLOC_SHIFT=4096" "on_sh65536|$ON GGML_ALLOC_ALIGN=256 GGML_ALLOC_SHIFT=65536")
rounds=${ROUNDS:-3}
for tur in $(seq 1 $rounds); do
  for ((j=0;j<${#CF[@]};j++)); do
    c=${CF[$(( (j + tur - 1) % ${#CF[@]} ))]}
    ad=${c%%|*}; envs=${c#*|}
    f=$D/w_${ad}_r$tur
    env $envs GGML_METAL_GPU_TIME=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "The history of computing" </dev/null > $f.txt 2> $f.err
  done
done
python3 -I - "$D" <<'PY'
import sys, re, statistics as st, glob
D = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{D}/w_*_r*.err")):
    m = re.match(r".*/w_(.+)_r(\d+)\.err", f)
    v = [float(x) for x in re.findall(r"GPU_TIME: dugum=957 sutun=3 gpu_us=([0-9.]+)", open(f, errors="ignore").read())]
    if v:
        rows.setdefault(m.group(1), []).append(st.median(v)/1000)
for ad, r in rows.items():
    print(f"{ad:12s} tur_medyan_ms: {[round(x,3) for x in r]}  ortalama: {st.mean(r):.3f}")
PY
