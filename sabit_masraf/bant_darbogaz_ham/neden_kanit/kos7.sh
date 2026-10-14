#!/bin/bash
# GPU suresi (komut tamponu zaman damgalari): 3 sutunluk dogrulama grafi, uc yerlesim, 4 tur donusumlu
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON_ADD="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
P="The history of computing"
EXTRA="${EXTRA:-}"
TAG="${TAG:-g}"
CF=("off_ub512|$OFF|512" "off_ub4|$OFF|4" "on_ub512|$ON_ADD|512")
for tur in 1 2 3 4; do
  for ((j=0;j<${#CF[@]};j++)); do
    c=${CF[$(( (j + tur - 1) % ${#CF[@]} ))]}
    IFS='|' read ad envs ub <<< "$c"
    f=$D/${TAG}_${ad}_r$tur
    env $envs $EXTRA GGML_METAL_GPU_TIME=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -p "$P" </dev/null > $f.txt 2> $f.err
  done
done
python3 -I - "$D" "$TAG" <<'PY'
import sys, re, statistics as st, glob, os
D, TAG = sys.argv[1], sys.argv[2]
rows = {}
for f in sorted(glob.glob(f"{D}/{TAG}_*_r*.err")):
    m = re.match(rf".*/{TAG}_(.+)_r(\d+)\.err", f)
    ad, tur = m.group(1), int(m.group(2))
    v = [float(x) for x in re.findall(r"GPU_TIME: dugum=957 sutun=3 gpu_us=([0-9.]+)", open(f, errors="ignore").read())]
    if v:
        rows.setdefault(ad, []).append((tur, len(v), st.median(v), min(v)))
for ad, r in rows.items():
    print(ad, "tur_medyan_ms:", [round(x[2]/1000, 3) for x in sorted(r)], "n_graf:", [x[1] for x in sorted(r)],
          "ortalama_medyan_ms: %.3f" % (st.mean(x[2] for x in r)/1000))
PY
