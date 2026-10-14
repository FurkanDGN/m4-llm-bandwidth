#!/bin/bash
# Katman farki hangi matris carpimlarindan geliyor: agirlik adina gore atlama. off_ub512 ve off_ub4, 2 tur donusumlu.
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
P="The history of computing"
for X in "$@"; do
  name=m_$(echo "$X" | tr ',.' '__' | cut -c1-30)
  for tur in 1 2; do
    for ub in 512 4; do
      [ $tur = 2 ] && ub=$([ $ub = 512 ] && echo 4 || echo 512)
      f=$D/${name}_ub${ub}_r$tur
      env $OFF GGML_METAL_SKIP_MM_NAME=$X $EXTRA GGML_METAL_GPU_TIME=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -p "$P" </dev/null > $f.txt 2> $f.err
    done
  done
  python3 -I - "$D" "$name" "$X" <<'PY'
import sys, re, statistics as st, glob
D, name, X = sys.argv[1:4]
res = {}
for ub in (512, 4):
    meds, ns = [], []
    for f in sorted(glob.glob(f"{D}/{name}_ub{ub}_r*.err")):
        v = [float(x) for x in re.findall(r"GPU_TIME: dugum=957 sutun=3 gpu_us=([0-9.]+)", open(f, errors="ignore").read())]
        if v:
            meds.append(st.median(v)/1000); ns.append(len(v))
    res[ub] = (meds, ns)
a, b = res[512], res[4]
if a[0] and b[0]:
    print(f"atla[{X}]: ub512 {[round(x,3) for x in a[0]]} | ub4 {[round(x,3) for x in b[0]]} | n={a[1]},{b[1]} | fark(ub4-ub512) = {st.mean(b[0])-st.mean(a[0]):+.3f} ms")
else:
    print(X, "veri yok", res)
PY
done
