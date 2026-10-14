#!/bin/bash
# on_ub512 ile off_ub512 arasindaki GPU suresi farki: islem atlama ile konumla. 2 tur donusumlu. EXTRA ile ek ortam.
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
SMALL="ADD,MUL,RMS_NORM,GLU,ROPE,SET_ROWS,GET_ROWS,CPY,CONT,SCALE,UNARY"
for X in "$@"; do
  name=q_$(echo "$X" | tr ',:.' '___' | cut -c1-30)
  for tur in 1 2; do
    for cfg in A B; do
      [ $tur = 2 ] && cfg=$([ $cfg = A ] && echo B || echo A)
      if [ $cfg = A ]; then E="$OFF"; else E="$ON"; fi
      case "$X" in NONE) SK="";; SMALL) SK="GGML_METAL_SKIP_OPS=$SMALL";; MM:*) SK="GGML_METAL_SKIP_MM_NAME=${X#MM:}";; *) SK="GGML_METAL_SKIP_OPS=$X";; esac
      f=$D/${name}_${cfg}_r$tur
      env $E $SK $EXTRA GGML_METAL_GPU_TIME=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "The history of computing" </dev/null > $f.txt 2> $f.err
    done
  done
  python3 -I - "$D" "$name" "$X" <<'PY'
import sys, re, statistics as st, glob
D, name, X = sys.argv[1:4]
res = {}
for cfg in ("A", "B"):
    meds, ns = [], []
    for f in sorted(glob.glob(f"{D}/{name}_{cfg}_r*.err")):
        v = [float(x) for x in re.findall(r"GPU_TIME: dugum=957 sutun=3 gpu_us=([0-9.]+)", open(f, errors="ignore").read())]
        if v:
            meds.append(st.median(v)/1000); ns.append(len(v))
    res[cfg] = (meds, ns)
a, b = res["A"], res["B"]
if a[0] and b[0]:
    print(f"atla[{X}]: off_ub512 {[round(x,3) for x in a[0]]} | on_ub512 {[round(x,3) for x in b[0]]} | n={a[1]},{b[1]} | fark(on-off) = {st.mean(b[0])-st.mean(a[0]):+.3f} ms")
else:
    print(X, "veri yok", res)
PY
done
