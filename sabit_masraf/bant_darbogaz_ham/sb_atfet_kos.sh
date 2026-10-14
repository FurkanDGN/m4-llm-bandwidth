#!/bin/bash
# 7B, 3 token'lik graf: hangi birlestirme yavaslatiyor (3 tur, donuslu)
B=${BIN:-$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin}
D=$HOME/claude-scratch/bant-darbogaz
OUT=${1:-$D/sb_atfet.csv}
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
CFGS=("kapali=GGML_METAL_FUSE_SB=0" "hepsi=GGML_METAL_FUSE_SB=1" "glusuz=GGML_METAL_FUSE_SB=1,GGML_METAL_FUSE_MM_GLU=0" "qkvsiz=GGML_METAL_FUSE_SB=1,GGML_METAL_FUSE_MM_QKV=0" "yalniz_ekle=GGML_METAL_FUSE_SB=1,GGML_METAL_FUSE_MM_GLU=0,GGML_METAL_FUSE_MM_QKV=0")
N=${#CFGS[@]}
echo "tur,yapilandirma,ms_batch" > $OUT
for tur in $(seq 1 ${ROUNDS:-3}); do
  for ((j=0; j<N; j++)); do
    c=${CFGS[$(( (j + tur - 1) % N ))]}; ad=${c%%=*}; envs=${c#*=}; envs=${envs//,/ }
    v=$(env $envs timeout 120 $B/llama-bench -m $M7 -ngl 99 -p 3 -n 0 -b 3 -ub 3 -r 10 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
print('%.4f' % (float(r[-4])/1e6))
")
    echo "$tur,$ad,$v" >> $OUT
  done
done
python3 -I - "$OUT" <<'PY'
import csv, sys, collections, statistics as st
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[r["yapilandirma"]].append(float(r["ms_batch"]))
base = st.mean(d["kapali"])
for k, v in d.items():
    print("%-12s ort %.3f ms  fark %+.3f ms  tur %s" % (k, st.mean(v), st.mean(v)-base, " ".join("%.2f" % x for x in v)))
PY
echo ATFET_BITTI
