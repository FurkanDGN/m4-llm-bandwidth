#!/bin/bash
# 7B, 2 ve 3 token'lik graf: yalniz glu birlestirmesi, thread basina satir sayisi (SB_FR) taramasi, 3 tur donuslu
B=${BIN:-$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin}
D=$HOME/claude-scratch/bant-darbogaz
OUT=${1:-$D/sb_glu.csv}
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
ONLY="GGML_METAL_FUSE_SB=1,GGML_METAL_FUSE_MM_ADD=0"
CFGS=("kapali=GGML_METAL_FUSE_SB=0" "glu_fr1=$ONLY,GGML_METAL_SB_FR=1" "glu_fr2=$ONLY,GGML_METAL_SB_FR=2" "glu_fr3=$ONLY,GGML_METAL_SB_FR=3" "glu_fr4=$ONLY,GGML_METAL_SB_FR=4")
N=${#CFGS[@]}
echo "p,tur,yapilandirma,ms_batch" > $OUT
for p in 3 2; do
  for tur in $(seq 1 ${ROUNDS:-3}); do
    for ((j=0; j<N; j++)); do
      c=${CFGS[$(( (j + tur - 1) % N ))]}; ad=${c%%=*}; envs=${c#*=}; envs=${envs//,/ }
      v=$(env $envs timeout 120 $B/llama-bench -m $M7 -ngl 99 -p $p -n 0 -b $p -ub $p -r 10 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
print('%.4f' % (float(r[-4])/1e6))
")
      echo "$p,$tur,$ad,$v" >> $OUT
    done
  done
done
python3 -I - "$OUT" <<'PY'
import csv, sys, collections, statistics as st
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[(r["p"], r["yapilandirma"])].append(float(r["ms_batch"]))
for p in ("3", "2"):
    base = st.mean(d[(p, "kapali")])
    for (pp, k), v in d.items():
        if pp != p: continue
        print("p=%s %-9s ort %.3f ms  fark %+.3f ms  tur %s" % (p, k, st.mean(v), st.mean(v)-base, " ".join("%.2f" % x for x in v)))
PY
echo GLU_BITTI
