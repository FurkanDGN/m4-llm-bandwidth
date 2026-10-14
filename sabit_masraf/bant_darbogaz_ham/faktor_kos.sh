#!/bin/bash
# Birden cok yapilandirmanin hiz karsilastirmasi: modeller 0.5B (-n 96 -r 3), 1.5B (-n 96 -r 3), 7B (-n 64 -r 2); 3 tur, sira her turda doner.
# kullanim: faktor_kos.sh <cikti.csv> "<ad>=ENV1=a,ENV2=b" ...
OUT=$1; shift
CFGS=("$@")
B=${BIN:-$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin}
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M15=$HOME/llm-bench/models/qwen2.5-1.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
echo "model,tur,sira,yapilandirma,ms_token,sapma_ms,tok_s" > $OUT
N=${#CFGS[@]}
for tur in 1 2 3; do
  for model in 05 15 7; do
    case $model in 05) M=$M05; NT=96; R=3;; 15) M=$M15; NT=96; R=3;; 7) M=$M7; NT=64; R=2;; esac
    for ((j=0; j<N; j++)); do
      idx=$(( (j + tur - 1) % N ))
      c=${CFGS[$idx]}; ad=${c%%=*}; envs=${c#*=}; envs=${envs//,/ }
      env $envs GGML_METAL_SB=0 $B/llama-bench -m $M -ngl 99 -p 0 -n $NT -r $R -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
n_gen=int(r[-7]); avg_ns=float(r[-4]); sd=float(r[-3]); ts=float(r[-2])
print('$model,$tur,$((j+1)),$ad,%.4f,%.4f,%.2f' % (avg_ns/n_gen/1e6, sd/n_gen/1e6, ts))
" >> $OUT
    done
  done
done
echo BITTI >> $OUT
python3 -I - <<PY
import csv, collections
d = collections.defaultdict(list)
names = []
for r in csv.DictReader(open("$OUT")):
    if r["model"] in ("05", "15", "7"):
        d[(r["model"], r["yapilandirma"])].append(float(r["ms_token"]))
        if r["yapilandirma"] not in names: names.append(r["yapilandirma"])
for m in ("05", "15", "7"):
    base = d[(m, names[0])]
    for nm in names:
        v = d[(m, nm)]
        if not v: continue
        wins = sum(1 for x, y in zip(base, v) if y < x)
        print("model %-3s %-14s ort %.4f ms  tur: %s  (tabana gore %+.4f ms, tabandan hizli %d/%d tur)" % (m, nm, sum(v)/len(v), " ".join("%.3f" % x for x in v), sum(v)/len(v)-sum(base)/len(base), wins, len(v)))
PY
echo FAKTOR_BITTI
