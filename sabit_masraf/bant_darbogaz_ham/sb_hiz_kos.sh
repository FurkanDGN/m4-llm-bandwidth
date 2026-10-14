#!/bin/bash
# 2 ve 3 token'lik tek grafin suresi (llama-bench -p N -n 0): cok sutunlu birlestirme kapali/acik, 5 tur donuslu
B=${BIN:-$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin}
D=$HOME/claude-scratch/bant-darbogaz
OUT=${1:-$D/sb_hiz.csv}
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
echo "model,tur,p,fuse_sb,ms_batch" > $OUT
for tur in $(seq 1 ${ROUNDS:-5}); do
  for model in 05 7; do
    if [ $model = 05 ]; then M=$M05; else M=$M7; fi
    for p in 2 3; do
      for j in 0 1; do
        fs=$(( (j + tur) % 2 ))
        v=$(GGML_METAL_FUSE_SB=$fs timeout 120 $B/llama-bench -m $M -ngl 99 -p $p -n 0 -b $p -ub $p -r 10 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
print('%.4f' % (float(r[-4])/1e6))
")
        echo "$model,$tur,$p,$fs,$v" >> $OUT
      done
    done
  done
done
echo BITTI >> $OUT
python3 -I - "$OUT" <<'PY'
import csv, sys, collections, statistics as st
d = collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if r["fuse_sb"] in ("0", "1") and r["ms_batch"]:
        d[(r["model"], r["p"], r["tur"])][r["fuse_sb"]] = float(r["ms_batch"])
for m in ("05", "7"):
    for p in ("2", "3"):
        a = [v["0"] for (mm, pp, t), v in d.items() if mm == m and pp == p and "0" in v and "1" in v]
        b = [v["1"] for (mm, pp, t), v in d.items() if mm == m and pp == p and "0" in v and "1" in v]
        if not a: continue
        w = sum(1 for x, y in zip(a, b) if y < x)
        print("model %-3s p=%s kapali %.3f ms acik %.3f ms fark %+.3f ms (%%%+.1f) acik hizli %d/%d tur" % (m, p, st.mean(a), st.mean(b), st.mean(b)-st.mean(a), (st.mean(b)/st.mean(a)-1)*100, w, len(a)))
PY
