#!/bin/bash
# 7B, 4 ve 5 token'lik tek graf suresi: matris (varsayilan), MMA'siz matris, skaler r2, skaler r3. Tur basina siralama doner.
B=${BIN:-$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin}
D=$HOME/claude-scratch/bant-darbogaz
OUT=${1:-$D/deney.csv}
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
VARS=(mat nm nm2 nm3)
echo "varyant,tur,p,ms_batch" > $OUT
for tur in $(seq 1 ${ROUNDS:-4}); do
  for p in 4 5; do
    for i in 0 1 2 3; do
      v=${VARS[$(( (i + tur) % 4 ))]}
      case $v in
        mat) E="GGML_METAL_SB=1" ;;
        nm)  E="GGML_METAL_SB_NM=1" ;;
        nm2) E="GGML_METAL_SB_NM=2" ;;
        nm3) E="GGML_METAL_SB_NM=3" ;;
      esac
      ms=$(env $E timeout 120 $B/llama-bench -m $M7 -ngl 99 -p $p -n 0 -b $p -ub $p -r 10 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
print('%.4f' % (float(r[-4])/1e6))
")
      echo "$v,$tur,$p,$ms" >> $OUT
    done
  done
done
echo BITTI >> $OUT
python3 -I - "$OUT" <<'PY'
import csv, sys, collections, statistics as st
d = collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if r["varyant"] in ("mat", "nm", "nm2", "nm3") and r["ms_batch"]:
        d[(r["p"], r["tur"])][r["varyant"]] = float(r["ms_batch"])
for p in ("4", "5"):
    base = [v["mat"] for (pp, t), v in d.items() if pp == p and "mat" in v]
    for k in ("mat", "nm", "nm2", "nm3"):
        xs = [v[k] for (pp, t), v in d.items() if pp == p and k in v and "mat" in v]
        if not xs: continue
        w = sum(1 for x, b in zip(xs, base) if x < b)
        print("p=%s %-3s ort %.3f ms  fark %+.3f ms  matristen hizli %d/%d tur  tur %s" % (p, k, st.mean(xs), st.mean(xs) - st.mean(base), w, len(xs), " ".join("%.2f" % x for x in xs)))
PY
