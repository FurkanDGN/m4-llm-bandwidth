#!/bin/bash
# 7B, 2 ve 3 token'lik graf: skaler yolda qkv birlestirmesi kalsin mi. kapali, yalniz ekle, ekle+qkv; 6 tur, kosu basina 40 tekrar
B=${BIN:-$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin}
D=$HOME/claude-scratch/bant-darbogaz
OUT=$D/sb_qkv.csv
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
CFGS=("kapali=GGML_METAL_FUSE_SB=0" "ekle=GGML_METAL_FUSE_SB=1,GGML_METAL_FUSE_MM_QKV=0" "ekle_qkv=GGML_METAL_FUSE_SB=1")
N=${#CFGS[@]}
$B/llama-bench -m $M7 -ngl 99 -p 3 -n 0 -b 3 -ub 3 -r 3 -o csv >/dev/null 2>&1
echo "p,tur,yapilandirma,ms_batch" > $OUT
for p in 2 3; do
  for tur in 1 2 3 4 5 6; do
    for ((j=0; j<N; j++)); do
      c=${CFGS[$(( (j + tur - 1) % N ))]}; ad=${c%%=*}; envs=${c#*=}; envs=${envs//,/ }
      v=$(env $envs timeout 120 $B/llama-bench -m $M7 -ngl 99 -p $p -n 0 -b $p -ub $p -r 40 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
print('%.4f' % (float(r[-4])/1e6))
")
      echo "$p,$tur,$ad,$v" >> $OUT
    done
  done
done
python3 -I - "$OUT" <<'PY' > $D/sb_qkv_ozet.txt
import csv, sys, collections, statistics as st
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[(r["p"], r["yapilandirma"])].append(float(r["ms_batch"]))
for p in ("2", "3"):
    base = d[(p, "kapali")]
    for ad in ("kapali", "ekle", "ekle_qkv"):
        v = d[(p, ad)]
        w = sum(1 for x, y in zip(base, v) if y < x)
        print("p=%s %-9s ort %.3f ms  fark %+.3f ms (%%%+.1f)  kapaliya gore hizli %d/%d  tur %s" % (p, ad, st.mean(v), st.mean(v)-st.mean(base), (st.mean(v)/st.mean(base)-1)*100, w, len(v), " ".join("%.2f" % x for x in v)))
    e = d[(p, "ekle")]; q = d[(p, "ekle_qkv")]
    print("p=%s qkv ek etkisi (ekle_qkv - ekle): %+.3f ms, qkv'li hizli %d/%d tur" % (p, st.mean(q)-st.mean(e), sum(1 for x, y in zip(e, q) if y < x), len(e)))
PY
