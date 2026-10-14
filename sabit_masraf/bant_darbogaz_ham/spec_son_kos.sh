#!/bin/bash
# Tahminli uretme k=1..4: cok sutunlu birlestirme kapali/acik, 5 tur (tampon hizalamasi 4096)
D=$HOME/claude-scratch/bant-darbogaz
for k in 1 2 3 4; do
  K=$k ROUNDS=5 $D/spec_kos4.sh $D/spec_son_k$k.csv "kapali=GGML_METAL_FUSE_SB=0" "acik=GGML_METAL_FUSE_SB=1" > $D/spec_son_log_k$k.txt 2>&1
done
echo SPEC_SON_BITTI > $D/spec_son_bitti.txt
