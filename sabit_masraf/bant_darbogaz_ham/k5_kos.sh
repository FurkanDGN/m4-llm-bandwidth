#!/bin/bash
# k=3 ve k=4: kucuk batch yolu varsayilan (2-3 skaler, 4-8 matris) ve yalniz skaler (2-8), birlestirme acik, hizalama 4096, 5 tur
D=$HOME/claude-scratch/bant-darbogaz
LOG=$D/k5_log.txt; : > $LOG
for k in 3 4; do
  echo "K=$k basladi" >> $LOG
  K=$k ROUNDS=5 $D/spec_kos5.sh $D/k5_k$k.csv "sb_varsayilan=GGML_METAL_SB=1" "sb_skaler=GGML_METAL_SB=s" >> $LOG 2>&1
  echo "K=$k bitti" >> $LOG
done
echo K5_BITTI >> $LOG
