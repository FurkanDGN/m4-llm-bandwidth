#!/bin/bash
# Tahminli uretme k=3 ve k=4: birlestirme kapali / acik, 5 tur (tampon hizalamasi varsayilan 4096)
D=$HOME/claude-scratch/bant-darbogaz
LOG=$D/k4_log.txt; : > $LOG
OFF="GGML_METAL_FUSE_MM_ADD=0,GGML_METAL_FUSE_MM_GLU=0,GGML_METAL_FUSE_MM_QKV=0,GGML_METAL_FUSE_ROPE_K=0,GGML_METAL_FUSE_ROPE_QK=0,GGML_METAL_FUSE_SET_V=0"
for k in 3 4; do
  echo "K=$k basladi" >> $LOG
  K=$k ROUNDS=5 $D/spec_kos4.sh $D/k4_k$k.csv "kapali=$OFF" "acik=GGML_METAL_X=0" >> $LOG 2>&1
  echo "K=$k bitti" >> $LOG
done
echo K4_BITTI >> $LOG
