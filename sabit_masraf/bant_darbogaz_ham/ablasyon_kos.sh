#!/bin/bash
# Islem atlatma olcumu: her yapilandirma icin tg suresi. Iki tur, taban basta ve sonda.
B=$HOME/claude-scratch/llamacpp-ablasyon/build/bin
D=$HOME/claude-scratch/bant-darbogaz
OUT=$D/ablasyon_sonuc.csv
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
SMALL="RMS_NORM,ROPE,SET_ROWS,FLASH_ATTN_EXT,ADD,GLU,GET_ROWS"
CONFIGS=("taban:" "RMS_NORM:RMS_NORM" "ROPE:ROPE" "SET_ROWS:SET_ROWS" "FLASH_ATTN:FLASH_ATTN_EXT" "ADD:ADD" "GLU:GLU" "KUCUK_HEPSI:$SMALL" "MUL_MAT:MUL_MAT" "HEPSI:ALL" "taban:")
echo "model,tur,sira,yapilandirma,ms_token,sapma_ms,tok_s" > $OUT
for tur in 1 2; do
  for model in 05 7; do
    if [ $model = 05 ]; then M=$M05; N=256; else M=$M7; N=128; fi
    sira=0
    for c in "${CONFIGS[@]}"; do
      ad=${c%%:*}; skip=${c#*:}; sira=$((sira+1))
      if [ -n "$skip" ]; then export GGML_METAL_SKIP_OPS=$skip; else unset GGML_METAL_SKIP_OPS; fi
      GGML_METAL_SB=0 $B/llama-bench -m $M -ngl 99 -p 0 -n $N -r 5 -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
# son alanlar: ... n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts
n_gen=int(r[-7]); avg_ns=float(r[-4]); sd=float(r[-3]); ts=float(r[-2])
print('$model,$tur,$sira,$ad,%.4f,%.4f,%.2f' % (avg_ns/n_gen/1e6, sd/n_gen/1e6, ts))
" >> $OUT
    done
  done
done
echo BITTI >> $OUT
