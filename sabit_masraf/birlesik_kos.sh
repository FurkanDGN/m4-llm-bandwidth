#!/bin/bash
# Ekleme birlestirmesi (FUSE) ve ilk komut grubu boyutu (N_MAIN=128) birlikte: 3 tur, donusumlu.
B=$HOME/claude-scratch/llamacpp-ablasyon/build/bin
D=$HOME/claude-scratch/bant-darbogaz
OUT=$D/birlesik_sonuc.csv
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M15=$HOME/llm-bench/models/qwen2.5-1.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
CF4=("taban:GGML_METAL_FUSE_MM_ADD=0" "ekle:GGML_METAL_FUSE_MM_ADD=1" "nmain128:GGML_METAL_FUSE_MM_ADD=0,GGML_METAL_N_MAIN=128" "ekle_nmain128:GGML_METAL_FUSE_MM_ADD=1,GGML_METAL_N_MAIN=128")
CF2=("taban:GGML_METAL_FUSE_MM_ADD=0" "ekle:GGML_METAL_FUSE_MM_ADD=1" "ekle_nmain128:GGML_METAL_FUSE_MM_ADD=1,GGML_METAL_N_MAIN=128")
echo "model,tur,sira,yapilandirma,ms_token,sapma_ms,tok_s" > $OUT
for tur in 1 2 3; do
  for model in 05 15 7; do
    case $model in 05) M=$M05; N=256; CF=("${CF4[@]}");; 15) M=$M15; N=192; CF=("${CF4[@]}");; 7) M=$M7; N=128; CF=("${CF2[@]}");; esac
    sira=0
    for c in "${CF[@]}"; do
      ad=${c%%:*}; vars=${c#*:}; sira=$((sira+1))
      unset GGML_METAL_N_MAIN GGML_METAL_FUSE_MM_ADD
      for v in ${vars//,/ }; do export $v; done
      GGML_METAL_SB=0 $B/llama-bench -m $M -ngl 99 -p 0 -n $N -r 5 -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
n_gen=int(r[-7]); avg_ns=float(r[-4]); sd=float(r[-3]); ts=float(r[-2])
print('$model,$tur,$sira,$ad,%.4f,%.4f,%.2f' % (avg_ns/n_gen/1e6, sd/n_gen/1e6, ts))
" >> $OUT
    done
  done
done
echo BITTI >> $OUT
