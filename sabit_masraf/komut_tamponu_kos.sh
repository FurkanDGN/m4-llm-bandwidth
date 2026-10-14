#!/bin/bash
# Ilk komut tamponuna dusen dugum sayisi (N_MAIN) ve ek komut tamponu sayisi (N_CB) taramasi.
B=$HOME/claude-scratch/llamacpp-ablasyon/build/bin
D=$HOME/claude-scratch/bant-darbogaz
OUT=$D/komut_tamponu_sonuc.csv
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
C05=("taban:" "nmain16:GGML_METAL_N_MAIN=16" "nmain32:GGML_METAL_N_MAIN=32" "nmain128:GGML_METAL_N_MAIN=128" "nmain256:GGML_METAL_N_MAIN=256" "ncb2:GGML_METAL_N_CB=2" "ncb3:GGML_METAL_N_CB=3" "ncb2_nmain32:GGML_METAL_N_CB=2,GGML_METAL_N_MAIN=32" "ncb2_nmain128:GGML_METAL_N_CB=2,GGML_METAL_N_MAIN=128" "taban:")
C7=("taban:" "nmain128:GGML_METAL_N_MAIN=128" "ncb2:GGML_METAL_N_CB=2" "ncb2_nmain128:GGML_METAL_N_CB=2,GGML_METAL_N_MAIN=128" "taban:")
echo "model,tur,sira,yapilandirma,ms_token,sapma_ms,tok_s" > $OUT
for tur in 1 2; do
  for model in 05 7; do
    if [ $model = 05 ]; then M=$M05; N=256; CF=("${C05[@]}"); else M=$M7; N=128; CF=("${C7[@]}"); fi
    sira=0
    for c in "${CF[@]}"; do
      ad=${c%%:*}; vars=${c#*:}; sira=$((sira+1))
      unset GGML_METAL_N_MAIN GGML_METAL_N_CB
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
