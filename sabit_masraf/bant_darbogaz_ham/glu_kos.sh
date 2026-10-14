#!/bin/bash
# gate+up+glu birlestirmesi A/B (ekleme birlestirmesi her ikisinde acik): 4 tur, donusumlu.
B=$HOME/claude-scratch/llamacpp-ablasyon/build/bin
D=$HOME/claude-scratch/bant-darbogaz
OUT=$D/glu_sonuc.csv
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M15=$HOME/llm-bench/models/qwen2.5-1.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
echo "model,tur,sira,yapilandirma,ms_token,sapma_ms,tok_s" > $OUT
for tur in 1 2 3 4; do
  for model in 05 15 7; do
    case $model in 05) M=$M05; N=256;; 15) M=$M15; N=192;; 7) M=$M7; N=128;; esac
    sira=0
    for f in 0 1; do
      sira=$((sira+1)); ad=$([ $f = 0 ] && echo glu_kapali || echo glu_acik)
      GGML_METAL_FUSE_MM_GLU=$f GGML_METAL_SB=0 $B/llama-bench -m $M -ngl 99 -p 0 -n $N -r 5 -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
n_gen=int(r[-7]); avg_ns=float(r[-4]); sd=float(r[-3]); ts=float(r[-2])
print('$model,$tur,$sira,$ad,%.4f,%.4f,%.2f' % (avg_ns/n_gen/1e6, sd/n_gen/1e6, ts))
" >> $OUT
    done
  done
done
echo BITTI >> $OUT
