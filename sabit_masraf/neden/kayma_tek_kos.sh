#!/bin/bash
# Tek token (llama-bench tg64): tum aktivasyonlari SHIFT bayt kaydir, birlestirme acik, hizalama 32 (ajan kopyasi), 3 tur dönüşümlü
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz
OUT=$D/kayma_tek.csv
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
echo "model,tur,shift,tok_s" > $OUT
SHIFTS="0 256 512 768 1024 1280"
for tur in 1 2 3; do
  for m in 05 7; do
    if [ $m = 05 ]; then M=$M05; N=96; else M=$M7; N=64; fi
    for sh in $SHIFTS; do
      v=$(GGML_ALLOC_SHIFT=$sh timeout 120 $B/llama-bench -m $M -ngl 99 -p 0 -n $N -r 1 -o csv 2>/dev/null | tail -1 | awk -F, '{gsub(/"/,"",$(NF-1)); print $(NF-1)}')
      echo "$m,$tur,$sh,$v" >> $OUT
    done
  done
done
echo BITTI >> $OUT
