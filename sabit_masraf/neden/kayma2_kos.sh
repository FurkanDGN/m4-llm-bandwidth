#!/bin/bash
# Tek token kayma tekrari: 0.5B ve 1.5B, kayma 0/512/1024/1536, 6 tur donuslu, llama-bench -n 96 -r 3
B=$HOME/claude-scratch/neden-kopya/build/bin
D=$HOME/claude-scratch/bant-darbogaz
OUT=$D/kayma_tek2.csv
echo "model,tur,shift,tok_s" > $OUT
SH=(0 512 1024 1536)
for tur in 1 2 3 4 5 6; do
  for m in 05 15; do
    if [ $m = 05 ]; then M=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf; else M=$HOME/llm-bench/models/qwen2.5-1.5b-instruct-q4_0.gguf; fi
    for ((j=0; j<4; j++)); do
      sh=${SH[$(( (j + tur - 1) % 4 ))]}
      v=$(GGML_ALLOC_SHIFT=$sh timeout 120 $B/llama-bench -m $M -ngl 99 -p 0 -n 96 -r 3 -o csv 2>/dev/null | tail -1 | awk -F, '{gsub(/"/,"",$(NF-1)); print $(NF-1)}')
      echo "$m,$tur,$sh,$v" >> $OUT
    done
  done
done
echo "ENDMARK" >> $OUT
