#!/bin/bash
# Perplexity, 2 ve 3 token'lik batch (kucuk batch yolu), cok sutunlu birlestirme kapali/acik
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
W=$HOME/llm-bench/data/wiki.test.raw
OUT=$D/ppl_sb.csv
echo "model,batch,fuse_sb,ppl" > $OUT
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M15=$HOME/llm-bench/models/qwen2.5-1.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
for pair in "05:$M05:8" "15:$M15:8" "7:$M7:3"; do
  IFS=: read tag M CH <<< "$pair"
  for b in 2 3; do
    for fs in 0 1; do
      v=$(GGML_METAL_FUSE_SB=$fs timeout 600 $B/llama-perplexity -m $M -f $W -ngl 99 -b $b -ub $b --chunks $CH 2>&1 | grep -E "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
      echo "$tag,$b,$fs,$v" >> $OUT
    done
  done
done
echo BITTI >> $OUT
