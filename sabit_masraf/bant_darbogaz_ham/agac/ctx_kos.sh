#!/bin/bash
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
OUT=$HOME/claude-scratch/bant-darbogaz/agac/ctx.csv
echo "tur,k,c,tok_s,metin" > $OUT
for tur in 1 2 3; do for k in 1 2 3 4; do for c in 0 2048; do
  [ $(( (tur + k) % 2 )) = 1 ] && c=$((2048 - c))
  f=$HOME/claude-scratch/bant-darbogaz/agac/ctx_k${k}_c${c}_r$tur
  $B/llama-speculative -m $HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf -md $HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max $k --spec-draft-p-min 0 -c $c -p "Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs." </dev/null > $f.txt 2> $f.err
  echo "$tur,$k,$c,$(grep -h decoded $f.err | sed -E 's/.*speed: *([0-9.]+).*/\1/'),$(md5 -q $f.txt | cut -c1-8)" >> $OUT
done; done; done
echo BITTI >> $OUT
