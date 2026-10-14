#!/bin/bash
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz/agac
OUT=$D/pencere2.csv
CFGS=("z1_c0:1:1:0" "z2_c0:2:1:0" "z3_c0:3:1:0" "z4_c0:4:1:0" "z1_c2k:1:1:2048" "z2_c2k:2:1:2048" "z3_c2k:3:1:2048" "z4_c2k:4:1:2048" "a2_np2:2:2:2048" "a3_np2:3:2:2048" "a4_np2:4:2:2048")
echo "tur,ad,tok_s,n_drafted,n_accept,metin" > $OUT
N=${#CFGS[@]}
for tur in 1 2 3; do for ((j=0; j<N; j++)); do
  c=${CFGS[$(( (j + tur*4) % N ))]}; IFS=: read ad k np ctx <<< "$c"; f=$D/p2_${ad}_r$tur
  $B/llama-speculative -m $HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf -md $HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max $k --spec-draft-p-min 0 -np $np --spec-draft-p-split 0.1 -c $ctx -p "Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs." </dev/null > $f.txt 2> $f.err
  echo "$tur,$ad,$(grep -h decoded $f.err | sed -E 's/.*speed: *([0-9.]+).*/\1/'),$(grep -h n_drafted $f.err | sed -E 's/.*= *//'),$(grep -h 'n_accept ' $f.err | sed -E 's/.*= *//'),$(md5 -q $f.txt | cut -c1-8)" >> $OUT
done; done
echo BITTI >> $OUT
