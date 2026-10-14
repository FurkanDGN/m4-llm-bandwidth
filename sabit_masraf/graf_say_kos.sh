#!/bin/bash
# Karar adimindaki GPU cagrisi sayisi: model x birlestirme acik/kapali
B=$HOME/claude-scratch/llamacpp-sb/llama.cpp/build/bin
D=$HOME/claude-scratch/bant-darbogaz
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
for model in 05 7; do
  eval M=\$M$model
  for fus in on off; do
    if [ $fus = off ]; then export GGML_METAL_FUSION_DISABLE=1; else unset GGML_METAL_FUSION_DISABLE; fi
    GGML_METAL_SB=0 GGML_METAL_GRAPH_DEBUG=1 $B/llama-completion -m $M -ngl 99 -n 4 -p "Hello" -v --no-warmup -no-cnv --temp 0 > $D/say_${model}_${fus}.out 2> $D/say_${model}_${fus}.err
    python3 -I $D/graf_say.py $D/say_${model}_${fus}.err
  done
done
echo BITTI
