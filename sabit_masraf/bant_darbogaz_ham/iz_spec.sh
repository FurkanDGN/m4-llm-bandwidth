#!/bin/bash
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
for a in 0 1; do
  rm -rf $D/izs$a.trace $D/izs${a}_gpuint.xml
  GGML_METAL_FUSE_MM_ADD=$a GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_SB=1 xcrun xctrace record --template 'Metal System Trace' --time-limit 15s --output $D/izs$a.trace --launch -- $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 160 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "$ESSAY" > $D/izs${a}_kayit.log 2>&1
  xcrun xctrace export --input $D/izs$a.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' --output $D/izs${a}_gpuint.xml > $D/izs${a}_export.log 2>&1
  echo "izs$a hazir"
done
echo IZ_BITTI
