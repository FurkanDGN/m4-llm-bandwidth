#!/bin/bash
B=$HOME/claude-scratch/neden-kopya/build/bin
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
for cfg in off on; do
  if [ $cfg = off ]; then E="$OFF"; else E="$ON"; fi
  env $E GGML_SCHED_LOG_REALLOC=1 GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 40 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "The history of computing" </dev/null > k12_$cfg.txt 2> k12_$cfg.err
  echo "== $cfg: realloc olaylari:"; grep -n 'SCHED_REALLOC\|dugum=957 sutun=\(1\|3\) birlesen' k12_$cfg.err | cut -c1-120 | head -12
done
