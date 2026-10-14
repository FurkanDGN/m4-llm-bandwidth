#!/bin/bash
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
for cfg in off on; do
  if [ $cfg = off ]; then E="$OFF"; else E="$ON"; fi
  env $E GGML_METAL_N_MAIN=100000 GGML_METAL_GRAPH_DEBUG=2 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 4 --spec-draft-n-max 2 --spec-draft-p-min 0 -v -p "The history of computing" </dev/null > /dev/null 2> go_$cfg.err
  echo "$cfg: fuse2=$(grep -c 'fuse 2 ops' go_$cfg.err) fuse3=$(grep -c 'fuse 3 ops' go_$cfg.err)"
done
