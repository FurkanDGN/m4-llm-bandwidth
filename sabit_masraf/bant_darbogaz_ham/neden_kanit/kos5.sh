#!/bin/bash
# Mikro olcum: tek model, 3 tokenluk grafin suresi, yerlesim kaynagi (-ub) degisir. Birlestirmeler kapali. 4 tur, donusumlu.
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
UBS=(512 3 4 512 3 4 512 3 4 512 3 4)
OUT=$D/kos5.csv
echo "sira,ub,pp3_ts" > $OUT
for i in "${!UBS[@]}"; do
  ub=${UBS[$i]}
  r=$(env $OFF GGML_METAL_SB=1 timeout 100 $B/llama-bench -m $T -ngl 99 -p 3 -n 0 -ub $ub -r 20 -o csv 2>/dev/null | tail -1)
  echo "$i,$ub,$r" >> $OUT
done
cat $OUT | cut -c1-400
