#!/bin/bash
# Graf ayrintili gunluk: 3 sutunluk dogrulama grafinin adresleri. Ayni komut, uc yerlesim.
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
P="The history of computing"
for ub in 512 4 3; do
  env $OFF GGML_METAL_N_MAIN=100000 GGML_METAL_GRAPH_DEBUG=3 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 6 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -v -p "$P" </dev/null > gd_$ub.txt 2> gd_$ub.err
  echo "ub=$ub: $(wc -l < gd_$ub.err) satir, MUL_MAT satiri: $(grep -c 'node\[.*MUL_MAT ' gd_$ub.err)"
done
