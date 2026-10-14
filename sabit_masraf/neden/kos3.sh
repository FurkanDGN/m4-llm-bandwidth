#!/bin/bash
# -ub taramasi: yerlesimin geldigi rezerv boyutuna gore 3 sutunluk dogrulama grafinin bariyer sayisi (tum birlestirmeler kapali)
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
P="The history of computing"
for ub in 3 4 5 6 8 12 16 32 64 128 256 512; do
  env $OFF GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 40 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -p "$P" </dev/null > $D/ub_$ub.txt 2> $D/ub_$ub.err
  echo "ub=$ub: $(grep -h 'encoded' $D/ub_$ub.err | sed 's/.*encoded *\([0-9]*\) tokens.*/enc=\1/') $(grep 'graf dugum=957 sutun=3' $D/ub_$ub.err | sed 's/.*sutun=3 //' | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')"
done
