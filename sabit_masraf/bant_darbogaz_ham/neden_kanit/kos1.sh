#!/bin/bash
# Hedef grafinda 3 sutunluk grafin bariyer sayisi: yerlesim kaynagi (rezerv boyutu) ve ekleme bayragi capraz
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON_ADD="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
run() { # ad, env, ekstra arglar
  local ad=$1 envs=$2; shift 2
  env $envs GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 48 --spec-draft-n-max 2 --spec-draft-p-min 0 "$@" -p "$ESSAY" </dev/null > $D/$ad.txt 2> $D/$ad.err
  echo "$ad: rc=$? $(grep 'graf dugum=957 sutun=3' $D/$ad.err | sed 's/.*graf //' | sort | uniq -c | sort -rn | head -2 | tr '\n' ';')"
}
run a_off_ub512 "$OFF"
run b_off_ub3 "$OFF" -ub 3
run c_off_ub4 "$OFF" -ub 4
run d_on_ub512 "$ON_ADD"
run e_on_ub3 "$ON_ADD" -ub 3
