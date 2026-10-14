#!/bin/bash
B=$HOME/claude-scratch/neden-kopya/build/bin
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
run() { local ad=$1 envs=$2 ub=$3; env $envs GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -p "The history of computing" </dev/null > k13_$ad.txt 2> k13_$ad.err; echo "$ad: $(grep 'graf dugum=957 sutun=3' k13_$ad.err | sed 's/.*sutun=3 //' | sort | uniq -c | sort -rn | head -2 | tr '\n' ';') md5=$(md5 -q k13_$ad.txt | cut -c1-8)"; }
run off_ub512 "$OFF" 512
run ub3 "$OFF" 3
run ub3_a256 "$OFF GGML_ALLOC_ALIGN=256" 3
run on_ub512 "$ON" 512
run on_ub512_a256 "$ON GGML_ALLOC_ALIGN=256" 512
