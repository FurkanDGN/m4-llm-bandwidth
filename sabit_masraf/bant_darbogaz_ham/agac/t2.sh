#!/bin/bash
cd ~/claude-scratch/bant-darbogaz/agac
for env in "GGML_METAL_SB=1"; do
env $env ~/claude-scratch/llamacpp-ablasyon-sb/build/bin/llama-speculative -m ~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf -md ~/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -np 2 --spec-draft-p-split 0.1 -c 2048 -p "Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs." </dev/null > t2.txt 2> t2.err
echo "[$env] $(grep -h decoded t2.err | sed -E 's/.*speed: *([0-9.]+).*/\1/') $(grep -h 'n_drafted\|n_accept ' t2.err | sed 's/.*I //' | tr '\n' ' ') metin $(md5 -q t2.txt | cut -c1-8)"; head -c 300 t2.txt; echo
done
