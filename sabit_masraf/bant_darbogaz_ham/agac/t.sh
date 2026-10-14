#!/bin/bash
cd ~/claude-scratch/bant-darbogaz/agac
for cfg in "1 1" "2 2" "3 2" "2 3" "3 3"; do set -- $cfg; k=$1; np=$2
~/claude-scratch/llamacpp-ablasyon-sb/build/bin/llama-speculative -m ~/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf -md ~/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max $k --spec-draft-p-min 0 -np $np --spec-draft-p-split 0.1 -c 2048 -v -p "Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs." </dev/null > t_dbg.txt 2> t_dbg.err
echo "k=$k np=$np split=$(grep -c 'splitting seq' t_dbg.err) speed $(grep -h decoded t_dbg.err | sed -E 's/.*speed: *([0-9.]+).*/\1/') $(grep -h 'n_drafted\|n_accept ' t_dbg.err | sed 's/.*I //' | tr '\n' ' ') metin $(md5 -q t_dbg.txt | cut -c1-8)"
done
