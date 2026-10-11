#!/bin/zsh
set -e
M=~/llm-bench/models
S=./scratchpad

echo "########## 1) DECODE HIZI (tg128, GPU) ##########"
for q in q2_k q3_k_m q4_0; do
  echo "--- $q ---"
  llama-bench -m $M/qwen2.5-1.5b-instruct-$q.gguf -p 0 -n 128 -o csv 2>/dev/null | tail -1
done

echo "########## 2) SPEKULATIF DECODE ##########"
PROMPT="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
echo "--- baseline: 1.5B q4_0 tek basina ---"
llama-cli -m $M/qwen2.5-1.5b-instruct-q4_0.gguf -ngl 99 --temp 0 -n 512 -no-cnv -p "$PROMPT" 2>&1 >/dev/null | grep -E "eval time|sampl" || true
echo "--- spekulatif: 1.5B q4_0 + 0.5B q4_0 taslak ---"
llama-speculative -m $M/qwen2.5-1.5b-instruct-q4_0.gguf -md $M/qwen2.5-0.5b-instruct-q4_0.gguf -ngl 99 -ngld 99 --temp 0 -n 512 --draft-max 16 -p "$PROMPT" 2>&1 >/dev/null | grep -E "encoded|decoded|accept|draft|target|eval time" || true

echo "########## 3) KALITE BEDELI (perplexity, wikitext-2) ##########"
if [ ! -f $S/wiki.test.raw ]; then
  curl -sL -o $S/wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  cd $S && unzip -o -q wt.zip && cp wikitext-2-raw/wiki.test.raw $S/wiki.test.raw
fi
for q in q8_0 q4_0 q3_k_m q2_k; do
  echo "--- $q ---"
  llama-perplexity -m $M/qwen2.5-1.5b-instruct-$q.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 2>&1 | grep -E "Final estimate" || echo "PPL HATA: $q"
done
echo "BITTI"
