#!/bin/zsh
set -e
M=~/llm-bench/models
S=./scratchpad
PROMPT="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."

echo "########## 2) SPEKULATIF DECODE ##########"
echo "--- baseline: 1.5B q4_0 tek basina ---"
llama-cli -m $M/qwen2.5-1.5b-instruct-q4_0.gguf -ngl 99 --temp 0 -n 512 -no-cnv -p "$PROMPT" </dev/null 2>&1 >/dev/null | grep -E "eval time" || true
echo "--- spekulatif: 1.5B q4_0 + 0.5B q4_0 taslak ---"
llama-speculative -m $M/qwen2.5-1.5b-instruct-q4_0.gguf -md $M/qwen2.5-0.5b-instruct-q4_0.gguf -ngl 99 -ngld 99 --temp 0 -n 512 --draft-max 16 -p "$PROMPT" </dev/null 2>&1 >/dev/null | grep -E "encoded|decoded|accept|draft:|target:" || true

echo "########## 2b) MOE KIYASI (tg128) ##########"
for m in olmoe-1b-7b-0924-instruct-q4_0.gguf qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf qwen2.5-1.5b-instruct-q4_0.gguf; do
  echo "--- $m ---"
  llama-bench -m $M/$m -p 0 -n 128 -o csv 2>/dev/null | tail -1 | awk -F'","' '{printf "  decode: %s tok/s\n", $(NF-1)}'
done

echo "########## 3) KALITE BEDELI (perplexity, wikitext-2, 64 chunk) ##########"
for q in q8_0 q4_0 q3_k_m q2_k; do
  echo "--- $q ---"
  llama-perplexity -m $M/qwen2.5-1.5b-instruct-$q.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep -E "Final estimate" || echo "PPL HATA: $q"
done
echo "BITTI"
