#!/bin/zsh
set -e
M=~/llm-bench/models
S=./scratchpad
echo "=== OLMoE decode (tg128) ==="
llama-bench -m $M/olmoe-1b-7b-0924-instruct-q4_0.gguf -p 0 -n 128 </dev/null 2>/dev/null | grep tg
echo "=== PPL 1.5B q8_0 ==="
llama-perplexity -m $M/qwen2.5-1.5b-instruct-q8_0.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep "Final estimate"
echo "=== PPL 1.5B q4_0 ==="
llama-perplexity -m $M/qwen2.5-1.5b-instruct-q4_0.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep "Final estimate"
echo "=== PPL 1.5B q3_k_m ==="
llama-perplexity -m $M/qwen2.5-1.5b-instruct-q3_k_m.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep "Final estimate"
echo "=== PPL 1.5B q2_k ==="
llama-perplexity -m $M/qwen2.5-1.5b-instruct-q2_k.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep "Final estimate"
echo "=== PPL OLMoE q4_0 ==="
llama-perplexity -m $M/olmoe-1b-7b-0924-instruct-q4_0.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep "Final estimate"
echo "=== PPL 7B q4_0 ==="
llama-perplexity -m $M/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep "Final estimate"
echo "BITTI"
