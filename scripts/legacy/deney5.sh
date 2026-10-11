#!/bin/zsh
M=~/llm-bench/models
S=./scratchpad
if [ ! -s $S/wiki.test.raw ]; then
  curl -sL -o $S/wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  cd $S && unzip -o -q wt.zip && cp wikitext-2-raw/wiki.test.raw $S/wiki.test.raw
fi
ls -lh $S/wiki.test.raw
for m in "qwen2.5-1.5b-instruct-q8_0" "qwen2.5-1.5b-instruct-q4_0" "qwen2.5-1.5b-instruct-q3_k_m" "qwen2.5-1.5b-instruct-q2_k" "olmoe-1b-7b-0924-instruct-q4_0" "qwen2.5-7b-instruct-q4_0-00001-of-00002"; do
  echo "=== PPL $m ==="
  llama-perplexity -m $M/$m.gguf -ngl 99 -f $S/wiki.test.raw --chunks 64 </dev/null 2>&1 | grep "Final estimate" || echo "HATA: $m"
done
echo "BITTI"
