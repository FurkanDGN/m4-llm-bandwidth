#!/bin/zsh
# 7B + 0.5B ve 1.5B + 0.5B spekulatif tarama (yamasiz, brew llama.cpp). --spec-draft-p-min etkisizdi (n_drafted ayni).
M=~/llm-bench/models
S=./scratchpad
D=$M/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
run() {
  local tag=$1 tgt=$2 k=$4 pm=$5 n=$6
  if [[ $3 == essay ]]; then
    timeout 200 llama-speculative -m $tgt -md $D -ngl 99 -ngld 99 --temp 0 -n $n --spec-draft-n-max $k --spec-draft-p-min $pm -p "$ESSAY" </dev/null > $S/spec/${tag}_$3_k${k}_p${pm}.full 2>&1
  else
    timeout 200 llama-speculative -m $tgt -md $D -ngl 99 -ngld 99 --temp 0 -n $n --spec-draft-n-max $k --spec-draft-p-min $pm -f $S/gen/p_$3.txt </dev/null > $S/spec/${tag}_$3_k${k}_p${pm}.full 2>&1
  fi
  echo "$tag $3 k=$k pmin=$pm exit=$?"
}
T7=$M/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
T15=$M/qwen2.5-1.5b-instruct-q4_0.gguf
for c in "2 0" "4 0" "8 0" "15 0" "31 0" "15 0.5" "15 0.8" "15 0.9" "31 0.8" "31 0.9"; do run 7b $T7 essay ${=c} 256; done
for c in "4 0" "15 0" "31 0" "15 0.8" "31 0.8"; do run 7b $T7 code ${=c} 400; done
for c in "4 0" "8 0" "15 0" "31 0" "15 0.8"; do run 1.5b $T15 essay ${=c} 256; done
echo SWEEP-BITTI
