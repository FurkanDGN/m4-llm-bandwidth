#!/bin/zsh
# llama-lookup matrisi (ilk tur, yamasiz): model x is x k. Cikti: m_<model>_<is>_k<k>.full
M=~/llm-bench/models
S=./gen
declare -A MODELS
MODELS[1.5b]=$M/qwen2.5-1.5b-instruct-q4_0.gguf
MODELS[7b]=$M/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
for mk in 1.5b 7b; do
  for p in prose code essay; do
    for k in 0 8 31; do
      timeout 200 llama-lookup -m ${MODELS[$mk]} -ngl 99 --temp 0 -n 600 -c 4096 -f $S/p_$p.txt --spec-draft-n-max $k </dev/null > $S/m_${mk}_${p}_k$k.full 2>&1
      echo "$mk $p k=$k exit=$?"
    done
  done
done
echo MATRIX-BITTI
