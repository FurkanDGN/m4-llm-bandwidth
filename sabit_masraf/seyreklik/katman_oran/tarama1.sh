#!/bin/zsh
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/katman_oran
OUT=$D/tarama1.csv
liste() { # liste <varsayilan> <bas> <son> <deger>  -> 28 elemanli virgullu liste
  local v=$1 a=$2 b=$3 x=$4 s="" i
  for i in {0..27}; do if (( i >= a && i <= b )); then s+="$x,"; else s+="$v,"; fi; done
  echo ${s%,}
}
ppl() { local ad=$1; shift
  local t0=$(date +%s)
  v=$(env "$@" SEYREK_SIM=1 GGML_METAL_SB=1 timeout 1500 $B/llama-perplexity -m $M -ngl 99 -f $W -c 512 --chunks 4 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "$ad,$v,$(( $(date +%s) - t0 ))s" | tee -a $OUT
}
echo "duzen,ppl,sure" > $OUT
ppl "down0.5 hepsi" SEYREK_SIM_DOWN=$(liste 0.5 0 -1 0)
ppl "up0.5 0-19 + down0.5" SEYREK_SIM_DOWN=$(liste 0.5 0 -1 0) SEYREK_SIM_UP=$(liste 0 0 19 0.5)
for g in "0 7" "8 13" "14 19" "20 27"; do read -r a b <<< "$g"; set -- $a $b
  for x in 0.6 0.7 0.8; do ppl "down $x kat $1-$2, diger 0.5" SEYREK_SIM_DOWN=$(liste 0.5 $1 $2 $x); done
done
for x in 0.3 0.4; do ppl "down $x kat 20-27, diger 0.5" SEYREK_SIM_DOWN=$(liste 0.5 20 27 $x); done
for g in "0 7" "8 13" "14 19"; do read -r a b <<< "$g"; set -- $a $b
  for x in 0.5 0.6 0.7; do ppl "up $x kat $1-$2 yalniz, down0.5" SEYREK_SIM_DOWN=$(liste 0.5 0 -1 0) SEYREK_SIM_UP=$(liste 0 $1 $2 $x); done
done
echo OLCUM_BITTI
