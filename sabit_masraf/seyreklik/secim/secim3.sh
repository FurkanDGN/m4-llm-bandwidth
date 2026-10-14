#!/bin/zsh
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/secim
OUT=$D/secim3.csv
grp() { local s="" i; for i in {0..27}; do
  if (( i <= 7 )); then s+="$1,"; elif (( i <= 13 )); then s+="$2,"; elif (( i <= 19 )); then s+="$3,"; else s+="$4,"; fi; done; echo ${s%,}; }
ppl() { local ad=$1; shift
  v=$(env "$@" SEYREK_SIM=1 GGML_METAL_SB=1 timeout 1500 $B/llama-perplexity -m $M -ngl 99 -f $W -c 512 --chunks 4 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "$ad,$v" | tee -a $OUT
}
echo "duzen,ppl" > $OUT
DN=$(grp 0.5 0.5 0.5 0.5); D6=$(grp 0.6 0.6 0.6 0.6); D7=$(grp 0.7 0.7 0.7 0.7)
G=SEYREK_AGIRLIK=$D/ist.bin; N=SEYREK_AGIRLIK_DOWN=$D/down_norm.bin
ppl "down0.5 hepsi duz" SEYREK_SIM_DOWN=$DN
ppl "down0.5 hepsi |h|*norm" SEYREK_SIM_DOWN=$DN $N
ppl "down0.6 hepsi duz" SEYREK_SIM_DOWN=$D6
ppl "down0.6 hepsi |h|*norm" SEYREK_SIM_DOWN=$D6 $N
ppl "down0.7 hepsi duz" SEYREK_SIM_DOWN=$D7
ppl "down0.7 hepsi |h|*norm" SEYREK_SIM_DOWN=$D7 $N
ppl "up0.5<=19 d0.5 |u|*g" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0) $G
ppl "up0.5<=19 d0.5 |u|*g*norm, down |h|*norm" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0) $G $N SEYREK_AGIRLIK_UP_DOWN=1
ppl "up0.5<=19 d0.5 |u|*norm (g'siz)" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0) $N SEYREK_AGIRLIK_UP_DOWN=1
ppl "up0.5<=19 d0.5 |u|*g, down |h|*norm" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0) $G $N
ppl "up0.4<=19 d0.5 |u|*g*norm, down |h|*norm" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.4 0.4 0.4 0) $G $N SEYREK_AGIRLIK_UP_DOWN=1
ppl "up0.6<=19 d0.5 |u|*g*norm, down |h|*norm" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.6 0.6 0.6 0) $G $N SEYREK_AGIRLIK_UP_DOWN=1
echo OLCUM_BITTI
