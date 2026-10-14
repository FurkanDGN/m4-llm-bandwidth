#!/bin/zsh
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/katman_oran
OUT=$D/tarama2.csv
# grp <a> <b> <c> <d>: 0-7, 8-13, 14-19, 20-27 gruplarina deger
grp() { local s="" i; for i in {0..27}; do
  if (( i <= 7 )); then s+="$1,"; elif (( i <= 13 )); then s+="$2,"; elif (( i <= 19 )); then s+="$3,"; else s+="$4,"; fi; done; echo ${s%,}; }
ppl() { local ad=$1; shift
  v=$(env "$@" SEYREK_SIM=1 GGML_METAL_SB=1 timeout 1500 $B/llama-perplexity -m $M -ngl 99 -f $W -c 512 --chunks 8 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "$ad,$v" | tee -a $OUT
}
echo "duzen,ppl" > $OUT
ppl "K0 yogun" SEYREK_SIM_DOWN=$(grp 0 0 0 0)
ppl "K0b down0.5 hepsi" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.5)
ppl "K1 up0.5 0-19 + down0.5 (24u)" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0)
ppl "K8 up0.4 0-19 + down0.5 (20u)" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0.4 0.4 0.4 0)
ppl "K2 d0.7 | u0.6 | u0.5 | d0.5 (22.8u)" SEYREK_SIM_DOWN=$(grp 0.7 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0 0.6 0.5 0)
ppl "K3 d0.7 | u0.6 | u0.5 | d0.6 (23.6u)" SEYREK_SIM_DOWN=$(grp 0.7 0.5 0.5 0.6) SEYREK_SIM_UP=$(grp 0 0.6 0.5 0)
ppl "K4 u0.4 | u0.6 | u0.5 | d0.5 (23.6u)" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0.4 0.6 0.5 0)
ppl "K5 u0.5 | u0.6 | u0.5 | d0.6 (26u)" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.6) SEYREK_SIM_UP=$(grp 0.5 0.6 0.5 0)
ppl "K6 u0.5 | u0.6 | u0.6 | d0.5 (26.4u)" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0.5 0.6 0.6 0)
ppl "K7 u0.4 | u0.6 | u0.4 | d0.5 (22.4u)" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0.4 0.6 0.4 0)
ppl "K9 d0.7 | u0.6 | u0.4 | d0.4 (20.8u)" SEYREK_SIM_DOWN=$(grp 0.7 0.5 0.5 0.4) SEYREK_SIM_UP=$(grp 0 0.6 0.4 0)
ppl "K10 d0.6 | u0.6 | u0.5 | d0.5 (22u)" SEYREK_SIM_DOWN=$(grp 0.6 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0 0.6 0.5 0)
ppl "K11 u0.3 | u0.6 | u0.5 | d0.5 (22u)" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.5) SEYREK_SIM_UP=$(grp 0.3 0.6 0.5 0)
echo OLCUM_BITTI
