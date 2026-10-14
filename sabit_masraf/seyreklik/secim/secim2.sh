#!/bin/zsh
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/secim
OUT=$D/secim2.csv
grp() { local s="" i; for i in {0..27}; do
  if (( i <= 7 )); then s+="$1,"; elif (( i <= 13 )); then s+="$2,"; elif (( i <= 19 )); then s+="$3,"; else s+="$4,"; fi; done; echo ${s%,}; }
ppl() { local ad=$1; shift
  v=$(env "$@" SEYREK_SIM=1 GGML_METAL_SB=1 SEYREK_AGIRLIK=$D/ist.bin timeout 1500 $B/llama-perplexity -m $M -ngl 99 -f $W -c 512 --chunks 4 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "$ad,$v" | tee -a $OUT
}
echo "duzen,ppl" > $OUT
DN=$(grp 0.5 0.5 0.5 0.5)
A=SEYREK_AGIRLIK_TUR=1
ppl "A1 up0.5 <=19 | up0.3 20-27" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0.3) $A
ppl "A1 up0.5 <=19 | up0.4 20-27" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0.4) $A
ppl "A1 up0.6 0-13 | up0.5 14-19 | d0.5" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.6 0.6 0.5 0) $A
ppl "A1 up0.6 hepsi" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.6 0.6 0.6 0.6) $A
ppl "A1 up0.4 hepsi" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.4 0.4 0.4 0.4) $A
ppl "A1 up0.5 <=19 d0.6 20-27" SEYREK_SIM_DOWN=$(grp 0.5 0.5 0.5 0.6) SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0) $A
ppl "A1 up0.5 0-19 | up0.5 20-23 | d0.5" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(python3 -c "print(','.join(['0.5']*24+['0']*4))") $A
ppl "A1 up0.7 <=19" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.7 0.7 0.7 0) $A
ppl "A2 up0.5 <=19 | up0.3 20-27" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0.3) SEYREK_AGIRLIK_TUR=2
echo OLCUM_BITTI
