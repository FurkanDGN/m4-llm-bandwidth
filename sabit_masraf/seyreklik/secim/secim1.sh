#!/bin/zsh
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/secim
OUT=$D/secim1.csv
# 1) kalibrasyon: wiki.test'in son 600 KB'ı (test ilk 4 parça, örtüşmez), 20 parça
SEYREK_TOPLA=$D/ist.bin GGML_METAL_SB=1 $B/llama-perplexity -m $M -ngl 99 -f $D/kalib.raw -c 512 --chunks 20 2>&1 | grep -E "SEYREK_TOPLA|Final"
grp() { local s="" i; for i in {0..27}; do
  if (( i <= 7 )); then s+="$1,"; elif (( i <= 13 )); then s+="$2,"; elif (( i <= 19 )); then s+="$3,"; else s+="$4,"; fi; done; echo ${s%,}; }
ppl() { local ad=$1; shift
  v=$(env "$@" SEYREK_SIM=1 GGML_METAL_SB=1 timeout 1500 $B/llama-perplexity -m $M -ngl 99 -f $W -c 512 --chunks 4 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "$ad,$v" | tee -a $OUT
}
echo "duzen,ppl" > $OUT
DN=$(grp 0.5 0.5 0.5 0.5)
for r in 0.5 0.4 0.6; do
  UP=$(grp $r $r $r 0)
  ppl "up$r <=19 duz |u|" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$UP
  ppl "up$r <=19 |u|*ort|silu g|" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$UP SEYREK_AGIRLIK=$D/ist.bin SEYREK_AGIRLIK_TUR=1
  ppl "up$r <=19 |u|*ort|h|/ort|u|" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$UP SEYREK_AGIRLIK=$D/ist.bin SEYREK_AGIRLIK_TUR=2
  ppl "up$r <=19 |u|*sqrt(ort|silu g|)" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$UP SEYREK_AGIRLIK=$D/ist.bin SEYREK_AGIRLIK_TUR=3
done
# up tüm katmanlarda (20-27 dahil) ağırlıklı: son katmanlar artık kaldırır mı?
ppl "up0.5 hepsi duz" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0.5)
ppl "up0.5 hepsi |u|*ort|silu g|" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0.5) SEYREK_AGIRLIK=$D/ist.bin SEYREK_AGIRLIK_TUR=1
ppl "up0.5 hepsi |u|*ort|h|/ort|u|" SEYREK_SIM_DOWN=$DN SEYREK_SIM_UP=$(grp 0.5 0.5 0.5 0.5) SEYREK_AGIRLIK=$D/ist.bin SEYREK_AGIRLIK_TUR=2
echo OLCUM_BITTI
