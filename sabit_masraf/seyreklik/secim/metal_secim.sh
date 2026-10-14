#!/bin/bash
# ağırlıklı seçim Metal kıyası: perplexity (-b 1, 4 parça) sonra hız (llama-bench -n 128, 5 tur dönüşümlü)
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4-downT-f16.gguf
M0=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/secim
grp() { local s="" i; for i in $(seq 0 27); do
  if [ $i -le 7 ]; then s+="$1,"; elif [ $i -le 13 ]; then s+="$2,"; elif [ $i -le 19 ]; then s+="$3,"; else s+="$4,"; fi; done; echo ${s%,}; }
S="GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_UP=1 SEYREK_ORANLAR_DOWN=$(grp .5 .5 .5 .5)"
A="SEYREK_AGIRLIK=$D/ist.bin SEYREK_AGIRLIK_TUR=1"
declare -a AD=(taban M1_u.5x20_duz M2_u.5x20_agirlikli M3_u.4x20_agirlikli M4_u.5x20_u.3x8_agirlikli M5_u.5x20_u.4x8_agirlikli M6_u.4x28_agirlikli M7_u.6x20_agirlikli)
declare -a ENVS=("GGML_METAL_SB=1"
 "$S SEYREK_ORANLAR_UP=$(grp .5 .5 .5 0)"
 "$S SEYREK_ORANLAR_UP=$(grp .5 .5 .5 0) $A"
 "$S SEYREK_ORANLAR_UP=$(grp .4 .4 .4 0) $A"
 "$S SEYREK_ORANLAR_UP=$(grp .5 .5 .5 .3) $A"
 "$S SEYREK_ORANLAR_UP=$(grp .5 .5 .5 .4) $A"
 "$S SEYREK_ORANLAR_UP=$(grp .4 .4 .4 .4) $A"
 "$S SEYREK_ORANLAR_UP=$(grp .6 .6 .6 0) $A")
OUT2=$D/metal_secim_ppl.csv; echo "ayar,ppl" > $OUT2
for i in "${!AD[@]}"; do
  [ $i = 0 ] && continue
  v=$(env ${ENVS[$i]} timeout 1500 $B/llama-perplexity -m $M -ngl 99 -f $W -c 512 -b 1 -ub 1 --chunks 4 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "${AD[$i]},$v" | tee -a $OUT2
done
OUT=$D/metal_secim_hiz.csv; echo "tur,ayar,ms_token" > $OUT
for tur in 1 2 3 4 5; do
  for i in "${!AD[@]}"; do
    MM=$M; [ $i = 0 ] && MM=$M0
    v=$(env ${ENVS[$i]} timeout 300 $B/llama-bench -m $MM -ngl 99 -p 0 -n 128 -r 1 -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "import sys,csv;r=list(csv.reader(sys.stdin))[0];print('%.2f'%(1000.0/float(r[-2])))")
    echo "$tur,${AD[$i]},$v" | tee -a $OUT
  done
done
echo OLCUM_BITTI
