#!/bin/bash
# gömülü seçim ağırlıklı model (downT-sel) doğrulama: perplexity (-b 1, 4 parça) + hız (3 tur)
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
MS=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4-downT-sel.gguf
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4-downT-f16.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/secim
grp() { local s="" i; for i in $(seq 0 27); do
  if [ $i -le 19 ]; then s+="$1,"; else s+="$2,"; fi; done; echo ${s%,}; }
S="GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_UP=1 SEYREK_ORANLAR_DOWN=$(grp .5 .5)"
G="SEYREK_AGIRLIK=$D/ist.bin SEYREK_AGIRLIK_TUR=1 SEYREK_AGIRLIK_DOWN=$D/down_norm.bin"
declare -a AD=(sel_u.5 sel_u.4 sel_kapat_u.5 eski_dosya_u.5 sel_d.5_yalniz)
declare -a MM=($MS $MS $MS $M $MS)
declare -a ENVS=("$S SEYREK_ORANLAR_UP=$(grp .5 0)" "$S SEYREK_ORANLAR_UP=$(grp .4 0)" "$S SEYREK_ORANLAR_UP=$(grp .5 0) SEYREK_AGIRLIK_KAPAT=1" "$S SEYREK_ORANLAR_UP=$(grp .5 0) $G" "$S SEYREK_ORANLAR_UP=$(grp 0 0)")
OUT2=$D/sel_dogrula_ppl.csv; echo "ayar,ppl" > $OUT2
for i in "${!AD[@]}"; do
  v=$(env ${ENVS[$i]} timeout 1500 $B/llama-perplexity -m ${MM[$i]} -ngl 99 -f $W -c 512 -b 1 -ub 1 --chunks 4 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "${AD[$i]},$v" | tee -a $OUT2
done
OUT=$D/sel_dogrula_hiz.csv; echo "tur,ayar,ms_token" > $OUT
for tur in 1 2 3; do
  for i in 0 3; do
    v=$(env ${ENVS[$i]} timeout 300 $B/llama-bench -m ${MM[$i]} -ngl 99 -p 0 -n 128 -r 1 -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "import sys,csv;r=list(csv.reader(sys.stdin))[0];print('%.2f'%(1000.0/float(r[-2])))")
    echo "$tur,${AD[$i]},$v" | tee -a $OUT
  done
done
echo OLCUM_BITTI
