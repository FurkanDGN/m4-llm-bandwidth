#!/bin/bash
# katman başına oran düzenleri: Metal hız (llama-bench -n 128, 5 tur dönüşümlü) + Metal perplexity (-b 1, 4 parça)
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
M=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4-downT-f16.gguf
M0=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
W=$HOME/llm-bench/data/wiki.test.raw
D=$HOME/claude-scratch/bant-darbogaz/seyrek/katman_oran
grp() { local s="" i; for i in $(seq 0 27); do
  if [ $i -le 7 ]; then s+="$1,"; elif [ $i -le 13 ]; then s+="$2,"; elif [ $i -le 19 ]; then s+="$3,"; else s+="$4,"; fi; done; echo ${s%,}; }
S="GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_UP=1"
declare -a AD=(taban K1_u.5x20_d.5 K5_u.5_u.6_u.5_d.6 K2_d.7_u.6_u.5_d.5 K10_d.6_u.6_u.5_d.5 K9_d.7_u.6_u.4_d.4 K8_u.4x20_d.5)
declare -a ENVS=("GGML_METAL_SB=1"
 "$S SEYREK_ORANLAR_DOWN=$(grp .5 .5 .5 .5) SEYREK_ORANLAR_UP=$(grp .5 .5 .5 0)"
 "$S SEYREK_ORANLAR_DOWN=$(grp .5 .5 .5 .6) SEYREK_ORANLAR_UP=$(grp .5 .6 .5 0)"
 "$S SEYREK_ORANLAR_DOWN=$(grp .7 .5 .5 .5) SEYREK_ORANLAR_UP=$(grp 0 .6 .5 0)"
 "$S SEYREK_ORANLAR_DOWN=$(grp .6 .5 .5 .5) SEYREK_ORANLAR_UP=$(grp 0 .6 .5 0)"
 "$S SEYREK_ORANLAR_DOWN=$(grp .7 .5 .5 .4) SEYREK_ORANLAR_UP=$(grp 0 .6 .4 0)"
 "$S SEYREK_ORANLAR_DOWN=$(grp .5 .5 .5 .5) SEYREK_ORANLAR_UP=$(grp .4 .4 .4 0)")
OUT=$D/metal_hiz.csv; echo "tur,ayar,ms_token" > $OUT
for tur in 1 2 3 4 5; do
  for i in "${!AD[@]}"; do
    MM=$M; [ $i = 0 ] && MM=$M0
    v=$(env ${ENVS[$i]} timeout 300 $B/llama-bench -m $MM -ngl 99 -p 0 -n 128 -r 1 -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "import sys,csv;r=list(csv.reader(sys.stdin))[0];print('%.2f'%(1000.0/float(r[-2])))")
    echo "$tur,${AD[$i]},$v" | tee -a $OUT
  done
done
OUT2=$D/metal_ppl.csv; echo "ayar,ppl" > $OUT2
for i in "${!AD[@]}"; do
  [ $i = 0 ] && continue
  v=$(env ${ENVS[$i]} timeout 1500 $B/llama-perplexity -m $M -ngl 99 -f $W -c 512 -b 1 -ub 1 --chunks 4 2>&1 | grep -h "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  echo "${AD[$i]},$v" | tee -a $OUT2
done
echo OLCUM_BITTI
