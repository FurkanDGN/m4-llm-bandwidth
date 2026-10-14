#!/bin/bash
cd ~/claude-scratch/bant-darbogaz/seyrek/ffn_olc
OUT=ffn_olc3.csv; echo "tur,ayar,katman_us,cikti" > $OUT
U="GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_ORAN=0.5 SEYREK_NCHUNK1=16 SEYREK_UP=1 SEYREK_UP_KAT_MAX=19"
declare -a AD=(up50_v4 up50_v5 up50_v5_g8 up50_v5_u8 up50_v5_g8_u8 up50_v4_u8 up50_all_v5_g8_u8)
declare -a MD=(up up up up up up up)
declare -a ENVS=("$U SEYREK_GATE_V=4" "$U" "$U SEYREK_GATE_NR=8" "$U SEYREK_UP_NR=8" "$U SEYREK_GATE_NR=8 SEYREK_UP_NR=8" "$U SEYREK_GATE_V=4 SEYREK_UP_NR=8" "$U SEYREK_GATE_NR=8 SEYREK_UP_NR=8 SEYREK_GATE_ORAN=0")
for tur in 1 2 3; do
  for i in "${!AD[@]}"; do
    r=$(env ${ENVS[$i]} ./ffn_olc ${MD[$i]} 28 20 2>/dev/null | grep SONUC)
    us=$(echo "$r" | sed -E 's/.*katman_us=([0-9.]+).*/\1/'); c=$(echo "$r" | sed -E 's/.*cikti_toplam=([^ ]+).*/\1/')
    echo "$tur,${AD[$i]},$us,$c" | tee -a $OUT
  done
done
echo OLCUM_BITTI
