#!/bin/bash
cd ~/claude-scratch/bant-darbogaz/seyrek/ffn_olc
OUT=ffn_olc4.csv; echo "tur,ayar,katman_us,cikti" > $OUT
S="GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_ORAN=0.5 SEYREK_NCHUNK1=16"
U="$S SEYREK_UP=1 SEYREK_UP_KAT_MAX=27"
declare -a AD=(mv mvhist mvhist_nr8 glu upgate_v4 upgate_v5 upgate_all_v4 upgate_all_v5 upgate_nomath downonly50 downonly_all)
declare -a MD=(mv mvhist mvhist glu upgate upgate upgate upgate upgate downonly downonly)
declare -a ENVS=("GGML_METAL_SB=1" "$U" "$U SEYREK_UP_NR=8" "GGML_METAL_SB=1" "$U SEYREK_GATE_V=4" "$U" "$U SEYREK_GATE_V=4 SEYREK_GATE_ORAN=0" "$U SEYREK_GATE_ORAN=0" "$U SEYREK_GATE_NOMATH=1" "$S" "GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_ORAN=0.00001 SEYREK_NCHUNK1=16")
for tur in 1 2 3; do
  for i in "${!AD[@]}"; do
    r=$(env ${ENVS[$i]} ./ffn_olc ${MD[$i]} 28 20 2>/dev/null | grep SONUC)
    us=$(echo "$r" | sed -E 's/.*katman_us=([0-9.]+).*/\1/'); c=$(echo "$r" | sed -E 's/.*cikti_toplam=([^ ]+).*/\1/')
    echo "$tur,${AD[$i]},$us,$c" | tee -a $OUT
  done
done
echo OLCUM_BITTI
