#!/bin/bash
# FFN-only çizge zamanlaması, katman başına µs; her düzen 3 kez (ortanca 20 koşu)
cd ~/claude-scratch/bant-darbogaz/seyrek/ffn_olc
OUT=ffn_olc.csv; echo "tur,ayar,katman_us,cikti" > $OUT
S="GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_ORAN=0.5 SEYREK_NCHUNK1=16"
U="$S SEYREK_UP=1 SEYREK_UP_KAT_MAX=19"
declare -a AD=(dense down50 downall down50_nc32 down50_toplasiz up50 up50_all up50_nomath up50_gate1 up50_kat27)
declare -a MD=(dense down down down down up up up up up)
declare -a ENVS=("GGML_METAL_SB=1" "$S" "GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_ORAN=0.00001 SEYREK_NCHUNK1=16" "GGML_METAL_SB=1 SEYREK_DOWN=1 SEYREK_ORAN=0.5 SEYREK_NCHUNK1=32" "$S SEYREK_SKIP_TOPLA=1" "$U" "$U SEYREK_GATE_ORAN=0" "$U SEYREK_GATE_NOMATH=1" "$U SEYREK_GATE_V=1" "$S SEYREK_UP=1 SEYREK_UP_KAT_MAX=27")
for tur in 1 2 3; do
  for i in "${!AD[@]}"; do
    r=$(env ${ENVS[$i]} ./ffn_olc ${MD[$i]} 28 20 2>/dev/null | grep SONUC)
    us=$(echo "$r" | sed -E 's/.*katman_us=([0-9.]+).*/\1/'); c=$(echo "$r" | sed -E 's/.*cikti_toplam=([^ ]+).*/\1/')
    echo "$tur,${AD[$i]},$us,$c" | tee -a $OUT
  done
done
echo OLCUM_BITTI
