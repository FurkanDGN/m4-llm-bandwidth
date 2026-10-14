#!/bin/bash
D=$HOME/claude-scratch/bant-darbogaz
ROUNDS=5 MODELS="05" $D/faktor2_kos.sh $D/nmain3_05.csv "varsayilan=GGML_METAL_X=0" "n64=GGML_METAL_N_MAIN=64" "n96=GGML_METAL_N_MAIN=96" "n128=GGML_METAL_N_MAIN=128" "n160=GGML_METAL_N_MAIN=160" "n192=GGML_METAL_N_MAIN=192" > $D/nmain3_05_ozet.txt 2>&1
ROUNDS=3 MODELS="15" $D/faktor2_kos.sh $D/nmain3_15.csv "varsayilan=GGML_METAL_X=0" "n128=GGML_METAL_N_MAIN=128" "n192=GGML_METAL_N_MAIN=192" > $D/nmain3_15_ozet.txt 2>&1
echo bitti > $D/nmain3_bitti.txt
