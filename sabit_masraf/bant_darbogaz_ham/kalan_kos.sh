#!/bin/bash
D=$HOME/claude-scratch/bant-darbogaz
$D/kayma2_kos.sh
ROUNDS=6 MODELS="05 15" $D/faktor2_kos.sh $D/nmain2.csv "varsayilan=GGML_METAL_X=0" "n64=GGML_METAL_N_MAIN=64" "n96=GGML_METAL_N_MAIN=96" "n128=GGML_METAL_N_MAIN=128" "n160=GGML_METAL_N_MAIN=160" "n192=GGML_METAL_N_MAIN=192" > $D/nmain2_ozet.txt 2>&1
echo KALAN_BITTI > $D/kalan_bitti.txt
