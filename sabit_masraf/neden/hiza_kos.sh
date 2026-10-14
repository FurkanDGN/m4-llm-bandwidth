#!/bin/bash
# Tampon hizalamasi (32 / 4096) x birlestirme (kapali / acik): tahminli uretme k=2, tek token, bariyer sayisi
D=$HOME/claude-scratch/bant-darbogaz
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
LOG=$D/hiza_log.txt; : > $LOG
OFF="GGML_METAL_FUSE_MM_ADD=0,GGML_METAL_FUSE_MM_GLU=0,GGML_METAL_FUSE_MM_QKV=0,GGML_METAL_FUSE_ROPE_K=0,GGML_METAL_FUSE_ROPE_QK=0,GGML_METAL_FUSE_SET_V=0"
CFG=("a32_kapali=GGML_METAL_BUFFER_ALIGN=32,$OFF" "a32_acik=GGML_METAL_BUFFER_ALIGN=32" "a4096_kapali=GGML_METAL_BUFFER_ALIGN=4096,$OFF" "a4096_acik=GGML_METAL_BUFFER_ALIGN=4096")
echo "FAZ 1: bariyer sayisi (3 sutunlu hedef grafi)" >> $LOG
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
for c in "${CFG[@]}"; do
  ad=${c%%=*}; envs=${c#*=}; envs=${envs//,/ }
  env $envs GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=1 timeout 200 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 48 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "$ESSAY" </dev/null > /dev/null 2> $D/hiza_$ad.err
  echo "BARIYER $ad: $(grep 'graf dugum=957 sutun=3' $D/hiza_$ad.err | sort | uniq -c | sort -rn | head -1 | sed 's/.*graf //')" >> $LOG
done
echo "FAZ 2: tahminli uretme k=2, 5 tur" >> $LOG
K=2 ROUNDS=5 $D/spec_kos4.sh $D/hiza_spec.csv "${CFG[@]}" >> $LOG 2>&1
echo "SPEC_BITTI" >> $LOG
echo "FAZ 3: tek token, 5 tur" >> $LOG
ROUNDS=5 MODELS="05 15 7" $D/faktor2_kos.sh $D/hiza_tek.csv "${CFG[@]}" >> $LOG 2>&1
echo HIZA_BITTI >> $LOG
