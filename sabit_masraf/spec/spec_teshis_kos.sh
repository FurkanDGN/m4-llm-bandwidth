#!/bin/bash
# k=2 teshis: hangi birlestirme, taslak hafifleyince hedef dogrulama suresi uzuyor mu?
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
OUT=$D/spec_teshis.csv
mkdir -p $D/spec_teshis
echo "tur,yapilandirma,tok_s,hedef_dogrulama_ms,metin_ozeti" > $OUT
CF=("orijinal:0:0:0" "ekle:1:0:0" "glu:0:1:0" "tumu:1:1:0" "agir_taslak:0:0:1")
for tur in 1 2 3; do
  n=${#CF[@]}
  for ((j=0;j<n;j++)); do
    c=${CF[$(( (j + tur) % n ))]}
    IFS=: read ad a g fd <<< "$c"
    if [ $fd = 1 ]; then export GGML_METAL_FUSION_DISABLE=1; else unset GGML_METAL_FUSION_DISABLE; fi
    GGML_METAL_FUSE_MM_ADD=$a GGML_METAL_FUSE_MM_GLU=$g GGML_METAL_SB=1 timeout 200 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 256 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "$ESSAY" </dev/null > $D/spec_teshis/${ad}_r${tur}.txt 2> $D/spec_teshis/${ad}_r${tur}.err
    ts=$(grep -h "decoded" $D/spec_teshis/${ad}_r${tur}.err | sed -E 's/.*speed: *([0-9.]+) t\/s.*/\1/' | tail -1)
    pe=$(grep -h "prompt eval time" $D/spec_teshis/${ad}_r${tur}.err | sed -E 's/.*= *([0-9.]+) ms \/.*/\1/' | tail -1)
    h=$(md5 -q $D/spec_teshis/${ad}_r${tur}.txt | cut -c1-8)
    echo "$tur,$ad,$ts,$pe,$h" >> $OUT
  done
done
unset GGML_METAL_FUSION_DISABLE
echo BITTI >> $OUT
echo TESHIS_BITTI
