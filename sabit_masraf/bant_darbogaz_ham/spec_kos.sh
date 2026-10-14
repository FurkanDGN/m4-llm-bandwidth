#!/bin/bash
# Tahminli uretme uctan uca A/B: 7B (lm_head q4_0) hedef + 0.5B taslak, yamali cekirdek acik, k=1,2.
# orijinal = ekleme ve glu birlestirmeleri kapali; tumu = ikisi acik. Dort tur, sira her turda degisir.
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
OUT=$D/spec_sonuc.csv
mkdir -p $D/spec_cikti
echo "tur,k,yapilandirma,tok_s,n_drafted,n_accept,metin_ozeti" > $OUT
for tur in 1 2 3 4; do
  for k in 1 2; do
    if [ $((tur % 2)) = 1 ]; then ORD="0 1"; else ORD="1 0"; fi
    for f in $ORD; do
      ad=$([ $f = 0 ] && echo orijinal || echo tumu)
      GGML_METAL_FUSE_MM_ADD=$f GGML_METAL_FUSE_MM_GLU=$f GGML_METAL_SB=1 timeout 200 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 256 --spec-draft-n-max $k --spec-draft-p-min 0 -p "$ESSAY" </dev/null > $D/spec_cikti/${ad}_k${k}_r${tur}.txt 2> $D/spec_cikti/${ad}_k${k}_r${tur}.err
      ts=$(grep -h "decoded" $D/spec_cikti/${ad}_k${k}_r${tur}.err | sed -E 's/.*speed: *([0-9.]+) t\/s.*/\1/' | tail -1)
      nd=$(grep -h "n_drafted" $D/spec_cikti/${ad}_k${k}_r${tur}.err | sed -E 's/.*= *([0-9]+).*/\1/' | tail -1)
      na=$(grep -h "n_accept" $D/spec_cikti/${ad}_k${k}_r${tur}.err | sed -E 's/.*= *([0-9]+).*/\1/' | tail -1)
      h=$(md5 -q $D/spec_cikti/${ad}_k${k}_r${tur}.txt | cut -c1-8)
      echo "$tur,$k,$ad,$ts,$nd,$na,$h" >> $OUT
    done
  done
done
echo BITTI >> $OUT
echo SPEC_BITTI
