#!/bin/bash
# Tahminli uretme A/B (k=2, 96 token, 3 tur): 7B hedef + 0.5B taslak. Yapilandirmalar ortam degiskeniyle verilir.
# kullanim: spec_kos2.sh <cikti.csv> <yapilandirma_adi=ENV1=a,ENV2=b> ...
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
OUT=$1; shift
CFGS=("$@")
mkdir -p $D/spec_cikti2
echo "tur,yapilandirma,tok_s,hedef_dogrulama_ms,metin_ozeti" > $OUT
N=${#CFGS[@]}
for tur in $(seq 1 ${ROUNDS:-3}); do
  for ((j=0; j<N; j++)); do
    idx=$(( (j + tur - 1) % N ))
    c=${CFGS[$idx]}
    ad=${c%%=*}
    envs=${c#*=}
    envs=${envs//,/ }
    dosya=$D/spec_cikti2/${ad}_k${K:-2}_r${tur}
    env $envs GGML_METAL_SB=1 timeout 200 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max ${K:-2} --spec-draft-p-min 0 -p "$ESSAY" </dev/null > $dosya.txt 2> $dosya.err
    ts=$(grep -h "decoded" $dosya.err | sed -E 's/.*speed: *([0-9.]+) t\/s.*/\1/' | tail -1)
    pe=$(grep -h "prompt eval time" $dosya.err | tail -1 | sed -E 's/.*= *([0-9.]+) ms.*/\1/')
    h=$(md5 -q $dosya.txt | cut -c1-8)
    echo "$tur,$ad,$ts,$pe,$h" >> $OUT
  done
done
echo BITTI >> $OUT
