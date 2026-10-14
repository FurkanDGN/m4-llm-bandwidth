#!/bin/bash
# Agac taslak (np dal) ile zincir k=1 karsilastirmasi, donusumlu, 7B + 0.5B, 96 token
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz/agac
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
CFGS=("z_k1:1:1:0.1:" "a_np2_k2_u:2:2:0.1:--kv-unified -c 2048" "a_np2_k3_u:3:2:0.1:--kv-unified -c 2048" "z_k1_u:1:1:0.1:--kv-unified -c 2048")
OUT=$D/agac2.csv
echo "tur,ad,tok_s,n_drafted,n_accept,metin" > $OUT
N=${#CFGS[@]}
for tur in $(seq 1 ${ROUNDS:-4}); do
  for ((j=0; j<N; j++)); do
    c=${CFGS[$(( (j + tur) % N ))]}
    IFS=: read ad k np ps ex <<< "$c"
    f=$D/${ad}_r$tur
    timeout 300 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max $k --spec-draft-p-min 0 -np $np --spec-draft-p-split $ps $ex -p "$ESSAY" </dev/null > $f.txt 2> $f.err
    ts=$(grep -h "decoded" $f.err | sed -E 's/.*speed: *([0-9.]+) t\/s.*/\1/' | tail -1)
    nd=$(grep -h "n_drafted" $f.err | sed -E 's/.*= *([0-9]+).*/\1/' | tail -1)
    na=$(grep -h "n_accept" $f.err | sed -E 's/.*= *([0-9]+).*/\1/' | tail -1)
    echo "$tur,$ad,$ts,$nd,$na,$(md5 -q $f.txt | cut -c1-8)" >> $OUT
  done
done
echo BITTI >> $OUT
