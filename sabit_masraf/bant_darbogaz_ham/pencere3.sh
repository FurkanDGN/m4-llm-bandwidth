#!/bin/bash
D=$HOME/claude-scratch/bant-darbogaz
NEW=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
OLD=$D/bin_A
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
OUT=$D/pencere3.txt
for r in $(seq 0 10); do for i in 0 1; do s=$(( (i + r) % 2 )); for p in 2 3; do
  if [ $s = 0 ]; then v=$(DYLD_LIBRARY_PATH=$OLD $OLD/llama-bench -m $M7 -ngl 99 -p $p -n 0 -b $p -ub $p -r 10 -o csv 2>/dev/null | tail -1 | python3 -I -c "import sys,csv;r=list(csv.reader(sys.stdin))[0];print('%.3f'%(float(r[-4])/1e6))"); n=eski; else v=$($NEW/llama-bench -m $M7 -ngl 99 -p $p -n 0 -b $p -ub $p -r 10 -o csv 2>/dev/null | tail -1 | python3 -I -c "import sys,csv;r=list(csv.reader(sys.stdin))[0];print('%.3f'%(float(r[-4])/1e6))"); n=yeni; fi
  echo "ab tur=$r $n p=$p $v" >> $OUT
done; done; done
for r in 1 2 3; do for c in 2048 8192 32768 0; do
  f=$D/agac/p3_c${c}_r$r
  $NEW/llama-speculative -m $M7 -md $HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 1 --spec-draft-p-min 0 -c $c -p "Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs." </dev/null > $f.txt 2> $f.err
  echo "ctx tur=$r c=$c $(grep -h decoded $f.err | sed -E 's/.*speed: *([0-9.]+).*/\1/') $(md5 -q $f.txt | cut -c1-8)" >> $OUT
done; done
echo BITTI >> $OUT
