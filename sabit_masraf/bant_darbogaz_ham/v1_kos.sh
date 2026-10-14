#!/bin/bash
cd ~/claude-scratch/llamacpp-ablasyon-sb
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
for r in $(seq 0 10); do for i in 0 1; do v=$(( (i + r) % 2 )); for p in 4 5; do echo "tur=$r V=$v p=$p $(GGML_METAL_SB_V=$v build/bin/llama-bench -m $M7 -ngl 99 -p $p -n 0 -b $p -ub $p -r 10 -o csv 2>/dev/null | tail -1 | python3 -I -c "import sys,csv;r=list(csv.reader(sys.stdin))[0];print('%.3f'%(float(r[-4])/1e6))")"; done; done; done
ps aux | sort -nrk3 | head -4 | awk '{print $3, $11}'
