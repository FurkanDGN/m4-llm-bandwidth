#!/bin/bash
# 0.5B tg: Metal System Trace, GPU bosluklari
B=$HOME/claude-scratch/llamacpp-ablasyon/build/bin
D=$HOME/claude-scratch/bant-darbogaz
M=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
rm -rf $D/iz05.trace $D/iz05_gpuint.xml
GGML_METAL_SB=0 xcrun xctrace record --template 'Metal System Trace' --time-limit 12s --output $D/iz05.trace --launch -- $B/llama-bench -m $M -ngl 99 -p 0 -n 512 -r 3 -mmp 0 > $D/iz05_kayit.log 2>&1
echo "kayit_cikis=$?"
xcrun xctrace export --input $D/iz05.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' --output $D/iz05_gpuint.xml > $D/iz05_export.log 2>&1
echo "export_cikis=$?"
python3 -I ~/claude-scratch/m4-deneyler/parse_trace.py $D/iz05_gpuint.xml
echo BITTI
