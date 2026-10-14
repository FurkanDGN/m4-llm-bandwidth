#!/bin/bash
# Zaman: yerlesim (rezerv boyutu) ve ekleme bayragi, tahminli uretme k=2, 4 tur, donusumlu sira
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz/neden_kanit
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
OFF="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
ON_ADD="$OFF GGML_METAL_FUSE_MM_ADD=1 GGML_METAL_FUSE_MIN_NODES=900"
P="The history of computing"
CF=("off_ub512|$OFF|512" "off_ub3|$OFF|3" "off_ub4|$OFF|4" "on_ub512|$ON_ADD|512")
OUT=$D/kos4.csv
echo "tur,yapilandirma,tok_s,hedef_ms,metin" > $OUT
for tur in 1 2 3 4; do
  for ((j=0;j<${#CF[@]};j++)); do
    c=${CF[$(( (j + tur - 1) % ${#CF[@]} ))]}
    IFS='|' read ad envs ub <<< "$c"
    f=$D/k4_${ad}_r$tur
    env $envs GGML_METAL_SB=1 timeout 100 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 96 --spec-draft-n-max 2 --spec-draft-p-min 0 -ub $ub -p "$P" </dev/null > $f.txt 2> $f.err
    ts=$(grep -h "decoded" $f.err | sed -E 's/.*speed: *([0-9.]+) t\/s.*/\1/' | tail -1)
    pe=$(grep -h "prompt eval time" $f.err | tail -1 | sed -E 's/.*= *([0-9.]+) ms.*/\1/')
    h=$(md5 -q $f.txt | cut -c1-8)
    echo "$tur,$ad,$ts,$pe,$h" >> $OUT
  done
done
cat $OUT
