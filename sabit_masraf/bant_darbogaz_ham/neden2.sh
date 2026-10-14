#!/bin/bash
# Her birlestirme tek basina, yalniz hedef grafinda (MIN_NODES=900): once sayilar, sonra sure (k=2, 3 tur)
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
OFF="GGML_METAL_FUSE_MM_ADD=0,GGML_METAL_FUSE_MM_GLU=0,GGML_METAL_FUSE_MM_QKV=0,GGML_METAL_FUSE_ROPE_K=0,GGML_METAL_FUSE_ROPE_QK=0,GGML_METAL_FUSE_SET_V=0"
declare -a AD=("orijinal" "t_ekle" "t_glu" "t_qkv" "t_ropek" "t_ropek_qk" "t_ropek_v")
declare -a EV=(
 "$OFF"
 "$OFF,GGML_METAL_FUSE_MM_ADD=1,GGML_METAL_FUSE_MIN_NODES=900"
 "$OFF,GGML_METAL_FUSE_MM_GLU=1,GGML_METAL_FUSE_MIN_NODES=900"
 "$OFF,GGML_METAL_FUSE_MM_QKV=1,GGML_METAL_FUSE_MIN_NODES=900"
 "$OFF,GGML_METAL_FUSE_ROPE_K=1,GGML_METAL_FUSE_MIN_NODES=900"
 "$OFF,GGML_METAL_FUSE_ROPE_K=1,GGML_METAL_FUSE_ROPE_QK=1,GGML_METAL_FUSE_MIN_NODES=900"
 "$OFF,GGML_METAL_FUSE_ROPE_K=1,GGML_METAL_FUSE_SET_V=1,GGML_METAL_FUSE_MIN_NODES=900"
)
SAY=$D/neden2_say.txt; : > $SAY
for i in "${!AD[@]}"; do
  envs=${EV[$i]//,/ }
  env $envs GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=1 timeout 200 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 48 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "$ESSAY" </dev/null > /dev/null 2> $D/neden/n2_${AD[$i]}.err
  echo "${AD[$i]}: $(grep 'graf dugum=957 sutun=3' $D/neden/n2_${AD[$i]}.err | sort | uniq -c | sort -rn | head -1 | sed 's/.*graf //')" >> $SAY
done
echo SAYI_BITTI >> $SAY
ARGS=()
for i in "${!AD[@]}"; do ARGS+=("${AD[$i]}=${EV[$i]}"); done
K=2 $D/spec_kos3.sh $D/neden2.csv "${ARGS[@]}"
echo NEDEN2_BITTI >> $SAY
