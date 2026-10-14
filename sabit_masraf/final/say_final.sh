#!/bin/bash
# Token basina GPU cagrisi (kodda sayilan dispatch): model x birlestirme adimi
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
KAP="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
: > $D/say5.txt
for model in 05 7; do
  case $model in 05) M=$M05;; 7) M=$M7;; esac
  for v in upstream_kapali bizim_kapali ekle ekle_glu ekle_glu_qkv ekle_glu_qkv_ropek ekle_glu_qkv_ropek_qk ekle_glu_qkv_ropek_qk_v; do
    case $v in
      upstream_kapali) E="GGML_METAL_FUSION_DISABLE=1";;
      bizim_kapali) E="$KAP";;
      ekle) E="GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0";;
      ekle_glu) E="GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0";;
      ekle_glu_qkv) E="GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0";;
      ekle_glu_qkv_ropek) E="GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0";;
      ekle_glu_qkv_ropek_qk) E="GGML_METAL_FUSE_SET_V=0";;
      ekle_glu_qkv_ropek_qk_v) E="GGML_METAL_X=0";;
    esac
    env $E GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=0 timeout 120 $B/llama-completion -m $M -ngl 99 -n 4 -p "Hello" --no-warmup -no-cnv --temp 0 2> $D/say5_${model}_${v}.err > /dev/null
    echo "$model $v: $(grep 'graf dugum' $D/say5_${model}_${v}.err | tail -1 | sed 's/.*graf //')" >> $D/say5.txt
  done
done
echo SAY_BITTI >> $D/say5.txt
