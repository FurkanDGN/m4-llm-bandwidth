#!/bin/bash
# Final olcum: cikti esitligi (200 token), perplexity (-b 1: tek token yolu), hiz (5 tur), tahminli uretme (k=1,2).
D=$HOME/claude-scratch/bant-darbogaz
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
LOG=$D/final_log.txt
: > $LOG
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M15=$HOME/llm-bench/models/qwen2.5-1.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
KAPALI="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
P="Explain how a transformer language model generates text, step by step."
echo "FAZ A: cikti esitligi 200 token" >> $LOG
for pair in "05:$M05" "15:$M15" "7:$M7"; do
  tag=${pair%%:*}; M=${pair#*:}
  env $KAPALI GGML_METAL_SB=0 $B/llama-completion -m $M -ngl 99 -n 200 -p "$P" --temp 0 --seed 1 -no-cnv --no-warmup > $D/final_dogt_${tag}_kapali.out 2>/dev/null
  GGML_METAL_SB=0 $B/llama-completion -m $M -ngl 99 -n 200 -p "$P" --temp 0 --seed 1 -no-cnv --no-warmup > $D/final_dogt_${tag}_final.out 2>/dev/null
  if cmp -s $D/final_dogt_${tag}_kapali.out $D/final_dogt_${tag}_final.out; then echo "DOGRULUK $tag: AYNI ($(wc -c < $D/final_dogt_${tag}_final.out) bayt)" >> $LOG; else echo "DOGRULUK $tag: FARKLI" >> $LOG; fi
done
echo "FAZ B: perplexity (-b 1 -ub 1)" >> $LOG
for pair in "05:$M05:3" "15:$M15:2" "7:$M7:1"; do
  tag=${pair%%:*}; rest=${pair#*:}; M=${rest%%:*}; CH=${rest##*:}
  for cfg in kapali final; do
    if [ $cfg = kapali ]; then E="$KAPALI"; else E=""; fi
    env $E GGML_METAL_SB=0 $B/llama-perplexity -m $M -ngl 99 -f $HOME/llm-bench/data/wiki.test.raw -c 512 -b 1 -ub 1 --chunks $CH > $D/final_ppl_${tag}_${cfg}.out 2>&1
    echo "PPL $tag $cfg: $(grep -E 'Final estimate' $D/final_ppl_${tag}_${cfg}.out | tail -1)" >> $LOG
  done
done
echo "FAZ C: hiz 5 tur" >> $LOG
SNAP=$D/bin_final; rm -rf $SNAP; cp -R $B $SNAP
KAPALI_CSV=${KAPALI// /,}
ROUNDS=5 MODELS="05 15 7" BIN=$SNAP $D/faktor2_kos.sh $D/final_hiz.csv "kapali=$KAPALI_CSV" "final=GGML_METAL_FUSE_MM_ADD=1" >> $LOG 2>&1
echo "FAZ D: tahminli uretme" >> $LOG
for k in 1 2; do
  K=$k $D/spec_kos3.sh $D/final_spec_k$k.csv "orijinal=$KAPALI_CSV" "tumu=GGML_METAL_FUSE_MM_ADD=1" "sadece_taslak=GGML_METAL_FUSE_MAX_NODES=900" >> $LOG 2>&1
  echo "SPEC k=$k bitti" >> $LOG
done
echo FINAL_BITTI >> $LOG
