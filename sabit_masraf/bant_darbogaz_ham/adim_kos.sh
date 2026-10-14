#!/bin/bash
# Bir birlestirme adiminin kisa olcumu: 0.5B 48 token cikti kontrolu (tum birlestirmeler kapali ile karsilastirma),
# sonra 0.5B (-n 96 -r 3) ve 7B (-n 64 -r 2) hiz olcumu, 3 tur, sira her turda doner.
# kullanim: adim_kos.sh <ad> "<onceki ortam>" "<yeni ortam>"
AD=$1; ONCEKI=$2; YENI=$3
B=${BIN:-$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin}
D=$HOME/claude-scratch/bant-darbogaz
M05=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
M7=$HOME/llm-bench/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
P="Explain how a transformer language model generates text, step by step."
KAPALI="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_SET_V=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_NORM_MM=0"
env $KAPALI GGML_METAL_SB=0 $B/llama-completion -m $M05 -ngl 99 -n 48 -p "$P" --temp 0 --seed 1 -no-cnv --no-warmup > $D/adim_${AD}_kapali.out 2>/dev/null
env $YENI GGML_METAL_SB=0 $B/llama-completion -m $M05 -ngl 99 -n 48 -p "$P" --temp 0 --seed 1 -no-cnv --no-warmup > $D/adim_${AD}_yeni.out 2>/dev/null
if cmp -s $D/adim_${AD}_kapali.out $D/adim_${AD}_yeni.out; then echo "DOGRULUK: AYNI ($(wc -c < $D/adim_${AD}_yeni.out) bayt)"; else echo "DOGRULUK: FARKLI"; fi
OUT=$D/adim_${AD}.csv
echo "model,tur,sira,yapilandirma,ms_token,sapma_ms,tok_s" > $OUT
for tur in 1 2 3; do
  for model in 05 7; do
    case $model in 05) M=$M05; N=96; R=3;; 7) M=$M7; N=64; R=2;; esac
    if [ $((tur % 2)) = 1 ]; then ORD="onceki yeni"; else ORD="yeni onceki"; fi
    sira=0
    for ad in $ORD; do
      sira=$((sira+1)); E=$([ $ad = onceki ] && echo "$ONCEKI" || echo "$YENI")
      env $E GGML_METAL_SB=0 $B/llama-bench -m $M -ngl 99 -p 0 -n $N -r $R -mmp 0 -o csv 2>/dev/null | tail -1 | python3 -I -c "
import sys,csv
r=list(csv.reader(sys.stdin))[0]
n_gen=int(r[-7]); avg_ns=float(r[-4]); sd=float(r[-3]); ts=float(r[-2])
print('$model,$tur,$sira,$ad,%.4f,%.4f,%.2f' % (avg_ns/n_gen/1e6, sd/n_gen/1e6, ts))
" >> $OUT
    done
  done
done
echo BITTI >> $OUT
python3 -I - <<PY
import csv, collections
d = collections.defaultdict(list)
for r in csv.DictReader(open("$OUT")):
    if r["model"] in ("05", "7"):
        d[(r["model"], r["yapilandirma"])].append(float(r["ms_token"]))
for m in ("05", "7"):
    a = d[(m, "onceki")]; b = d[(m, "yeni")]
    if a and b:
        n = sum(1 for x, y in zip(a, b) if y < x)
        print("model %s: onceki %.4f ms, yeni %.4f ms, fark %+.4f ms (%%%+.1f), yeni daha hizli: %d/%d tur" % (m, sum(a)/len(a), sum(b)/len(b), sum(b)/len(b)-sum(a)/len(a), (sum(b)/sum(a)-1)*100*len(a)/len(b), n, len(a)))
PY
echo ADIM_BITTI
