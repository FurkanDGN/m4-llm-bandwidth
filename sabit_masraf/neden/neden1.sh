#!/bin/bash
# Tahminli uretme: her grafin GPU cagrisi ve bellek bariyeri sayisi, yapilandirma basina
B=$HOME/claude-scratch/llamacpp-ablasyon-sb/build/bin
D=$HOME/claude-scratch/bant-darbogaz
T=$HOME/llm-bench/models/qwen2.5-7b-instruct-oq4.gguf
DR=$HOME/llm-bench/models/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."
KAP="GGML_METAL_FUSE_MM_ADD=0 GGML_METAL_FUSE_MM_GLU=0 GGML_METAL_FUSE_MM_QKV=0 GGML_METAL_FUSE_ROPE_K=0 GGML_METAL_FUSE_ROPE_QK=0 GGML_METAL_FUSE_SET_V=0"
mkdir -p $D/neden
for cfg in orijinal tumu sadece_hedef sadece_taslak; do
  case $cfg in
    orijinal) E="$KAP";;
    tumu) E="GGML_METAL_X=0";;
    sadece_hedef) E="GGML_METAL_FUSE_MIN_NODES=900";;
    sadece_taslak) E="GGML_METAL_FUSE_MAX_NODES=900";;
  esac
  env $E GGML_METAL_FUSE_STATS=1 GGML_METAL_SB=1 timeout 200 $B/llama-speculative -m $T -md $DR -ngl 99 -ngld 99 --temp 0 -n 64 --spec-draft-n-max 2 --spec-draft-p-min 0 -p "$ESSAY" </dev/null > $D/neden/$cfg.txt 2> $D/neden/$cfg.err
done
python3 -I - <<'PY'
import re, collections, statistics as st, os
D = os.path.expanduser("~/claude-scratch/bant-darbogaz/neden/")
for cfg in ("orijinal", "tumu", "sadece_hedef", "sadece_taslak"):
    g = collections.defaultdict(list)
    for s in open(D + cfg + ".err", errors="ignore"):
        m = re.search(r"graf dugum=(\d+) sutun=(\d+) gpu_cagrisi=(\d+) bariyer=(\d+)", s)
        if m:
            n, c, d, b = map(int, m.groups())
            g[(n, c)].append((d, b))
    print("==", cfg)
    for (n, c), v in sorted(g.items()):
        ds = sorted(x[0] for x in v); bs = sorted(x[1] for x in v)
        print(f"  dugum={n} sutun={c} adet={len(v)}  cagri medyan={st.median(ds):.0f} (min {ds[0]}, max {ds[-1]})  bariyer medyan={st.median(bs):.0f} (min {bs[0]}, max {bs[-1]})")
PY
echo NEDEN1_BITTI
