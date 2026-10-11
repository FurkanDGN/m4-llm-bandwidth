#!/bin/zsh
# llama.cpp yamali yapi ile yapilan olcumlerin komutlari. Calistirmak icin degil, KNOWHOW.md'deki komutlarin kaydi.
# Bu dosya ekrandan yeniden yazildi (orijinal komutlar satir ici calistirilmisti). Bellek disiplini: ayni anda TEK agir is, derleme -j4.
# Dizin: <klasor>/llama/{src,build,bin_base,res,spec,lk}; modeller ~/llm-bench/models
set -u
M=~/llm-bench/models
M7=$M/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
D=$M/qwen2.5-0.5b-instruct-q4_0.gguf
ESSAY="Write a detailed, factual essay about the history of computing, from mechanical calculators to modern GPUs."

# --- kurulum ---
# python3 -m venv venv && venv/bin/pip install cmake
# git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git src && (cd src && git checkout e475fa2b5)
# cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF
# cmake --build build -j4 --target llama-bench llama-speculative llama-lookup llama-completion test-backend-ops
# cp -R build/bin bin_base   # yamadan ONCE
# python3 -I llama_patch/apply_patch.py src ; cmake --build build -j4 --target ...   # yama sonrasi

# --- esik taramasi (yamasiz yol; GGML_METAL_MM_MIN=T) ---
thr() { for T in 8 5 4 3 2 1; do GGML_METAL_SB=0 GGML_METAL_MM_MIN=$T build/bin/llama-bench -m $M7 -ngl 99 -p 1,2,3,4,5,6,7,8,9,12,16 -n 0 -r 3 -o csv </dev/null 2>/dev/null > res/thr_T$T.csv; done; }

# --- yamali modlar (GGML_METAL_SB=0|1|s|m, GGML_METAL_SB_TM=1|2, GGML_METAL_SB_R) ---
run() { local tag=$1; shift; env "$@" build/bin/llama-bench -m $M7 -ngl 99 -p 1,2,3,4,5,6,7,8,9,12,16 -n 0 -r 3 -o csv </dev/null 2>/dev/null > res/pp_$tag.csv; }
modes() {
  run base GGML_METAL_SB=0
  run sb1_tm2 GGML_METAL_SB=1 GGML_METAL_SB_TM=2
  run sb1_tm1 GGML_METAL_SB=1 GGML_METAL_SB_TM=1
  run sbs GGML_METAL_SB=s
  run sbm_tm2 GGML_METAL_SB=m GGML_METAL_SB_TM=2
  run sbm_tm1 GGML_METAL_SB=m GGML_METAL_SB_TM=1
  run sbs_T4 GGML_METAL_SB=s GGML_METAL_MM_MIN=4
}

# --- dogruluk (ggml'nin kendi testi; CPU referansi) ---
correctness() { build/bin/test-backend-ops test -o MUL_MAT -b MTL0 -p "type_a=q4_0"; }

# --- kernel bazli per-op (tek matris m=4096 k=14336) ---
perop() { for R in 1 2 3 4 6 8; do GGML_METAL_SB=s GGML_METAL_SB_R=$R build/bin/test-backend-ops perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32,m=4096,n=[1-3],k=14336" > res/tbo_perf_r$R.txt 2>&1; done
         GGML_METAL_SB=0 build/bin/test-backend-ops perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32,m=4096,n=[1-3],k=14336" > res/tbo_perf_r0.txt 2>&1; }

# --- spekulatif (yamasiz sb=0, yamali sb=1) ---
spec() { for rep in 1 2 3; do for sb in 0 1; do for k in 1 2 3; do
  GGML_METAL_SB=$sb timeout 200 build/bin/llama-speculative -m $M7 -md $D -ngl 99 -ngld 99 --temp 0 -n 256 --spec-draft-n-max $k -p "$ESSAY" </dev/null > spec/r${rep}_sb${sb}_k$k.full 2>&1
done; done; done
  for rep in 1 2 3; do for sb in 0 1; do GGML_METAL_SB=$sb timeout 200 build/bin/llama-lookup -m $M7 -ngl 99 --temp 0 -n 256 -c 2048 -p "$ESSAY" --spec-draft-n-max 0 </dev/null > spec/r${rep}_base_sb$sb.full 2>&1; done; done; }

# --- lookup (kopya-agir isler; istemler scripts/mk_prompts*.py ile uretilir) ---
lookup() { local G=./gen
  for task in copy prose; do for sb in 0 1; do for k in 0 1 2 3 4 7 8 15 31; do
    GGML_METAL_SB=$sb timeout 120 build/bin/llama-lookup -m $M7 -ngl 99 --temp 0 -n 500 -c 4096 -f $G/p_$task.txt --spec-draft-n-max $k </dev/null > lk/${task}_sb${sb}_k$k.full 2>&1
  done; done; done
  for task in code essay; do for sb in 0 1; do for k in 0 1 2 3 4 7 8 15 31; do
    GGML_METAL_SB=$sb timeout 120 build/bin/llama-lookup -m $M7 -ngl 99 --temp 0 -n 400 -c 4096 -f $G/p_$task.txt --spec-draft-n-max $k </dev/null > lk/${task}_sb${sb}_k$k.full 2>&1
  done; done; done; }

# --- tg128 regresyon ---
tg() { for mdl in qwen2.5-7b-instruct-q4_0-00001-of-00002 qwen2.5-1.5b-instruct-q4_0 qwen2.5-0.5b-instruct-q4_0; do
  bin_base/llama-bench -m $M/$mdl.gguf -ngl 99 -p 0 -n 128 -r 5 -o csv </dev/null 2>/dev/null | tail -1
  for sb in 0 1; do GGML_METAL_SB=$sb build/bin/llama-bench -m $M/$mdl.gguf -ngl 99 -p 0 -n 128 -r 5 -o csv </dev/null 2>/dev/null | tail -1; done
done; }

# --- xctrace denemesi (basarisiz; bkz. raw_results/07_xctrace_notes.txt) ---
# xcrun xctrace record --instrument 'Metal GPU Counters' --instrument 'Metal Application' --time-limit 40s --output t1.trace --launch -- build/bin/llama-bench -m $M/qwen2.5-1.5b-instruct-q4_0.gguf -ngl 99 -p 0 -n 32 -r 2
# xcrun xctrace export --input t1.trace --toc
