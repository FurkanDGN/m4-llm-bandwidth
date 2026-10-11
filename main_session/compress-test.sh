#!/bin/zsh
# q4/q8 GGUF'u zstd ile sıkıştırıp oran ve açma hızını ölçer (lz4 kısmı lz4-test.sh'de düzeltildi)
set -e
OUT=${1:-/tmp/compress-out}
mkdir -p $OUT
for f in ~/llm-bench/models/*.gguf; do
  name=$(basename $f)
  orig=$(stat -f%z $f)
  echo "=== $name  orig=$orig bytes ==="
  for codec in "zstd -3" "zstd -19"; do
    tool=${codec%% *}; lvl=${codec##* }
    cf=$OUT/$name.$tool$lvl
    t0=$(python3 -c 'import time;print(time.time())')
    $tool $lvl -f -q -o $cf $f
    t1=$(python3 -c 'import time;print(time.time())')
    comp=$(stat -f%z $cf)
    best=999999
    for i in 1 2 3; do
      d0=$(python3 -c 'import time;print(time.time())')
      $tool -d -q -c $cf > /dev/null
      d1=$(python3 -c 'import time;print(time.time())')
      best=$(python3 -c "print(min($best,$d1-$d0))")
    done
    python3 -c "
orig=$orig; comp=$comp; ct=$t1-$t0; dt=$best
print(f'  {\"$tool $lvl\":10s} oran={orig/comp:5.3f}x  kazanc={100*(1-comp/orig):4.1f}%  sikistirma={orig/1e9/ct:5.2f} GB/s  acma={orig/1e9/dt:6.2f} GB/s')
"
  done
done
rm -rf $OUT
