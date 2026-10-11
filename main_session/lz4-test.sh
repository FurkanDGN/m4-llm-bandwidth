#!/bin/zsh
# lz4 sıkıştırma/açma ölçümü (lz4 -o bayrağı kabul etmez; konumsal arg kullanılır)
set -e
OUT=${1:-/tmp/lz4-out}
mkdir -p $OUT
for f in ~/llm-bench/models/*.gguf; do
  name=$(basename $f)
  orig=$(stat -f%z $f)
  echo "=== $name  orig=$orig bytes ==="
  for lvl in 1 9; do
    cf=$OUT/$name.lz4$lvl
    t0=$(python3 -c 'import time;print(time.time())')
    lz4 -$lvl -f -q $f $cf
    t1=$(python3 -c 'import time;print(time.time())')
    comp=$(stat -f%z $cf)
    best=999999
    for i in 1 2 3; do
      d0=$(python3 -c 'import time;print(time.time())')
      lz4 -d -q -c $cf > /dev/null
      d1=$(python3 -c 'import time;print(time.time())')
      best=$(python3 -c "print(min($best,$d1-$d0))")
    done
    python3 -c "
orig=$orig; comp=$comp; ct=$t1-$t0; dt=$best
print(f'  lz4 -{$lvl}    oran={orig/comp:5.3f}x  kazanc={100*(1-comp/orig):4.1f}%  sikistirma={orig/1e9/ct:5.2f} GB/s  acma={orig/1e9/dt:6.2f} GB/s')
"
  done
done
rm -rf $OUT
