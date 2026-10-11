#!/bin/zsh
# CPU'nun tüm çekirdekleriyle paralel açma tavanı: dosyayı 12 parçaya böl,
# her parçayı ayrı sıkıştır, hepsini aynı anda aç, toplam GB/s hesapla.
set -e
O=${1:-/tmp/pchunks}; mkdir -p $O
F=~/llm-bench/models/qwen2.5-1.5b-instruct-q4_0.gguf
split -n 12 $F $O/ch_
for c in $O/ch_*; do zstd -3 -q -f $c -o $c.zst; lz4 -1 -q -f $c $c.lz4; done
rm $O/ch_??
orig=$(stat -f%z $F)
for tool in zstd lz4; do
  ext=$([ $tool = zstd ] && echo zst || echo lz4)
  best=999999
  for i in 1 2 3; do
    t0=$(python3 -c 'import time;print(time.time())')
    for c in $O/*.$ext; do $tool -d -q -c $c > /dev/null & done
    wait
    t1=$(python3 -c 'import time;print(time.time())')
    best=$(python3 -c "print(min($best,$t1-$t0))")
  done
  python3 -c "print(f'  $tool paralel 12 is : acma={$orig/1e9/$best:6.2f} GB/s')"
done
rm -rf $O
