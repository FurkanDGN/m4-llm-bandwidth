#!/usr/bin/env python3
"""Byte-plane entropi/sıkıştırma ölçümü: GPU-içi kayıpsız açmanın tavan kazancı.

Dosyanın ortasından büyük bir dilim alır (tensör verisi), element boyutuna göre
byte-plane'lere ayırır, her plane'i ayrı zstd'ler ve düz zstd ile karşılaştırır.
Kullanım: byteplane.py <gguf> <elem>  (fp16=2, q4_0 blok=18, q8_0 blok=34, düz bakış=1)
"""
import sys, math, collections
import zstandard as zstd

path, elem = sys.argv[1], int(sys.argv[2])
N = 512 * 1024 * 1024  # 512 MB dilim
with open(path, 'rb') as f:
    f.seek(0, 2); size = f.tell()
    f.seek(size // 2 - N // 2)
    data = f.read(N)

def H(b):
    c = collections.Counter(b); n = len(b)
    return -sum(v/n * math.log2(v/n) for v in c.values())

cctx = zstd.ZstdCompressor(level=3)
plain = len(cctx.compress(data))
print(f"{path.split('/')[-1]}  dilim={N/1e6:.0f}MB")
print(f"  düz zstd-3        : oran={N/plain:.3f}x  kazanç={100*(1-plain/N):.1f}%")

if elem > 1:
    planes = [data[i::elem] for i in range(elem)]
    tot = 0
    for i, p in enumerate(planes):
        c = len(cctx.compress(bytes(p)))
        tot += c
        print(f"  plane {i} (byte {i}) : entropi={H(p[:4*1024*1024]):.3f} bit  oran={len(p)/c:.3f}x")
    print(f"  byte-plane toplam : oran={N/tot:.3f}x  kazanç={100*(1-tot/N):.1f}%")
else:
    print(f"  entropi           : {H(data[:4*1024*1024]):.3f} bit/byte (tavan oran={8/H(data[:4*1024*1024]):.3f}x)")
