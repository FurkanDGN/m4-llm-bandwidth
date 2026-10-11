# Ana oturum ölçümleri (zip / entropi / CPU açma)

Kaynak: ana oturum (Fable 5), 11 Eki 2026. Tek koşu (açma süreleri 3 tekrarın en iyisi), bağımsız tekrar: YOK. Bu dizindeki scriptler birebir o oturumda kullanılanlardır; çıktılar oturum kayıtlarından aktarıldı.

## 1. GGUF dosyalarını kayıpsız sıkıştırma (compress-test.sh, lz4-test.sh, zstd-q8.sh)

qwen2.5-1.5b-instruct q4_0 (1 066 227 232 B) ve q8_0 (1 894 532 128 B).

| model | codec | oran | kazanç | sıkıştırma | açma |
|---|---|---|---|---|---|
| q4_0 | zstd -3 | 1.050x | %4.8 | 1.64 GB/s | 1.61 GB/s |
| q4_0 | zstd -19 | 1.052x | %4.9 | 0.02 GB/s | 1.29 GB/s |
| q4_0 | lz4 -1 | 1.007x | %0.7 | 3.23 GB/s | 5.05 GB/s |
| q4_0 | lz4 -9 | 1.008x | %0.8 | 0.54 GB/s | 4.90 GB/s |
| q8_0 | zstd -3 | 1.046x | %4.4 | 4.22 GB/s (T0) | 1.34 GB/s |
| q8_0 | zstd -19 | 1.047x | %4.5 | 0.06 GB/s | 1.30 GB/s |
| q8_0 | lz4 -1 | 1.005x | %0.5 | 2.03 GB/s | 5.40 GB/s |
| q8_0 | lz4 -9 | 1.006x | %0.6 | 0.59 GB/s | 5.15 GB/s |

## 2. Byte-plane entropi ve sıkıştırma (byteplane.py)

Dosya ortasından 537 MB dilim, zstd-3, entropi ilk 4 MB üzerinden.

| model | düz zstd | byte-plane toplam | not |
|---|---|---|---|
| q4_0 (elem=1) | 1.055x (%5.2) | — | entropi 7.554 bit/B → tavan 1.059x |
| q4_0 (elem=18, blok yapısı) | 1.055x | 1.091x (%8.4) | plane 16-17 (ölçek) 5.85/4.87 bit |
| q8_0 (elem=1) | 1.040x (%3.9) | — | entropi 7.662 bit/B → tavan 1.044x |
| q8_0 (elem=34) | 1.040x | 1.057x (%5.4) | plane 21/33 (ölçek) 6.94/6.88 bit |
| fp16 (elem=2) | 1.092x (%8.4) | 1.188x (%15.8) | plane 1 (üst byte) 5.520 bit → 1.432x |

## 3. CPU paralel açma (cpu-paralel.sh)

q4_0 dosyası 12 parça, tüm parçalar aynı anda açılıyor, 3 tekrarın en iyisi:

- zstd: 10.0 GB/s
- lz4: 13.9 GB/s

Kıyas: q4_0 decode'un ağırlık okuma hızı ~191 GB/s (llama-bench tg128 türevi). CPU açma 14-19 kat yetersiz.

## 4. Bağlamdaki diğer ana oturum sayıları

- llama-bench (donanım sahibinin ilk koşusu, [F]): 1.5B q4_0 prefill 1990 tok/s, decode 180.2 tok/s; q8_0 decode 121.1; batch 1→16 toplam 327→865 tok/s (B=2-8 bölgesi 1.1-1.25x).
- Güç (powermetrics, [F]): decode 17.3 W, prefill 25.3 W, ikisi de %100 "doluluk".
- PPL (wikitext-2, 64 chunk, [M]): 1.5B q8 9.82 / q4 10.29 / q3_K_M 11.02 / q2_K 16.26; OLMoE q4 11.91 (farklı tokenizer, Qwen ile kıyaslanmaz); 7B q4 7.77.
- Decode tg128 ([M]): 0.5B 316, 1.5B q4 179.5-180, q3_K_M 117, q2_K 139, 7B 53.7-54.5, OLMoE-1B-7B 229 tok/s.
