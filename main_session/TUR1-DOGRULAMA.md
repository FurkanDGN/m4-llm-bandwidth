# Tur 1 doğrulama koşuları — bulut denetimi sonrası yerel deneyler

KAYNAK: ana oturum (koordinatör), 11 Eki 2026, 14:29-14:47. Donanım: M4 Pro 24 GB.
İkili: yerel derleme, llama.cpp commit e475fa2b5 + `llama_patch/sb_q4_0.diff` (yamalı/yamasız aynı ikili, `GGML_METAL_SB=0/1` ile).
Bu koşular, ajan kaynaklı tek-koşu sonuçlarının İLK bağımsız tekrarıdır (farklı deneyci, yeniden derlenmiş ikili).
llama-bench koşuları mmap kapalı (`-mmp 0`), tekrar sayısı 5; lookup/spec koşuları 3'er tekrar.
Bulut denetimi: `BULUT-ANALIZ.md` (repo dalı `claude/new-session-wu71za`). Deney adları (E2, E3, E8) oradandır.

## 1. Yamanın doğruluğu (E8) — KAPANDI

test-backend-ops MUL_MAT, SB=1: 3/3 backend OK, sıfır FAIL.

Perplexity, 7B q4_0, wikitext 12 chunk, c=512, parti büyüklüğü = ub:

| ub | SB=0 (yamasız yol) | SB=1 (yamalı) | geçiş süresi sb0→sb1 |
|---|---|---|---|
| 2 | 7.8342 | 7.8345 | 9.07 → 5.02 sn (1.81x) |
| 3 | 7.8340 | 7.8345 | 8.88 → 4.64 sn (1.91x) |
| 4 | 7.8340 | 7.8339 | 9.22 → 4.70 sn (1.96x) |
| 8 | 7.8340 | 7.8339 | 9.28 → 2.71 sn (3.42x) |

Maks |ΔPPL| = 0.0005 (eşik 0.01). Greedy (temp 0) çıktı metinleri k=0/2/5'te SB=0 ve SB=1 arasında bayt bayt AYNI (skaler yol da MMA half-x yolu da). Yama çıktıyı değiştirmiyor.

## 2. Sabit masraf modeli (E2) — her iki eski uydurma da düzeltildi

Önemli ön bulgu (gguf_dump, tensör tipleri): bu GGUF'larda embedding bağlı DEĞİL —
`output.weight` ayrı tensör (1.5B q4_0: Q6_K; 1.5B q8/fp16: kendi tipi; 0.5B: Q8_0; 7B: Q6_K).
Decode'da okunan bayt = dosya − token_embd. Eski uydurma (1.54 ms + 263 GB/s) token_embd'yi sayıyordu (imkânsız eğim);
bulutun düzeltmesi (1.14 ms + 239 GB/s) küçük modellerde tied varsayımı yapıyordu (yanlış).

tg128, r=5, mmap kapalı:

| model | tok/s | okunan GB |
|---|---|---|
| 0.5B q4_0 | 330.0 ± 3.5 | 0.352 |
| 1.5B q4_0 | 183.3 ± 0.8 | 0.935 |
| 1.5B q8_0 | 122.5 ± 0.7 | 1.647 |
| 1.5B fp16 | 72.9 ± 0.5 | 3.094 |
| 7B q4_0 | 54.8 ± 0.1 | 4.125 |

Uydurma (5 nokta): **t = 1.63 ms + 3.99 ms/GB (250.8 GB/s)**, R² = 0.9993, tüm sapmalar ≤ %1.9.
Eğim, ölçülen okuma tavanının (254.3 GB/s) %98.6'sı — fiziksel olarak tutarlı. "1.5B q8 imkânsız 266 GB/s"
tutarsızlığı da çözüldü: doğru bayt sayımıyla 251 GB/s.

## 3. lm_head q6_K → q4_0 (E3) — bulut hipotezi D13/O6 DOĞRULANDI

`llama-quantize --allow-requantize --output-tensor-type q4_0` (3.7 sn; dosya 140.5 MB küçülür).

| ölçüm | orijinal (q6_K) | lm_head q4_0 | fark |
|---|---|---|---|
| tg128 | 54.61 ± 0.32 | 56.08 ± 0.12 | +%2.7 (beklenen bant 55.8-56.5 ✓) |
| spekülatif k=1 (essay, 3 tekrar) | 62.2-62.5 | 67.3-68.0 | **+%8.6** |
| spekülatif k=2 (essay, 3 tekrar) | 59.3-59.5 | 65.4-65.5 | **+%10.3** |
| PPL (32 chunk) | 6.752 ± 0.196 | 6.545 ± 0.190 | fark gürültü içinde; ceza yok |

Mekanizma: q6_K ext kernel listesinde B=2-3 için yok → lm_head (0.447 GB) doğrulama partisinde 2-3 kez okunuyordu.
Yamalı + lm_head q4_0 ile 7B serbest-metin spekülatif artık 67-68 tok/s (yamasız taban 54.8'e göre 1.23x).

## 4. Tek-koşu tablolarının bağımsız tekrarı — 5/6 hücre tuttu, 1 hücre düzeltildi

llama-lookup 7B, temp 0, 3'er tekrar (ajanın tek koşusu vs bu koşular):

| hücre | ajan (tek koşu) | bu tur (3 tekrar) | durum |
|---|---|---|---|
| kod k=0 SB0 | 50.8 | 52.1-52.6 | ✓ |
| kod k=0 SB1 | 53.5 | 53.2-53.3 | ✓ |
| kod k=7 SB0 | 44.9 | 44.9-45.2 | ✓ birebir |
| kod k=7 SB1 (manşet) | 127.7 | **126.3-126.5** | ✓ (−%1) |
| esse k=7 SB1 | 48.7 | 50.8-51.4 | ✓ (+%5, sınırda) |
| esse k=7 SB0 | 34.1 | **40.2-40.3** | ✗ %18 fark — tek koşu artefaktı |

Düzeltme: yamasız lookup'ın kopyasız metin cezası k=7'de 0.64x değil **0.75x**.
Olası sebep: orijinal oturumda bellek baskısı/mmap tahliyesi (bulut D4). Yamalı ceza: 51.4/53.3 = **0.96x** (ajan 0.92 demişti; yön aynı).
Greedy (essay, n=300) ek seri: k=0 53.2/53.3 (SB0/SB1 başa baş), k=2 SB0 48.0 → SB1 53.2, k=5 SB0 43.1 → SB1 51.4.
Manşet bulgu (kod işinde yamayla 2.4x, k=7'de yamasıza göre 2.8x) bağımsız tekrarla DOĞRULANDI.

## 5. Kalan işler (bulut listesinden koşulmadı)

E1 (xctrace per-kernel süreler), E4/E5 (skaler kernel düzeni ve x trafiği, mikro), E6 (mul_mv_ext ayar taraması),
E7 (split-K), E10 (tavan varyantları), E11 (K-quant/IQ bant verimi — en büyük açık kaldıraç), E12-E18.
