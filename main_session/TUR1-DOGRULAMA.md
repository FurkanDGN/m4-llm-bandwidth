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

## 6. E11: K-quant/IQ formatlarının bant verimi (1.5B, tg128 r=5 mmap kapalı; PPL wikitext 32 chunk c=512)

Not: bartowski dosyalarında embedding bağlı (output.weight yok) → okunan ≈ dosya; q4_0'da okunan = dosya − embd.
Etkin GB/s = okunan / (t − 1.63 ms). Aynı protokolde ölçülen PPL'ler kendi aralarında karşılaştırılabilir.

| format | dosya MiB | tok/s | etkin GB/s | PPL |
|---|---|---|---|---|
| q4_0 (çapa) | 1011 | 185.6 ± 0.3 | 249 | 9.319 |
| IQ4_XS | 849 | 157.5 ± 1.0 | 189 | **9.077** |
| IQ3_M | 735 | 138.6 ± 0.1 | 138 | 9.764 |
| IQ3_XS | 692 | 141.4 ± 0.1 | 133 | 10.091 |
| Q3_K_S | 720 | 104.4 ± 0.4 | 95 | 10.828 |
| IQ2_M | 568 | 136.9 ± 0.3 | 105 | 13.132 |
| Q3_K_M | 876 | 119.5 ± 0.2 | ~120 | — |
| Q2_K | 712 | 143.2 ± 0.5 | ~127 | — |

Bulgular:
- Bulut A1 doğrulandı [Ö]: 4-bit altı tüm formatlar bandın %40-55'inde (95-138 GB/s); q4_0 tek başına %99'da.
- Sürpriz: IQ4_XS kalitede q4_0'ı GEÇİYOR (9.08 < 9.32) ve %16 küçük; hız cezası yalnız kernel verimi.
  Kernel tam banda çıkarılsa ≈193 tok/s olurdu (q4_0 185.6'dan hızlı): bu makinedeki en kârlı kernel işi.
- IQ ailesi (tablo-tabanlı, "pahalı" sanılan) Metal'de Q3_K'dan hızlı: dönüşüm maliyeti bit sayısının değil
  kernel yazımının fonksiyonu. Q3_K_S en kötüsü (95 GB/s).
- 3.3 bpw sınıfı tam bantta ≈221 tok/s verirdi (q4_0'dan +%19, PPL 10.1 bedeliyle).

## 7. E1 (kısmi): sabit masrafın bileşimi — encoder seviyesi iz analizi

Metal System Trace, 7B q4_0 tg64 r=2 (12 sn kayıt). Shader-profiler tablosu bu kayıt modunda boş geldi
(kernel adlı döküm için ayrı profilleme gerekir); analiz encoder seviyesinde (metal-gpu-intervals).

- Decode penceresi 2.42 s; GPU compute kanalı meşgul %96.9, boşluk %3.1 (medyan boşluk 95 µs; token başına ~3 encoder).
- Boşluk payı token başına ≈ 0.3-0.6 ms. Sabit masraf 1.63 ms'nin kalanı (~1.0-1.3 ms) kernel İÇİNDE:
  ağırlık taşımayan küçük kernel'ler (norm/rope/attention) ve matmul kuyrukları "meşgul" sayılıyor.
- Bulut E1 hipotezi ("boşluk + küçük kernel 0.8-1.5 ms") doğrulandı; çöküş eşiği (< 0.3 ms) tetiklenmedi.
- Sınır: kernel adıyla döküm yok; per-kernel ayrım için MTLCaptureManager (.gputrace) yolu (E18) gerekiyor.

## 8. E12: half2 FMA tepe — paketli çift hız YOK

32 bağımsız zincir, 2^20 thread: fp32 6.03-6.10, skaler half 6.24-6.30, half2 6.17-6.35 TFLOPS.
half2 iki işlem olarak yürüyor (M4 Pro'da paketli half matematiği yok). B=4-8 skaler yolda
FMA'yı half2 birikeçle yarılama fikri ölü; fp16 ALU avantajı %3-4'te kalıyor.

## 9. E6: mul_mv_ext ayar taraması — "ekle değil, ayarla" yolu KAPANDI

SB=0, q4_0 m=4096 k=14336, nsg{2,4,8} × nxpsg{4,8,16,32} (+ n=8 için r1=5):
- En iyi ayar her n'de nxpsg=4; kazanç varsayılana göre n=2 %4, n=3 %9, n=4 %13, n=5 %22, n=8 %13.
- nsg hiç fark etmiyor; nxpsg=32 felaket (2x yavaş); r1=5 geri tepiyor (1.9-2.2 ms).
- SB kernel'ine uzaklık ayarla kapanmıyor: n=2 282 vs 143 µs (2.0x), n=5 749 vs 328 (2.3x), n=8 1105 vs 346 (3.2x).
- Bulut E6 çöküş eşiği (en iyi n=2 ≥ 250 µs) tetiklendi → ext tasarımı yetmiyor; yeni kernel tasarım olarak gerekçeli [Ö].

## 10. E4/E5: skaler kernel'de sınırlayıcı x trafiği; yük düzeni mikroda önemli, uçtan uca değil

Mikro (2^29 ağırlık, 8192², ms; MODE 0=düz çapa, 4=ggml-18B yama deseni, 5=lane kendi 4 baytı, 6=x sabit, 7=x half):
- E5 ana bulgu: x SABİTLENİNCE B=2/3/4 hepsi ~1.15 ms'ye (B=1 düzeyi) çöküyor. Skaler kernel'de B ile büyüyen bedelin tamamı x yükleri [Ö]. x'i half okumak B=4'te %36, B=3'te %9 kazandırıyor (hassasiyet bedeli ayrı konu).
- E4: yama deseni (MODE 4) B=3 R=4'te 1.84 ms; düzeltilmiş eşleme (MODE 5) 1.40; düz çapa 1.26. Bulutun iki tahmini de (≥1.6, ≤1.4) tam isabet.
- UYGULAMA DENEMESİ (negatif sonuç): MODE 5 eşlemesi gerçek yamaya işlendi, doğruluk geçti, çıktı birebir aynı;
  ama uçtan uca spekülatif k=1 +%1.2 / k=2 −%1.2 — kazanç yok. Mikrodaki fark DRAM rejimine ve per-op şekline özgü
  (per-op 33 MB SLC içinde). Yama orijinal hâline geri alındı. Ders: mikro kazancı uçtan uca ölçmeden yamaya taşıma.

## 11. E15/E16/E13 + IQ kernel ameliyatı (serbest tur)

E15 — çoklu istek (llama-batched-bench, 7B, npp=128 ntg=64):
yamasız toplam decode DÜZ: 1 akış 54.4, 2/4/8 akış 51-52 tok/s (akış eklemek hiçbir şey kazandırmıyor);
yamalı: 2 akış 93.5, 4 akış 96.4, 8 akış 153.8 tok/s toplam (3.0x). Çok-ajanlı/çok-kullanıcılı yerel servis için yamanın en büyük getirisi bu.

E16 — KV derinliği (7B, -fa 1): d=0 54.3, d=4k 48.6 (−%10), d=16k 39.4 (−%27). SÜRPRİZ: q8_0 KV f16'dan YAVAŞ
(4k 42.6, 16k 33.4) — bayt yarılanıyor ama attention'daki açma maliyeti kazancı yutuyor. "Bakımsız kernel" temasının üçüncü örneği.

E13 mikro — saf açma ALU tavanları (32 ağırlık/iter, bellek okumasız, Gw/s):
q4_0 sihirli 1368; IQ4_NL sözlük (threadgroup) 902, (constant dizi) 622; 3-bit düzlem-sihirli 883; bit-düzlemi LUT3 835.
Bellek sınırları: IQ4_XS 478, 3.5bpw 581 Gw/s → tüm alt-4-bit açmaları bellek sınırının ÜSTÜNDE akabiliyor;
yavaşlık matematik değil kernel yapısı [Ö].

IQ kernel ameliyatı (uçtan uca hakemli):
- Hipotez 1 (aux32/q8 byte-takma-adı yavaşlatıyor): ÖLDÜ — doğrudan-shift eşdeğeri ±0 (156 vs 157.5).
- Hipotez 2 (N_R0 düşük, y yükleri az amortismanlı): TUTTU — üç tek satırlık sabit değişikliğiyle:
  N_R0_IQ4_XS 2→8: IQ4_XS 157.5 → 171.6 tok/s (+%9, etkin 189→212 GB/s), PPL bire bir aynı (9.0771).
  N_R0_Q2_K 4→8: Q2_K 143.2 → 156.7 (+%9.4). N_R0_Q3_K 2→8: Q3_K_S 104.4 → 112.3 (+%7.6), Q3_K_M nötr.
  Geri tepenler: IQ3 ailesi 8'de −%5 (4'te kaldı), Q2_K=16 −%2, IQ4_XS=16 −%10, NR0 tatlı noktası format başına farklı — mikro-mimari görünmez olduğundan ancak ölçümle bulunuyor.
- Değişiklik `llama_patch/nr0_tune.diff`te; test-backend-ops MUL_MAT tamamı OK.
- Yeni resim: IQ4_XS 171.6 vs q4_0 183.3 — %16 küçük, PPL daha iyi (9.08<9.32), hız farkı %6.4'e indi.
