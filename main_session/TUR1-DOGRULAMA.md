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

## 12. E14/E9/E17 + mikro-mimari kaynak notu

- E14 (spekülatif araç/p_min): `llama-speculative-simple` (`--spec-type draft-simple`) aynı işte 61.4 tok/s — eski araçtan
  (65.4) YAVAŞ; p_min>0 ise bu commit'te (e475fa2b5) `common_speculative_accept` assert'iyle ÇÖKÜYOR (çerçeve hatası).
  E14 kapandı: araç değişikliği kazandırmıyor. Not: yeni çerçevede ngram-* spec türleri var (server'da taslaksız lookup kapısı).
- E9 (taslak kapsamı, ne00%256→%32): 0.5B greedy çıktısı SB=0/1 birebir aynı (değişiklik doğru) ama spec k=2 hızı değişmedi
  (65.2-65.6 ≈ 65.4) — bulut kriteri tetiklendi: taslak partileri zaten ucuz. Sıfır kazanç için yüzey büyütmemek adına geri alındı.
- E17 (MLX kıyası): mlx-community/Qwen2.5-0.5B-Instruct-4bit üretim 431.7 tok/s vs llama.cpp q4_0 330 (+%31 MLX).
  Eşik (≥420) aşıldı [Ö]: sabit masraf (1.63 ms) donanım tabanı değil; MLX'te ~0.9 ms'ye düşüyor. Küçük model/taslak
  hızının gerçek sınırı çerçeve; llama.cpp küçük modelde token başına ~0.7 ms masayı bırakıyor. (Quant/tokenizer farkı
  not edildi; %31 bu farklardan büyük.)
- Mikro-mimari kütüphanesi: `~/Documents/deneyler/apple-silicon-docs/` (CPU: applecpu, M1 Explainer 8 cilt, SME, Oakley;
  GPU: gpu-applegpu, gpu-metal-benchmarks). metal-benchmarks README'si NR0 bulgusunun geriye dönük açıklamasını veriyor:
  yazmaç fazla ayrılınca occupancy düşer (çekirdek başına ≤24 simd), ALU kullanımı komut karışımından tahmin edilebilir —
  "NR0=16 neden kötü" artık belgeyle açıklanabilir durumda.

## 13. E10: gerçek okuma tavanı ve ısınma etkisi — D1 anomalisi çözüldü

Bağımsız bant ölçeri (micro/band.metal: r4=eski 4-akış deseni, r8=8 uçuşta, rc=threadgroup'a bitişik 64 KB):

| desen | 302 MB soğuk | 2 GB | 302 MB + 500 ms ısınma |
|---|---|---|---|
| r4 (eski stream_read) | 194 / 215 | 248 / 254 | 258 / 260 |
| r8 | 255 / 259 | 246 / 250 | 259 / 260 |
| rc | 260 / 264 | 256 / 259 | 260 / 263 |

(medyan / en iyi, GB/s)

- GERÇEK TAVAN ≈ 258-264 GB/s (teorik 273'ün %95-97'si). Eski referans 254.3 düşük ölçümmüş (desen + soğukluk).
- D1 anomalisi kapandı: bench15 B=1'in 258 GB/s'si tavan ÜSTÜ değil, tam tavan. q4_0 B=1 kernel'i gerçekten optimal.
- E2 uydurmasının eğimi (250.8) gerçek tavanın %96.5'i.
- Isınmasız kısa ölçüm −%25'e kadar sapıyor (DVFS): bundan sonra her mutlak GB/s ölçümünde ≥500 ms ön ısıtma zorunlu;
  D2'nin ±%3-6 "program farkı" büyük ölçüde buydu.

## 14. NR0 kalıbının sınırı: tavandaki formatlarda etkisiz

Q6_K ve Q8_0'da NR0 2→4: Q8_0 121.5 (taban 122.5), 7B q4_0 53.5 (taban 54.8) — ikisi de gürültü payında, kazanç yok.
Tutarlı açıklama: NR0/occupancy kolu yalnız bellek tavanından uzak kernel'lerde (IQ4_XS, Q2_K, Q3_K; %40-75 bölgesi)
çalışıyor; zaten %96-99'da koşan q4_0/q8_0'da kazanılacak yer yok. İkisi de 2'ye geri alındı; nr0_tune.diff değişmedi.

## 15. E7 faz 1-4: B=2-8 MMA kernel'i gecikme-zinciri sınırlı (üç tahmin çöktü, teşhis netleşti)

Ölçüm düzeni: llama-batched-bench, 7B oq4 (lm_head q4_0), npl=B, grup maliyeti = B×1000/S_TG ms.
Taban hesap (roofline, okunan 3.98 GB): B≤6 için 17.5 ms (1.00x), B=8 için 20.2 ms (1.15x).

Faz 1 (mod kıyası, tek koşu):

| B | SB=1 (ms) | tek tokene oran | SB=m MMA (ms) | skaler (ms) |
|---|---|---|---|---|
| 2 | 18.9 | 1.07x | 34.7 | 18.9 |
| 3 | 24.2 | 1.37x | 35.2 | 24.2-28.0 |
| 4 | 34.7 | 1.96x | 35.7 | — |
| 8 | 36.3 | 2.05x | — | — |

- MMA kernel'i B=2'de bile ~35 ms: maliyet B'den bağımsız DÜZ → hesap sınırlı değil (hesap B ile büyürdü).
  Etkin okuma ~117 GB/s (tavanın %45'i).
- Skaler yol B=2'de tabanda (1.07x) — bulutun B=2 öngörüsü doğrulandı.

Denenen dört varyant (hepsi doğruluk kapısından geçti: test-backend-ops 3/3) ve sonuçları, aynı zincirde v1 önce:

| varyant | değişiklik | B=2-8 grup maliyeti | hüküm |
|---|---|---|---|
| v1 tm2 (mevcut) | — | 33.6-36.4 ms | baz |
| skaler n4-n8 | şablonu 4-8 sütuna genişlet | 40.7-297 ms | ÇÖP: x yükleri sütunla doğrusal + yazmaç baskısı (n8'de ~90 float/thread) |
| v2 (tg-stage) | x'i threadgroup belleğine ortak yükle (blok başına 2 bariyer) | 76.9-80.5 ms | ÇÖP: bariyer 4 simdgroup'u kilit adıma sokuyor |
| v3 (geniş qs) | dağınık 2B qs yükleri → tek uint4 | 70.4-74.1 ms | ÇÖP: geniş yük zincir gecikmesini uzatıyor |
| v4 (v3+shuffle x) | x'i simd_shuffle ile dağıt | 166-283 ms | ÇÖP: shuffle'lar zincire seri ALU gecikmesi ekliyor |
| v1/v2 tm4 | simdgroup başına 4 tile | 77-138 ms | ÇÖP: yazmaç → occupancy |

Teşhis (tüm veriyle tutarlı tek açıklama): kernel GECİKME-ZİNCİRİ sınırlı. Blok döngüsünün her adımı
yük→aç→MMA bağımlılık zinciri; maliyet B'den bağımsız, zinciri uzatan her değişiklik (geniş yük, shuffle,
bariyer, yazmaç baskısı) doğrudan zarar. Kaba hesap: ~24 yerleşik simdgroup × iterasyon başına 288 B /
~600 çevrim ≈ tam ölçülen banda denk — sınır tam bu dengede.

Tahmin karnesi (dürüst): v2 için "yük komutu azalır, hızlanır" ve v3 için "2B dağınık okumalar birleşir,
~20-24 ms'ye iner" tahminlerinin İKİSİ DE ters yönde çöktü. Önden doğru çıkanlar: B=2 tabanda, B=4-8'de
~1.8x açık var, suçlu ALU değil. Ders: gecikme-sınırlı rejimde "daha az/daha geniş yük" sezgisi ters çalışır;
doğru kaldıraç uçuştaki bağımsız yük sayısını artırmak.

Sıradaki aday (v5, henüz koşulmadı): blok döngüsünü 2'li aç, iki bloğun tüm yüklerini hesaptan önce birlikte
ihraç et (yazılım boru hattı). Beklenti kayıtta: zincir başına bayt 2x → 35 ms'nin 24-28 bölgesine inmesi.

## 16. E19: MLX kıyası — "hazır çözüm var mı" denetimi (tek koşu + kısmi tekrar)

mlx-lm 0.32, mlx-community Qwen2.5 4bit (7B + 0.5B taslak; 4.5 bpw, q4_0 ile aynı bayt sınıfı), aynı
istemler (mk_prompts.py, seed 7), greedy. Kıyas: yamalı llama.cpp + oq4.

| iş | MLX | yamalı llama.cpp | kazanan |
|---|---|---|---|
| 7B düz decode (esse) | 58.9 (2 koşu: 58.9/58.9) | 56.6 | MLX +%4 (çerçeve ek yükü farkı) |
| spec esse k=1 | 75.2 | 68.5 | MLX +%10 |
| spec esse k=2 | 71.5 (tekrar 71.2) | 65.4 | MLX |
| spec esse k=3 / k=5 | 59.2 / 42.6 | — | kimse (ikisi de çöküyor) |
| JSON kopya (spec k=2-8) | 85.8-90.4 | lookup k=7: 106.6 | llama.cpp lookup |
| kod düzenleme | (lookup yok) | lookup k=7: 127.7 | llama.cpp lookup |

- Spec VERİMİ iki çerçevede aynı (k=1 kazancı: MLX 1.28x, biz 1.25x — kendi tabanlarına göre). MLX'in
  mutlak üstünlüğü tabanından geliyor: token başına ~0.65 ms daha az çerçeve masrafı (E17 ile tutarlı).
- B=4-8 klifi MLX'te de ÇÖZÜLMEMİŞ: k≥3 kazandırmıyor. E7'nin alanı hâlâ kimsenin çözmediği yer.
- Pratik tavsiye güncellendi: serbest metin sohbet + spec için bugün en iyi araç MLX (75 tok/s);
  kod/JSON/belge düzenleme için yamalı llama.cpp lookup (106-128 tok/s) rakipsiz.
- Sınırlar: MLX çıktıları quant farkı yüzünden bayt-eş değil (araç-düzeyi uçtan uca kıyas); esse k=1
  MLX'te tek koşu.

## 17. E7 kapanışı: 35 ms duvarı gecikme değil, SABİT MMA SAYISI (teşhis düzeltildi, beş çürütme ile)

Faz 5 (v5 = yazılım boru hattı, blok döngüsü 2'li/4'lü açık, yükler hesaptan önce; aynı zincirde v1 önce):

| varyant | B=2 | B=4 | B=8 (ms) |
|---|---|---|---|
| v1 tm2 | 33.7 | 34.7 | 36.3 |
| v5 u2 | 34.0 | 34.5 | 36.0 |
| v5 u4 | 37.7 | 38.5 | 39.9 |
| v5 tm1 u4 | 33.7 | 34.0 | 34.9 |

v5 ETKİSİZ (±%1-4) — boru hattı tahmini de çöktü (24-28 beklenmişti). Bu, bölüm 15'teki "gecikme-zinciri"
teşhisini de ÇÜRÜTÜYOR: gecikme sınırı olsaydı uçuştaki yükleri ikiye katlamak kazandırırdı.

Tüm veriyle tutarlı kalan tek açıklama (bulutun D bulgusunun geç teyidi): kernel HESAP sınırlı, ama B'deki
düzlüğün sebebi şu — MMA yolu B kaç olursa olsun TAM 8 SÜTUN hesaplıyor; adım başına MMA sayısı sabit
(~110M MMA = 113 GFLOP, 35 ms'de ≈ 3.2 TFLOPS ≈ tepenin %52'si; kalan pay dequant + yük komutlarının aynı
boru hattını paylaşması). Beş çürütme bununla tutarlı: yük azaltma/genişletme (v3), shuffle (v4), boru hattı
(v5) MMA sayısını değiştirmiyor → etkisiz ya da zararlı; bariyer (v2) ve yazmaç baskısı (tm4, skaler n4-8)
ek fren → zararlı.

Sonuç ve açık uç:
- B=2-3: skaler yol tabanda/yakınında (18.9 / 24 ms) — çözülmüş.
- B=4-8: 35 ms duvarı YAPISAL — donanımda 4-sütun simdgroup matrisi yok, 8 sütunun yarısı B=4'te boşa.
  Skaler yol da x-yük deseni + yazmaç duvarı yüzünden oraya inemiyor (faz 2). Beş yeniden-yapılandırma
  denendi, beşi de tek ikilide env ile açılabilir durumda ağaçta duruyor (GGML_METAL_SB_V=2..5, SB_NSC).
- Varsayılan davranış DEĞİŞMEDİ (NSC=3, v1): E7 hız değil mekanizma getirdi. B=4-8'i tabana indirme işi
  "yarım kalmış kernel" değil "format/donanım kısıtı" dosyasına taşındı.
- Tahmin karnesi: 5 müdahale tahmini çöktü; önden doğru çıkanlar roofline (B=2 tabanda, B=8 tavan 1.15x)
  ve bulutun hesap-sınırlılık okuması. Ders: ölçmeden kernel teşhisi koymak iki kez üst üste yanılttı;
  "değişken başına tek varyant + aynı zincirde baz" düzeni olmasa bu beş yanlış yol fark edilmeden
  birikebilirdi.

## 18. E20: IQ4_XS kernel'inde iki nötr probe — NR0 kazanımından ötesi bu yapıda yok

Düzeltilmiş muhasebe: bartowski 1.5B IQ4_XS tied-embedding (okunan = dosya 0.911 GB) → 171.6 tok/s =
5.83 ms − 1.63 sabit = 4.20 ms → 217 GB/s = tavanın ~%87'si. Tavan senaryosu ~190 tok/s (+%10 pay var).

| probe | sonuç (tg128, 5 tekrar, q4_0 kontrollü) | hüküm |
|---|---|---|
| NSG 2→4 | 171.48 ± 0.27 (baz 171.6; kontrol 185.3/186.3) | etkisiz |
| q4 çift 4B yükü → tek uint2 | 172.06 ± 0.09 | gürültü; GERİ ALINDI (E4/E5 kuralı) |

Kernel'in yük deseni zaten düzgün (coalesced qs, satırlara amortize y, tg-tablo). Kalan ~%13'ün tek
değişkenli ucuz kaldıracı kalmadı; E7 dersiyle (üst üste çürüyen kernel tahminleri) buradaki kazı
durduruldu. IQ4_XS son durumu: NR0=8 ile 171.6 tok/s, 1.5B tavsiyesi değişmedi.

### E20 eki: NR0 yaması 7B IQ4_XS'te de ölçüldü — kazanç +%4.4 (formata özel, modele özel değil)

Soru: N_R0_IQ4_XS=8 yaması 1.5B'ye mi özgü? Bartowski Qwen2.5-7B-Instruct-IQ4_XS indirildi
(4.213 GB; token_embd IQ4_XS 0.290 GB, output Q6_K 0.447 GB, tied değil → okunan 3.923 GB;
taban 1.626 + 3.987×3.923 = 17.27 ms → 57.9 tok/s). Tek değişken A/B, her adımda derleme
damgası + test-backend-ops 3/3 + 7B q4_0 kontrolü (tg128, 5 tekrar):

| N_R0_IQ4_XS | 7B IQ4_XS tok/s | tabana oran | q4_0 kontrol |
|---|---|---|---|
| 8 (yama) | 52.09 ± 0.10 | %90.0 | 57.03 ± 0.08 |
| 4 (upstream) | 49.91 ± 0.03 | %86.2 | 56.93 ± 0.09 |
| 8 (geri alım doğrulama) | 51.94 ± 0.05 | %89.7 | 57.01 ± 0.04 |

- Yama 7B'de de çalışıyor ama kazanç +%4.4 (1.5B'de +%38 idi): 7B'de kernel NR0=4 ile zaten
  tabanın %86'sındaydı, amorti edilecek pay küçük. (İlk yazımda "kazanç model küçüldükçe
  büyüyor" demiştim; ek-2'deki Gemma ölçümü bunu ÇÜRÜTTÜ — belirleyici olan boyut değil,
  bazın tabana uzaklığı.)
- Pratik not: 7B'de IQ4_XS (52.1) q4_0'dan (57.0) YAVAŞ, dosya ise neredeyse aynı boyutta
  (3.92 vs 3.99 GiB; bartowski output'u Q6_K tuttuğu için bit tasarrufu erimiş). 7B'de hız
  için q4_0, kalite için IQ4_XS; 1.5B'deki tablonun aynısı.

### E20 eki-2: Aynı A/B Gemma-3'te — kazanç boyutla değil, bazın tabana uzaklığıyla ölçekleniyor

Bartowski Gemma-3 1B ve 4B IQ4_XS (ikisi de tied; 1B: toplam 0.708 GB, token_embd Q8_0 0.321 GB
→ taban 224.8 tok/s; 4B: toplam 2.257 GB, token_embd Q6_K 0.551 GB → taban 94.1 tok/s). Tied
olduğu için gömme tablosu her token okunuyor ve Q8_0/Q6_K kernel'inden geçiyor — bayt payı
%45/%24, yani IQ4_XS yamasının etki alanı dışında (seyreltme). Aynı zincir düzeni, kontrol
Qwen 1.5B q4_0 (185.1-185.4 üç koşuda sabit):

| N_R0_IQ4_XS | Gemma 1B tok/s | Gemma 4B tok/s |
|---|---|---|
| 8 (yama) | 182.57 ± 1.13 | 78.92 ± 0.05 |
| 4 (upstream) | 186.95 ± 0.26 | 77.66 ± 0.05 |
| 8 (geri alım doğrulama) | 183.53 ± 0.37 | 78.92 ± 0.07 |

- Gemma 4B: yama +%1.6. Gemma 1B: yama −%2.3 (İLK NEGATİF ÖRNEK — gürültü dışı, iki koşuda
  tekrarlanabilir). Dört modelin tablosu: Qwen 1.5B +%38, Qwen 7B +%4.4, Gemma 4B +%1.6,
  Gemma 1B −%2.3.
- Desen boyut değil: NR0=4 bazı tabandan ne kadar uzaksa yama o kadar kazandırıyor (Qwen 1.5B
  bazı %63'teydi → büyük kazanç; Gemma'lar %82-83'te → pay yok). Gemma 1B'deki küçük zararın
  muhtemel mekanizması satır sayısı azken NR0=8'in threadgroup sayısını yarılaması (occupancy);
  ölçülmedi, hipotez olarak not.
- Varsayılan karar: N_R0_IQ4_XS=8 KALIYOR (+%38'lik kazanım −%2.3'lük zarara ağır basar);
  Gemma-1B sınıfı model ana yük olursa 4'e döndürülebilir. 3.5 bpw düzlem formatı mat-vec'te TAVANDA — hızlı 3-bit'in yapıcı kanıtı

Mikro: micro/mv3p.metal + mv3p.swift (16384×16384, SoA: 2-bit düzlem + 1-bit düzlem + blok-32 fp16 ölçek
= 3.5 bpw; E13'ün T3 sihirli açması; CPU çift-duyarlık doğrulaması 2 satır, göreli fark ≤1e-6; 600 ms
ısınma, 7 tekrar medyanı):

| kernel | bpw | medyan | etkin bant | ağırlık hızı |
|---|---|---|---|---|
| mv3p (düzlem 3-bit) | 3.5 | 0.471 ms | 249.6 GB/s (tavanın %96'sı) | 570.5 Gw/s |
| mv4m (q4_0 eşdeğeri referans) | 4.5 | 0.586 ms | 257.7 GB/s | 458.1 Gw/s |

- E13'ün öngörüsü (açma ALU'su 883 Gw/s > 3.5bpw bellek sınırı 571) uçtan uca mikroda DOĞRULANDI —
  günün kernel tahminleri içinde ilk tutan. Alt-4-bit yavaşlığının kernel-yapısal olduğu artık yapıcı
  kanıtla kapalı: tavanda akan bir 3.5 bpw düzeni mevcut ve pakette.
- Projeksiyon (sabit masraf modeliyle): 1.5B @3.5bpw ≈ 216 tok/s (IQ4_XS 171.6), 7B ≈ 71.5 tok/s düz
  decode (q4_0 56.6). KALİTE AÇIK SORU: bu basit (doğrusal, blok-32) 3-bit şema; IQ3 sınıfının kalite
  avantajı kitaplık/süperblok yapısından geliyor. Gerçek kazanım ancak kalite ölçümüyle hak edilir.
- Pratik yollar: (a) llama.cpp'ye yeni GGUF tipi (büyük iş; upstream yolu kapalı olduğundan yerel kalır),
  (b) MLX'in hazır 3-bit affine formatı (E21 ekinde hız ölçümü).

### E21 eki: MLX 3-bit — hız ikinci kanıt, kalite çöküşü

mlx_lm convert ile Qwen2.5-1.5B → 3-bit grup-64 (3.501 bpw ölçüldü):

- Hız: 242.3 tok/s (4.13 ms/token ≈ 0.72 GB / ~250 GB/s + ~0.9 ms çerçeve) — MLX'in hazır 3-bit kernel'i
  de bant hızında. Kıyas: MLX 4-bit 216.9, llama.cpp q4_0 185.7, IQ4_XS 171.6, Q3_K_S 112.
- Kalite: TEK ÖRNEKTE ÇÖKÜŞ — esse istemine 25 token'lık alakasız ret cevabı verip durdu (greedy;
  PPL ölçülmedi). Basit doğrusal 3.5 bpw şemanın q2_K emsali gibi modeli bozduğu yönünde güçlü işaret.
- E21 hükmü: hız sorusu YAPICI KAPALI (iki bağımsız kanıt: bizim mikro 249.6 GB/s + MLX uçtan uca 242 tok/s);
  açık problem artık yalnız KALİTE/bit — IQ3 sınıfı kalite (kitaplık/süperblok) + düzlem-sınıfı yerleşimi
  birleştiren format tasarımı. E13'ün LUT3 ölçümü (835 Gw/s > 571 sınır) bunun ALU tarafında da mümkün
  olduğunu söylüyor. Ayrı araştırma kalemi olarak dosyalandı.

## 19. E22: GPU sayaçları (Xcode Metal Debugger) — "sabit 8 sütun işi" doğrulandı, "hesap tavanına dayanıyor" doğrulanmadı

xctrace yolu (bölüm 7, `raw_results/07_xctrace_notes.txt`) sayaç vermemişti. Xcode 26.6 Metal Debugger yolu çalıştı.

Yöntem:
- Tek işlemlik GPU yakalaması: `METAL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=5 GGML_METAL_CAPTURE_DIR=<dizin> GGML_METAL_SB=1 test-backend-ops perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32,m=M,n=N,k=K,"`
- `llama_patch/e22_capture.diff`: yakalama dizinini ortam değişkeninden alır (varsayılan /tmp) ve test-backend-ops'a Qwen2.5-7B boyutlarını (18944x3584, 3584x18944) ekler.
- Sayaç: Xcode'da .gputrace aç, Performance, Counters, Encoders, CSV dışa aktar (elle). n8'de profil ayarı varsayılan (Medium, Overlapping); n4'te ayar kaydı yok.
- Ham veri: `raw_results/gpu_sayac/`. Yakalama dosyaları (~33 MB her biri) repoda yok.
- Her komut kuyruğunun maliyeti tamamen `kernel_mul_mm_sb_q4_0_f32_tm2`; çağrı başına `dispatchThreadgroups {64,1,1}` (4096 satır, 64 satır/grup). Sayılar başka işle karışmıyor.

İşlem başına süre (us, test-backend-ops perf, q4_0):

| boyut (m x k) | n=1 | n=2 | n=3 | n=4 | n=8 |
|---|---|---|---|---|---|
| 4096 x 14336 | 141 | 152 | 240 | 397 | 388 |
| 18944 x 3584 | - | - | - | 301 | 309 |
| 3584 x 18944 | - | - | - | 435 | 461 |

Sayaçlar, 4096 x 14336 (n4 iki kuyruk, n8 iki kuyruk):

| sayaç | B=4 | B=8 |
|---|---|---|
| parçacık başına ALU komutu | 93 740 (iki kuyrukta da) | 93 740 (iki kuyrukta da) |
| komut karışımı float / half / tamsayı-koşul | %66.9 / %7.7 / %21.6 | aynı |
| ALU kullanımı | %41.6 | %41.2-41.6 |
| F32 sınırlayıcı / kullanım | %67.9 / %55.7 | %66.7-67.5 / %55.2-55.6 |
| komut hacmi sınırlayıcı | %64.4-64.7 | %64.9-65.5 |
| doluluk (Kernel Occupancy) | %14.5-14.6 | %14.8-15.0 |
| cihaz belleği bandı | 74.5-75.8 GB/s | 76.0-76.5 GB/s |
| son seviye önbellek / L1 / başlatma sınırlayıcı | %11 / %14 / %0.02 | %14-16 / %16-17 / %0.02 |

Bulgular:
- Sabit iş doğrulandı: B=4 ve B=8'de parçacık başına komut sayısı ve karışımı birebir aynı. Çekirdek B'den bağımsız aynı işi yapıyor, B=4'te sütunların yarısı boşa.
- Bellek ve başlatma elendi: bant tavanın (~258 GB/s) ~%29'u; önbellek ve başlatma sınırlayıcıları %0-16.
- "Hesap tavanında" doğrulanmadı: hiçbir birim sınırda değil; ALU %42, en yüksek sınırlayıcı F32 %68. Bölüm 17'deki "tepenin %52'si" hesabı F32 kullanımıyla (%56) uyumlu, ama bu "hesap sınırlı" demek için yetmez.
- ALU %100'e çıkabilseydi işlem ~165 us olurdu (397 us x 0.416; tahmin, ölçüm değil). Bu B=1'in 141 us'sine yakın.
- Doluluk %14.5: 64 grup x 128 parçacık = 8192 parçacık, 16 çekirdeği doldurmuyor (hesap; ~3072 parçacık/çekirdek varsayımıyla %16.7).
- Gerçek 7B boyutları aşağıda; paralellik süreyi belirliyor.

### 19.1 Gerçek 7B boyutları (B=4, aynı yöntem; ham veri `raw_results/gpu_sayac/up_n4.csv`, `down_n4.csv`)

İşlem başına süre (us, test-backend-ops perf) ve B=1'e oran:

| boyut | n=1 | n=2 | n=3 | n=4 | n=8 |
|---|---|---|---|---|---|
| 18944 x 3584 (gate/up, 296 grup) | 150 | 156 | 195 | 301 (2.0x) | 309 (2.1x) |
| 3584 x 18944 (down, 56 grup) | 157 | 177 | 258 | 435 (2.8x) | 461 (2.9x) |

Sayaçlar (B=4, iki kuyruk aralığı):

| sayaç | 4096 x 14336 (64 grup) | 18944 x 3584 (296 grup) | 3584 x 18944 (56 grup) |
|---|---|---|---|
| doluluk | %14.5 | %38.6-41.6 | %12.2 |
| ALU kullanımı | %41.6 | %51.0 | %37.4-37.6 |
| F32 sınırlayıcı | %67.9 | %84.8 | %61.0-61.2 |
| F32 kullanımı | %55.7 | %68.0 | %50.1-50.3 |
| komut hacmi sınırlayıcı | %64.4-64.7 | %83.1-83.3 | %50.1 |
| cihaz belleği bandı (GB/s) | 74.5-75.8 | 90.7-91.1 | 66.9-68.6 |
| parçacık başına ALU komutu | 93 740 | 23 514 | 123 836 |

Bulgular:
- Parçacık başına komut k ile orantılı; ağırlık başına komut üç boyutta da ~13.1 (parçacık sayısı grup x 128 varsayımıyla hesap).
- Doluluk, ALU kullanımı ve F32 sınırlayıcı birlikte artıyor: %12 doluluk ile %37 ALU ve %61 F32; %39-42 doluluk ile %51 ALU ve %85 F32.
- Gate/up boyutunda çekirdek F32 tavanına yakın (sınırlayıcı %85): orada B=4 maliyeti B=1'in 2.0 katı ve "sabit 8 sütun işi" duvarı büyük ölçüde gerçek.
- Down boyutu aynı baytı ve aynı komut toplamını işliyor ama 56 grupla F32 yalnız %61: süre 435 us, gate/up'ın 1.45 katı. ALU kullanımı oranı (51.0 / 37.5 = 1.36) süre oranıyla yaklaşık uyumlu.
- Tahmin, ölçüm değil: down gate/up düzeyine (301 us) çıksa FFN süresi layer başına ~%13, tüm adımda ~%10 kısalır (35 ms yerine ~31 ms). Bölüm 17'deki tm1 u4 (%0-4 tüm adım) bununla çelişmiyor. B=1 tabanı olan 17.5 ms'ye yetmez.
- Qwen2.5-7B'de q/o (3584 satır, 56 grup) ve k/v (512 satır, 8 grup) da düşük gruplu; bunların sayacı alınmadı.

Bölüm 17'ye düzeltme: "35 ms duvarı yapısal donanım kısıtı" hükmü gate/up için doğru (F32 sınırlayıcı %85, B=1'in 2.0 katı). 4096 testinde ve down boyutunda paralellik kaybı ek yavaşlık yapıyor. Duvarın büyük kısmı sabit 8 sütun işi, küçük kısmı (tahmini ~%10) düşük grup sayısı.

Açık işler:
- Down, q ve o için daha küçük satır dilimi (tm1) ya da split-K (bölüm 5, E7): yalnız bu boyutlarda uçtan uca ölçülmedi, kazanç tahmini ~%10.
- F32 sınırlayıcı %85'in ötesine geçmeyi engelleyen neden bilinmiyor (MMA, half ve tamsayı komutları aynı hattı paylaşıyor olabilir; ölçülmedi).
- B=1 kontrol sayacı (`n1_sb1`) alınmadı, gerekmedi.
