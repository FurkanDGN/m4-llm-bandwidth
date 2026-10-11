# SONUÇLAR: bugünün tüm ölçüm tabloları

## Okuma kılavuzu (bilimsel hijyen)

Her tablonun başında bir **Durum** satırı var: kim ölçtü, kaç koşu/tekrar, doğrulama yöntemi, bağımsız tekrar var mı.

Kaynak etiketleri:
- **[A]** = bu oturumdaki ajan (bu paketi yazan). **Doğrulanmamış tek kaynak** sayılmalı. Aksi yazılmadıkça bağımsız tekrar YOK.
- **[F]** = donanım sahibinin llama-bench koşuları ve önceki oturumda ölçülüp bana aktarılanlar. Ben ölçmedim, ham çıktısını görmedim.
- **[M]** = ana oturumun zip/entropi/CPU ölçümleri. Veriler bende **yok** (bölüm 13).

"Doğrulama": Metal çekirdek ölçümlerinde = CPU'da çift hassasiyetle yeniden hesaplanan örnek satırlarla göreli hata. llama.cpp ölçümlerinde = çoğunlukla YOK (araç çıktısı olduğu gibi).

**Neden ihtiyat gerekli (örnek):** İlk q4 sonuçlarımda "bayt yerine hesap q4'te yaramaz, karışım 243→70 GB/s'ye düşer" dedim; yavaş bir dönüşüm yolu yüzünden yanlıştı (hızlı dönüşümle q4'te rastgele üretim okumadan 1.34 kat hızlı çıktı). Tek kaynaklı, baz çizgisi tavanla doğrulanmamış bir ölçüm sağlam görünüp yanlış sonuca götürdü. Aşağıdaki hiçbir [A] sayısını bağımsız tekrar olmadan "kanıt" gibi okuma.

---

## 1. Başlangıç bağlamı (aktarılan)

Durum: ölçen=[F]/ana oturum; tekrar=bilinmiyor; doğrulama=yok; bağımsız tekrar=YOK (benim ölçümlerim bazılarını yeniden üretti: bölüm 2).

| Model / ölçüm | Değer |
|---|---|
| Qwen2.5-1.5B tg128 (GPU) q4_0 / q8_0 / q3_K_M / q2_K | 180 / 121 / 117 / 139 tok/s |
| Qwen2.5-7B q4_0 tg | 53.7 tok/s |
| OLMoE-1B-7B q4_0 tg | 229 tok/s |
| 1.5B q4_0 prefill | ~1990 tok/s |
| Etkin bant (1.5B q4_0 decode) | ~191 GB/s |
| Spekülatif 1.5B hedef + 0.5B taslak | 86 tok/s (taban 180) |
| Spekülatif 7B hedef + 0.5B taslak | 20-24 tok/s (taban 53.7) |
| 1.5B hedef perf: doğrulama | 904 token / 1588 ms = 569 tok/s; sürenin %68'i "unaccounted" (taslak) |

## 2. Perplexity (wikitext-2, 64 parça)

Durum: ölçen=ana oturum zinciri (llama-perplexity); ben yalnız çıktı dosyasını okudum; tekrar=1; doğrulama=yok (araç çıktısı); bağımsız tekrar=YOK. Farklı model aileleri (Qwen vs OLMoE) farklı tokenizer kullandığından PPL'leri birbiriyle kıyaslanmaz.

| Model | PPL |
|---|---|
| 1.5B q8_0 | 9.8209 ± 0.20171 |
| 1.5B q4_0 | 10.2914 ± 0.21137 |
| 1.5B q3_K_M | 11.0236 ± 0.22989 |
| 1.5B q2_K | 16.2623 ± 0.34398 |
| OLMoE-1B-7B q4_0 | 11.9059 ± 0.26635 |
| 7B q4_0 | 7.7653 ± 0.16481 |

## 3. llama-bench (brew llama.cpp, yamasız)

Durum: ölçen=[A]; tekrar=-r 5 (ilk tablolar), -r 3 (ince eğri); doğrulama=yok (araç çıktısı); bağımsız tekrar=YOK (tg128 değerleri [F] ile uyumlu: 1.5B 179.5 vs 180, 7B 54.5 vs 53.7).

tg128 (tok/s): 0.5B 316.4 ± 1.8, 1.5B 179.5 ± 1.0, 7B 54.5 ± 0.02 (7B etkin bant 241 GB/s, 1.5B 190, 0.5B 134).

Parti süresi (ms) ve tek-token'a oran (ince eğri, `n/avg_ts`):

| n | 0.5B | 1.5B | 7B |
|---|---|---|---|
| 1 | 3.7 (1.00x) | 5.9 (1.00x) | 18.2 (1.00x) |
| 2 | 3.9 (1.06x) | 9.0 (1.52x) | 35.0 (1.92x) |
| 3 | 4.9 (1.35x) | 12.1 (2.05x) | 50.4 (2.77x) |
| 4 | 5.9 (1.61x) | 15.2 (2.57x) | 68.3 (3.75x) |
| 5 | 7.2 (1.96x) | 19.7 (3.32x) | 101.9 (5.59x) |
| 6 | 7.9 (2.15x) | 21.7 (3.67x) | 97.4 (5.34x) |
| 7 | 9.2 (2.51x) | 26.9 (4.54x) | 127.5 (7.00x) |
| 8 | 9.6 (2.62x) | 28.2 (4.77x) | 136.1 (7.47x) |
| 9 | 10.5 (2.88x) | 26.2 (4.42x) | 92.8 (5.09x) |
| 16 | 11.1 (3.02x) | 26.4 (4.46x) | 93.9 (5.15x) |
| 32 | 10.7 (2.92x) | 26.6 (4.49x) | 92.2 (5.06x) |

Bulgu (kayıt): 1-8'de doğrusal; 9'dan itibaren düz taban (0.5B 3.0x, 1.5B 4.5x, 7B 5.1x).

Sabit masraf uydurması (tg128, 3 nokta, tek doğrulama): token süresi ≈ 1.54 ms + 3.80 ms/GB × model boyutu (bant tavanı ~263 GB/s); 0.5B tahmin 317 tok/s, ölçülen 316.

## 4. Spekülatif decode (yamasız, 7B hedef + 0.5B taslak, esse, temp 0)

Durum: ölçen=[A]; tekrar=1 koşu/hücre (k=1..8 yield taraması 1 koşu); doğrulama=yok (çıktı metni karşılaştırılmadı); bağımsız tekrar=YOK. Aynı koşul k=4 7B: eski `--draft-max` koşusunda 20.43 tok/s, yeni `--spec-draft-n-max` koşusunda 20.51 tok/s (160 kabul, %41.237 birebir): iç tutarlılık VAR.

| k | tok/s | tok/tur | doğrulama ms/tur | taslak ms/tur | kabul |
|---|---|---|---|---|---|
| 1 | 37.8 | 1.68 | 38.9 | 5.4 | %68 |
| 2 | 34.1 | 2.19 | 54.8 | 9.3 | %60 |
| 3 | 27.1 | 2.41 | 76.3 | 12.9 | %47 |
| 4 | 20.5 | 2.63 | 111.5 | 16.7 | %41 |
| 5 | 21.8 | 2.77 | 107.0 | 20.3 | %36 |
| 6 | 17.6 | 2.90 | 141.0 | 24.1 | %32 |
| 7 | 16.5 | 2.97 | 152.0 | 27.4 | %28 |
| 8 | 24.4 | 3.11 | 96.9 | 30.5 | %27 |
| 15 | 20.6 | 3.15 | 99.5 | 53.3 | %14 |
| 31 | 15.7 | 3.19 | 97.6 | 105.4 | %7 |

Taban (düz decode) 52.5-53.9 tok/s. Sonuç: doğrulama maliyeti B'ye göre llama.cpp'nin parti eğrisini izliyor (tablo 3); taslak token başına ~3.4-4.2 ms.

Diğer: kod istemi 7B+0.5B (acceptance %97): k=4 37.0, k=15 96.4, k=31 132.5 tok/s (taban 52.8). 1.5B+0.5B esse: k=4 88.7, k=8 79.9, k=15 63.0, k=31 41.7 (taban ~173). `--spec-draft-p-min` {0, 0.5, 0.8, 0.9}: n_drafted birebir aynı (etkisiz; backend örnekleme açık/kapalı fark etmedi); neden bilinmiyor.

## 5. llama-lookup (yamasız)

Durum: ölçen=[A]; tekrar=1 koşu/hücre; doğrulama=çıktı metni tabanla karşılaştırıldı (12 karşılaştırmanın 11'inde özdeş; 1.5B esse'de %80'de ayrışma: farklı matris çekirdeği kaynaklı küçük sayı farkı); bağımsız tekrar=YOK.

JSON (1252 token istem, 18 kayıt, status pending→done), -n 900:

| k | 1.5B tok/s | 7B tok/s |
|---|---|---|
| 0 | 166.2 | 51.8 |
| 3 | 181.2 | 46.4 (baz altı) |
| 8 | 230.1 | 64.1 |
| 15 | 257.0 | 72.4 |
| 31 | 267.0 | 75.8 |

Matris (-n 600), k=8 / k=31 (taban k=0):

| İş | 1.5B (taban) | 1.5B k=8 | 1.5B k=31 | 7B (taban) | 7B k=8 | 7B k=31 |
|---|---|---|---|---|---|---|
| düz yazı | 173.5 | 251.5 (1.45x) | 488.8 (2.82x) | 53.0 | 76.2 (1.44x) | 159.3 (3.00x) |
| kod | 172.0 | 260.1 (1.51x) | 529.6 (3.08x) | 52.8 | 77.2 (1.46x) | 167.2 (3.17x) |
| esse (kopya yok) | 172.7 | 146.6 (0.85x) | 162.6 (0.94x) | 52.5 | 37.6 (0.72x) | 37.6 (0.72x) |

## 6. İlk dönem Metal çekirdekleri (YAVAŞ dönüşüm; **geçersiz kılındı**, bölüm 8 ile değiştirildi)

Durum: ölçen=[A]; tekrar=reps 15 medyan (dispatch başına 3-4 tekrar); doğrulama=CPU referansı VAR (göreli ≤ 3e-6; q4/q8 çıktısı okuyan=üreten göreli ≤ 1e-6); bağımsız tekrar=YOK. **Aşağıdaki q4/q8 sonuçları, yavaş `uchar4→float4` dönüşümü yüzünden yanıltıcıdır.** Karşılığı bölüm 8'dedir.

Bant tavanı (akış okuma): 254.3 GB/s (medyan), en iyi 255.4. İlk okuyan çekirdek 175-214 GB/s (kod düzeni yüzünden tavanın altında); daha sonra şablon çekirdek 244-257 GB/s.

q4 (8192², sanal GB/s): okuma 243.7 (433 G ağırlık/s), rastgele üretim 228.0 (405 G ağırlık/s), 0.94x. q8: okuma 239.7, üretim 326.8 (1.36x), karışık en iyi 3/4 üretimde 437.8 (1.83x). fp16: okuma 232.8, üretim 959.7 (4.12x); karışık 3/4 üretim 1045 GB/s.

Parçalama (2^31 ağırlık, 8192², q4): okuma 6.13 ms (eski çekirdek), üretim 5.13, **PRNG-yok** 3.81 ms (563 G ağırlık/s), yalnız PRNG 0.81 ms (2649 G ağırlık/s). "Okuma, ölçek yok" 8.70 ms: anormal, derleyici etkisi, yorumlanmadı.

Türetme (q4 tohum+katsayı, P=1..4; G ağırlık/s): P=1 246, P=2 179, P=3 122, P=4 93 (okuma 435, rastgele 405). Karışık türetme P=3 q4 8192²: 243.3 / 151.6 / 110.1 / 85.7 / 69.5 GB/s (GF=0..4) ← "243→70" (hatalı sonuç).

Türetme P'ye göre (sanal GB/s, 8192², GF=0..4): q8 P=2 238/273/249/232/198; q8 P=3 240/236/198/163/133; fp16 P=2 232/286/381/486/372; fp16 P=3 232/286/371/330/250; fp16 P=4 227/287/347/247/187.

## 7. Mikro-ölçümler

Durum: ölçen=[A]; tekrar=7 tekrar (5 sonrası medyan); doğrulama=yok (işlem hızı ölçümü; çıktıyı tampona yazar); bağımsız tekrar=YOK; ilk FMA ölçümü sonradan **düzeltildi** (aynı kişi, aynı yöntem, 32 zincir).

| Ölçüm | Sonuç |
|---|---|
| fp32 FMA, 8 zincir (ilk) | 5.5 TFLOPS (2.76 T FMA/s) |
| fp32 FMA, 32 zincir | **6.15 TFLOPS** (tg 128-1024 aynı) |
| fp32 float4 vektör, 32 zincir | 5.8-6.0 TFLOPS |
| half FMA, 8 zincir | 2.96 T/s (fp32 8 zincirden %7 hızlı) |
| simdgroup_matrix half×half→float | 6.15-6.25 TFLOPS |
| simdgroup_matrix half×half→half | 6.26-6.40 TFLOPS |
| tamsayı kaydır+xor | ~0.8 T deyim/s (~1.6 T temel işlem/s) |

Dönüşüm+nokta çarpım mikro-ölçümü (32 ağırlık/yineleme, G ağırlık/s; "yuva" = 2.76 T FMA/s'ye göre, **o rakam düzeltilmeden hesaplandı**):

| Varyant | G ağırlık/s | FMA-yuvası/ağırlık |
|---|---|---|
| taban (yalnız sözcük üretimi) | 4645 | 0.59 |
| uchar4→float4 (`cvt`) | 570 | 4.84 |
| float sihirli sayı | 1085 | 2.54 |
| **half2 sihirli sayı** | **1374** | **2.01** |
| tamamen half | 1433 | 1.93 |
| 256 girişli LUT | 636 | 4.34 |

## 8. Hızlı dönüşümle düzeltilmiş format, karışık ve türetme sonuçları

Durum: ölçen=[A]; tekrar=reps 7 (format), reps 5 (üretici varyantları), medyan; doğrulama=CPU referansı VAR (en büyük göreli hata 5e-7); bağımsız tekrar=YOK, ama iki ayrı program (bench12 ve bench13) aynı sayıları verdi (fp16 P=2 686 GB/s ikisinde de).

Bant tavanları (bu programda): q4 243.0, q8 243.3, fp16 244.5 GB/s → bellekten en fazla 432 / 229 / 122 G ağırlık/s.

Saf üretim (tüm satır; okumaya oran; 8192²):

| Format | Okuma GB/s (G ağırlık/s) | Rastgele | Türet P=2 | Türet P=3 |
|---|---|---|---|---|
| q4_0 benzeri | 250.0 (444) | 335.8 (1.34x) | 193.1 (0.77x) | 137.1 (0.55x) |
| q8_0 benzeri | 247.8 (233) | 633.1 (2.56x) | 365.0 (1.47x) | 258.6 (1.04x) |
| fp16 benzeri | 245.2 (123) | 1193.6 (4.87x) | 686.2 (2.80x) | 486.9 (1.99x) |

Rastgele üretim 597 G ağırlık/s (hesap sınırlı, üç formatta aynı). 4 boyutta tutarlı.

Karışık (GF=0..4; sanal GB/s; 8192²):

| | GF=0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| q4 rastgele | 250.0 | 319.2 | **417.5** | 374.1 | 335.8 |
| q4 türet P=2 | 250.0 | 248.4 | 249.6 | 228.5 | 193.1 |
| q4 türet P=3 | 250.0 | 249.6 | 223.8 | 169.7 | 137.1 |
| q8 rastgele | 247.8 | 322.6 | 467.7 | **704.5** | 633.1 |
| q8 türet P=2 | 247.8 | 278.0 | 323.2 | 382.4 | 365.0 |
| q8 türet P=3 | 247.8 | 278.3 | 319.3 | 321.8 | 258.6 |
| fp16 rastgele | 245.2 | 319.8 | 478.3 | 902.1 | **1193.6** |
| fp16 türet P=2 | 245.2 | 291.2 | 385.9 | 530.4 | **686.2** |
| fp16 türet P=3 | 245.2 | 300.1 | 383.3 | 533.4 | 486.9 |

Üretici varyantları (P=2, tüm satır, okumaya oran; 8192²; 18944x3584 ile aynı):

| Üretici | q4 | q8 | fp16 |
|---|---|---|---|
| rastgele (taban) | 1.30 | 2.47 | 4.67 |
| A xorshift (mevcut) | 0.75 | 1.42 | 2.69 |
| B ertelenmiş ofset | 0.74 | 1.41 | 2.68 |
| C tek-adım rotasyon | 0.78 | 1.48 | 2.81 |
| D tek çarpımlı hash, 2 bit | 0.85 | 1.60 | 3.04 |
| E tek çarpımlı hash, ±1 seçim | **1.00** | **1.91** | **3.57** |

P=3: A 0.53/1.01/1.91; B 0.54/1.03/1.95; C 0.61/1.16/2.19. E tohum gösteriminin bellek tavanına ulaşır (q8 1.89x, fp16 3.56x); A hesap sınırlı (343 G ağırlık/s üç formatta aynı), E bellek sınırlı (~460 G ağırlık/s). E'nin kalite sınırı: 8 ağırlıkta yalnız 4 değer. Hibrit GF=2/3'te tüm türetme varyantları aynı süre (okunan satırların belleğinde sınırlı).

## 9. Toplu (B sütun) çekirdekler

Durum: ölçen=[A]; tekrar=reps 5-9, medyan; doğrulama=CPU referansı VAR (skaler 96 çekirdek 4.7e-7; MMA ≤ 4e-7, x yarım hassasiyete yuvarlanmış referansla); bağımsız tekrar=YOK, ama dört boyutta (4096², 8192², 18944×3584, 3584×18944) aynı örüntü.

Ana tablo (8192², ms; 2^29 ağırlık; B=1 skaler 1.17 ms):

| B | (a) B kez matvec | skaler okuma (R) | MMA okuma (en iyi tm) | skaler türet-E | MMA türet-E | oran okuma / B=1 |
|---|---|---|---|---|---|---|
| 1 | 1.20 | 1.17 (R=1) | 1.92 (tm2) | 1.16 | 3.01 | 1.00x |
| 2 | 2.39 | 1.19 (R=2) | 1.92 | 1.30 | 3.01 | 1.00x |
| 3 | 3.60 | 1.27 (R=4) | 1.92 | 1.97 | 3.01 | 1.09x |
| 4 | 4.77 | 2.54 (R=4) | **1.92** | 2.72 | 3.01 | 1.66x |
| 5 | 5.94 | 2.61 (R=8) | 1.92 | 2.77 | 3.01 | 1.66x |
| 6 | 7.13 | 3.25 | 1.92 | 3.40 | 3.01 | 1.66x |
| 7 | 8.36 | 4.15 | 1.92 | 4.34 | 3.01 | 1.66x |
| 8 | 9.51 | 4.61 | 1.92 | 5.04 | 3.01 | 1.66x |
| 9-16 | 10.6-19.3 | - | **3.60 (tm1, 2 karo)** | - | 4.66 | 3.1x |

En iyi okuma eğrisi (4 boyut): B=1 1.00x, B=2 1.00x (0.98-1.01), B=3 1.09x (3584×18944'te 1.23x), B=4-8 1.66-1.69x, B=9-16 3.0-3.2x. Türetme (E)/okuma: B=1'de ~1.0x (başa baş), B=2'de 1.09x, B=3-8'de 1.4-2.4x yavaş, B=9-16'da 1.3x yavaş; hiçbir B'de okumayı geçmez (bayt eşit).

Ara dönemler (kayıt): ilk skaler çekirdek (lane=32 ağırlıklı blok, yavaş dönüşüm) B=2'de 2.05x, B=8'de 9.4x, B≥11'de patlama; sihirli dönüşümlü lane=blok sürümü hâlâ doğrusal (B=8 9.9x): darboğaz dönüşüm değil x trafiği. Lane=sözcük eşlemesi: B=2 1.20x (dot() sürümü) / 1.00x (açık FMA). R taraması: B=4'te 2.19-2.33x, B=5-8'de R=8 ile 2.25-3.98x, B≥9 skaler çökme. Yazılım ön-yükleme: iyileşme yok.

## 10. llama.cpp yamalı yapı

Durum: ölçen=[A]; tekrar: llama-bench -r 3 (parti eğrileri), -r 5 (tg128), spekülatif k=1-3: 3 tekrar, k=4-5: 1 tekrar; lookup: 1 tekrar/hücre; doğrulama=ggml test-backend-ops (CPU referansı; yalnız küçük boyut) VAR, gerçek model boyutunda sayısal doğrulama YOK (çıktı metni karşılaştırılmadı); bağımsız tekrar=YOK. Yama aynı ikilide `GGML_METAL_SB=0` (kapalı) ile yamasız davranışı veriyor.

Eşik taraması (yamasız kod yolu; 7B parti ms; tek tekrar seti, gürültülü):

| B | T=8 (varsayılan) | T=5 | T=4 | T=3 | T=2 | T=1 |
|---|---|---|---|---|---|---|
| 2 | 38.2 | 39.4 | 35.5 | 34.9 | 37.6 | 106.2 |
| 4 | 73.3 | 77.6 | 68.5 | 93.9 | 91.5 | 91.8 |
| 5 | 108.4 | 106.0 | 92.2 | 93.9 | 91.9 | 91.3 |
| 8 | 143.3 | 109.2 | 93.1 | 92.6 | 92.8 | 105.8 |

Kodda küçük-parti yolu (`mul_mv_ext`, ne11 2-8) zaten var ve devrede. Eşiği 4'e indirmek B=5-8'i ~92-93 ms'ye çeker (mat-mat tabanı 5x), B=1-4'e dokunmaz.

7B parti süresi (ms; B=1 = 18.6):

| B | yamasız | yamalı (varsayılan) | kazanç |
|---|---|---|---|
| 2 | 35.2 (1.90x) | **19.7 (1.06x)** | 1.79x |
| 3 | 50.7 (2.73x) | 28.8 (1.55x) | 1.76x |
| 4 | 68.7 (3.69x) | 33.9 (1.82x) | 2.03x |
| 5 | 102.3 (5.50x) | 34.0 (1.83x) | 3.01x |
| 6 | 99.5 (5.35x) | 34.3 (1.85x) | 2.90x |
| 7 | 127.8 (6.87x) | 34.5 (1.86x) | 3.70x |
| 8 | 137.3 (7.39x) | 34.7 (1.87x) | 3.96x |
| 9 | 93.0 (5.00x) | 92.7 (4.99x) | 1.00x |
| 16 | 94.4 (5.07x) | 94.7 (5.09x) | 1.00x |

Kernel bazlı (test-backend-ops perf, m=4096, k=14336, tek matris): B=1 134 µs; B=2 yamasız 295 → yamalı 143 µs (1.07x); B=3 yamasız 441 → yamalı R=3 212 µs (1.57x; R=4 236 µs); B≥4 MMA 326-346 µs (2.4-2.6x). Mikro-ölçümdeki 1.09x (B=3) ve 1.66x (B=4-8) tek-matris, az-threadgroup durumunda tutmadı.

Spekülatif 7B+0.5B, esse, temp 0, -n 256 (tok/s; taban ~53.3, düz decode):

| k (B) | yamasız | yamalı (3 tekrar) | tabana göre |
|---|---|---|---|
| 1 (2) | 38.1, 38.4, 38.5 | 61.9, 61.6, 61.2 | **1.16x** |
| 2 (3) | 34.3, 34.4, 34.3 | 59.4, 60.1, 59.3 | **1.12x** |
| 3 (4) | 27.2, 27.2, 27.3 | 45.4, 45.8, 45.2 | 0.85x |
| 4 (5) | 20.6 (1 tekrar) | 44.5 (1 tekrar) | 0.83x |
| 5 (6) | 21.9 (1 tekrar) | 44.5 (1 tekrar) | 0.83x |

(İlk yamalı koşuda R=4 B=3 için k=2 52.3 tok/s idi; R=3'e çekilince 59.3-60.1.)

llama-lookup 7B, yamalı/yamasız (tok/s; B=k+1; 1 tekrar; **gürültü payı var**):

| k (B) | JSON yamasız / yamalı | düz yazı | kod | esse (kopya yok) |
|---|---|---|---|---|
| 0 (1) | 41.0* / 51.4 | 53.4 / 52.8 | 50.8 / 53.5 | 53.6 / 53.2 |
| 1 (2) | 41.6 / 79.7 | 48.2 / 84.6 | 46.0 / 87.4 | 50.7 / 53.8 |
| 2 (3) | 47.5 / 88.4 | 50.4 / 83.7 | 51.4 / 97.7 | 48.0 / 52.6 |
| 3 (4) | 44.4 / 82.0 | 48.4 / 76.8 | 45.9 / 86.9 | 44.5 / 49.5 |
| 4 (5) | 39.0 / 96.0 | 41.2 / 92.9 | 40.5 / 100.3 | 39.1 / 49.7 |
| 7 (8) | 39.2 / **106.6** | 46.4 / **118.2** | 44.9 / **127.7** | 34.1 / 48.7 |
| 8 (9) | 58.3 / 86.9 | 76.6 / 76.5 | 78.0 / 78.2 | 40.1 / 40.7 |
| 15 (16) | 68.3 / 100.1 | 111.1 / 108.1 | 115.1 / 115.5 | 39.0 / 40.3 |
| 31 (32) | 73.1 / 103.1 | 160.2 / 156.9 | 170.1 / 171.1 | 39.7 / 40.7 |

*JSON yamasız k=0 41.0 anormal dusuk (ayrı önceki ölçüm 51.8): açıklanmadı. Bulgular: kopya-ağır işte k eşiği 8'den 1'e düştü (yamasızda k≤7 çoğunlukla tabandan yavaştı); k≥8 (B≥9) yola yama dokunmuyor, en iyi nokta hâlâ k=31 (kod 171, düz yazı 157). JSON'da B≥9'da 1.4x fark hipotezi (kısa taslaklı turlar) ölçülmedi.

tg128 regresyonu: 7B orijinal 53.15, 52.72; yamalı SB0 54.33, 54.46; SB1 54.49, 54.59. 1.5B 183.8-185.4 hepsinde. 0.5B 322-345 (K=896: yama devreye girmez; oynama gürültü). **Regresyon yok** (iki tekrar; bağımsız tekrar yok).

ggml `test-backend-ops test -o MUL_MAT -b MTL0 -p type_a=q4_0`: 3/3 backends passed, FAIL yok (n=2..7, m=16, k=256/1024).

## 11. Projeksiyon ve gerçekleşen

Durum: projeksiyon=[A] hesabı (ölçülmüş bileşenlerden), ölçüm değil; gerçekleşen=bölüm 10. Bağımsız tekrar YOK.

| k | Projeksiyon tok/s (taban 54.5'e oran) | Gerçekleşen (yamalı) |
|---|---|---|
| 1 | 65.5 (1.20x) | 61.2-61.9 (1.16x) |
| 2 | 71.0 (1.30x) | 59.3-60.1 (1.12x) |
| 3 | 51.6 (0.95x) | 45.2-45.8 (0.85x) |

Projeksiyon iyimserdi (k=2 için B=3 maliyeti 1.09x varsayılmıştı, gerçek 1.55x).

## 12. GPU sayaçları (xctrace)

Durum: ölçen=[A]; tekrar=3 deneme; doğrulama=yok; sonuç=**sayaç yakalanamadı**. Detay: `raw_results/07_xctrace_notes.txt`. "Metal GPU Counters" (varsayılan "Performance Limiters"): "Selected counter profile is not supported on target device"; "Metal System Trace" ve "Game Performance" yalnız zaman çizelgesi verdi. ALU limiter / bellek limiter / occupancy değerleri YOK. Çıkarımlar (sayaç değil, değişken-çıkarma): xorshift türetme hesap sınırlı, E ve B=1 q4 okuma bellek sınırlı, rastgele üretim hesap sınırlı.

## 13. Bende bulunmayan ölçümler

- **Ana oturumun zip/entropi/CPU ölçümleri [M]:** sonuç tabloları bende yok (betik adlarını gördüm, içeriklerini ve çıktılarını görmedim). SONUCLAR'a konamadı. koordinatör oturumun raporlarından eklenmeli.
- Orijinal ham dosyalar panikle kayboldu; `raw_results/` aktarım kopyasıdır.
