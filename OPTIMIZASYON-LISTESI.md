# Optimize edilecekler listesi

Buraya bellek bandı tavanına dokunmayan, ölçülmüş ya da ölçülebilir kayıplar girer.
Her madde: ne kadar kayıp var, neye dayanıyor, nasıl ölçülür.
Sayıların kaynağı ve güvenilirliği için `SONUCLAR.md` ve `main_session/TUR1-DOGRULAMA.md`.
Bu listedeki ölçümlerin ham verisi, programları ve llama.cpp yaması `sabit_masraf/` altında.
Yama `sabit_masraf/yama.diff`. Önce llama.cpp commit e475fa2b5 üstüne `llama_patch/apply_patch.py` (küçük batch çekirdekleri) uygulanır, sonra bu yama. Sırayla uygulanınca temiz oturduğu doğrulandı.
Çalışma kopyası `~/claude-scratch/llamacpp-ablasyon-sb`. Ablasyon yamasının eski hâli `sabit_masraf/yama_eski_ablasyon.diff`.
Ölçen her yerde bu oturum (Sonnet 5.5). Bağımsız tekrar yok. Tur sayıları ilgili maddede yazılı.
Ölçümlerin gürültü sınırı için madde 7'ye bak. Madde 3'teki hız sayıları o sınırın üstünde olanları ayırıyor.

## 1. Token başına sabit masraf (1.63 ms): ilk görev, döküm tamam

**Kayıp.** Token süresi `t = 1.63 ms + 3.99 ms/GB × okunan bayt`. 5 model noktası, R² 0.9993, tg128, 5 tekrar, mmap kapalı (`TUR1-DOGRULAMA.md` bölüm 2; ana oturum ölçtü).
Bu sabit terim, ağırlık okumakla ilgisi olmayan kısımdır.

| model | token süresi | sabit masrafın payı |
|---|---|---|
| 0.5B q4_0 (taslak model) | 3.03 ms | %54 |
| 1.5B q4_0 | 5.46 ms | %30 |
| 7B q4_0 | 18.25 ms | %9 |

Paylar uydurma formülden hesaplandı, ayrıca ölçülmedi.
Tahminli üretmede taslak model her token için bu masrafı öder. Taslak token başına 3.4-4.2 ms tutuyor.

**Döküm.** llama.cpp'nin Metal koduna küçük bir yama yazdık. Ortam değişkeniyle seçilen işlem türünün GPU çağrısı atlanıyor.
Komut hazırlığı ve bekleme noktaları yerinde kalıyor. Çıktı yanlış sayılar üretiyor, ama yalnız süre ölçülüyor.
Her yapılandırma 2 tur, her turda 5 tekrar. Taban 4 kez ölçüldü. Ham veri: `sabit_masraf/ablasyon_sonuc.csv`.

| parça (ms/token) | 0.5B q4_0 | 7B q4_0 |
|---|---|---|
| token süresi (taban, medyan) | 3.00 | 18.41 |
| matris çarpımları (hepsi atlanınca düşen) | 1.31 | 16.54 |
| küçük işlemlerin hepsi (norm, rope, attention, ekleme, glu, set_rows, get_rows) | 0.63 | 1.34 |
| hiç GPU işi yokken geçen süre (komut hazırlama tabanı) | 0.49 | 0.55 |
| üç parçanın toplamı | 2.43 | 18.43 |
| açıklanamayan | 0.57 | -0.02 |

Matris çarpımları atlanınca kalan süre: 0.5B için 1.69 ms, 7B için 1.87 ms. Bu, uydurmadaki 1.63 ms sabit terimle aynı büyüklükte.

7B'de üç parça toplamı ölçülen süreyi tutturuyor. Eski izleme kaydıyla da uyumlu:
- Matris çarpımı payı 16.54 ms. 4.125 GB / 16.54 ms = 249 GB/s, yani bellek tavanı.
- Komut hazırlama tabanı 0.55 ms. Eski izleme kaydındaki GPU boşluğu 0.57 ms (%3.1).
- Küçük işlemler 1.34 ms. Eski kayıttaki "kernel içinde kalan" tahmini 1.0-1.3 ms.

0.5B'de 0.57 ms açıklanamıyor. Madde 4 bunun bir kısmını açıklıyor olabilir.

**Çağrı sayısı** (kodda sayıldı: `GGML_METAL_FUSE_STATS=1`, `sabit_masraf/final/say5.txt`):
- Token başına GPU çağrısı: 0.5B 508, 7B 592. Bu, yamadaki birleştirmeler kapalıyken ve llama.cpp'nin kendi birleştirmeleri açıkken.
- llama.cpp'nin kendi birleştirmesi de kapalıyken 557 ve 649.
- Matris çarpımı sayısı 169 ve 197 (yedi çarpım × katman sayısı + 1). Kalan çağrılar küçük işlemler.
- Eski sayılar (484 ve 564) günlük satırlarından sayılmıştı. Yeniden sıralamadan sonra günlük satırları eksik çıktığı için o yöntem bırakıldı. Kodda sayılan değer flash attention'ın yardımcı çağrılarını da içeriyor, bu yüzden 24-28 fazla.
- Küçük çağrı başına ortalama GPU süresi: 0.5B ~2.0 µs, 7B ~3.7 µs (eski hesap, yeniden hesaplanmadı).

**Küçük işlemlerin kırılımı** (7B, tek tek atlanınca düşen süre, ms): attention 0.49, ekleme 0.43, glu 0.26, set_rows 0.25, rms_norm 0.21, rope 0.17.
Toplamları 1.81 ediyor, birlikte atlayınca düşen 1.34'ten fazla. Yani tek tek sayılar birbirine eklenmez.
0.5B'de tek tek farklar ölçüm gürültüsü içinde kalıyor (±0.1-0.2 ms). Yalnız toplam 0.63 ms güvenilir.

**Birleştirme ve eşzamanlılık kapatma testi.**
- İşlem birleştirme kapalı: 7B +0.10 ms (57 çağrı fazla, çağrı başına ~1.7 µs). 0.5B +0.03 ms, gürültü içinde.
- Eşzamanlılık kapalı: 7B +0.26 ms. 0.5B -0.14 ms, gürültü içinde.

**Sınırlar.**
- Atlanan işlem GPU'yu boşaltıyor. GPU saat hızı düşmüş olabilir. Bu sapma ölçülmedi.
- Her çekirdeğin kendi süresi ve başlangıç/bitiş kuyruğu ayrıca ölçülmedi. Bunun için Xcode GPU yakalaması gerekir (elle adım).

**Not.** `BULUT-ANALIZ.md`'deki "1.14 ms" sayısı yanlış bayt sayımına dayanıyordu, kullanma.

## 2. MLX sabit masrafı nasıl düşürüyor

**Soru.** MLX aynı 0.5B 4-bit modelde 431.7 tok/s verdi, llama.cpp 330 (`TUR1-DOGRULAMA.md` bölüm 12).
Aynı bayt sayısı varsayarsak bu, MLX'te yaklaşık 0.9 ms sabit masrafa denk geliyor. llama.cpp'de 1.63 ms.
MLX bunu nasıl yapıyor, anlamak istiyoruz.

**Bildiğimiz.**
- Fark tek ölçüm. Sıkıştırma ve kelime dağarcığı farkı var, ama fark bunlardan büyük.
- 7B düz üretimde MLX üstünlüğü yalnız +%4 (`TUR1-DOGRULAMA.md` bölüm 16). Sabit masraf büyük modelde küçük pay.
- Tahminli üretmede MLX'in mutlak üstünlüğü tabanından geliyor. Kazanç oranları iki çerçevede aynı.

**Hipotezler (hiçbiri ölçülmedi).**
- MLX token başına daha az GPU çağrısı yapıyor.
- MLX bir sonraki token'ın komutlarını GPU çalışırken hazırlıyor.
- MLX CPU tarafında daha az iş yapıyor (graf yeniden kullanımı).

**Nasıl ilerlenir:**
1. MLX'in token başına GPU çağrı sayısını say (Metal System Trace).
2. Aynı atlama yöntemini MLX'te kur: GPU işi olmadan geçen taban süresini ölç.
3. llama.cpp'nin tablosuyla yan yana koy. Fark hangi satırda?
4. MLX kaynak kodunda token üretim adımını oku.

**Durum.** Açık. Madde 1'in sonuçları (komut hazırlama tabanı 0.5 ms, küçük işlemler 0.63 ms) MLX'in nerede kazandığını aramak için başlangıç noktası.

## 3. Küçük işlemleri birleştirme: yapıldı

**Amaç.** Küçük GPU işlemlerini daha az GPU çağrısına indirmek. Madde 1'e göre küçük işlemler 7B'de 1.34 ms, 0.5B'de 0.63 ms.
Attention kendi başına ağır bir program olduğu için birleştirilmedi. Altı birleştirme yapıldı, bir aday vazgeçildi (aşağıda).

**Sonuç.** Tüm birleştirmeler kapalıyken ve açıkken karşılaştırıldı.
Ölçüm: llama-bench, kapalı ve açık dönüşümlü, 5 tur. 0.5B ve 1.5B için 96 token ve 3 tekrar, 7B için 64 token ve 2 tekrar. Ham veri `sabit_masraf/final/final_hiz.csv`.

| model | tüm kapalı | tüm açık | fark | kaç turda hızlı |
|---|---|---|---|---|
| 0.5B | 3.005 ms (332.8 tok/s) | 2.549 ms (392.3 tok/s) | -0.455 ms (%-15.2) | 5/5 |
| 1.5B | 5.403 ms (185.1 tok/s) | 4.979 ms (200.9 tok/s) | -0.425 ms (%-7.9) | 5/5 |
| 7B | 18.339 ms (54.5 tok/s) | 17.738 ms (56.4 tok/s) | -0.602 ms (%-3.3) | 5/5 |

Başka bir koşuda (3 tur) 7B için -0.842 ms çıkmıştı. İki koşu arasındaki fark, madde 7'deki gürültü sınırıyla uyumlu.
Bu tablo tek token (ne11=1) içindir. Çok sütunlu doğrulama grafı için genişletme madde 6'nın sonunda.
Çağrı başına kazanç: 0.5B için 0.455 ms / 284 çağrı = 1.6 µs. 7B için 0.602 ms / 335 çağrı = 1.8 µs.

**Doğruluk.**
- Üç modelde 200 token greedy çıktı birebir aynı.
- Perplexity, her token tek tek işlenerek ölçüldü (`-b 1 -ub 1`, wiki.test.raw, 512 bağlam). Birleşik çekirdekler yalnız tek token yolunda çalıştığı için toplu işleme ölçümü bunları sınamazdı.
  - 0.5B (3 parça): 15.1521 kapalı, 15.1499 açık.
  - 1.5B (2 parça): 9.8477 kapalı, 9.8475 açık.
  - 7B (1 parça): 5.7262 kapalı, 5.7267 açık.
  - Farklar en çok 0.0022. Bu, tahmin hatasının (±1.3-1.8) binde biri.
- `test-backend-ops` tek işlem sınadığı için birleştirmeyi kapsamaz.

**Çağrı sayısı, adım adım** (token başına, kodda sayıldı, `sabit_masraf/final/say5.txt`):

| adım | 0.5B | 7B |
|---|---|---|
| birleştirmeler kapalı | 508 | 592 |
| + ekleme | 389 | 453 |
| + glu | 341 | 397 |
| + q/k/v | 293 | 341 |
| + rope K ile K önbelleği | 270 | 313 |
| + Q rope | 247 | 285 |
| + V önbelleği | 224 | 257 |

Toplam 0.5B'de 284, 7B'de 335 çağrı eksildi (%-56).

**Adımlar.** Her birleştirmenin kapatma değişkeni 0 değerini alır. Kod `sabit_masraf/yama.diff` içinde.
1. Ekleme matris çarpımının içine (`GGML_METAL_FUSE_MM_ADD`). Toplam değişkeni eklenecek vektörle başlatılıyor. Çekirdek `kernel_mul_mv_q4_0_f32_add`.
2. gate ve up matris çarpımı ile glu tek çağrıda (`GGML_METAL_FUSE_MM_GLU`). Çekirdek `kernel_mul_mv_q4_0_f32_glu`.
3. q, k, v matris çarpımı ve bias tek çağrıda (`GGML_METAL_FUSE_MM_QKV`). Üç ağırlığın satır aralıkları tek çağrıda bölünüyor. Çekirdek `kernel_mul_mv_q4_0_f32_qkv`.
4. K için rope ve K önbelleğine yazma tek çağrıda (`GGML_METAL_FUSE_ROPE_K`). Rope sonucu hiç yazılmıyor, doğrudan f16 önbelleğe gidiyor.
5. Q rope aynı çağrıya katıldı (`GGML_METAL_FUSE_ROPE_QK`).
6. V önbelleğine yazma aynı çağrıya katıldı (`GGML_METAL_FUSE_SET_V`).
Koşullar `ggml-metal-common.cpp` içinde. Çizge sıralayıcı birleşik grupların çıktılarını doğru izlesin diye `node_info` yapısına `extra_dsts` eklendi.

**Adım ölçümleri** (ms/token, negatif = daha hızlı). Her satırın güvenilirliği madde 7'deki gürültü sınırına bağlı. 7B'de 0.17 ms altı ayırt edilemiyor.

| adım | 0.5B | 1.5B | 7B |
|---|---|---|---|
| ekleme | -0.1 ile -0.2 (4/4 tur) | -0.108 (3/3) | -0.16 ile -0.23 (4/4 ve 3/3) |
| glu | -0.127 (4/4) | -0.211 (4/4) | -0.108 (3/4) |
| q/k/v | -0.137 (3/3) | ölçülmedi | -0.034 (2/3), gürültü içinde |
| rope K | -0.072 (3/3); ikili ölçümde -0.039 ve -0.029 | -0.045 (3/3) | ayırt edilemedi (+0.39, +0.34, +0.04) |
| Q rope | -0.175 (2/3); ikili ölçümde -0.119 (2/3) | -0.165 (3/3) | ayırt edilemedi (ikili +0.17 0/3, dörtlü -0.19 3/3) |
| V | -0.051 (8/8, hata ±0.010) | -0.031 (6/8, hata ±0.011) | ayırt edilemedi |

V adımı önce 7B'deki iki ikili ölçüme bakılarak kapatılmıştı (+0.13 ve +0.14 ms). A/A kontrolü bu farkın gürültü büyüklüğünde olduğunu gösterdi. 0.5B ve 1.5B'de 8 turla yeniden ölçülünce kazançlı çıktı, V açık bırakıldı.
Ham veri: `sabit_masraf/adim/`.

**Vazgeçilen: rms_norm'u sonraki matris çarpımına katmak.** Her iş grubunun norm toplamını kendisi yeniden hesaplaması gerekiyor.
Yalnız bu fazladan işin maliyetini ölçtük (birleştirmeyi yazmadan, toplamı hesaplayıp çıktıyı değiştirmeyen bir deneme çekirdeğiyle). Ölçülen maliyet, kazanılacak tek çağrıdan büyük:

| model | fazladan iş (ms/token, 3/3 tur yavaş) | kazanılacak (tahmin) |
|---|---|---|
| 0.5B | +0.116 | ~0.1 ms |
| 1.5B | +0.333 | ~0.1 ms |
| 7B | +0.571 | ~0.05 ms |

Kazanç sütunu ölçülmedi: 48 veya 56 çağrı × 1.6-1.8 µs çağrı başına kazancı. Gerçek birleştirme norm ağırlığını okumayı da ekleyeceği için maliyet bu ölçümden düşük olamaz. Birleştirme yazılmadı. Ham veri `sabit_masraf/adim/norm_probe.log`.

**Sınırlar.** Yalnız Q4_0, yalnız tek token (matris-vektör yolu), neox tipi rope, f16 önbellek, Metal. K-quant ve batch > 1 yolları birleştirmede yok.
Tahminli üretmede birleştirme, Metal tampon hizalaması 4096 olunca %+4.9 kazandırıyor (madde 6). Hizalama 32 iken %-6.2 kaybettiriyordu.

## 4. İlk komut grubuna düşen düğüm sayısı (küçük model)

**Durum.** llama.cpp, çizgenin ilk `MAX(64, %10 × düğüm sayısı)` düğümünü ana iş parçacığında kodlayıp hemen GPU'ya gönderiyor. Kalanı başka bir iş parçacığı kodluyor.
Küçük modelde ilk grup çok kısa sürüyor olabilir. GPU onu bitirip kalan komutların hazır olmasını bekliyor olabilir. Bu bir hipotez, doğrudan ölçülmedi.

**Ölçüm** (ortam değişkenleri `GGML_METAL_N_MAIN`, `GGML_METAL_N_CB`; ham veri `komut_tamponu_sonuc.csv`, `birlesik_sonuc.csv`):
- 0.5B (2 tur): N_MAIN 16 +0.23 ms, 32 +0.12, varsayılan (~82) 0, 128 -0.225 (%-7.4), 256 -0.075. N_CB 2 -0.08, 3 -0.16.
- 0.5B (3 tur, ayrı koşu): N_MAIN 128 -0.11 ms (%-3.9). Ekleme birleştirmesiyle birlikte -0.20 ms (%-6.7, 3/3 tur).
- 1.5B: N_MAIN 128 +0.018 ms, etki yok. 7B: etki yok.
- Küçük modelde iz kaydında (`iz_05.out`) token başına 3 komut grubu ve iki büyük boşluk görünüyor: ~0.34 ms ve ~0.70 ms. İzleme token süresini ~%33 uzattığı için mutlak değerler güvenilmez.

**Yeniden ölçüm, altı birleştirme açıkken** (ölçen bu oturum, bağımsız tekrar yok, `llama-bench -n 96 -r 3`, sıra turlarda dönüyor; `sabit_masraf/adim/nmain3_05.csv`, `nmain3_15.csv`). Varsayılan ana grup şu an `MAX(64, %10 × düğüm sayısı)`.

| N_MAIN | varsayılan | 64 | 96 | 128 | 160 | 192 |
|---|---|---|---|---|---|---|
| 0.5B ms/token (5 tur) | 2.548 | 2.578 | 2.540 | 2.516 | 2.547 | 2.617 |
| 0.5B fark | 0 | +0.030 | -0.008 | -0.032 | -0.001 | +0.069 |
| varsayılandan hızlı tur | - | 1/5 | 3/5 | 4/5 | 2/5 | 0/5 |

| N_MAIN | varsayılan | 128 | 192 |
|---|---|---|---|
| 1.5B ms/token (3 tur) | 4.943 | 4.965 | 4.960 |
| 1.5B fark | 0 | +0.021 | +0.017 |
| varsayılandan hızlı tur | - | 0/3 | 1/3 |

- Birleştirmeden önce 0.5B'de N_MAIN 128 -0.225 ms (%-7.4) kazandırıyordu. Şimdi en iyi değer 128 ve fark -0.032 ms (%-1.3, 4/5 tur). Kazanç 0.2 ms'den 0.03 ms'ye indi. GPU çağrısı sayısı 508'den 224'e düşünce ilk grubun kısa kalma sorunu da küçülmüş olabilir. Bu bir yorum, doğrudan ölçülmedi.
- 0.5B'nin turları iki kümede toplanıyor (2.47-2.52 ve 2.60-2.66 ms), tüm yapılandırmalarda. Tur aralığı 0.15 ms, farkın 5 katı. 0.03 ms'lik fark bu gürültüden ayrılamıyor.
- 1.5B'de hiçbir değer varsayılandan iyi değil. 7B bu turda ölçülmedi (önceki ölçümde etki yoktu).
- Karar: N_MAIN için sabit sayı ya da yeni kural önerilmiyor. Bu madde etkisi gürültü içinde olduğu için kapatıldı.
- İki kümeli tur dağılımının nedeni bilinmiyor, ölçülmedi.

## 5. Komut oynatma (Metal indirect command buffer)

**Fikir.** GPU komutlarını her token'da yeniden kodlamak yerine bir kez kaydedip tekrar oynatmak.

**Kapı ölçümü** (`GGML_METAL_TIMING=1`, `zamanlama_sonuc.txt`): CPU'nun GPU komutlarını kodlama süresi token başına 0.5B'de ~400 µs (birleştirme kapalı) / ~330 µs (açık), 7B'de ~634-656 µs.
Bu süre komut hazırlama tabanıyla (0.49 / 0.55 ms) aynı büyüklükte. Ama kodlamanın ne kadarı GPU çalışırken paralel yapılıyor, ölçülmedi.

**Bilinmeyen.** Tabandaki payın kodlama mı, llama.cpp'nin her token'da çizgeyi yeniden kurması mı olduğu. Komut oynatma yalnız kodlamayı ortadan kaldırır.
Her token'da değişen değerler (dikkat hesabının baktığı geçmiş uzunluğu) komutların içinde güncellenmek zorunda. Bu büyük bir değişiklik, bu bilgim kodda doğrulanmadı.

**Durum.** Açık. Önce madde 3 ve 4 ile alınabilecek kısım alınır, sonra kalan tabana bakılır.

## 6. Tahminli üretmede birleştirme hedef modeli yavaşlatıyordu: neden bulundu, düzeltildi

**Sonuç** (7B oq4 hedef, 0.5B q4_0 taslak, k=2, 96 token, 5 tur, dönüşümlü sıra, metin her yapılandırmada birebir aynı; ölçen bu oturum; `sabit_masraf/neden/hiza_spec.csv`):

| | birleştirme kapalı | birleştirme açık |
|---|---|---|
| tampon hizalaması 32 (eski) | 64.65 tok/s | 60.65 (%-6.2; ilk tur 51.89 aykırı, medyan 62.81) |
| tampon hizalaması 4096 (yeni) | 64.38 tok/s | **67.81 (%+4.9)**, beş tur 67.5-67.95 |

Hedefin doğrulama süresi sırayla 1225, 1379, 1235 ve 1214 ms.
Düzeltmeyle birleştirme tahminli üretmede yavaşlatmıyor, kazandırıyor. Önceki ölçüm (hizalama 32, k=2): tüm birleşik %-3.9. O ölçümün ham verisi `sabit_masraf/final/final_spec_k2.csv`, artık eski.

**Neden.** Kök nedeni ayrı bir Sonnet 5.5 ajanı kodu okuyarak ve ölçerek buldu (88 araç çağrısı). Düzeltmenin sonucunu ben bağımsız olarak 5 turla doğruladım. Ajanın kendi GPU süresi ölçümlerini (23.8 ve 26.4 ms gibi) ben yeniden ölçmedim. Ayrıntı `sabit_masraf/neden/neden_analiz.md`.
Zincir:
1. Ayırıcı (ggml'in ara tensörlere bellek adresi dağıtan kodu), önceki grafın yerleşimini düğüm sıra numarasına göre yeniden kullanıyor (`ggml-alloc.c:1008-1049`, `1083-1094`). Koşu başında en son 512 sütunlu graf rezerve ediliyor, tüm graflar onun yerleşimini alıyor. Birleştirme kapalıyken koşu boyunca sıfır yeniden ayırma oldu.
2. Ekleme birleştirmesi açıkken çizge düzenleme (`ggml_graph_optimize`, ayırıcıdan önce çalışıyor) hedefin tek token'lık grafının düğüm sırasını değiştiriyor. Yerleşim uymuyor. Ayırıcı yerleşimi iki kez yeniden hesaplıyor: tek token'lık grafta ve ilk 3 sütunlu doğrulama grafında. Sonra llama.cpp grafı yeniden kullandığı için o yerleşim koşu sonuna kadar kalıyor.
3. Yeni yerleşim tam boyutlu ve sıkı. Tensörler bitişik duruyor. Bellek aralığı kontrolü (`ggml-metal-common.cpp:140`) bitişik aralıkları çakışma sayıyor. Bu yüzden 3 sütunlu grafta 30 fazladan bellek bariyeri çıkıyor (421 → 451). Bariyer, GPU'nun önceki işlerin yazdığı belleği okumadan önce beklediği noktadır. Maliyeti yaklaşık 0.3 ms (iki ölçümün farkından çıktı, doğrudan ölçülmedi).
4. Asıl kayıp bariyerden değil, adres hizasından geliyor (yaklaşık 2.3 ms, grafın %9'u). Yeni yerleşimde matris çarpımlarının girdi vektörünün adresi 1024 bayta hizalı değil. Matris çarpımı çekirdekleri bu adrese duyarlı. Ayrıntısı madde 8'de.

**Düzeltme.** Metal tampon türlerinin (shared ve private) hizalaması 32'den 4096'ya çıkarıldı (`ggml-metal.cpp`, `get_alignment`). Ortam değişkeni `GGML_METAL_BUFFER_ALIGN`, varsayılan 4096. Ayırıcı bu değeri arka ucun ihtiyacı olarak alıyor, yerleşim nereden gelirse gelsin her aktivasyon aynı hizaya düşüyor. Mapped tür (ağırlıklar) değişmedi.
- Bariyer sayısı da düşüyor: 3 sütunlu grafta birleştirme açıkken 395 yerine 365.
- Tek token hızı değişmiyor. Kapalı için hizalama 32 ile 4096 arası fark +0.03 (0.5B), +0.03 (1.5B), +0.07 ms (7B), gürültü içinde. Birleştirme kazancı iki hizalamada aynı: 0.5B -0.42 ve -0.42, 1.5B -0.44 ve -0.47, 7B -0.58 ve -0.64 ms (5 tur, hepsi 5/5). Yani madde 3'teki tek token sayıları hizalamadan etkilenmemiş. Ham veri `sabit_masraf/neden/hiza_tek.csv`.
- Bellek maliyeti: tensör başına en çok 4 KB boşluk. Ölçülmedi.

**Bilmediğimiz.**
- 1 KiB periyodun donanım nedeni ölçülmedi. Belgelerden bir çıkarım var: GPU çekirdeği başına veri önbelleği 8 KB ve önbellek satırı 128 bayt (`apple-silicon-docs/gpu/metal-benchmarks/README.md`, "On-Chip Memory" tablosu). Bu 64 satır eder. Önbellek 8 yollu ise 8 küme olur ve adres her 1024 baytta bir aynı kümeye düşer. Bu ölçülen periyotla uyuşuyor. Ama belgelerde ilişkisellik (kaç yollu) yazmıyor. Bu bir hipotez.
- 0.5B ve 1.5B modelde tahminli üretme düzeltmeden sonra ölçülmedi.
- Ayırıcının sıra numarasıyla uygulama davranışı (`ggml_gallocr_needs_realloc` tensör kimliğine bakmıyor) başka graflarda yanlış tensöre yer verebilir mi, bilinmiyor. Kod şüphesi, ölçülmedi. Ayrı iş.

**Denenip işe yaramayanlar.**
- Bitişik aralıkları çakışma saymayan sıkı kontrol (`>=` yerine `>`): bariyer sayısını düşürüyor ama hızı düzeltmedi. Bariyer maliyeti küçük. Ham veri `sabit_masraf/spec/spec_sikikenar.csv`. Değişiklik geri alındı.
- Birleştirmeyi yalnız saf tek token'lık graflara kısıtlamak: kayıp kaldı.
- `GGML_METAL_FUSE_MAX_NODES=900` ile yalnız taslakta birleştirme: kazanç veriyordu (%+3.6) ama düğüm sayısına dayanan bir ayrımdı. Artık gerekmiyor. Ortam değişkenleri deney aracı olarak yamada duruyor.

**k taraması** (tampon hizalaması 4096, 5 tur, 96 token, ölçen bu oturum; `sabit_masraf/neden/k4_k3.csv`, `k4_k4.csv`, `hiza_spec.csv`). Metin aynı yapılandırmada kapalı ve açıkta birebir aynı. k değerleri arasında metin aynı değil: k=1 ve k=3 aynı metni, k=2 ve k=4 ise ayrı birer farklı metni üretti. Nedeni ölçülmedi, muhtemelen batch şekli değişince sayısal sonuç değişiyor.

| k (doğrulama grafı) | kapalı | açık | fark | kabul | doğrulama adımı başına (kapalı, yaklaşık) |
|---|---|---|---|---|---|
| 1 (2 token) | 67.77 tok/s | 69.21 | %+2.1 (5/5 tur) | %66 (ilk tur) | ~21.4 ms |
| 2 (3 token) | 64.38 tok/s | 67.81 | %+4.9 | %58 | ~27.4 ms |
| 3 (4 token) | 53.13 | 54.81 | %+3.2 | %49 | ~37.2 ms |
| 4 (5 token) | 55.87 | 60.04 | %+7.5 | %45 | ~37.8 ms |

- k=1 hizalama 4096 ile de ölçüldü (ölçen bu oturum, 5 tur, metin kapalı ve açıkta aynı; `sabit_masraf/neden/k6_k1.csv`): kapalı 67.77, açık 69.21 tok/s, fark +1.44 (%+2.1), açık 5/5 turda hızlı. Hizalama 32'deki eski ölçüm: 66.95 ve 68.73 (%+2.7).
- En hızlı k bu ölçümlerde 1 (açık 69.21), sonra 2 (67.81). Çok sütunlu birleştirmeden sonraki güncel tablo aşağıda (en hızlı yine k=1, 70.34). Önceki "k=2 en hızlı" yargısı k=1'in eski hizalamayla ölçülmesine dayanıyordu. Dikkat: k=1 ve k=2 farklı metin üretti, yani iki satır birebir aynı işi yapmıyor. Aynı metinle (örneğin hedefin kendi üretimiyle) karşılaştırılmadı.
- Hedef modelin tahminsiz hızı bu tablodaki k değerleriyle aynı koşulda ölçülmedi. Tahminli üretmenin asıl kazancı (tahminsize göre) bu yüzden bu tabloda yok.
- k=4 açıkta 60.04, k=2'nin altında.
- Adım başına süre 3 token'dan 4 token'a geçerken ~27'den ~37 ms'ye çıkıyor, 4 ve 5 token'da aynı kalıyor. Bu, küçük batch yamasının 4 token ve üstünde matris çarpımı (simdgroup matrix) yoluna geçmesiyle uyumlu. Bu yolun 4 token'da da yavaş olduğu daha önce bulunmuştu. Bu koşuda yolu doğrudan ölçmedim, adım süresi yaklaşık değer (n_drafted / k).
- Kabul oranı k büyüdükçe düşüyor (%58, %49, %45).
- `GGML_METAL_SB=s` (2-8 token'da yalnız skaler yol) k=3 ve k=4'ü hızlandırmıyor, çok yavaşlatıyor (5 tur, birleştirme açık, metin aynı; `sabit_masraf/neden/k5_k3.csv`, `k5_k4.csv`):

| k | varsayılan yol | yalnız skaler | adım başı süre (varsayılan / skaler) |
|---|---|---|---|
| 3 | 54.94 tok/s | 29.48 (%-46.3) | ~37 / ~75 ms |
| 4 | 60.18 tok/s | 22.97 (%-61.8) | ~37 / ~110 ms |

  Skaler yolun maliyeti sütun sayısıyla hızla artıyor. 4 ve üstü token'da matris yolu doğru seçim. Eski fikir çürüdü.
- 4 token'da matris yolu 3 token'lık skaler yoldan ~10 ms pahalı (27 ve 37 ms). "4 sütunluk blok yazılırsa k=3 iyileşir" fikri kodu okuyunca geçersiz çıktı (kod okuması, ölçülmedi; `ggml-metal.metal` `kernel_mul_mm_sb_q4_0_f32`): matris birimi her çağrıda 8 satır x 8 sütun x 8 derinlik çarpıyor ve 4, 5 token'da aynı sayıda çarpma yapıyor. Yarı dolu sütunlar çarpma sayısını artırmıyor. 4 ve 5 token'ın adım süresinin aynı çıkması (37.2 ve 37.8 ms) bununla uyumlu. Yani 4 sütunluk blok MMA sayısını azaltmaz. Yolun sabit maliyeti, her ağırlık parçasını yarım duyarlıklı parçaya açıp ayrı çarpmaya vermekten geliyor olabilir. Bu bir hipotez. Bu maliyeti düşürmek ayrı bir çekirdek işi.

**Doğrulama grafı için çok sütunlu birleştirme: yapıldı** (kod ve ölçüm bu oturum; yama `sabit_masraf/yama.diff`; ham veri ve betikler `sabit_masraf/sb_birlestirme/`).

Önceden tek token'a (ne11=1) bağlı olan ekleme, gate+up+glu ve q/k/v birleştirmeleri, küçük batch çekirdeklerine (2-8 token) genişletildi. Ortam değişkeni `GGML_METAL_FUSE_SB=0` bunları kapatır, varsayılan açık.

GPU çağrısı sayısı (`GGML_METAL_FUSE_STATS=1`, koddan sayılıyor, 1 koşu; p=2..4 değerleri 1.5B ile ölçüldü, graf düğüm sayısı 957, 7B ile aynı):

| graf | birleştirmeler kapalı | önceki durum | şimdi |
|---|---|---|---|
| taslak 0.5B, 1 token | 508 | 224 | 224 |
| hedef 7B, 1 token | 592 | 257 | 257 |
| hedef, 2-3 token (skaler yol) | 592 | 505-508 | 367 |
| hedef, 4 token (matris yolu) | 592 | 505-508 | 257 |

**Ne kaldı, ne gitti.**
- Skaler yolda (2-3 token) yalnız ekleme kaldı. Matris yolunda (4-8 token) ekleme, qkv ve glu üçü de kaldı.
- Skaler glu çıkarıldı. Satır sayısını değiştirmek kurtarmadı (7B, 3 tur, kapalı ilk tur ısınma nedeniyle dışarıda, kapalı referans p=3'te 23.73 ms, p=2'de 18.75 ms): p=3 için 1/2/3/4 satır 27.08 / 25.61 / 28.42 / 32.39 ms. p=2 için 19.63 / 18.57 / 20.92 / 22.49 ms. Satır sayısı arttıkça yavaşlıyor. Nedeni bilinmiyor. Kayıt baskısı hipotezini `maxTotalThreadsPerThreadgroup` ile sınadım, tüm varyantlarda 1024 çıktı, yani bu gösterge hipotezi ne doğruladı ne çürüttü.
- Skaler qkv çıkarıldı. Yalnız ekleme ile ekleme+qkv karşılaştırması p=3'te iki ölçümde de qkv aleyhine çıktı: +0.31 ms (3 tur) ve +0.74 ms (6 tur, 0/6). p=2'de +0.73 ms (3/6 tur), gürültülü. İkinci ölçüm sırasında makinede video açıktı (GPU paylaşımı), temiz makinede tekrarlanmadı. Nedeni bilinmiyor.

**Karar tablosu** (7B, 2-5 token'lık tek graf süresi, `llama-bench -p N -n 0`, 4 tur dönüşümlü, ben ölçtüm; `sb_karar.csv`). Bu ölçüm sırasında makinede video açık olmuş olabilir. Satırlar kapalıya göre fark (ms) ve kapalıdan hızlı tur sayısı:

| token | yalnız ekleme | ekleme+qkv | hepsi (o günkü sürümde skaler glu de vardı) |
|---|---|---|---|
| 2 (skaler) | -0.09 (3/4) | +1.21 (1/4) | -0.02 (2/4) |
| 3 (skaler) | -0.49 (3/4) | -0.44 (4/4) | +3.27 (0/4) |
| 4 (matris) | -0.63 (4/4) | -1.77 (4/4) | -2.06 (4/4), %-6.1 |
| 5 (matris) | -0.46 (4/4) | -0.40 (4/4) | -0.85 (4/4), %-2.6 |

2 token'lık ölçümler gürültülü: kapalı turları 18.6-20.5 ms arasında. Bu yüzden 2 token'da ±0.7 ms altı fark ayırt edilemiyor.

**Doğruluk.** Perplexity (wiki.test.raw, `-b N -ub N`, birleştirme kapalı ve açık) 4 ondalığa kadar aynı çıktı:
- 1.5B nihai sürüm, 6 tur: 2 ve 3 token 10.4081, 4 ve 5 token 10.4077 (ayrıca 8 token 10.4077, ilk sürümle).
- 0.5B (8 parça, 2 ve 3 token) 17.2897 ve 7B (3 parça, 2 ve 3 token) 7.2102 ilk sürümle ölçüldü, yani skaler glu ve qkv vardı. Nihai sürümle 0.5B ve 7B ayrıca ölçülmedi.
- Tahminli üretme çıktı metni her k'de kapalı ve açıkta birebir aynı.

**Uçtan uca tahminli üretme** (7B hedef + 0.5B taslak, 96 token, 5 tur dönüşümlü, temiz makine, ben ölçtüm; `spec_son_k1..k4.csv`). "Önceki" = `GGML_METAL_FUSE_SB=0`, yani yalnız tek token ve rope birleştirmeleri:

| k | önceki | şimdi | fark | adım süresi, önceki | adım süresi, şimdi | kabul oranı | adım başına token |
|---|---|---|---|---|---|---|---|
| 1 | 68.62 tok/s | 70.34 | %+2.5 | 24.38 ms | 23.78 ms | %65.5 (38/58) | 1.67 |
| 2 | 67.98 | 68.89 | %+1.3 | 32.04 ms | 31.61 ms | %57.8 (52/90) | 2.18 |
| 3 | 54.41 | 57.16 | %+5.1 | 45.71 ms | 43.51 ms | %48.7 (57/117) | 2.49 |
| 4 | 59.48 | 60.74 | %+2.1 | 47.55 ms | 46.57 ms | %45.0 (63/140) | 2.83 |

Kabul oranı = kabul edilen taslak token / üretilen taslak token (`llama-speculative` çıktısındaki n_accept / n_drafted). Beş turda ve kapalı/açıkta aynı çıktı, çünkü metin aynı. Adım başına token = üretilen token / adım sayısı (adım sayısı = n_drafted / k). Adım süresi = `decoded N tokens in S seconds` süresi / adım sayısı, yani taslak üretme, doğrulama ve örnekleme birlikte. Beş koşunun ortalaması.

- En hızlı ayar k=1, 70.34 tok/s.
- "Önceki" sütunu eski ölçümlerle (açık: 69.21, 67.81, 54.81, 60.04) %1 içinde uyuşuyor. Bu, önceki durumun ikinci ölçümü.
- Video açıkken yapılan ilk deneme gürültülüydü (k=2 kapalı turları 31-52 tok/s) ve kullanılmadı. Veri `spec_son_gurultulu_k*.csv`.

**Açık.**
- Skaler qkv'nin temiz makinede tekrar ölçümü.
- Matris yolunda glu için satır bloğu 1 seçildi (`tm=1`), 2 ölçülmedi.
- Skaler glu ve qkv'nin neden yavaşladığı bilinmiyor.

### 4-8 token'lık doğrulama geçişini hızlandırma (12 Eki)

Hedef: 7B'de 4-5 token'lık tek grafı 1 token tabanına (17.7 ms) yaklaştırmak. Başlangıç 31.2-31.3 ms. Tüm sayılar 7B, `llama-bench -p N -n 0 -b N -ub N -r 10`, temiz makine, ben ölçtüm.

**Kalıcı iki değişiklik:**

| değişiklik | p=4 | p=5 | ölçüm |
|---|---|---|---|
| başlangıç | 31.17 ms | 31.30 ms | 5 tur |
| K sırası değişti: x float2 ile yükleniyor, ağırlık dördü tek kaydırmayla açılıyor | 29.15 | 29.34 | 5 tur, eski ve yeni ayrı derleme |
| down, o ve qkv'de K bölme (`KS=4`: 4 simdgroup aynı satırları K'nin dörtte birleriyle işler, sonda tek bariyerle toplanır) | 26.26 | 26.42 | 4 tur dönüşümlü, aynı ikili (`GGML_METAL_SB_KS=1/4`), 4/4 tur hızlı |

- K sırası: MMA'nın K ekseni serbest etiket. Konum P(j,c) = 16·(j>>1) + 2c + (j&1) seçilince kanal başına x iki float2 ile, ağırlık iki bitişik ushort ile geliyor. Ağırlık başına dequant komutu da azaldı.
- K bölme: geçmiş sayaçlarda (`main_session/TUR1-DOGRULAMA.md` bölüm 19.1) down 56 grupla doluluk %12, F32 %61'de kalıyordu. Gate/up'a uygulanmadı: doluluğu zaten %41.6 (belge: ALU kullanımı 24 simd/çekirdekte doyuyor).
- Doğruluk: 1.5B perplexity (6 parça, `-b 4`) 10.4083 (K sırası) ve 10.4082 (K bölme). Eski 10.4077. Fark toplama sırasından.

**Uçtan uca tahminli üretme** (aynı düzen: 7B + 0.5B taslak, 96 token, 5 tur, temiz makine, ben ölçtüm; `~/claude-scratch/bant-darbogaz/yeni_duzen/spec_ks4_k*.csv`). "Eski" = yukarıdaki tablonun "şimdi" sütunu, yeniden ölçülmedi. Metin her k'de eskisiyle bayt bayt aynı.

| k | eski | şimdi | fark | adım süresi eski → şimdi |
|---|---|---|---|---|
| 1 | 70.34 tok/s | 70.21 | %-0.2 | 23.78 → 23.82 ms |
| 2 | 68.89 | 68.81 | %-0.1 | 31.61 → 31.65 ms |
| 3 | 57.16 | 64.68 | %+13.2 | 43.51 → 38.45 ms |
| 4 | 60.74 | 67.61 | %+11.3 | 46.57 → 41.85 ms |

k=1-2 skaler yolu kullanıyor, değişmedi. En hızlı ayar hâlâ k=1.

**Çürüyen ya da işe yaramayanlar (bugün):**
- 4-5 sütunlu skaler çekirdek: p=4 36.8, p=5 48.3 ms. Geçmişte de çöpe gitmişti (bölüm 15, "skaler n4-n8"). Tekrar edildi.
- 4 bloğu art arda yüklemek: +1.5 ms. Geçmişteki v5 u4 ile aynı sonuç. Tekrar edildi.
- Blok çifti için tek 8 baytlık `packed_ushort4` yükleme + `simd_shuffle_xor`: +1.4 ms. Geçmişteki v3/v4'e benzer. Tekrar edildi.
- Ablasyon dersi: "yalnız yükleme" varyantı (MMA, x, dequant yerine ucuz işlem) 22.7 ms verdi, ama yerine konan işlem de ALU komutu. Hata günlüğündeki uyarı (KNOWHOW bölüm 8) burada da geçerli: işi çıkaran varyantla darboğaz teşhisi konmaz.

**Kalanın yapısı (hesap, ölçüm değil):** MMA 5 token için de 8 sütun hesaplıyor. Ölçülmüş MMA tepesiyle (6.3 TFLOPS, `SONUCLAR.md`) yalnız MMA ~16.6 ms, okuma ~14.2 ms. Gate/up'ta komutların ~2/3'ü MMA. Bu yolla 18 ms'ye inilmez; gerçekçi tavan 24-25 ms.

**Kaydırma azaltma (sonradan eklendi).** Belgede 32 bitlik kaydırma çeyrek hızda (Apple 7/8: 32/çevrim, bit işlemi 128). Yüksek dördün kaydırmadan `0x00F000F0` maskesiyle (1024+16q) alınıp x/16 ile ölçeklenmesi blok başına kaydırmayı 3'ten 1'e indirdi. Sessiz pencerede 10 tur dönüşümlü: p=4 ortanca -0.08 ms (6/10), p=5 -0.14 ms (8/10). Perplexity 10.4079. Varsayılan yapıldı.

**Denenip bırakılanlar (aynı gün):** x'i bir blok önden yüklemek +1.16 ms; T2'yi half biriktirmek -0.06 ms (ayırt edilemez, doğruluğu değiştiriyor).

**Deney anahtarları temizlendi.** `GGML_METAL_SB_NM`, `GGML_METAL_SB_SMAX`, `GGML_METAL_SB_V`, skaler n4/n5 çıkarıldı. Kalan ayar: `GGML_METAL_SB_KS` (varsayılan 4). Yama `sabit_masraf/yama.diff` yeniden üretildi, temiz tabana (e475fa2b5 + `apply_patch.py`) uygulanınca 10 dosya çalışma kopyasıyla birebir aynı. Temiz sürümde 1.5B perplexity: 2-3 token 10.4081, 4-5 token 10.4079.

**Bağlam boyutu.** `llama-speculative` varsayılan olarak hedefte 131072, taslakta 32768 token'lık bağlam açıyor. Gürültülü makinede k=2 `-c 2048` ile iki turda ~71 tok/s, varsayılanla 64.7-69.3. Temiz ölçüm aşağıda.

**Ağaç taslak (boş MMA sütunlarını kullanmak).** `llama-speculative -np N --spec-draft-p-split P` ağaç modu bu sürümde üç yerden bozuktu, `sabit_masraf/yama_speculative_agac.diff` ile düzeltildi:
- 0. dalın örnekleyicisi her adımda hedefin açgözlü örnekleyicisinden kopyalanıyordu, dal olasılıkları hep 0 kalıyordu.
- Yeni `dist` örnekleyicisi aday olasılıklarını doldurmuyor; taslakta softmax elle hesaplanıyor (yalnız np>1).
- Birleşik olmayan önbellekte `seq_cp` akışlar arası çalışmıyor ("failed to find a memory slot"), metin bozuluyordu; np>1'de `kv_unified` açılıyor. Bu sorun yamamız kapalıyken de vardı.
- Düzeltmeden sonra metin aynı k'deki zincirle bayt bayt aynı. Adım başına kabul arttı (k=2: 2 dalla taslak 66, kabul 53; zincirde 90/52). Hız tek koşuda zincir k=1 civarında. Temiz ölçüm aşağıda.

**Bağlam ve ağaç, temiz ölçüm** (sessiz pencere, 3 tur dönüşümlü, ortanca, ben ölçtüm; `~/claude-scratch/bant-darbogaz/agac/pencere2.csv`). Metin her ayarda aynı k'deki zincirle aynı:

| ayar | varsayılan bağlam | `-c 2048` |
|---|---|---|
| zincir k=1 | 68.54 tok/s | 72.18 (%+5.3) |
| zincir k=2 | 68.17 | 70.61 (%+3.6) |
| zincir k=3 | 65.07 | 66.53 (%+2.2) |
| zincir k=4 | 68.52 | 70.58 (%+3.0) |
| ağaç k=2 / k=3 / k=4, 2 dal | — | 68.57 / 69.50 / 68.78 |

- En hızlı ayar: zincir k=1, `-c 2048`, 72.2 tok/s.
- Ağaç adım başına kabulü artırıyor ama taslak tarafındaki ek iş yüzünden zinciri geçmiyor.
- Bağlamın neden etkili olduğu ölçülmedi. `-c 8192` ölçülmedi.

**Bağlam taraması** (k=1, sessiz pencere, 3 tur, ortanca; `dogrulama_cekirdegi/pencere3.txt`): `-c 2048` 72.00, `8192` 71.77, `32768` 71.23, varsayılan (131072) 69.75 tok/s. Kazanç bağlamla düzgün değişiyor. Öneri: `-c 8192`.

**Skaler yolda kaydırma azaltma** (2-3 token, `sb_deq2m`: yüksek dörtler maskeyle, ölçek d/16 ile). Sessiz pencere, 10 tur dönüşümlü, eski ikili `bin_A`: p=2 +0.01 ms (5/10), p=3 -0.15 ms (8/10). Perplexity 1-3 token 10.4081 (değişmedi). Tutuldu.

**Açık:**
- Bağlamın neden etkili olduğu bilinmiyor (k=1'de varsayılan ile 8192 arası adım başına ~0.5 ms). Ölçülmedi.

### Adım dökümü ve taslakta boşa giden forward (13 Eki)

**Adım dökümü** (`llama-speculative`'e eklenen sayaçlar, her `llama_decode` sonrası `llama_synchronize` ile GPU beklenerek; 7B + 0.5B, `-c 8192`, 96 token, 3 tur, ben ölçtüm; `sabit_masraf/taslak_atlama/adim_dokumu.txt`). Adım başına ms:

| k | hedef decode | taslak decode | taslak/adım | hedef örnekleme | taslak örnekleme | KV + diğer | toplam |
|---|---|---|---|---|---|---|---|
| 1 | 18.3 | 5.2 | 2 | 0.4 | 0.1 | 0.2 | 24.2 |
| 4 | 26.4 | 12.8 | 4.9 | 0.7 | 0.3 | 0.2 | 40.4 |

Döngüde gizli masraf yok: adım süresi hedef + taslak + 1 ms. Taslak forward'ı 2.6 ms, llama-bench'le aynı. Taslak adım başına k+1 kez çalışıyor: k taslak token + hedefin düzelttiği token.

**Boşa giden forward.** Son taslak token'ının çıktısı hiç kullanılmıyor (döngü ondan sonra kırılıyor) ama taslak modelde yine de hesaplanıyor. Bu forward atlandı; yalnız bütün taslaklar kabul edilirse o token, hedefin düzelttiği token'la birlikte sonraki forward'a ekleniyor (2 token'lık batch). Ortam değişkeni `SPEC_SKIP_LAST=1`, zincir modunda (np=1). Metin her k'de bayt bayt aynı.

Sonuç (5 tur dönüşümlü, ortanca tok/s, GPU beklemesiz; `sabit_masraf/taslak_atlama/skip_k1_6.csv`):

| k | kapalı | açık | fark | tur |
|---|---|---|---|---|
| 1 | 69.96 | 72.46 | %+3.6 | 5/5 |
| 2 | 70.88 | 71.87 | %+1.4 | 5/5 |
| 3 | 66.32 | 67.40 | %+1.6 | 5/5 |
| 4 | 70.00 | 71.33 | %+1.9 | 5/5 |
| 5 | 70.86 | 72.11 | %+1.8 | 5/5 |
| 6 | 68.70 | 71.64 | %+4.3 | 5/5 |

- GPU beklemeli ölçümde kazanç daha büyüktü (k=1 %+8, k=4 %+6). Beklemesiz koşuda boşa giden forward, CPU'nun hedef batch'ini kodlama süresiyle kısmen çakışıyormuş; o yüzden net kazanç küçük.
- k=1 ve k=5 artık başa baş (72.5 / 72.1). k=4-5'in k=1'i geçmesi için kalan iki kol: taslak forward'ının 2.6 ms'si (bunun ~1.2 ms'si sabit masraf, madde 1) ve %45 kabul oranı.

**Taslak olarak 1.5B** (q4_0, atlama açık, 3 tur, ortanca tok/s; `sabit_masraf/taslak_atlama/taslak15b.csv`): k=1 73.6, k=2 70.3, k=3 67.6, k=4 69.8, k=5 75.1, k=6 75.6. Metin 0.5B taslakla aynı (hedef açgözlü). Taslak forward'ı 5 ms ama kabul daha yüksek; k=5-6 ilk kez k=1'i geçti. Kabul oranı günlüğe yazılmadı (ayıklama hatası), p-min taramasında yazılıyor.

**Taslakta "emin değilsen dur" (p-min).** `--spec-draft-p-min` bu sürümün örnek programında hiç kullanılmıyordu. Eklendi: taslağın en iyi token'ının olasılığı eşiğin altındaysa zincir orada kesilir, hedef daha kısa batch doğrular. Olasılık ham logit'ten hesaplanıyor; açgözlü örnekleyicinin aday listesi diğer logit'leri sildiği için oradan okununca hep 1.0 çıkıyordu (ilk deneme bu yüzden etkisizdi). Sonuç (3 tur, ortanca tok/s, atlama açık; `sabit_masraf/taslak_atlama/pmin2.csv`):

| taslak | k | p-min 0 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|---|
| 0.5B | 5 | 72.6 | 75.9 | 74.8 | 69.8 |
| 0.5B | 6 | 71.6 | 74.4 | 74.8 | 70.3 |
| 0.5B | 7 | 65.6 | 69.9 | 75.3 | 70.8 |
| 1.5B | 5 | 75.4 | **79.2** | 67.3 | 67.3 |
| 1.5B | 6 | 75.3 | **79.3** | 65.7 | 67.4 |
| 1.5B | 7 | 69.4 | 74.2 | 66.1 | 68.5 |

- 1.5B k=5 p-min 0.3: kabul edilen token sayısı aynı (76), taslak forward'ı 120'den 100'e indi. Yani 0.3 eşiği yalnız boşa gidecek taslakları kesiyor. Metin her ayarda aynı.
- 0.5 ve üstü kabulü düşürüyor (1.5B k=5: 76 → 66 → 60).
- k=8 denenmedi: 9 token'lık batch küçük batch çekirdeğinin dışına düşüyor, 33 tok/s.
- Eski en iyi k=1 0.5B 72.5; yeni en iyi 1.5B k=5-6 p-min 0.3 ile 79.2, %+9.

**Dikkat: 96 token'lık tek metin yanılttı.** İnce tarama (5 tur, `pmin3.csv`) 1.5B k=5-6 p-min 0.3 için 78.1-78.6 verdi. Ama ikinci bir metinde (Python web sunucusu açıklaması, 96 token) sıra tersine döndü: 0.5B k=1 76.5, 1.5B k=5 p-min 0.3 69.9. Üç metin, 256 token, 2 tur kıyas (`cok_metin.csv`, ortalama tok/s):

| ayar | deneme metni | web sunucu | öykü | ortalama | kabul (256 token) |
|---|---|---|---|---|---|
| 0.5B k=1 | 72.4 | 73.4 | 72.4 | **72.7** | %65-66 |
| 0.5B k=2 p0.3 | 71.8 | 70.5 | 65.9 | 69.4 | %55-62 |
| 0.5B k=5 p0.3 | 72.8 | 68.6 | 63.6 | 68.3 | %43-48 |
| 1.5B k=2 p0.3 | 68.3 | 65.9 | 61.7 | 65.3 | %65-71 |
| 1.5B k=5 p0.3 | 72.1 | 61.2 | 57.0 | 63.5 | %51-65 |

- Deneme metninin ilk 96 token'ı (giriş cümleleri) olağandışı kolay tahmin ediliyor; 256 token'da ve başka metinlerde kabul düşüyor. 79 tok/s sonucu o metne özgü, genel değil.
- Üç metinde de en hızlı ayar 0.5B k=1. k=4-5 bu taslaklarla k=1'i geçmiyor; nedeni kabul oranı (%43-65), çekirdek değil.
- Ders: tahminli üretme ölçümü en az 3 metin ve 256 token ister. Bundan sonraki uçtan uca ölçümler bu düzenle yapılır.

## 7. Ölçüm güvenilirliği ve gürültü sınırı

**A/A kontrolü.** Aynı yapılandırmayı kendisiyle karşılaştırdık. Böylece testin ayırabileceği en küçük fark görülüyor. 8 tur, dönüşümlü sıra, `sabit_masraf/adim/aa_test.csv`.

| model | tur farkı std | en büyük tek tur farkı | 3 turluk ortalamanın hatası (1σ) |
|---|---|---|---|
| 0.5B | 0.013 ms | 0.028 ms | ~0.008 ms |
| 7B | 0.147 ms | 0.236 ms | ~0.085 ms |

- 0.5B'de 0.03 ms üstündeki farklar güvenilir.
- 7B'de 3 turluk ikili ölçüm 0.17 ms altını ayıramıyor. Aynı yapılandırma kendisine karşı bir turda 0.24 ms fark gösterdi.
- Bu yüzden 7B'deki adım adım "yavaşlamalar" (+0.13 ile +0.17 ms) kanıt sayılmaz. Büyük farklar (toplam kazanç, tahminli üretme kaybı, norm maliyeti) bu sınırın çok üstünde.

**Kurallar.**
- Bir adımı açıp kapatma kararı, gürültü sınırının altındaki ikili ölçüme dayanmaz. V adımı bu hatayla yanlış kapatılmıştı, 8 turla düzeltildi.
- Küçük etkiyi çözmek için tur sayısı artırılır (0.5B'de ucuz). 7B için toplam fark ölçülür.
- Ölçüm programları `#!/bin/bash` ile koşar. Elle yazılan ortam değişkeni döngüleri zsh'de yanlış çıktı: zsh tırnaksız değişkeni boşluktan bölmüyor, birden çok değişkenin yalnız ilki uygulanıyor. Çağrı sayımının ilk sonuçları bu yüzden tutarsız çıktı, bash'te yeniden ölçüldü.
- Çıktı eşitliği kontrolü (48 veya 200 token greedy) hata yakalar ama zayıf. Perplexity tek token yolunda ölçülmeli (`-b 1 -ub 1`).
- Bellek yerleşimi ve adres hizası hızı değiştirebiliyor (madde 8). Bir yapılandırma farkı ölçülürken tampon adreslerinin hizası da yapılandırmayla birlikte değişmiş olabilir. Beklenmedik bir yavaşlama ya da hızlanma görülürse önce adres hizasına bakılır.
- Kök neden aramayı ölçerek yürütmek yavaş kaldı. Bu kez kodu okuyan ajan nedeni buldu. Varsayım kodu okuyarak bulunur, ölçüm yalnız o varsayımı doğrular.

## 8. Aktivasyon adresinin 1024 bayta göre kalanı matris çarpımı hızını değiştiriyor

**Gözlem** (ajan ölçtü, ben yeniden ölçmedim; 3 sütunlu hedef grafı, 957 düğüm, kayma taraması değer başına 1 koşu; `sabit_masraf/neden/neden_analiz.md`).
Tüm ara tensörleri sabit bir bayt sayısı kaydırınca, yerleşim ve bariyerler aynı kalmasına rağmen grafın GPU süresi değişiyor:

| kaydırma (bayt) | 0 | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 | 3072 | 3840 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU süresi (ms) | 26.78 | 29.36 | 28.96 | 30.72 | 26.77 | 29.24 | 28.85 | 30.70 | 26.77 | 26.66 | 30.69 |

Periyot 1024 bayt. Süre 26.7 ile 30.7 ms arasında oynuyor, yani grafın %15'i. 256 baytlık kayma 2.6 ms ekliyor.

**Yeri.** İşlem atlama deneyiyle kayıp yalnız matris çarpımlarında bulundu (hepsi atlanınca fark -0.09). Kaybın olduğu çarpımlar Q/K/V, o-proj ve ffn_down. ffn_gate ve ffn_up'ta fark yok. Bu iki grup, girdi adreslerinin 1024'e hizalı olup olmamasıyla örtüşüyor (ilinti, doğrudan kanıt kaydırma ve hizalama denemeleri).

**Hizalamayı zorlama** (ajan, ayırıcı hizalaması, 3 tur, 3 sütunlu grafın GPU süresi): ekleme açık yerleşimde hizalama 32'de 26.43 ms, 256'da 26.87, 1024'te 24.12, 4096'da 23.84. Taban 23.79 ms. 256 yetmiyor, çünkü 256'ya hizalı ama 1024'e göre kalanı 512 olan adresler hâlâ yavaş.

**Tek token'lık grafta kayma taraması** (ölçen bu oturum, ajanın kopyası, birleştirme açık, hizalama 32, `GGML_ALLOC_SHIFT`, `llama-bench` 0.5B 96 token, 7B 64 token, 3 tur, değer başına koşu başına 1 tekrar; `sabit_masraf/neden/kayma_tek.csv`).

| kayma (bayt) | 0 | 256 | 512 | 768 | 1024 | 1280 |
|---|---|---|---|---|---|---|
| 7B (ms/token) | 17.097 | 17.148 | 17.005 | 17.038 | 17.013 | 16.972 |
| 7B fark | 0 | %-0.3 | %+0.5 | %+0.3 | %+0.5 | %+0.7 |
| 0.5B (ms/token) | 2.579 | 2.624 | 2.481 | 2.628 | 2.570 | 2.597 |
| 0.5B fark | 0 | %-1.7 | %+3.9 | %-1.8 | %+0.4 | %-0.7 |

- 7B: en büyük fark %0.7, yani 0.12 ms. Bu, 7B'nin A/A gürültü sınırının (0.17 ms) altında. 1024 baytlık periyot görünmüyor. 3 sütunlu grafta aynı kayma %15 oynatıyordu.
- 0.5B: gürültü yüksek (kayma 0'ın üç turu 377.8-393.5 tok/s). 512'de %+3.9 çıktı ve 3/3 turda kayma 0'dan hızlı. Ama 1024'te fark yok (%+0.4), yani 1024 periyoduyla uyuşmuyor. Tek koşu iddiadır; 512 ve 1536 ayrıca tekrar ölçülürse netleşir. Şimdilik "0.5B için bilinmiyor".
- Tekrar (ölçen bu oturum, 0.5B ve 1.5B, kayma 0/512/1024/1536, 6 tur dönüşümlü, `llama-bench -n 96 -r 3`; `sabit_masraf/neden/kayma_tek2.csv`, özet `kayma_ozet.py`):

| kayma (bayt) | 0 | 512 | 1024 | 1536 |
|---|---|---|---|---|
| 0.5B (ms/token) | 2.573 | 2.541 | 2.576 | 2.615 |
| 0.5B fark, kayma 0'dan hızlı tur | 0 | %+1.2, 3/6 | %-0.1, 3/6 | %-1.6, 1/6 |
| 1.5B (ms/token) | 4.996 | 4.985 | 4.978 | 4.990 |
| 1.5B fark, kayma 0'dan hızlı tur | 0 | %+0.2, 5/6 | %+0.3, 6/6 | %+0.1, 4/6 |

- 0.5B'deki ilk %+3.9 gürültüymüş: tekrarda 512'de %+1.2 ve yalnız 3/6 tur. 0.5B'nin tur içi aralığı ±%2.
- 1.5B: en büyük fark %0.3 (0.018 ms), yönü tutarsız. 1024 baytlık periyot görünmüyor.
- Sonuç: 0.5B, 1.5B ve 7B tek token grafları bu adres duyarlılığından etkilenmiyor. Tek token için hizalamadan kazanılacak bir şey yok. 3 sütunlu grafın %15'lik duyarlılığı tek token'da yok.

**Açık sorular.**
- Tek token'lık grafta üç model de duyarsız çıktı (yukarıdaki tablolar). Neden 3 sütunlu grafta duyarlı, tek token'da duyarsız olduğu ölçülmedi.
- Hangi çekirdek ve hangi okuma duyarlı. Yalnız sınıf düzeyinde biliyoruz.
- Çıktı (`dst`) adresinin etkisi. Tek ipucu: ffn_gate ve ffn_up çıktıları 512 kalanında durdu ve yavaşlamadılar.
- Donanım nedeni: madde 6'daki hipotez (8 KB veri önbelleği, 128 bayt satır, 8 küme).
- Ağırlık okumaları (mmap'li, 32 hizalı) için aynı duyarlılık var mı, bilinmiyor. Eğer varsa ağırlık yerleşimi de hızı etkiliyor olabilir.

## 9. Aktivasyon seyrekliği: FFN'de küçük çıkacak nöronların ağırlığını okumamak (fikir, 13 Eki)

**Fikir.** Donanım sahibinin sorusu: sıcak/soğuk nöronları önceden tahmin edip etkisiz nöronlar için hesap yapmamak. Literatürde DejaVu, PowerInfer, "LLM in a flash" bunu yapıyor: küçük bir tahminci her katmanda hangi FFN nöronlarının sıfıra yakın çıkacağını söylüyor, o satırlar okunmuyor.

**Bizim için anlamı.** Hedef adımının ~14 ms'si ağırlık okuma, FFN ağırlıkların üçte ikisi. Nöronların yarısı atlanırsa okuma 4-5 ms düşer.

**Engeller.**
- Qwen2.5 SwiGLU kullanıyor; nöronlar sıfır değil "küçük" oluyor. ReLU modellerde %90 seyreklik var, SwiGLU'da bilinen değer %10-30. Yayınlar modeli ReLU'ya çevirip yeniden eğitiyor (TurboSparse, ProSparse). Elimizde eğitim yok.
- MMA yolu satır atlamayı desteklemiyor; seyrek satır toplayan çekirdek gerekir. Skaler yolda kolay.

**Sayım** (13 Eki, 194 token'lık metin, `ffn_swiglu` çıkışı, 28 katman; `sabit_masraf/seyreklik/qwen7b.csv`): satır maksimumunun %5'i altında kalan nöron ortalama %94, %1'i altında %58. Üst yarı nöronlar mutlak değer toplamının %88'ini taşıyor. Dağılım çok çarpık; "küçük" sayısı tek başına atlanabilirliği söylemiyor.

**Kahin ölçümü** (en küçük %X nöron hesaplandıktan sonra sıfırlanıyor, perplexity wiki.test 512 bağlam 4 parça, 7B oq4; `sabit_masraf/seyreklik/oracle_ppl.csv`; `llama-perplexity` yaması `SEYREK_ORAN`):

| sıfırlanan | perplexity | fark |
|---|---|---|
| %0 | 6.883 | — |
| %30 | 6.929 | +%0.7 |
| %50 | 6.945 | +%0.9 |
| %70 | 7.627 | +%10.8 |
| %90 | 16.03 | bozuk |

Yukarıdaki "SwiGLU'da %10-30" tahminim yanlıştı: Qwen2.5-7B'de nöronların yarısı atılınca kalite neredeyse değişmiyor. Bu, tahminci olmadan hangi nöronun atılacağını bilmeyi gerektiren kahin sonucu; bayt tasarrufu için bir seçim yöntemi lazım.

**Tahminci gerektirmeyen yol.** Gate matrisi tam okunur, silu(gate) büyük olan nöronlar için up ve down satırları okunur. FFN model baytlarının %87'si (katman başına 3×3584×18944 ağırlık, attention 29M'e karşı 203M). up+down'ın yarısı atlanırsa toplam okuma ~%29 düşer, 14.2 ms okumada ~4 ms. Gate'e göre seçim (silu(gate) mutlak değeri en küçük %X sıfırlanıyor; `gate_ppl.csv`): %30 7.052 (+%2.5), %50 8.459 (+%23), %60 10.32, %70 16.31. Olmuyor; up değeri seçime girmek zorunda.

**Tahminci gerektirmeyen ikinci yol: yalnız down atlamak.** gate ve up tam okunur, silu(gate)×up hesaplanır, küçük nöronlar için down okunmaz. Bu tam olarak kahin tablosu: %50'de +%0.9. down FFN'in üçte biri; yarısı atlanınca toplam okuma ~%14 düşer, ~2 ms/token.

**Engel: bellek yerleşimi.** down ağırlığı ne=[18944, 3584]; bir nöron 3584 çıkış satırının her birinde tek bir K girdisi. q4_0 bloğu 32 ardışık K değerini paketliyor; tek nöron atlamak bayt kazandırmıyor, 32'lik bloğun tamamı atlanmalı. Çözüm: down'ı transpoze yerleşimle yeniden kuantize etmek (nöron başına 3584 ağırlık bitişik, ~2 KB), çekirdeği "seçili satırların ağırlıklı toplamı" olarak yazmak. Model dosyasında dönüşüm + yeni çekirdek gerekir; kuantizasyon blokları başka eksende olduğu için perplexity ayrıca ölçülmeli.

**Oran eşiği (satır maksimumunun yüzdesi) kapalı** (`esik_ppl.csv`): %1 eşik nöronların %56'sını siliyor, perplexity 7.239 (+%5.2); %2 → %74 silinir, 8.93. Token başına sabit "en küçük %50" seçimi çok daha iyi (+%0.9). Seçim token başına sıralamayla yapılmalı.

**Uygulama (13 Eki, gece).** Yapılanlar:
- `sabit_masraf/seyreklik/down_transpoze.py`: GGUF'ta `ffn_down` tensörlerini açıp transpoze edip q4_0 ile yeniden kuantize ediyor (nöron başına 3584 ağırlık bitişik, 2016 bayt). Şimdilik kaynak q4_0 dosyası; f16 kaynağı indiriliyor.
- Yükleyici (`SEYREK_DOWN=1`): down şekli {n_embd, n_ff}. Çizge: transpoze down için `ggml_mul_mat` düğümü, `op_params[1]` işaretli; Metal tarafı bu işareti görünce kendi yoluna sapıyor.
- Metal: `kernel_seyrek_esik2` (tek TG, 1024 iş parçacığı: 2048 kutulu logaritmik histogramla token başına eşik, sonra seçili satır indekslerinin sıkıştırılmış listesi), `kernel_seyrek_down1i` (TG = 4×112 iş parçacığı, her iş parçacığı bir q4_0 bloğu = 32 sütun, satır listesini 8'er önden yükleyerek dolaşır, grup içinde 4 alt parçayı toplar), `kernel_seyrek_topla` (parça toplamları). Çok token'lı yol için ayrı, yavaş ama doğru çekirdek (`kernel_seyrek_down`, atomikli).
- Doğruluk: transpoze model, yeni çekirdek, 4 parça perplexity (512 batch): oran 0 → 6.973, oran 0.5 → 7.040. Orijinal 6.883; +%1.3 fark q4→f32→q4 çifte kuantizasyondan.

Hız (7B, `llama-completion` 48 token, ms/token, taban 17.11; çekirdek sürümleri sırasıyla):

| sürüm | yoğun (oran 0) | %50 seyrek |
|---|---|---|
| v1 atomikli, 128 iş parçacığı × 4 sütun | 42.1 | 33.8 |
| v2 kısmi toplam, aynı yerleşim | 44.3 | 34.2 |
| down1: satır genişliğinde 112 iş parçacığı, hizasız uint4 yük | 22.5 | 20.6 |
| down1: hizalı packed_ushort4 yük | 18.7 | 18.0 |
| down1: 448 iş parçacığı, 8 satır önden, grup içi toplama | 18.8 | 18.0 |
| esik2 + indeks listesi + down1i (dallanmasız) | 18.9 | **16.4** |

- Hizasız 16 baytlık yük en büyük kayıptı (22.5 → 18.7).
- İlk eşik çekirdeği (4 geçişli radix, 256 iş parçacığı) tek başına 1.5 ms yiyordu (katman başına 53 µs); tek geçişli histogram + sıkıştırma ile değişti, ayrı ölçülmedi.
- Toplama çekirdeği 0.18 ms (atlanarak ölçüldü).
- %50 seyrek yol tabanı 0.7 ms geçiyor (%4). Yoğun yolda hâlâ 1.7 ms sabit masraf var (eşik + 3 dağıtım + bariyerler); seyreklikten gelen 2.4 ms kazanç bunun altında kalıyor.

**Masraf dökümü (ölçüldü):** eşik çekirdeği (esik2, tek TG) 0.5 ms (3 kez koşturup farktan), toplama 0.18 ms, nibble açma matematiği 0.7 ms (matematiksiz kopyayla farktan), bellek kısmı ~240 GB/s. "Eşiği atla, listeyi CPU'dan doldur" denemesi geçersiz çıktı (liste dolmamış, 13.2 ms "down hiç okunmadı" demekti).

**Denenip geri alınan:** nibble açmada half maske hilesi (`sb_deq2m`): 16.44 → 17.11, yavaşlattı.

**Eşik çekirdeği kaldırıldı:** histogram GLU çekirdeğinin sonuna gömüldü (`kernel_mul_mv_q4_0_f32_glu_hist`, global atomik 2048 kutu, katman paritesine göre iki histogram; down çekirdeği bir sonraki histogramı sıfırlıyor). Down çekirdeği (`kernel_seyrek_down1h`) eşiği histogramdan 256 iş parçacığıyla, sıkıştırmayı kendi satır aralığı için grup içinde yapıyor. Tek token perplexity (-b 1, 4 parça): yoğun 6.973, %50 7.060.

| sürüm | yoğun | %50 seyrek |
|---|---|---|
| esik2 + down1i, 32 parça | 18.9 | 16.4-16.6 |
| histogramlı, 32 parça | 18.5 | 16.9 |
| histogramlı, 64 parça | | **16.2-16.7** |

- **5 tur dönüşümlü kıyas** (96 token, `hiz_5tur.csv`): taban 17.09, %50 seyrek 16.44 (5/5 tur hızlı, %-3.8), yoğun-transpoze 18.84. Seyrek yol tabanı 0.65 ms geçiyor; tasarımın tabanı (bellek 2.1 + matematik 0.7 + toplama 0.2 = 3.0 ms down yolu) 15.8 civarı olurdu, 0.6 ms daha var.
- **f16 kaynaktan kuantize** (`Qwen/Qwen2.5-7B-Instruct-GGUF` fp16 indirildi, birleştirildi, down transpoze edilip `llama-quantize q4_0 --output-tensor-type q4_0`; dosya `qwen2.5-7b-instruct-oq4-downT-f16.gguf`, tensör türleri oq4 ile aynı): perplexity yoğun 6.863 (orijinal oq4 6.883), %50 seyrek 6.977 (+%1.4, orijinale göre). Hız 3 tur: taban 17.09-17.28, seyrek 16.42-16.75. Çifte kuantizasyon sorunu kalktı; bu dosya ana sürüm.
- **Kaydırmasız nibble açma** (belge: `apple-silicon-docs/gpu/metal-benchmarks`, M1/M2 için sağa kaydırma ~8 çevrim, bitwise and ~1; M4 verisi belgede yok): `v & 0xF / 0xF0 / 0xF00 / 0xF000` maskelenip float'a çevriliyor, ölçek (1, 1/16, 1/256, 1/4096) satır skalerine katılıyor; −8 ofseti döngüden çıkarılıp sonda 8·Σxd olarak düşülüyor. 3 tur: %50 seyrek 16.45 → 16.13-16.16, yoğun-transpoze 18.8 → 18.2-18.4. Çıktı metni bit bit aynı. Taban 17.09'a göre %-5.6.
- Belge ajanından diğer notlar: FMA16 ile FMA32 aynı hızda (fp16 hilesi için dayanak yok); ALU doyması için çekirdek başına 24 simd gerekiyor, bizde 14; threadgroup bariyeri, float atomik ve dispatch boşluğu için belgede sayı yok.
- Parça sayısı 96 ve üstü kötüleşiyor (topla büyüyor, grup başı iş küçülüyor). Koşudan koşuya gürültü ~0.5 ms; kesin kıyas için dönüşümlü 5 tur gerekir, yapılmadı.
- Oran 0.6: hız 16.1, perplexity 7.258 (+%4.1). Oran 0.5 doğru nokta (7.036, +%0.9).
- Yama: `sabit_masraf/yama_seyrek_tam.diff` (çalışma kopyasının tam diff'i, önceki yamayı da içerir). Dönüştürücü ve ölçüm araçları `sabit_masraf/seyreklik/`.

**Sonraki adım için ön ölçüm: up/gate'i de atlamak.** 442 token'lık wiki metninde katman 0, 4, ..., 24 için token başına "en büyük %50" maskeleri (`eval-callback` `SEYREK_MASKE`):
- Ardışık iki token'ın maskeleri %57-65 örtüşüyor (rastgele seçim %50 verirdi). Önceki token'dan tahmin işe yaramaz.
- Son iki token'ın birleşimi yeni token'ın seçili nöronlarının %77-87'sini kapsıyor, ama birleşim nöronların %70'i. Kazanç yok.
- Hep sıcak (%90 üstü sıklık) nöron yok; hep soğuk yalnız katman 0'da %21. Sabit sıcak/soğuk küme yolu kapalı.
- Kalan yol: girdi vektöründen (ffn_norm çıkışı) maskeyi tahmin eden küçük öğrenilmiş ağ (DejaVu / PowerInfer tarzı). Denendi, aşağıda.

**Tahminci denemesi (13 Eki, gece; araçlar `sabit_masraf/seyreklik/tahminci/`).** Veri: wiki.test'ten 12 parça, 18.3k token; her katman için FFN girişi (ffn_norm çıkışı, f16) ve hedef maske (en büyük %50). Eğitim MLX ile CPU/GPU'da saniyeler (küçük ağ; ağır hesap değil).
- İkili sınıflandırma (3584→128→18944, BCE, 4 epoch): doğrulama kapsaması katman 0 %71, katman 12 %56, katman 24 %56. Rastgele %50. 512 gizli + 8 epoch: %58. Düz lojistik regresyon: %56.
- Büyüklük dökümüyle (katman 0/8/16/24, 13k token) kütle kapsaması, seçilen %50 için:

| yöntem | katman 8 kütle / kapsama | katman 16 kütle / kapsama |
|---|---|---|
| kahin | 0.858 / 1.000 | 0.865 / 1.000 |
| rastgele | 0.500 / 0.500 | 0.500 / 0.500 |
| tam gate (gate tam okunur) | 0.695 / 0.659 | 0.718 / 0.679 |
| SVD r128 / r256 / r512 (gate ve up düşük rank) | 0.553 / 0.582 / 0.621 | 0.572 / 0.604 / 0.643 |
| eğitimli ağ r256, log|h| hedefi, MSE | 0.530 / 0.536 | 0.555 / 0.553 |

- Gate/up matrislerinde rank 512 tekil değer enerjisinin yalnız %28-36'sını taşıyor; düşük rank yaklaşımı yok.
- **Mimari taraması** (katman 8, `mimariler.py`, kütle; kahin 0.858): MLP r1024 log|h| 0.552, ağırlıklı kayıp 0.562; çift doğrusal (SwiGLU yapısını taklit, r256/512) 0.501/0.502, ağırlıklı 0.506; gate/up ayrı MLP r512 çarpımı 0.524. Hepsi rastgele düzeyinde.
- **Düşük bitli ön izleme** (gate ve up 32'lik bloklarla 2/3 bite indirilip yaklaşık h ile seçim): 2 bit 0.767, 3 bit 0.834, 2 bit gate × tam up 0.824. İşe yarayan tek şey ağırlıkların kaba kopyası. Bayt hesabı: 3 bit ön izleme 51 MB + seçili satırlar tam 38 MB + down 19 MB = 108 MB, yoğun 114 MB (%5). 2 bit: 91 MB (%20) ama kalite belirsiz. Uygulanmadı.
- **SVD başlangıçlı ince ayar** (`mimariler2.py`, çift doğrusal, ham h'ye MSE): r512 0.621 → 0.633, r1024 0.676 → 0.676 (log ağırlıklı kayıp düşürdü: 0.578), r2048 yalnız SVD 0.763. Rank arttıkça düzelen eğri; r2048 q4'te 46 MB tutar ve 2 bitlik ön izlemenin (34 MB) düzeyine ancak gelir. Eğitim eğriyi oynatmıyor.
- Neden: her tahmincinin girişinde 3584→r darboğazı var; g_i = satır_i·x ve satırlar neredeyse rastgele 18944 yön. r boyutlu altuzay bu yönlerin ancak r/3584'ünü taşır. Eğitimden bağımsız bir sınır.
- **Düşük bit ön izleme taraması** (`onizleme.py`, katman 8, kütle / h göreli hata, seçilmeyen sıfır): 1 bit 0.71 / 0.34, üçlü 0.75 / 0.29, 2 bit 0.77 / 0.27, 3 bit 0.83 / 0.18, 4 bit 0.86 / 0.15 (kahin %50: 0.143). Saf kuantizasyon (seçimsiz): q2 hata 0.545, q3 0.223. Bayt hesabı kazancı küçük tutuyor.

**Up-önce seçimi (asıl bulgu, 14 Eki).** Tek matrisle seçimde gate 0.695 kütle verirken **up 0.804** veriyor (daha önce yalnız gate denenmişti). Düzen: up tam okunur (38 MB), |u| büyük olan nöronlar için gate (19-23 MB) ve down (19-23 MB) satırları okunur; tahminci yok, seçilenler tam hassas. FFN çıkış hatası (down ile çarpılmış, `teshis.py`):

| katman | kahin %50 | kahin %60 | up %40 | up %50 | up %50 ∪ gate %50 |
|---|---|---|---|---|---|
| 0 | 0.030 | 0.054 | 0.040 | 0.076 | 0.014 |
| 8 | 0.125 | 0.181 | 0.150 | 0.214 | 0.080 |
| 16 | 0.124 | 0.181 | 0.149 | 0.213 | 0.077 |
| 24 | 0.094 | 0.134 | 0.193 | 0.248 | 0.055 |

- Perplexity (4 parça, geri çağrı `SEYREK_UP`): up %40 8.141, up %50 9.217, up %60 10.80. Hata metriğiyle uyumsuz; katman aralığına göre (2 parça, taban 8.131): up %50 yalnız 0-7 → 8.225, 8-19 → 8.348, **20-27 → 10.370**; kahin %50 aynı aralıklarda 8.133 / 8.114 / 8.145. Son katmanlar up seçimini kaldırmıyor (katman 24'te en büyük %0.5 nöronun %4.5'i kaçıyor).
- Çözüm: up seçimini katman 0-19'da uygulamak, son katmanlarda kapatmak. Perplexity (4 parça, geri çağrı): up %50 katman 0-19 → 7.338 (+%6.6); up %50 0-19 + up %30 20-27 → 7.803 (son katmanlar %30'u bile kaldırmıyor); **up %40 katman 0-19 → 7.152 (+%3.9)**, kahin %60 düzeyi.
- Metal up yolu ilk ölçüm (128 token, 3 tur): taban 17.2-17.4, yalnız down %50 16.2-16.5, up %50 ≤19 15.9-16.3, up %40 ≤19 16.6. Son düzeltmeyle (son 8 katmanda down seyrek) up %50 ≤19 15.4-15.8.
- **Ölçüm hatası (14 Eki):** Metal up yolunun tek token perplexity'si 9.03 / 7.96 çıkmıştı (simülasyon 7.34 / 7.15). Teşhis: oran 0.0001 ile (hepsi seçili) yoğunla aynı; sayaçla seçilen nöron sayısı doğru (9825/18944); tek token dökümünde Metal h ile numpy referansı birebir (9472/9472 aynı nöron, fark < 0.004); ama yalnız katman 0'da bile 10.45. Neden: `llama-perplexity` yaması `SEYREK_ORAN` görünce kendi sıfırlama geri çağrısını kuruyordu ve `SEYREK_UP` ile 28 katmanın hepsinde up seçimi yapıyordu. Metal ölçümü iki kez sıfırlanmıştı. Geri çağrı artık `SEYREK_SIM` ile açılıyor. Önceki Metal down-only perplexity'leri de geri çağrıyla birlikte ölçülmüştü (kahin sıfırlama + Metal, aynı şeyi yaptıkları için sonuç tutarlıydı); yeniden ölçülüyor.
- Ders: deney anahtarlarını aynı ortam değişkenine bağlama; simülasyon ve gerçek yol ayrı anahtar ister.
- **Seçim ağırlıkları model dosyasına gömüldü (14 Eki sabah; `sabit_masraf/seyreklik/secim/sel_ekle.py`):** yan dosya yerine GGUF'ta katman başına iki f32 vektör: `blk.N.ffn_sel_up.weight` (ort|silu g|) ve `blk.N.ffn_sel_down.weight` (down satır normu), 18944 eleman, toplam 4,2 MB. Dosya `~/llm-bench/models/qwen2.5-7b-instruct-oq4-downT-sel.gguf` (ana sürüm artık bu). llama.cpp tarafı: `LLM_TENSOR_FFN_SEL_UP/DOWN` (isteğe bağlı, `qwen2.cpp`), `llm_graph_params.model` ile grafik bağlamı katman tensörlerine erişiyor, `build_ffn` bunları up-hist (`src[2]`), gate (`src[3]`), GLU (`src[2]`) ve down (`src[2]`) oplarına bağlıyor; Metal tarafı tensör varsa onu, yoksa `SEYREK_AGIRLIK*` yan dosyasını, o da yoksa 1.0 kullanıyor. `SEYREK_AGIRLIK_KAPAT=1` kapatır (kıyas için).
  - Doğrulama (`sel_dogrula_ppl.csv`, 4 parça -b 1): gömülü up %50 ≤19 7.078 (yan dosya 7.079), up %40 6.945, kapalı 7.252 (düz 7.274), yalnız down %50 6.957 (yan dosya 6.957). Hız 3 tur: gömülü 14,53-14,58, yan dosya 14,62-14,73 (`sel_dogrula_hiz.csv`).
  - **Ölçüm gürültüsü notu:** aynı ayarın Metal perplexity'si koşudan koşuya ±0,05 oynuyor (up %50 ≤19 g+n dört koşu: 7.026 / 7.039 / 7.085 / 7.096 / 7.132). Histogram eşiği atomiklerle toplanıyor, eşit değerlerde seçim sırası değişiyor ve sonraki katmanlara yayılıyor. Yukarıdaki tablolarda 0,1'den küçük farklar anlamlı değil; ana sonuçlar (0,2-0,3'lük farklar) etkilenmiyor.
- **Ağırlıklı seçim (14 Eki gece, ben ölçtüm; `sabit_masraf/seyreklik/secim/`):** Soru: up yolunda %50'nin bedeli +%6 perplexity, kâhinde +%0,9; fark seçim ölçütünden mi geliyor? Fikir: nöronu yalnız anlık |u| ile değil, |u| × (o nöronun sabit önem ağırlığı) ile seçmek. İki ağırlık: (a) ortalama |silu(g_i)| (kalibrasyon: wiki.test'in son 600 KB'ı, 20 parça, 10k token, test parçalarıyla örtüşmez; `SEYREK_TOPLA` ile toplanır), (b) down satır normu ‖d_i‖ (model dosyasından, `down_norm.py`; nöronlar arasında 14 kat değişiyor). Bayt maliyeti sıfır (katman başına 18944 float, 76 KB).
  - Simülasyon (oq4, 4 parça; `secim1.csv`, `secim3.csv`): up %50 ≤19 düz 7.40 → |u|·ort|silu g| **7.11**; up %40 ≤19 7.21 → **6.96** (yalnız down %50 = 6.94); up %50 tüm katmanlarda 9.22 → 7.47 (son katmanlar kullanılabilir oldu). Down seçimi |h|·‖d‖: %50 6.945 → 6.894 (yoğun 6.883), %60 7.135 → 7.049. Up tarafında ‖d‖ ile ort|silu g| eşdeğer (gürültü ±0,03); ikisi birlikte 7.17/7.10, fark gürültü içinde.
  - Metal uygulaması: `kernel_mul_mv_q4_0_f32_hist` / `_glu_hist` epilogunda histogram |u|·w / |h|·w üzerinden; `kernel_seyrek_gate4` ve `kernel_seyrek_down1h` karşılaştırması aynı çarpımla. Ağırlıklar scratch tamponunda 2 MB ofsetinden (`SEYREK_AGIRLIK` = istatistik dosyası, `SEYREK_AGIRLIK_TUR` 1 = ort|silu g|; `SEYREK_AGIRLIK_DOWN` = norm dosyası; yoksa 1.0). İlk sürüm 4 MB ofsetle 8 MB tamponu aşıp sessizce çöküyordu (tüm perplexity'ler boş), 2 MB'a alındı.
  - **Metal sonuç** (`metal_secim2_ppl.csv` 4 parça -b 1, `metal_secim2_hiz.csv` 5 tur ortanca ms/token; bu turda yoğun 17,12 / 6.863; g = ort|silu g| ağırlığı, n = ‖d‖ ağırlığı):

    | düzen | ms/token | perplexity |
    |---|---|---|
    | yalnız down %50 düz | 15,71 (%-8,2) | 6.996 (+%1,9) |
    | yalnız down %50 n | 15,73 | 6.957 (+%1,4) |
    | yalnız down %60 n | 15,25 (%-10,9) | 7.057 (+%2,8) |
    | up %50 ≤19 düz (eski varsayılan) | 14,73 (%-14,0) | 7.274 (+%6,0) |
    | up %50 ≤19 g | 14,69 | 7.131 (+%3,9) |
    | **up %50 ≤19 g+n** | **14,69 (%-14,2)** | **7.079 (+%3,1)** |
    | **up %40 ≤19 g+n** | **15,28 (%-10,7)** | **6.928 (+%0,9)** |
    | up %60 ≤19 g+n | 14,04 (%-18,0) | 7.611 (+%10,9) |
    | up %50 ≤19 + up %30 20-27 g+n | 14,65 | 7.094 |

  - Sonuç: aynı hızda kalite bedeli yarıya indi (+%6,0 → +%3,1); up %40 düzeni kâhin düzeyinde kalite (+%0,9) ile %-10,7. Ağırlık hesabı hıza dokunmuyor (düz ile g+n aynı ms). Ağırlıklar şimdilik yan dosyada; kalıcı çözüm GGUF'a katman başına bir vektör olarak gömmek. Bu turun mutlak hızları sabah ölçümünden ~0,2-0,3 ms yavaş (yoğun 17,12 vs 16,93), yüzdeler esas. Kalibrasyon ve norm dosyaları (`ist.bin`, `down_norm.bin`, 12,7 MB'ar) `secim1.sh` ve `down_norm.py` ile yeniden üretilir, depoya konmadı.
- **Katman başına oran taraması (14 Eki, ben ölçtüm; `sabit_masraf/seyreklik/katman_oran/`):** Soru: tüm katmanlarda tek %50 yerine katman grubuna göre farklı oran daha iyi hız/kalite verir mi? Araçlar: `llama-perplexity` simülasyonuna `SEYREK_SIM_DOWN` / `SEYREK_SIM_UP` (28 elemanlı virgüllü liste), Metal yoluna `SEYREK_ORANLAR_DOWN` / `SEYREK_ORANLAR_UP` (oran grafikten `op_params[2]` ile çekirdeğe gider; 0 = o katmanda up yolu kapalı).
  - Grup duyarlılığı (simülasyon, oq4, 4 parça, taban down %50 her yerde 6.945; `tarama1.csv`): katman 8-13 seyrekliği neredeyse bedava kaldırıyor (down %60 −0.03, up %60 +0.11); katman 0-7'de down %70 ucuz (+0.06) ama up %50 pahalı (+0.28); 14-19 ve 20-27 her artışta +0.06-0.26. Down %80 her grupta +0.23-0.55. Gürültü ±0.03 (tek koşu).
  - Tam düzen kıyası, Metal (`metal_hiz.csv` 5 tur ortanca ms/token, `metal_ppl.csv` 4 parça -b 1; sıra 0-7 | 8-13 | 14-19 | 20-27, u = up yolu, d = yalnız down; yoğun 16,93 / 6.863):

    | düzen | ms/token | perplexity |
    |---|---|---|
    | u.5 u.5 u.5 d.5 (mevcut) | 14,43 | 7.286 (+%6,2) |
    | u.5 u.6 u.5 d.6 | **14,12** | 7.485 (+%9,1) |
    | d.7 u.6 u.5 d.5 | 14,42 | 7.340 |
    | d.6 u.6 u.5 d.5 | 14,56 | 7.201 (+%4,9) |
    | d.7 u.6 u.4 d.4 | 14,71 | 7.184 |
    | u.4 u.4 u.4 d.5 | 15,02 | 7.055 (+%2,8) |

  - Sonuç: mevcut tek oranlı düzen zaten hız/kalite sınırının üstünde. Katman bazlı oran sınırı kaydırmıyor, üzerinde gezdiriyor: aynı kalitede en fazla ~%1 hız ya da aynı hızda ~1 puan perplexity. Simülasyonun 8 parça sıralaması (d.7|u.6|u.5|d.5 mevcut düzenden iyi görünüyordu) Metal'de tutmadı; simülasyon yoğun oq4 dosyası ve 512'lik toplu işle çalışıyor, Metal ölçümü esas. Varsayılan değişmedi.
- **Son hız kıyası 2 (14 Eki, yeni gate4 + 16 parça; 128 token, 5 tur dönüşümlü, ortanca ms/token; `hiz_son2.csv`):** taban 16,95; yalnız down %50 **15,43 (%-9,0)**; **up %50 ≤19 + down %50 14,35 (%-15,3, 69,7 tok/s)**; up %40 ≤19 + down %50 15,04 (%-11,3). Bir önceki sürümde %-5,1 / %-11,3 / %-6,5 idi. Kalite değişmedi (seçim aynı, yalnız çekirdek düzeni): up %50 7.27-7.31 (+%6), up %40 7.10 (+%3,4), yalnız down 7.00-7.05 (+%2,2).
- **Gate çekirdeği profili (14 Eki, ben ölçtüm, `llama-bench -n 128`, 3 tur dönüşümlü, ms/token; `gate_profil.csv`):** taban 17.1-17.5; down %50 16.1-16.4; up %50 ≤19 eski gate 15.2-15.7; **yeni gate (`kernel_seyrek_gate4`) 15.0-15.4**; gate hesapsız (yalnız eşik+sıkıştırma, h=u) 14.7-14.9; gate tüm satırlar eski 18.4-18.5 / yeni 17.9-18.0. Gate parça sayısı (32/64/128) fark etmiyor.
  - Yeni gate: simdgroup başına 4 satır, x yarım-blok halinde yazmaçta (yoğun `mul_vec_q_n_f32_core` düzeni). Eski çekirdek satır başına x'i yeniden okuyordu (128 B x / 18 B ağırlık).
  - Gate iç çarpımının kalan maliyeti ~0,2 ms/token (20 katman); gate artık darboğaz değil.
  - **Asıl açık down yolunda:** beklenen tasarruf %12,5, ölçülen %5. Parça taraması (`down_profil.csv`, `SEYREK_NCHUNK1`): 32 parça 15,9-16,3, 64 parça 16,2-16,4, 128 parça 16,7-16,8. Eski varsayılan 256 parça 18,0 (taban'dan yavaş). 16 parça 15,1-15,4 çıktı ama **geçersiz**: iş grubu başına 1184 nöron düşüyor, sıkıştırma 896'da kesiyordu (perplexity 16,5). Sıkıştırma 896'lık dilimler halinde döngüye alındı, `lidx` 2560; yeniden ölçülüyor. Tüm satırları seyrek yoldan okumak (oran ~0) 64 parça 18,3-18,7, 32 parça 18,0-18,2: seyrek yolun sabit masrafı yoğun matvec'e göre ~1 ms/token.
  - **Düzeltilmiş parça taraması** (`down_profil2.csv`, 3 tur, ms/token): taban 17,1-17,6; down %50 ayrı toplama 32 parça 15,9-16,3, **16 parça 15,6-15,9**, 8 parça 16,9-17,0 (paralellik yetmiyor); up %50 ≤19 + down %50 16 parça **14,5**. `topla` dağıtımını kaldıran birleşik sürüm (son biten iş grubu toplar) 16 parçada 16,1-16,2, up ile 14,9: daha yavaş, çünkü son iş grubu toplarken GPU boş kalıyor; varsayılan ayrı toplama kaldı (`SEYREK_DOWN_FUSE=1` ile açılır).
  - Doğruluk (`down2_ppl.csv`, 4 parça): gate4 tüm satırlar + down tüm satırlar 6.874, eski gate tüm satırlar 6.864 (yoğun 6.863); down %50 16 parça ayrı 6.996, birleşik 7.017 (64 parça 7.015). Aynı seçim, yalnız toplama sırası farklı; eşik seçimi sonraki katmanlara yayıldığından ±0,02 oynuyor.
  - **Sabit masraf ayrıştırma** (`down_ayristir.csv`, down %50 16 parça, blokları çekirdek içinde 3 kez çalıştırıp farktan; 3 tur ortanca): eşik hesabı ~0,05 ms/token, sıkıştırma ~0,08, iş grubu içi indirgeme ~0,05, `topla` ~0,17 (atlayınca 15,40 vs 15,57). Boş dağıtım + bariyer (`SEYREK_BOS_X=3`, 84 ek sınır): +0,2 ms → sınır başına ~2,3 µs; katman başına 2 ek sınır toplam ~0,13 ms. İlk turda sıkıştırma/indirgeme +0,5 ms görünmüştü, 2-3. tur yalanladı: tek koşu veri değil.
  - Sonuç: bariyerler ve dağıtım sınırları açığı açıklamıyor. Kalan şüpheli: 16 iş grubunun ana döngüde bant genişliğini dolduramaması ve GLU epilogundaki histogram atomikleri.
  - **down1h2 çekirdeği** (`SEYREK_DOWN_K=2`, varsayılan): simdgroup başına bariyersiz sıkıştırma (her simdgroup dilimin 1/14'ünü kendi listesine yazar, tek bariyer), iş grubu içi indirgeme yok (4 alt grup 4 kısmi satır yazar, `topla` 64 satır toplar). İlk sürüm paylaşımlı bellek 39 KB ile çöktü (sınır 32 KB); indirgeme tamponu kaldırılınca 32 token kısa koşuda 14,4 ms (K1 15,6). Tam ölçüm ve perplexity sürüyor.
  - down1h2 sonucu (`down_k2.csv`, 3 tur): ilk sürümde çekirdek `topla` için büyütülen `nchunk`'ı kendisi de kullanıp nöronların dörtte birini işliyordu (14,4 ms sahte, perplexity 51404). Düzeltilince liste gezme döngüsündeki çalışma zamanı indisli diziler yazmaçtan belleğe taşındı: 17,7 ms. İki geçişli sıkıştırma + K1 satır döngüsüyle doğru (perplexity 7.048) ama 15,7-15,9 vs K1 15,5-15,7: indirgemeyi kaldırmak `topla`'nın 64 satır okumasıyla sıfırlanıyor. K1 varsayılan kaldı; `SEYREK_DOWN_K=2` deney için duruyor.
  - up %50 ≤19 + down %50, 16 parça: **14,5-14,6 ms** (3 tur, K1).
  - Hesap: `topla` atlanınca 15,40 vs yoğun 17,1 → katman başına 61 µs tasarruf; yoğun down 153 µs ise seyrek down çekirdeği 19 MB'ı ~92 µs'de okuyor (~210 GB/s, yoğun ~250). Çekirdek akışı ideale yakın; kalan açık ~0,4 ms. Ders: Metal çekirdeğinde çalışma zamanı indisli yerel dizi (`ii[cnt]`) kullanma; derleyici diziyi belleğe atar.
  - 16 satır ön yükleme (K3, `down_k3.csv`): down %50 16,7-16,8 (K1 15,5-15,8), tüm satırlar 19,4-19,9 (K1 17,5-17,6): yazmaç taşması; 8 doğru. Tüm satırlar K1 16 parça 17,5-17,6 vs yoğun 17,0-17,3: seyrek yol aynı baytı okurken katman başına ~16 µs ek masraf (topla 6, iki sınır 5, eşik+sıkıştırma 5). Çekirdek akışı ~250 GB/s.
  - **Up yolu hesabı:** gate'in tüm satırlarını seyrek yoldan okumak (up-hist + gate4 tüm satırlar) yoğun birleşik GLU'dan katman başına ~80 µs pahalı (17,9 vs 16,3 @64 parça). Gate hesapsız (hiç okumadan) yalnız 1,5 ms kazandırıyor; 764 MB atlanıyor, ideal 3 ms. Şüpheli: up çekirdeğinin epilogundaki 18944 histogram atomiği (değerler az kutuya yığılıyor) ya da tek matris okuyan up çekirdeğinin birleşik GLU'dan verimsiz olması. Atomiksiz ölçüm sürümü (`SEYREK_HIST_SKIP`) ile ayrıştırılıyor.
  - Histogram atomikleri masrafsız (`hist_atomik.csv`, 3 tur): yoğun model + GLU epilogunda histogram zorlanmış 17,0-17,2 = taban 17,0-17,2. Hipotez öldü.
  - **FFN ölçüm programı** (`sabit_masraf/seyreklik/ffn_olc/ffn_olc.cpp`): yalnız 28 katman FFN (gate, up, down q4_0, rastgele), bağımlılık zinciri, Metal; katman başına µs doğrudan. Yoğun 464 µs/katman = 247 GB/s. Dolaylı çıkarım yerine bundan sonra bu kullanılacak.
  - **FFN programı sonuçları** (`ffn_olc/ffn_olc_1.csv`, `ffn_olc4.csv`, katman başına µs, 3 tekrar × 20 koşu ortanca): yoğun 463; down %50 403 (ideal 387; `topla` 5, sınırlar 5, eşik+sıkıştırma 5); tüm satırlar seyrek yoldan 473 (+10). up %50 ≤19 + down %50: 368 ortalama; seyrek up katmanı 354 (ideal 305). Tek başına: up matvec 163 (235 GB/s), histogramlı up 163, birleşik GLU 306 (250 GB/s), up-hist + gate4 %50 satır 253 → gate4 ~88 µs (217 GB/s, ideal 76); gate4 tüm satır 157 (243 GB/s); gate hesapsız 3.
  - Denenip kazandırmayanlar: gate V5 (eşik+liste tek çekirdek, 64 iş parçacıklı 2368 gruplu yoğun düzenli toplama matvec) 282 vs 253; simdgroup başına 8 satır (up-hist ve gate) +2-4 µs; parça sayısı 32/128 fark yok.
  - Sonuç: gate4 tam satırda bant genişliğinde; yarı satırda rastgele 2 KB satır toplamanın DRAM bedeli ~%10. Kalan açık katman başına ~49 µs (up tek matris +10, gate +12, down +16, sınırlar ~10); üç ayrı çekirdek ve dağıtım sınırları yapısal. Çekirdek işi pratik tabana geldi; bundan sonraki kazanç seçim oranından gelir.
  - Varsayılanlar koda yazıldı: `kernel_seyrek_gate4`, `SEYREK_NCHUNK1=16`, down K1. Deney anahtarları (`SEYREK_GATE_V`, `SEYREK_DOWN_K`, `SEYREK_GATE_NR`, `SEYREK_UP_NR`, `SEYREK_HIST_SKIP`, `SEYREK_GLU_HIST_FORCE`, `SEYREK_BOS_X`, `SEYREK_HIST_X/COMP_X/RED_X`) duruyor.
  - Ders: parça sayısını değiştirince çekirdeğin kapasite sınırlarını (iş parçacığı × dilim, paylaşımlı bellek) kontrol et; hızlı çıkan sonucu perplexity ile doğrulamadan yazma.
  - Yeni gate'in perplexity'si (4 parça, up %50 ≤19) 7.271 (eski 7.314); eşik seçimi tie'larda değişiyor, doğruluk testi tüm satırlarla ayrıca yapılıyor.
- **Son hız kıyası** (128 token, 5 tur dönüşümlü, ortanca ms/token; `hiz_son.csv`): taban 17.29; yalnız down %50 16.40 (%-5.1); **up %50 ≤19 + down %50 15.33 (%-11.3)**; up %40 ≤19 + down %50 16.17 (%-6.5). up %50 ≤19 düzeni 65.2 tok/s (taban 57.8), kalite +%6.6; up %40 düzeni 61.8 tok/s, kalite +%3.4.
- Beklenen bayt tasarrufu up %50 ≤19 için ~%20 (4.1 GB'ın 0.84 GB'ı); ölçülen %11. Aradaki fark gate çekirdeğinin (simdgroup başına satır iç çarpımı) ve ek dağıtımların verimsizliği; henüz ayrıştırılmadı.
- **Temiz Metal perplexity** (tek token, 4 parça, f16 tabanlı transpoze model, yoğun 6.863; `metal_up_ppl.csv`): yalnız down %50 → 7.015 (+%2.2); **up %50 katman ≤19 + down %50 → 7.314 (+%6.6)**; **up %40 katman ≤19 + down %50 → 7.096 (+%3.4)**. Simülasyonla uyumlu (7.338 / 7.152).
- Metal yolunda iki hata ayıklandı: (1) up-önce koşulu yanlış yazılmıştı (`down->ne[0] != cur->ne[0]`, FFN girişinde her zaman yanlış), yol hiç açılmıyordu; (2) açılınca `cur = tmp` dalı down'a gate yerine up'ı veriyordu, çıktı "appré appré" oldu. Teşhis: Metal'e gelen mul_mat düğümlerini sırayla yazdırmak (`SEYREK_DBG`), gate düğümünün hiç gelmediği görüldü. Ayrıca işaretli up düğümünde eski GLU/qkv/add birleştirmesi kapatıldı.
- Metal yolu yazıldı: `kernel_mul_mv_q4_0_f32_hist` (up, |u| histogramı epilogda), `kernel_seyrek_gate` (eşik + grup içi sıkıştırma + simdgroup başına satır iç çarpımı, h üretir), down mevcut çekirdekle sıfırları atlar. Çizgede `SEYREK_UP=1`, `SEYREK_UP_KAT_MAX`.
- Sonuç (tahminci için): Qwen2.5-7B'de SwiGLU aktivasyonunun büyüklüğü girdiden ucuza tahmin edilemiyor; ama up'ın kendisi iyi bir seçici. ReLU modellerde çalışmasının nedeni %90 sıfırlık; burada "en büyük %50" sınırı gürültü içinde. up/gate atlama bu modelde kapalı; yalnız down atlama (yukarıdaki uygulama) geçerli.


