# Bağımsız denetim: M4 Pro LLM decode bant genişliği paketi

Kapsam: bu depodaki KNOWHOW.md, SONUCLAR.md, raw_results/, metal/, llama_patch/, main_session/ ve scripts/ dosyalarının tamamı okundu. Makineye erişim yok; tüm sayılar paketin kendi tablolarından alındı, aritmetik `python3 -I` ile yeniden hesaplandı. llama.cpp kaynak ağacı bu ortamda yok; llama.cpp iç yapısına dair ifadeler bellekten yazıldı ve "kontrol et" diye işaretlendi.

Etiketler: **[Ö]** paketteki bir ölçüm iddiayı doğrudan destekliyor. **[H]** paket sayılarından türetilen hesap. **[S]** donanım veya literatür bilgisine dayanan spekülasyon; makinede ölçülmeden kanıt sayılmaz.

Atıf biçimi: `dosya:satır`, `SONUCLAR §n`, `raw_results/NN:satır`.

## Yönetici özeti

1. **[H] q4_0 B=1 decode zaten bant tavanında, hatta ölçülen tavanın %2 üstünde.** bench15'te B=1 1.17 ms = 302 MB (kod+ölçek) → 258 GB/s; stream_read tavanı 243-254 GB/s. B=1 için kernel tarafında kazanılacak bir şey kalmadı; tek kaldıraç byte sayısı ve sabit masraf.
2. **[H] Bu GPU hesap-yoksul: ridge noktası B≈7.** FMA tepesi 6.15 TFLOPS / 254 GB/s = 24 FLOP/byte; q4_0'da token başına hesap 2.30 ms, bellek 16.2 ms → B>7 doğrulama partisi hesap sınırlı. M4 Pro'da MMA hızı FMA hızına eşit (6.2 vs 6.15 TFLOPS), yani simdgroup_matrix FLOP kazandırmaz, yalnız operand dağıtımı kazandırır. 7B q4_0 için teorik tavan ≈430 tok/s (mükemmel kabul); en iyi ölçülen 171.
3. **[H] "1.54 ms sabit masraf, 263 GB/s" uydurması muhasebe hatalı.** 263 > ölçülen tavan 254 imkânsız. 7B'de token_embd (0.307 GB) decode'da okunmaz; 1.5B/0.5B tied-embedding. Okunan baytla: **1.14 ms + 4.18 ms/GB (239 GB/s)**; 0.5B tahmini 344 tok/s, yerel derlemede ölçülen 341.
4. **[Ö] "Türetme q4'ü geçemez" tanım gereği doğru:** türetilen satırlar `seeds[gw]` okur, bu q4 kodlarıyla aynı bayt sayısı. Deney "hesap bedava mı"yı ölçtü; "daha az bayt"ı hiç ölçmedi.
5. **[H] En büyük denenmemiş byte/token kaldıracı K-quant kernel'leri:** 1.5B q3_K_M 117 tok/s = etkin 111 GB/s, q2_K 112 GB/s; q4_0 240. Alt-4-bit formatlar bandın %45'inde çalışıyor; dönüşüm yolu düzeltilirse 1.5B'de ~210 tok/s potansiyel.
6. **[H] B=3'ün mikro 1.09x yerine gerçek 1.57x olmasının iki somut adayı:** (a) yama kernel'i ggml'in 18 B düzeninde lane başına 4 ushort yüklüyor, aynı 8 baytı iki lane okuyor (mikroda 1 uint); (b) lm_head q6_K B=2-3'te düz mul_mv yoluna düşüp 2-3 kez okunuyor. (b) llama-bench pp eğrisinde görünmez (logits yalnız son token) ama spekülatif doğrulamada görünür: projeksiyonun iyimserliğinin (1.30→1.12) kaynağı.
7. **[H] B=4-8 MMA yolu hesap sınırlı:** 8 sütunlu karo 2^29 ağırlıkta 1.40 ms tepe, ölçülen 1.92 (%73). B=4'te sütunların yarısı dolgu. Tek matriste 1.55x daha kötü (64 threadgroup); Qwen k/v_proj (512 satır) 8 threadgroup'a düşer. Split-K yok.
8. **[Ö] Gürültü tabanı ±3-10%:** eşik taramasında B=1 satırı 18.4-22.5 ms (T B=1'i etkileyemez); tg128 kontrol ikilileri 52.7-54.6; 0.5B 316 vs 341 (aynı commit). "B=2 1.00x", "E 1.00x", "regresyon yok", "tm1 vs tm2" oranları gürültünün içinde.
9. **[S] "Ekle değil, ayarla":** llama.cpp'nin mevcut `mul_mv_ext` yolu için nsg/nxpsg/r1ptg taraması hiç yapılmadı; ajanın "ext x'i paylaşmıyor" teşhisi kaynakla doğrulanmadı. Upstream için en olası yol yeni kernel değil, cihaz sezgisi ayarı.
10. **[Ö] Sayaç sorunu aşılabilir:** Metal System Trace kaydı BAŞARILI olmuştu (`metal-gpu-intervals`, `metal-shader-profiler-intervals`); per-kernel süreler oradan çıkarılabilir, 1.14 ms'nin bileşimi ölçülebilir. Programatik `MTLCaptureManager` ile .gputrace alıp Xcode'da sayaçlara bakmak xctrace hatasını atlar.

---

## 1. Ölçüm geçerliliği denetimi

### D1 [H] Kernel tavanın üstünde ölçülüyor; tavanın kendisi şüpheli
- İddia: stream_read "tavanı" gerçek DRAM tavanını %2-6 düşük ölçüyor ya da ölçek dizisi SLC'de kalıyor.
- Dayanak: bench15 B=1 R=1 1.17 ms (raw_results/05:134; SONUCLAR §9) × 2^29 ağırlık × 0.5625 B = 302 MB → 258 GB/s. bench13 okuma 257.9 GB/s (raw_results/04:50). Tavanlar: 243.0 (bench12 q4, raw_results/04:26), 254.3 (era1, raw_results/03:6). stream_read deseni: 1M thread, thread başına 4 bağımsız uint4, 16 MB adımlı 4 akış (`metal/era2_fast_dequant/kernels_mv9.metal:178-183`, dispatch `bench12.swift:116-127`). Ölçek dizisi 2^29/32×2 = 33.5 MB ve her koşuda aynı adreslerden okunuyor (`kernels_mv9.metal:69,72`).
- Sonuç: "q4 B=1 bellek sınırlı" doğru; ama "tavana ulaştı" ifadesi ±5% belirsiz, çünkü referansın kendisi değişken.
- Aksiyon: E10.

### D2 [Ö] Mutlak değerlerde programlar arası ±3-6% kayma; <3% oranlar anlamsız
- Dayanak: aynı q4 okuma işi 243.0 / 250.0 / 254.3 / 257.9 / 258 GB/s (raw_results/04:26,28,50; 03:6; 05:134). Zamanlama: komut tamponu başına 3-4 dispatch (~5 ms GPU), tamponlar arası CPU işi sırasında GPU boşta (`bench12.swift:28-35,102-106`); ısınma yalnız 2 tampon (~2-3 ms). Apple GPU DVFS'i bu boşluklarda saat düşürebilir; karıştırma (K1,K2,K1,K2) oranları korur, mutlakları korumaz.
- bench13'te zamanlama iki gruba bölünmüş (`bench13.swift:155-156`): E varyantı (gövde 14+) başka gruptaki okuma tabanıyla (gövde 0) oranlanıyor → "E tohum gösteriminin bellek tavanına ulaşır (1.00x)" iddiası ±3% belirsiz.
- Gürültü içinde kalan iddialar: B=2 "1.00x (0.98-1.01)" (§9), "half birikim +%4" (§7/§10), "tm1 vs tm2" (33.9 vs 35.5, raw_results/06:28), "fp16 P=2 686 ikisinde de" (aynı kernel dosyası, aynı zamanlama kodu kopyası; yöntem tekrarı değil, koşu tekrarı).
- Aksiyon: ölçüm başında 300-500 ms stream_read ön ısıtma; her tabloda aynı tampon grubunda okuma referansı; mutlak GB/s yerine Gw/s ve oran raporla; oranları iki ayrı çalıştırmada tekrarla.

### D3 [Ö] llama.cpp ölçümlerinin gürültü tabanı ±5-10%; birkaç sonuç bu tabanın içinde
- Dayanak: eşik taraması B=1 satırı 18.4-22.5 ms, T B=1'i etkileyemez (raw_results/06:10). tg128 7B: brew 54.51, yerel 53.58, bin_base 53.15/52.72, SB0 54.33/54.46, SB1 54.49/54.59 (raw_results/06:6,110); kontrol grubu içinde %3.5 oynama → "regresyon yok" (SONUCLAR §10) gürültü içinde doğru ama kanıt gücü düşük. Taban 3 tekrar: yamalı 53.6/50.4/50.5 (raw_results/06:63) %6. 0.5B tg128: brew 316.4 vs yerel 341.4/345.1 (raw_results/06:6,112) %8 fark, aynı commit, açıklanmadı (brew ikilisi "BLAS,MTL" Accelerate bağlı; derleme bayrakları farklı olabilir).
- Aksiyon: her hücre ≥3 koşu, min ve medyan; tüm karşılaştırmalar aynı ikiliyle (GGML_METAL_SB=0/1); brew ikilisini referans alma; koşu sırasında `powermetrics --samplers gpu_power -i 1000` ile GPU frekansı logla (sudo gerekir).

### D4 [S] Bellek baskısı ve mmap tahliyesi tek-hücre anomalilerini açıklayabilir
- Dayanak: oturum kernel panic ile bitti, 21 swapfile (raw_results/07:22). llama.cpp modeli mmap'ler; baskı altında sayfalar atılır ve SSD'den geri okunur. Hiçbir komutta `--mlock` yok (`scripts/llama_runs.sh`, `scripts/spec_sweep.sh`, `scripts/matrix.sh`). JSON k=0 yamasız 41.0 vs 51.8/51.4 (raw_results/06:68; SONUCLAR §5) ve 0.5B SB0 340.5/322.5 (raw_results/06:112) bu mekanizmayla uyumlu.
- Aksiyon: `--mlock` (speculative/lookup/perplexity) veya llama-bench `-mmp 0`; koşu boyunca `vm_stat 1` kaydı; tekrar etmeyen anomaliyi tabloya alma.

### D5 [Ö] CPU doğrulaması dar; MMA yolunun fp32'ye göre hatası ölçülmedi; yama gerçek boyutta doğrulanmadı
- Dayanak: bench12 10 satır (`bench12.swift:95`), bench15 5 satır × B (`bench15.swift:100`), bench3 3 satır. Satır seçimi (0-3, n/2, n-4..n-1) tüm `(r&3)` kalıntılarını kapsıyor (GF karışımı için yeter) ama Mx matrisin yalnız ilki ve sonu. MMA referansı `halfX=true` ile x'i yarıya yuvarlıyor (`bench15.swift:63-64,213`) → ölçülen 4e-7 hata "half x'e göre"; fp32'ye göre hata bilinmiyor. Yama: `test-backend-ops` yalnız m=16, k=256/1024 (raw_results/06:47); MMA'da ne11<8 sütun kırpması ve ne01 kırpması (`sb_q4_0.diff:250-254,307`) gerçek şekilde test edilmedi; greedy eşitlik ve PPL YOK (SONUCLAR §10). Spekülatif koşularda da çıktı metni karşılaştırılmadı (SONUCLAR §4), oysa temp 0'da spekülatif decode kayıpsızdır ve ucuz bir doğruluk kahini sunar.
- Not: llama.cpp'nin kendi mul_mm'i de src1'i half'e çevirir (B≥9 yolu), yani half x yeni bir risk değil; ama B=3 (fp32 skaler) ile B=4 (half MMA) arasında logit farkı doğar, partiye bağlı çıktı değişkenliği (1.5B esse'de görülen %80 ayrışma gibi, raw_results/02:157) yamayla B=4 sınırına taşınır.
- Aksiyon: E8.

### D6 [Ö/H] Türetme deneyi bayt tasarrufunu ölçmedi; sonuç tasarım gereği
- Dayanak: türetilen satırlar `seeds[gw]` okur (`kernels_mv9.metal:70`; `kernels14.metal:83` aynı `blk[gw]` tamponu seeds'e bağlanır, `bench15.swift:194`), 4 B / 8 ağırlık = q4 kodlarıyla bire bir. Tohum gösterimi (16 bit tohum + 4×4 bit katsayı) = 32 bit / 8 ağırlık (KNOWHOW §6.7). Dolayısıyla SONUCLAR §8-9'daki "türetme/okuma" oranları yalnız hesap maliyetidir; q4'ten az bayt okuyan hiçbir varyant yok.
- Çıkarım: E üreticisi tam q4 baytında 1.00x (raw_results/04:56) → üretim hesabı bedava. Aynı üretici 3 bpw tohumla teorik 1.33x hız; kalite ayrı soru (8 ağırlıkta 4 değer).
- Aksiyon: E13.

### D7 [H] "Sanal GB/s" format sütunları tek ölçümün yeniden ölçeklenmesi
- Dayanak: rastgele üretim 597 Gw/s üç formatta aynı (raw_results/04:28,37,44); "sanal" bayt = tam format baytı (KNOWHOW §4). fp16 4.87x = (64/18) × 1.34 ≈ 4.8 hesapla çıkar. Tablo 8'in q8/fp16 "rastgele" ve "türet" sütunları bağımsız bilgi taşımıyor.
- Aksiyon: Gw/s'yi birincil birim yap; "sanal GB/s"i yalnız roofline yorumunda kullan.

### D8 [H] FMA-yuvası birimi ve "B=3 bütçeyi aşıyor" hipotezi
- Dayanak: yuva hesabı 2.76 T FMA/s ile (`metal/micro/dq.swift:22`, raw_results/04:17), tepe sonra 3.07 T'ye düzeltildi (raw_results/04:15); SONUCLAR §7 yuva sütunu %11 şişkin. KNOWHOW §10'daki "q4 B=2-3 için ~7 yuva bedava" bütçesi 3.07 T ile 6.8; harcanan: dönüşüm ~1.8 + B FMA 3 + ölçek FMA 3/8 ≈ 5.2 → bütçe aşılmıyor. B=3'ün yavaşlığını FMA bütçesiyle açıklamak (KNOWHOW §9 (4)) zayıf; daha olası sınırlayıcı yük (load) sayısı ve x L1 trafiği (bkz. O3, O4).

### D9 [Ö] half FMA ölçümü skaler; paketli half2 oranı bilinmiyor
- Dayanak: `half_peak` skaler half zincirleri (`metal/micro/fma.metal:12-20`); `dq_halfall` half2 fma kullanır (`dq.metal:69-73`) ama dönüşüm sınırlı bir testte +4% (raw_results/04:21-22). half2 FMA'nın 2× issue olup olmadığı belirsiz; B=4-8 skaler yol için önemli (FMA sayısı yarıya iner).
- Aksiyon: E12.

### D10 [S] dq mikro-ölçümünde derleyici ortak alt-ifade riski
- Dayanak: `NEXTW` w1..w3 = w ^ sabit (`dq.metal:9-11`); `(w ^ C) & M = (w & M) ^ (C & M)` → derleyici maske/kaydırmayı dört varyant arasında paylaşabilir; dönüşüm maliyeti (cvt/magicf/magich) gerçek kernel'dekinden düşük görünebilir. Sonuçların göreli sırası (cvt 4.84 ≫ magich 2.01) muhtemelen korunur, mutlak yuva sayıları değil.
- Aksiyon: `xcrun -sdk macosx metal -S -emit-llvm dq.metal -o dq.ll` ile IR'de paylaşımı kontrol et; ya da her birikeç için bağımsız LCG durumu kullan.

### D11 [Ö] "Okuma, ölçek yok" anomalisi yorumlanmadan bırakıldı ama yöntemsel bir uyarı
- Dayanak: aynı okuma işinde ölçek çarpımı kaldırılınca 6.13 → 8.70 ms (raw_results/03:21). Küçük kod değişikliğinin ±40% etkisi, kernel karşılaştırmalarının derleyici zamanlamasına duyarlılığı kadar büyük. B=4-6 "ani sıçramalar" (KNOWHOW §6.4) ve kernels6'da B=6 10x (raw_results/05:11) aynı sınıfta.
- Aksiyon: kritik karşılaştırmalarda IR/AIR dump'ı ve yazmaç sayısı (E18 ile sayaçlar); tek varyant yerine 2-3 kod varyantının en iyisi.

### D12 [H] 7B-eşdeğeri (×14.2) ve mikro→gerçek projeksiyonu şekil farkını yok sayıyor
- Dayanak: KNOWHOW §4 "7B eşdeğeri = ms × 14.2". Gerçek 7B matrisleri: 3584×3584, 512×3584 (k/v_proj), 18944×3584, 3584×18944, lm_head 152064×3584 q6_K. MMA tm2'de 512 satır → 8 threadgroup; 1.5B kv 256 satır → 4. Mikro 1.92×14.2 = 27.3 ms tahmin, ölçülen 33.9-34.7 (SONUCLAR §10). Per-op tek matris 326 µs vs mikro-eşdeğer 210 µs (raw_results/06:44; 302 MB/1.92 ms oranıyla).
- Aksiyon: bench15'e gerçek şekil listesi ve katman ağırlığı (E7 ile birlikte).

### D13 [H] Spekülatif tur maliyeti ile llama-bench parti eğrisi farklı şeyi ölçüyor
- Dayanak: llama-bench `-p N` prompt işleme: logits yalnız son token için (llama_batch_get_one varsayılanı; kontrol et) → lm_head ne11=1. Spekülatif doğrulamada lm_head ne11=B. q6_K için ext yolu yalnız ne11 4-8 (`sb_q4_0.diff:88-93`); ne11=2-3'te düz mul_mv satır başına sütun döngüsü → lm_head B kez okunur (KNOWHOW §7 bunu not eder). lm_head 0.447 GB → 1.76 ms/okuma. Veri: spec k=1 yamalı doğrulama 22.2 ms vs bench B=2 19.7 (+2.5); k=2 33.5 vs B=3 28.8 (+4.7) (raw_results/06:52-53,26-27); fazladan 1-2 okuma (1.8/3.5 ms) + örnekleme ile uyumlu.
- Sonuç: projeksiyon (`scripts/analysis_snippets.py:44-58`) parti eğrisini tur maliyetine uyguladı; lm_head katsayısını kaçırdı → 1.30x tahmin, 1.12x gerçek (SONUCLAR §11).
- Aksiyon: E3.

### D14 [Ö] Lookup/spekülatif tabloları tek koşu; non-monotonik noktalar ext kernel'inin şekil patolojisini yansıtıyor (iç tutarlılık VAR)
- Dayanak: llama-bench 7B B=5 101.9 > B=6 97.4; B=7-8 127-136 > B=9 92.8 (raw_results/01:80-84). Spekülatif k=4 20.5 < k=5 21.8; k=8 24.4 > k=6-7 (SONUCLAR §4). İkisi aynı deseni veriyor → spekülatif doğrulama maliyeti gerçekten parti eğrisini izliyor; bu, paketin sağlam çapraz kontrollerinden biri.

---

## 2. Optimizasyon / darboğaz analizi

### O1 [H] Roofline: hangi B'de ne sınırlıyor
Varsayımlar: FMA tepesi 6.15 TFLOPS (raw_results/04:11), tavan 254 GB/s, 7B token başına okunan 4.117 GB (dosya 4.424 − token_embd 0.307), hesap 2×7.07e9 FLOP (embedding hariç) = 2.30 ms.

| B | taban ms (max(bellek, hesap)) | ölçülen ms | pay | sınırlayıcı |
|---|---|---|---|---|
| 1 | 16.2 | 18.6 | 1.15x | bellek + ~1.1 ms sabit |
| 2 | 16.2 | 19.7 (yamalı) | 1.22x | bellek |
| 3 | 16.2 | 28.8 (yamalı) | 1.78x | yük/x trafiği + lm_head (D13) |
| 4 | 16.2 | 33.9 (yamalı MMA) | 2.09x | MMA %73 + yarı dolgu |
| 8 | 18.4 | 34.7 (yamalı MMA) | 1.89x | MMA %73 |
| 9 | 20.7 | 92.7 | 4.48x | mul_mm 32'ye dolgu |
| 16 | 36.8 | 94.7 | 2.57x | mul_mm 32'ye dolgu |
| 32 | 73.6 | 92.2 | 1.25x | hesap (FMA tepesinin %80'i) |

- Çıkarımlar: (i) B≥9 "5x düz taban" (SONUCLAR §3) B=32'de donanım tabanına yakın, B=9-16'da ise dolgu israfı. Ajanın tm1 n2 (16 sütun) kernel'i 3.60 ms × 14.2 = 51 ms eşdeğeri (raw_results/05:142) → B=9-16 için ~1.8x potansiyel, hâlâ tabanın 1.4-2.5 katı. (ii) B=2-8'de yamalı sonuçlar tabanın 1.2-2.1 katı; kalan pay MMA verimi, lm_head ve sabit masraf. (iii) MMA = FMA hızı (6.25 vs 6.15, raw_results/04:11-14): M4 Pro'da ayrı matris birimi yok (M5 GPU'da var). Bu, "simdgroup_matrix kullan" tavsiyesinin bu cihazda yalnız operand dağıtımı için geçerli olduğu anlamına gelir.
- Spekülatif tavan: tok/s ≤ B / max(16.2 ms, 2.3 ms·B) → ≈430 tok/s. En iyi ölçülen 171 (kod, k=31). Boşluğun çoğu B=9-16 dolgusu ve kabul oranında.

### O2 [H/S] B=4-8 MMA kernel'i: hesap sınırlı, az threadgroup, dolgu
- Dayanak: `kernels11.metal` ve `sb_q4_0.diff:231-312`. 8 sütunlu karo: 2^29×8/512 = 8.4M MMA → 1.40 ms tepe; ölçülen 1.92 (%73) (raw_results/05:134-141). B=4'te sütunların yarısı, B=5'te 3/8'i dolgu (x sütunları kırpılarak çoğaltılır, `diff:253-254`). Per-op m=4096, tm=2: 64 threadgroup × 128 thread (`diff:69-71`); 326 µs vs mikro-eşdeğer 210 µs (raw_results/06:44). Her lane blok başına 2 ushort + 1 half yük (`diff:265-271`) ve 8 skaler float x yükü + half dönüşümü (`diff:279-281`): küçük yükler, uzun seri K döngüsü (k=14336 → 448 blok), düşük occupancy → gecikme sınırlı.
- Qwen2.5-7B k_proj/v_proj ne01=512 → tm2'de 8 threadgroup; 1.5B kv 256 → 4 threadgroup: 16 çekirdekli GPU'da yarısı boş. [S] Bu matrisler küçük (0.9 MB) ama 28 katman × 2 × gecikme-sınırlı dispatch 1-3 ms'ye ulaşabilir; per-kernel süre (E1) doğrular.
- Aksiyon: split-K (4 simdgroup × K/4, threadgroup belleğinde indirgeme) → threadgroup sayısı 4×; küçük ne01 için tm1 veya ext'e düşme; x'i önceden half'e çeviren tek geçiş; sonraki bloğun ushort çiftini önceden yükleme (E7).

### O3 [H] Yama skaler kernel'i mikro kernel'den iki kat fazla yük yapıyor
- Dayanak: `sb_q4_0.diff:178-182`: lane başına 4 ushort (8 bayt) yük, ama lane yalnız 4 baytlık nibble bilgisi kullanır; `cc` ve `cc+2` lane'leri aynı 8 baytı okur. `bq->d` her lane her chunk'ta (`diff:188`): 4 lane aynı half'i yükler. Ölçek FMA'sı chunk başına (`diff:199`) = ağırlık başına 1/8 FMA × B. Mikro karşılığı: 1 uint + 1 half yük / 8 ağırlık (`kernels14.metal:83,85`). Mikro B=3 R=4 1.09x (raw_results/05:136) vs per-op R=3 1.57x (raw_results/06:43).
- Alternatif eşleme: lane = 4 bayt (2 ushort, 1 adet 4 baytlık yük alignment'ı 18 B blokta garanti değil) → alt nibble'lar ağırlık j..j+3, üst nibble'lar j+16..j+19; x için `x[blk*32+4cc]` ve `x[blk*32+16+4cc]` iki float4. Yük sayısı yarıya iner, d dört lane'de bir.
- Aksiyon: E4 (önce mikroda ggml düzenini taklit et, sonra yamayı değiştir).

### O4 [H] Skaler B=4 sıçramasının en olası nedeni x'in L1 trafiği
- Dayanak: `kernels14.metal:67-71`: her iterasyonda lane 2B float4 x yükler (B×32 B) ve R satır × 8 ağırlık işler → x trafiği 4·B/R B/ağırlık. B=4, R=4: 4 B/ağırlık = ağırlık baytının 7 katı; 2^29 ağırlıkta 2.1 GB L1 trafiği; 1.2 ms'de ~1.8 TB/s gerekir (16 çekirdek için L1 sınırına yakın, [S]). R=8 ile yarıya iner ama R×B birikeç yazmaç baskısı (R×B ≤ 64 kısıtı, `gen_batch_instances.py:180`) B≥5'te R=8'i cezalandırır (raw_results/05:67-74). Era1'de bile x'i yazmaçta tutmak okumayı 6.13→5.47 ms hızlandırdı (raw_results/03:19-20): x maliyeti B=1'de %11.
- Bilinen çözümler (pakette yok): (a) lane eşlemesini "8 satır × 4 k-dilim" yap: aynı k-dilimindeki 8 lane aynı x adresini okur, simdgroup yükü yayın (broadcast) olur → x trafiği 8× düşer, birikeç lane başına yalnız B; satır sonucu 2 shuffle ile 4 lane arasında indirgenir. Bu, llama.cpp `mul_mv_ext`'in nxpsg tasarımıdır (kontrol et); (b) x'i half tut (2×); (c) threadgroup belleğinde K-karosu.
- Aksiyon: E5 (x'i sabit yapıp sınırı doğrula), sonra (a).

### O5 [S] "Ekle değil, ayarla": mul_mv_ext taraması yapılmadı
- Dayanak: yamasız per-op n=2 295 µs (2.2x) ext kernel'idir (raw_results/06:38). KNOWHOW §1 "mul_mv_ext x'i satırlar arasında paylaşmıyor" der ama kaynağa atıf yok; ext'in nsg/nxpsg/r1ptg seçimi ggml-metal-ops.cpp'de şekle göre sezgisel (kontrol et). Eşik taraması (`GGML_METAL_MM_MIN`) yalnız ext↔mm sınırını değiştirdi, ext'in içini değil.
- Aksiyon: E6. Upstream (AGENTS.md kısıtı, KNOWHOW §7) için sezgisel değişikliği insan yazarlığıyla önermek, yeni 190 satır kernel'den çok daha gerçekçi.

### O6 [H] lm_head: %11 bayt, B=2-3'te 2-3 kez okunuyor
- Dayanak: output.weight q6_K 152064×3584×(210/256) = 0.447 GB (KNOWHOW §7; llama-quantize Q4_0 ftype için output → Q6_K kuralı). Tek okuma 1.76 ms. D13'teki tur maliyeti farkları.
- Aksiyon: `--output-tensor-type q4_0` (B=1'de −0.55 ms ≈ +3%; B=2-3'te sb yoluna girer, fazladan okumalar kalkar ≈ +8-12% spekülatif tur) ya da q8_0 (ext listesinde, ne11 2-8 tek okuma; B=1'de +0.4 ms). PPL bedeli ölçülmeli. E3.

### O7 [Ö/H] Yama kapsam boşlukları, etkiye göre
1. `ne00 % 256` (`diff:37`): 0.5B taslağın K=896'sını dışlar. Taslak model her turda kabul edilen tokenları B=2-4'lük partiyle yutar → eski yol. Kuyruk koruması (`if (ci < nch)`) ile `% 32`'ye gevşetilebilir (896/8 = 112 = 3×32 + 16).
2. lm_head q6_K (O6).
3. B=9-16 32'ye dolgu (O1): lookup k=8-15 için 93 → ~51 ms.
4. `mul_mat_id` yok: OLMoE'de uzman matmul'ları küçük partilerle çalışır; klif orada daha sert.
5. Tipler: q8_0/q4_1/q5_0/iq4_nl/mxfp4 skaler yol için dönüşüm değişikliğiyle kolay; K-quant'lar zor.
6. Yapılandırma env ile: upstream'e uygun değil; cihaz sezgisi tablosu gerekir.
7. Sayısal yol farkı B=3 (fp32) ↔ B=4 (half x) (D5).

### O8 [H] Yama sağlamlık notları
- `GGML_METAL_MM_MIN` için `atoi` → anlamsız değer 0 → `ne11 <= 0` hiç tutmaz → ext tamamen kapanır, B=2-8 sessizce 5x yavaşlar (`diff:8-11,84,93`). `GGML_METAL_SB` bilinmeyen değer → 1 sayılır (`diff:21`).
- `min(row0 + r, ne01 - 1)` kırpması ve çıkış koruması doğru (`diff:153,208,307`); `nb1 == ne0*4` ve `nb10 == 4` kontrolleri var (`diff:39`); `nb11` keyfi (`diff:172`) iyi. Pipeline ilk kullanımda derlenir (`diff:118-122`): ilk token gecikmesi.
- Nibble düzeni ve x eşlemesi (`diff:178-199`, `284-286`) kâğıt üzerinde doğru; test-backend-ops geçişiyle uyumlu.

### O9 [H] Sabit masraf ≈ 1.14 ms ≈ 28 katman × 40 µs; taslak modelde %40
- Dayanak: düzeltilmiş uydurma (özet 3). 0.5B: 3.16 ms'nin 1.77'si bayt, kalan sabit → taslak maliyetinin %40-45'i dispatch/küçük kernel. Taslak ms/token 3.4-4.2 (SONUCLAR §4) vs 0.5B tek başına 3.16: tur başına ek ~1.5-2 ms sabit (örnekleme, KV, eşzamanlama).
- Aksiyon: E1 (bileşim), E17 (MLX karşılaştırması: küçük modelde dispatch masrafı farkı), ggml-metal'deki füzyon/eşzamanlılık knob'ları (`grep -n getenv ggml/src/ggml-metal/*.cpp`).

### O10 [S] Yüklemede yeniden paketleme
- 18 B serpiştirilmiş blok → 16 B hizalı kod dizisi + ayrı fp16 ölçek dizisi (mikrodaki düzen). uint4 ile 32 ağırlık tek yük; sb kernel'leri sadeleşir; ggml Metal arka ucunda yükleme kancası gerekir (CPU'da repack var, Metal'de yok). Önce mikroda ölçülmeli (E4 düzen testi tersinden aynı sonucu verir).

### O11 [Ö] Yazılım ön-yükleme ve LUT denemelerinin reddi doğru ama dar
- Prefetch (`mm8_prefetch_variant.metal.txt`) B=4-8'de kötüleşti (raw_results/05:84-93): yazmaç baskısını artırdığı için beklenen sonuç; occupancy düşükken prefetch zarar verir. LUT (`dq.metal:80-97`) threadgroup belleğinde 256 giriş, ağırlık çifti başına 1 lookup: 636 Gw/s. Bu, "LUT yaklaşımı ölü" demek değil; bit-düzlemi LUT-GEMM'de lookup/ağırlık 3-4 kat düşer (A2).

---

## 3. "Acaba?" soruları

### A1 [H] Alt-4-bit formatlar bandın %45'inde: en büyük byte/token kaldıracı
- Dayanak: 1.5B q3_K_M 117 tok/s (8.55 ms), dosya ~0.82 GB; sabit 1.14 ms çıkarılınca 111 GB/s. q2_K 139 tok/s → 112 GB/s. q4_0 → 240 GB/s (SONUCLAR §1; main_session/OLCUMLER.md:46). KNOWHOW §9 (3) bunu "dönüşüm kaynaklı olabilir (ölçülmedi)" diye bırakıyor.
- Potansiyel: 1.5B q3_K_M bant hızında ≈ 1.14 + 0.82×4.18 = 4.6 ms → ~215 tok/s (q4_0 180'den hızlı). 7B'de IQ3_XXS (~3.0 GB) ≈ +26%, Q3_K_S (~3.3 GB) ≈ +18%, IQ2_M (~2.6 GB) ≈ +40%; PPL bedeli 1.5B'de q3_K_M %7 (SONUCLAR §2), 7B'de tipik olarak daha küçük.
- Yöntem: 3-bit kodları 2-bit düzlem + 1-bit düzlem olarak tutup half2 sihirli sayıyla açmak: `((q2>>s)&0x00030003) | (((hm>>t)&0x00010001)<<2) | 0x64006400` → 2 ağırlık için ~5 int op (q4'te 3); bütçe (3.5 bpw'de 1.7 ps/ağırlık × 3.07 T = 5.3 yuva) yeter. q3_K'nın 6-bit ölçek açması thread başına tekrarlanıyorsa, genişletilmiş ölçek yan dizisi (+%1-2 bayt).
- Aksiyon: E11.

### A2 [S] Bit-düzlemi tablo çarpımı (LUT-GEMM, T-MAC)
- Fikir: 8 ardışık k için x'in 256 olası işaretli toplamını tabloya yaz (sütun başına); ağırlık bit-düzlemi baytı tabloya indeks → 1 lookup + shift-add; 3-bit'te 8 ağırlık için 3 lookup. Paketin LUT testi (2 ağırlık/lookup) 636 Gw/s → lookup sınırlıysa 3-bit'te ~1700 Gw/s; 3-3.5 bpw bellek sınırı 580-680 Gw/s → bellek sınırlı çalışabilir. B>1'de tablo B katı (threadgroup belleği 32 KB sınırı: B=8, 8 k-grubu × 256 × 4 B × 8 = 64 KB → half tablo veya 4 k-grubu).
- Aksiyon: E13.

### A3 [H] Ölçek hassasiyeti: fp16 ölçek bilgi taşımıyor
- Dayanak: q4_0 ölçek düzlemleri 5.85/4.87 bit entropi (OLCUMLER.md:26), yani 16 bitin ~10.7'si bilgi. 8-bit log/e4m3 ölçek → 32 ağırlıkta 1 B tasarruf = %5.5 bayt → B=1'de ≈ +5% tok/s; açma sihirli sayıyla (8 bit → half). Q4_K'nın 6-bit ölçekleri aynı fikrin süper-blok hâli; ama Q4_K'nın Metal kernel'i A1'deki hastalığı taşıyabilir.
- Aksiyon: mikroda 17 B/32 ağırlık "q4_0s" düzeni (kernels_mv9'a FMT=3): okuma ≥ 262 "sanal" GB/s beklenir (1.055× bayt avantajı); kalite ölçümü ayrı.

### A4 [S] Aktivasyon seyrekliğiyle sütun atlama (TEAL 2024, CATS 2024, PowerInfer-2)
- SwiGLU modellerde down_proj girişinin %40-60'ı eşiklenerek sıfırlanabiliyor (Llama-3 raporları; Qwen2.5 ölçülmemiş). GPU'da bayt tasarrufu için down_proj'u k-major saklamak gerekir (sıfır olmayan k'ler için 3584'lük sütun bitişik akar, y threadgroup belleğinde birikir); q4 blokları k boyunca 32'lik olduğundan satır-major düzende eleman bazlı seyreklik bayt kazandırmaz. Pay: down_proj ≈ model baytının %22'si → %40 seyreklikte ≈ %9 genel tasarruf; gate/up için tahminli seyreklik (CATS) ile daha fazla.
- Aksiyon: 1.5B'de transformers+MPS kancasıyla down_proj giriş dağılımını ve eşikleme PPL etkisini ölçmek 30 dk; kernel sonraki adım.

### A5 [S] KV cache: uzun bağlamda byte/token hesabı değişir, spekülatif decode lehine
- 7B GQA (4 KV baş × 128 × 28 katman × 2 × 2 B) = 57 KB/token bağlam. 4k bağlamda +0.23 GB/token (+5.7%), 16k'da +0.93 GB (+23%), 32k'da +45%. Doğrulama partisinde KV bir kez okunur ve B sorguya hizmet eder → uzun bağlamda spekülatif/lookup kazancı büyür. Pakette tüm ölçümler KV≈0'da. `-ctk q8_0 -ctv q8_0 -fa 1` KV baytını yarılar.
- Aksiyon: E16.

### A6 [S] q4'ten az bayt okuyan tohum gösterimi
- E üreticisi (`kernels_mv9.metal:141-151`) hesabı bedava kılıyor; 24 bit / 8 ağırlık (16 bit tohum + 2×4 bit katsayı) = 3 bpw + ölçek ≈ 3.44 bpw → bayt 1.31× azalır. Kalite: SeedLM tarzı en küçük kareler ile tohum arama (blok başına 65536 tohum × 2 katsayı). Pakette kalite tarafı hiç yok; "türetme kaybeder" sonucu bu gösterimle geçerli değil.
- Aksiyon: E13'ün ikinci kolu (hız), sonra numpy ile gerçek bir Qwen matrisinde yeniden yapılandırma hatası.

### A7 [S] Byte başına token: taslak modelsiz yöntemler (literatür notu)
- MTP başları (DeepSeek-V3, Qwen3-Next): Qwen2.5'te yok. EAGLE-3 (Qwen2.5-7B-Instruct için topluluk başları var), Medusa, Apple ReDrafter (RNN taslak başı; MLX uygulaması var): llama.cpp ana dalında desteklenmiyor. Self-speculative (LayerSkip, Kangaroo, Draft&Verify): taslak için katman atlama, KV paylaşımı, 0.5B masrafı yok; llama.cpp'de yok. Hiçbiri 30 dk işi değil; ama B≤7'nin ucuzlaması bunların değerini artırır (doğrulama bedeli artık 1.1-1.9x).
- Ağaç taslakları: eski `llama-speculative` örneği birden çok taslak dizisi destekler (n_seq_dft; bayrak adını `--help` ile doğrula); B≤7 ucuzken 2-3 dallı ağaç kabul uzunluğunu artırabilir (E14).

### A8 [S] `--spec-draft-p-min` etkisizliğinin olası nedeni araç seçimi
- `llama-speculative` eski ağaç tabanlı örnek; p_min'i `common_speculative` (llama-speculative-simple ve llama-server yolu) kullanır (kontrol et). Paketin tüm spekülatif sayıları server'ın kullanmadığı bir yolda ölçüldü.
- Aksiyon: E14.

### A9 [H] Çoklu-istek servis: B=2-8'in ucuzlaması verim kaldıracı
- Yamalı B=4 1.82x → 4 akış için 2.2× toplam verim (~120 tok/s toplam 7B); yamasız 3.69x → 1.08×. Tek kullanıcı gecikmesi değil, verim.
- Aksiyon: E15.

### A10 [S] Taslağı başka motorda eşzamanlı koşturmak
- Unified memory'de ANE (CoreML 0.5B int8) veya CPU (SME/Accelerate) taslak üretirken GPU doğrular → taslak maliyeti gizlenir. llama.cpp sıralı çalışır; `-ngld 0` taslağı CPU'ya alır ama yine sıralı. İlk veri: 0.5B'nin CPU hızı (`llama-bench -ngl 0 -t 8`).

### A11 [S] Sabit masrafa saldırı: füzyon ve tek geçişli küçük model
- 0.5B'de sabit masraf %40-45: taslak hızının gerçek sınırı. ggml-metal'de op füzyonu ve graph eşzamanlılığı knob'ları var (getenv listesi); MLX aynı modelde daha yüksek tok/s veriyorsa fark dispatch tarafındadır.
- Aksiyon: E17, E1.

### A12 [Ö] Kayıpsız sıkıştırma kapanmış kapı; byte-plane bulgusu A3'e işaret ediyor
- Dayanak: entropi 7.55 bit/B → tavan %5.9 (OLCUMLER.md:25-27); CPU açma 10-14 GB/s (OLCUMLER.md:36-37). Sonuç sağlam: GPU-içi entropi kodu açma bile en iyi %5-8 verir ve o da yalnız ölçek düzlemlerinden gelir → doğru hamle sıkıştırma değil, ölçeği baştan 8 bit saklamak (A3).
- Yöntem notu: entropi ilk 4 MB'tan, sıkıştırma 537 MB diliminden (`byteplane.py:17-20,33`): tutarlı; zaman ölçümü `time.time` ve `> /dev/null` (compress-test.sh) kaba ama 14× fark için yeter.

---

## 4. Yerel deney listesi (önem sırasına; her biri 10-30 dk)

Gürültü tabanı: llama.cpp hücreleri ±5-10%, Metal mikro oranları ±3%. Çöküş eşikleri bunun dışında seçildi. Komutlardaki bayraklar var olanlardır; emin olunmayanlar "(doğrula)" ile işaretli.

### E1. Sabit masrafın bileşimi: per-kernel süreler (D13, O9, O2)
- Hipotez: 7B tg'de kernel sürelerinin toplamı ≈ 17 ms, kalan ≈ 1 ms dispatch boşlukları + küçük kernel'ler (norm/rope/attention/k-v_proj); q6_K lm_head ≥ 1.8 ms.
- Komut: `xcrun xctrace record --template 'Metal System Trace' --time-limit 30s --output tg.trace --launch -- build/bin/llama-bench -m 7B -ngl 99 -p 0 -n 64 -r 2` (bu şablon raw_results/07:14'te çalıştı). Sonra `xcrun xctrace export --input tg.trace --toc`, tablo adlarını al (`metal-gpu-intervals`, `metal-shader-profiler-intervals`), `--xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]'` ile XML dök; python3 -I ile kernel adına göre süre topla, ardışık interval'ler arası boşlukları topla.
- Beklenen: toplam kernel 16.5-17.5 ms; boşluk+küçük kernel 0.8-1.5 ms; `kernel_mul_mv_q6_K_f32` ≥ 1.8 ms; k/v_proj dispatch'leri bayt payından 5× uzun.
- Çöker: boşluk + küçük kernel toplamı < 0.3 ms → sabit masraf dispatch değil; bayt muhasebem yanlış (E2'ye dön).
- Süre: 20 dk.

### E2. Düzeltilmiş uydurma: okunan bayt ve ek nokta (özet 3)
- Hipotez: t = 1.14 ms + 4.18 ms/GB × okunan bayt; 1.5B fp16 (3.09 GB) ≈ 14.1 ms (71 tok/s), 1.5B q8_0 (1.89 GB) ≈ 9.1 ms (110 tok/s; [F] 121 ile çelişir).
- Komut: tensör tiplerini doğrula: `python3 -I src/gguf-py/gguf/scripts/gguf_dump.py MODEL | grep -E "token_embd|output\.weight"`; `llama-bench -m 1.5B-fp16 -m 1.5B-q8_0 -ngl 99 -p 0 -n 128 -r 5`.
- Beklenen: fp16 13.5-14.7 ms; q8_0 8.7-9.6 ms.
- Çöker: fp16 > 16 ms veya < 13 ms → iki terimli model yanlış, sabit masraf model boyutuyla ölçekleniyor.
- Süre: 10 dk.

### E3. lm_head'i q4_0'a çevir (O6, D13)
- Hipotez: B=1 +2.5-3.5%; spekülatif k=1/k=2 tur doğrulaması −1.8/−3.5 ms (tok/s 61.9→~66, 59.7→~67); PPL ≤ +0.5%.
- Komut: `build/bin/llama-quantize --allow-requantize --output-tensor-type q4_0 7b-q4_0-00001-of-00002.gguf 7b-oq4.gguf Q4_0` (parçalı girdiyi destekler; doğrula). Sonra `llama-bench -p 0 -n 128 -r 5`; `GGML_METAL_SB=1 llama-speculative -m 7b-oq4.gguf -md 0.5B ... --spec-draft-n-max 1|2` (3 tekrar); `llama-perplexity -f wiki.test.raw -c 512 --chunks 32` iki modelde.
- Beklenen: tg128 55.8-56.5; k=2 ≥ 64 tok/s; ΔPPL ≤ 0.04.
- Çöker: k=2 kazancı < 4% → B=2-3'teki fazlalık lm_head değil (örnekleme/KV).
- Süre: 20 dk.

### E4. Skaler kernel düzeni: ggml 18 B düzenini mikroda taklit et (O3)
- Hipotez: B=3'ün 1.09x→1.57x farkının ≥ yarısı yük sayısından geliyor.
- Kod: `kernels14.metal` mm8'e MODE 4: blk'yi `device const ushort*` olarak 18 B stride ile oku (`bq = blk + gb*9`, `qs = bq+1`), lane başına 4 ushort (yamadaki `diff:178-182` birebir), `d` = `bq[0]`; bench15 "verify" ile doğrula; B=2,3 R=2,3,4 zamanla. İkinci varyant MODE 5: lane 2 ushort (4 bayt) → alt nibble j..j+3 ve üst nibble j+16..j+19, x için `X[blk*8 + cc]` ve `X[blk*8 + 4 + cc]`.
- Beklenen: MODE 4 B=3 ≥ 1.6 ms (1.27'den), MODE 5 ≤ 1.4 ms. Yamaya taşındığında per-op n=3: 212 → ≤ 175 µs.
- Çöker: MODE 4 ≤ 1.35 ms → düzen suçlu değil; fark threadgroup sayısı/SLC'de (per-op 33 MB).
- Süre: 30 dk.

### E5. x L1 trafiği B=4 sıçramasının nedeni mi (O4)
- Hipotez: x yükleri kaldırılınca B=4 R=4 2.54 → ≤ 1.3 ms; half x ile → ~1.7 ms.
- Kod: `kernels14.metal:67-71`'de MODE'a bağlı `xl[c] = float4(0.5f*c+0.1f)` sabitleri (sonuç doğrulanmaz, sadece zaman); ikinci varyantta X'i `device const half4*` olarak oku ve float4'e çevir.
- Beklenen: sabit-x B=4 1.2-1.35 ms; half-x 1.6-1.9 ms.
- Çöker: sabit-x ≥ 2.2 ms → sınırlayıcı x değil (yazmaç/occupancy veya FMA issue); o zaman E18 ile yazmaç sayısına bak.
- Süre: 20 dk.

### E6. mul_mv_ext'i ayarla, kernel ekleme (O5)
- Hipotez: ext'in nsg/nxpsg/r1ptg kombinasyonlarından biri n=2'de ≤ 180 µs verir (295 yerine).
- Kod: `grep -n "nxpsg\|r1ptg\|nsg" ggml/src/ggml-metal/ggml-metal-ops.cpp` ile seçim noktasını bul; üçüne `getenv` override ekle; `GGML_METAL_SB=0` ile `test-backend-ops perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32,m=4096,n=[1-3],k=14336"` üzerinde nxpsg ∈ {4,8,16,32}, r1ptg ∈ {2,3,4}, nsg ∈ {1,2,4} tara.
- Beklenen: en iyi n=2 ≤ 180 µs, n=3 ≤ 230 µs.
- Çöker: en iyi n=2 ≥ 250 µs → ext tasarımı yetmiyor, yeni kernel gerekçeli.
- Süre: 20 dk.

### E7. MMA split-K ve küçük ne01 (O2, D12)
- Hipotez: K'yi 4 simdgroup'a bölmek tek matriste 326 → ≤ 240 µs, 8192²'de 1.92 → ≤ 1.65 ms; ne01=512 şeklinde tm1 ve split-K tm2'den ≥ 2× hızlı.
- Kod: `kernels11.metal` mma_q4'e SPLITK şablon parametresi: simdgroup `sg` blok aralığı `[sg*nblk/4, (sg+1)*nblk/4)`; her simdgroup C'yi threadgroup belleğine (`threadgroup float red[4][8*TM][8]`) yazar, barrier, sg==0 toplar ve store eder. bench15'e `rows=512 cols=3584` şekli ekle (Mx otomatik). Yamada aynı değişiklik `kernel_mul_mm_sb_q4_0_f32` ve dispatch (`diff:69-71`: threadgroup başına 4 simdgroup aynı satır grubu).
- Beklenen: 8192² 1.55-1.70 ms; per-op ≤ 240 µs; 512×3584'te tm2 split-K ≥ 2× tm2.
- Çöker: 8192²'de ≥ 1.85 ms → gecikme değil, MMA hesap sınırı (%73 tavan); yalnız dolgu azaltma kalır.
- Süre: 30 dk.

### E8. Yamanın gerçek boyutta doğruluğu (D5)
- Hipotez: SB=0 ve SB=1 aynı PPL'yi (±0.01) verir; temp 0 spekülatif/lookup çıktısı tabanla birebir aynı.
- Komut: `for ub in 2 3 4 8; do GGML_METAL_SB=$sb llama-perplexity -m 7B -f wiki.test.raw -c 512 -b $ub -ub $ub --chunks 12; done` (sb=0,1). `llama-lookup --temp 0 -n 300 --spec-draft-n-max 0|2|5` çıktılarını `diff`.
- Beklenen: |ΔPPL| ≤ 0.01 her ub'de; metin farkı yok (sonda log satırları hariç).
- Çöker: ΔPPL > 0.05 veya ilk 100 tokende metin ayrışması → sayısal hata (MMA half x veya kırpma).
- Süre: 25 dk (7B PPL 12 chunk × 8 koşu).

### E9. Taslak modeli kapsama al: ne00 % 256 → % 32 (O7.1)
- Hipotez: k=2 spekülatif turunda taslak kısmı 8.3 → ≤ 7.5 ms; tok/s +2-4%.
- Kod: `diff:37` koşulunu `ne00 % 32 == 0` yap; skaler döngüye (`diff:163`) `if (ci < nch)` koruması yok, zaten `ci < nch` koşulu var; MMA döngüsü blok tabanlı, nb = ne00/32 tam → ek değişiklik gerekmez (doğrula). test-backend-ops k=896 şekli için `-p "k=896"` filtresi varsa koştur.
- Beklenen: k=2 60 → 61.5-62.5 tok/s.
- Çöker: < +1% → taslak partileri zaten ucuz (K küçük).
- Süre: 15 dk.

### E10. Gerçek bant tavanı (D1)
- Hipotez: farklı akış deseni ≥ 258 GB/s verir.
- Kod: `kernels_mv9.metal` stream_read varyantları: (a) threadgroup başına bitişik 64 KB parça, (b) thread başına 8 uint4 in-flight, (c) 2 GB tampon, (d) 500 ms ön ısıtma; `bench12.swift:116-127` çağrısını varyantlar için çoğalt.
- Beklenen: en iyi 256-266 GB/s.
- Çöker: hiçbiri > 255 → bench15'in 258'i ölçek dizisinin SLC'de kalmasından; ölçekleri 18 B düzende kodların içine serpiştirip B=1 okumayı tekrar ölç.
- Süre: 15 dk.

### E11. K-quant ve IQ formatlarının bant verimi (A1)
- Hipotez: tüm alt-4-bit Metal kernel'leri 100-130 GB/s etkin bantta.
- Komut: `llama-bench -m 1.5B-q3_K_M -m 1.5B-q2_K -ngl 99 -p 0 -n 128 -r 5`; 1.5B IQ3_XXS, IQ4_XS, Q3_K_S indir (~0.7-0.9 GB her biri, bartowski/Qwen2.5-1.5B-Instruct-GGUF); etkin GB/s = dosya / (1000/tok_s − 1.14). E1 yöntemiyle q3_K matmul kernel sürelerini al.
- Beklenen: hepsi ≤ 140 GB/s; IQ4_XS q4_0'a yakın (≥ 200).
- Çöker: q3_K_M ≥ 200 GB/s → [F] 117 sayısı gürültüydü, kaldıraç yok.
- İkinci kol (15 dk): `dq.metal`'e 3-bit sihirli varyant (A1 formülü) ekle, `dq.swift` listesine kat; beklenen ≥ 1000 Gw/s; çöker < 600 Gw/s (3.5 bpw bellek sınırı 580).
- Süre: 20 + 15 dk.

### E12. half2 FMA tepe (D9)
- Kod: `fma2.metal`'e `fma32h2` (32 bağımsız half2 zincir, aynı yapı); `fma2.swift`'te FLOP = 64×2.
- Beklenen: ya ~6.2 TFLOPS (paketli half yok) ya ~12.
- Çöker: yok; iki sonuç da yön verir: 12 çıkarsa B=4-8 skaler yol half2 birikeçle FMA sayısını yarılar (E5'in half-x varyantıyla birleştir).
- Süre: 10 dk.

### E13. Bit-düzlemi LUT ve 3 bpw tohum hız testi (A2, A6, D6)
- Hipotez: 3 lookup / 8 ağırlık ≥ 1200 Gw/s; E üreticisi 24 bit/8 ağırlık tohumla okumadan 1.25-1.33× hızlı.
- Kod: (a) `dq.metal` `dq_lut3`: threadgroup tablosu `float T[3 grup][256]` (8 k × 256 işaretli toplam, x'ten doldur), 8 ağırlık = 3 bayt (düzlemler) → `acc += T0[b0] + 2*T1[b1] + 4*T2[b2]` (ofset ayrı). (b) `kernels_mv9.metal` GMODE 6'ya FMT=3: seeds tamponu 3 B/8 ağırlık (uchar3 veya 4 ağırlık-çiftine 12 B hizalı paket), sanal bayt 14/32; bench13 oranı.
- Beklenen: (a) ≥ 1200 Gw/s; (b) okumaya göre 1.25-1.33×.
- Çöker: (a) < 700 Gw/s → 3 bpw bellek sınırının altında, LUT yolu ölü; (b) < 1.15× → tohum yükleri paketlenmiş düzende coalescing kaybediyor.
- Süre: 30 dk.

### E14. Spekülatif araç ve ağaç taslakları (A7, A8)
- Hipotez: `llama-speculative-simple` p_min'i uygular; p_min 0.5-0.75 ve n_max 2-3 ile tok/s ≥ 63 (59-60'tan).
- Komut: `GGML_METAL_SB=1 llama-speculative-simple -m 7B -md 0.5B -ngl 99 -ngld 99 --temp 0 -n 256 --spec-draft-n-max 2|3 --spec-draft-p-min 0|0.5|0.75 -p "$ESSAY"` (bayrak adlarını `--help` ile doğrula); eski örnekte çok dizili taslak (n_seq_dft 2,3; bayrak doğrula) ile k=2.
- Beklenen: n_drafted p_min ile düşer; en iyi ≥ 63 tok/s.
- Çöker: hiçbir yapılandırma > 62 → taslak kalitesi (0.5B vs 7B esse) sınır, araç değil.
- Süre: 20 dk.

### E15. Çoklu-istek verim (A9)
- Komut: `GGML_METAL_SB=$sb llama-batched-bench -m 7B -ngl 99 -npp 128 -ntg 64 -npl 1,2,4,8` (sb=0,1).
- Beklenen: SB=1 npl=4 toplam tg ≥ 105 tok/s, npl=8 ≥ 200; SB=0 npl=4 ≈ 57.
- Çöker: SB=1 npl=4 < 80 → batched yolda yama devreye girmiyor (ne12/nb kontrolleri, `diff:38-39`).
- Süre: 10 dk.

### E16. KV derinliği (A5)
- Komut: `llama-bench -m 7B -ngl 99 -p 0 -n 64 -d 0,4096,16384 -fa 1 -r 3`; aynı komut `-ctk q8_0 -ctv q8_0` ile.
- Beklenen: f16 KV'de 4k'da −5..−8%, 16k'da −18..−25%; q8 KV'de yarısı.
- Çöker: 16k'da düşüş < 5% → KV okuması bayt modelime girmiyor (attention gecikme sınırlı); A5 çıkarımı geçersiz.
- Süre: 15 dk.

### E17. MLX ile küçük model dispatch karşılaştırması (O9, A11)
- Komut: `pip install mlx-lm; python -m mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --max-tokens 256 --prompt "$ESSAY"` (≈300 MB indirme); llama.cpp 0.5B tg 341 ile karşılaştır (aynı istem, `llama-completion`/lookup k=0).
- Beklenen: MLX ≥ 420 tok/s ise dispatch/CPU masrafı kaldıraç; taslak ~1.3× hızlanabilir.
- Çöker: MLX ≤ 350 → sabit masraf çerçeveye özgü değil, donanım dispatch tabanı.
- Süre: 15 dk.

### E18. GPU sayaçları programatik capture ile (SONUCLAR §12)
- Hipotez: MMA kernel ALU sınırlı (~%70+), skaler B=4 yük/L1 sınırlı, yazmaç sayısı B=4-8'de occupancy'yi yarılıyor.
- Kod: bench15'e `--capture` seçeneği: `MTL_CAPTURE_ENABLED=1` ortamında `let d = MTLCaptureDescriptor(); d.captureObject = dev; d.destination = .gpuTraceDocument; d.outputURL = URL(fileURLWithPath: "b4.gputrace"); try MTLCaptureManager.shared().startCapture(with: d)`; tek B=4 dispatch; `stopCapture()`. Dosyayı Xcode'da aç → Performance/Counters (occupancy, ALU/memory limiter, yazmaç sayısı). Ayrıca xctrace hatası için Xcode sürümünün macOS 26.5 ile eşleştiğini kontrol et (raw_results/07:3).
- Çöküş eşiği yok; tanısal. E5/E7 kararlarını besler.
- Süre: 20 dk.

Uygulanamaz veya 30 dk'yı aşan: EAGLE/MTP/ReDrafter entegrasyonu (A7), aktivasyon seyrekliği kernel'i (A4; ölçüm kısmı 30 dk, kernel değil), ANE taslak (A10), yeniden paketleme kancası (O10), K-quant kernel yeniden yazımı (A1; mikro kısmı E11'de).

---

## 5. Tablo iç tutarsızlıkları

| Konu | Değerler | Büyüklük | Olası açıklama |
|---|---|---|---|
| 7B B=1 parti ms | 18.2 (§3), 18.6 (§10), 18.35 (uydurma), 18.4-22.5 (eşik taraması B=1 satırı, raw_results/06:10) | %20 | koşu gürültüsü, DVFS, bellek baskısı |
| 7B tg128 | 54.51 / 53.58 / 53.15 / 52.72 / 54.33 / 54.46 / 54.49 / 54.59 | %3.5 | aynı; "regresyon yok" bu bandın içinde |
| 0.5B tg128 | 316.4 (brew) vs 341.4 / 345.1 / 343.9 (yerel) | %8 | derleme/bağlantı farkı, açıklanmadı |
| q4 okuma tavanı | 243.0 / 250.0 / 254.3 / 257.9 / 258 | %6 | program/zaman farkı; kernel tavanı aşıyor (D1) |
| Uydurma bant | 263 GB/s (§3) vs tavan 254 | imkânsız | token_embd muhasebesi (özet 3) |
| JSON k=0 7B | 41.0 / 51.8 / 51.4 | %25 | tek koşu anomalisi; mmap/baskı (D4) |
| 1.5B q8_0 [F] | 121 tok/s → etkin 266 GB/s > tavan | %5 | [F] sayısı veya 1.5B'de sabit masraf farklı |
| §9 B=1 taban | "B kez matvec" 1.20 (R=8 kernel) vs "skaler okuma" 1.17 (R=1) | %2.5 | sütunlar farklı kernel; oran sütunu 1.17'yi kullanıyor |
| §3 "1-8'de doğrusal" | B=5 101.9 > B=6 97.4; B=7-8 > B=9 | yapısal | ext kernel şekil seçimi; spekülatif k=4/5/8 aynı deseni verir (D14) |
| Spec k=2 yamalı | 52.3 (R=4) vs 59.3-60.1 (R=3) | %13 | per-op R=4 236 vs R=3 212 µs (%11) ile uyumlu |
| Projeksiyon | k=2 1.30x tahmin, 1.12x gerçek | 16 puan | B=3 maliyeti 1.09 varsayıldı (1.55 gerçek) + lm_head (D13) |
| "İki ayrı program aynı sayı" (§8) | bench12/13 | yok | aynı kernel ve zamanlama kodu; bağımsız yöntem değil |

---

## 6. Bu denetimin sınırları

- Makineye erişim yok; hiçbir sayı yeniden ölçülmedi. Tüm [H] etiketli sonuçlar paketin kendi sayılarına dayanır; o sayılar "Bağımsız tekrar: YOK" ise türevleri de öyledir.
- llama.cpp kaynak ağacı bu ortamda yok. `mul_mv_ext` iç yapısı (nxpsg yayını), llama-bench'in logits davranışı, `llama-speculative` ile `-simple` arasındaki p_min farkı, llama-quantize'ın tensör tipi kuralları bellekten yazıldı; her biri "kontrol et" notuyla verildi ve ilgili deney (E3, E6, E14) bunu doğrular.
- M4 Pro SLC boyutu, L1 bant genişliği ve yazmaç dosyası sınırları Apple tarafından yayımlanmaz; O4'teki L1 sınırı tahmini [S]'dir.
- Ana oturumun [M] etiketli sıkıştırma ölçümleri yalnız OLCUMLER.md özetinden değerlendirildi; ham çıktıları yok.
- Yamanın `apply_patch.py` ile `sb_q4_0.diff` hâli karşılaştırıldı: içerik eşdeğer (örnekleme satırları `apply_patch.py:200-202`'de döngüyle üretiliyor).
