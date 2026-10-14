# Tahminli üretmede ekleme birleştirmesi neden hedefi yavaşlatıyor

Tarih: 11 Eki 2026. Ölçen ve okuyan: bu oturum (Sonnet 5.5). Bağımsız tekrar yok. Tur sayıları her tabloda yazılı.
Çalışma kopyasına (`~/claude-scratch/llamacpp-ablasyon-sb`) dokunmadım. Denemeler `~/claude-scratch/neden-kopya` içinde.
Betikler ve ham veri: `~/claude-scratch/bant-darbogaz/neden_kanit/`.

## Kısa cevap

Yavaşlama iki ayrı etkiden geliyor. İkisi de birleştirmenin kendisinden değil, hedef grafın bellek yerleşiminin değişmesinden.

1. Ekleme bayrağı, tek token'lık grafın düğüm sırasını değiştiriyor. Bu yüzden bellek ayırıcının önbellekteki yerleşimi bu grafa uymuyor. Ayırıcı yerleşimi yeniden hesaplıyor. Sonra aynı olay ilk 3 sütunluk grafta da oluyor. Sonuçta 3 sütunluk graf, kendi tam boyutuna göre sıkı paketlenmiş bir yerleşim alıyor.
2. Sıkı yerleşimde tensörler art arda, aralıksız duruyor. Çakışma kontrolü, birbirine değen aralıkları çakışma sayıyor. Bu 29-30 fazladan bariyer yapıyor. Maliyeti küçük, yaklaşık 0.3 ms.
3. Asıl kayıp (yaklaşık 2.3 ms, grafın %9'u) başka yerden: sıkı yerleşimde aktivasyon vektörlerinin adresi 1024 bayta hizalı değil. Matris çarpımı çekirdekleri, `src1` adresinin 1024'e göre kalanına duyarlı. Bu yalnız matris çarpımlarında görünüyor.

Eski "kararlı" yerleşim (birleştirme kapalıyken) şans eseri hep 1 MiB katlarında duruyor. Bu yüzden hızlı. Bayrak bu şansı bozuyor.

## Zincir ve kanıt

Terimler:
- Ayırıcı (gallocr): ggml'in ara tensörlere bellek adresi dağıtan kodu.
- Yerleşim: hangi tensörün tamponda hangi ofsette durduğu.
- Bellek bariyeri: GPU'nun, önceki çekirdekler bitmeden sonrakini başlatmamasını sağlayan bekleme noktası.

### Adım 1. Ayırıcı yerleşimi sıra numarasına göre yeniden kullanıyor (kesin)

- `llama-context.cpp:612, 632, 647`: başlangıçta sırayla 512 sütunlu, 1 sütunlu, tekrar 512 sütunlu graf rezerve ediliyor. Son rezerve eden 512 sütunlu graf. Yerleşim önbelleği bu.
- `ggml-alloc.c:1008-1049` (`ggml_gallocr_needs_realloc`): yeni graf, önbellekteki yerleşime yalnız düğüm sayısı ve boyutlar uyuyorsa uyar. Uyum, düğümün sıra numarasına göre bakılıyor. Hangi tensör olduğuna bakılmıyor.
- `ggml-alloc.c:1083-1094`: uyarsa önbellekteki ofsetler sıra numarasıyla uygulanıyor.
- Tüm grafların düğüm sayısı aynı (957), ve 512 sütunluk yuvalar küçük grafların tensörlerinden büyük. Yani sıra aynıysa her graf 512 sütunluk yerleşimi kullanıyor.
- Kanıt: `GGML_SCHED_LOG_REALLOC` (kopyada eklediğim günlük) birleştirme kapalıyken tüm koşuda sıfır yeniden ayırma gösterdi (`neden_kanit/k12_off.err`). `GGML_SCHED_DEBUG_REALLOC=2` ile aynı koşu çökmeden bitti (`kos2.sh`, `r_off_ub512`).

### Adım 2. Ekleme bayrağı 1 sütunlu grafın sırasını değiştiriyor (kesin)

- `ggml-backend.cpp:1417`: `ggml_graph_optimize` her parçada, ayırıcıdan önce çalışıyor ve `gf->nodes` dizisini yerinde yeniden sıralıyor.
- `ggml-metal-common.cpp:713-720`: ekleme açıkken matris çarpımı ile ekleme tek birim sayılıyor ve sıralama bu birimler üzerinden yapılıyor.
- Birleştirme kapalı: sıra `Q-mm, V-mm, K-mm, Q-ekle, V-ekle, K-ekle`. Açık: `Q-mm, Q-ekle, V-mm, V-ekle, K-mm, K-ekle`.
- Kanıt: `neden_kanit/go_off.err` ve `go_on.err` (GRAPH_DEBUG=2). Kapalıda `ADD:Qcur-0` ayrı düğüm olarak görünüyor, açıkta görünmüyor (matris çarpımının içine katılmış).
- 3 sütunlu grafta ekleme koşulu (`ne11 == 1`) sağlanmıyor. Sırası değişmiyor. Bu yüzden 3 sütunlu graf bayraktan doğrudan etkilenmiyor.

### Adım 3. Uyumsuzluk iki kez yeniden ayırma tetikliyor (kesin)

- `ggml-backend.cpp:1509`: `ggml_gallocr_alloc_graph` false dönerse `ggml_gallocr_reserve_n` ile yerleşim baştan hesaplanıyor.
- Kanıt `neden_kanit/k12_on.err`, ekleme açıkken tam iki olay:
  - `GGML_SCHED_REALLOC: nodes=970 sutun=1` (1 sütunlu graf, tüm koşuda bir kez çalışıyor)
  - `GGML_SCHED_REALLOC: nodes=970 sutun=3` (ilk doğrulama grafı)
- `GGML_SCHED_DEBUG_REALLOC=2` ile ekleme açıkken çökme, 1 sütunlu grafta (`r_on_ub512`).
- Sonra llama.cpp grafı yeniden kullanıyor (`graphs reused = 24`, `llama-context.cpp:1314`). Yani ilk doğrulama grafındaki yerleşim koşunun sonuna kadar sabit kalıyor.
- Diğer bayrakların neden zararsız olduğu da buradan çıkıyor:
  - glu: birleşik birimin sırası ayrık sırayla aynı. Yeniden ayırma yok.
  - q/k/v: ekleme kapalıyken hiç birleşmiyor.
  - rope: 512 sütunlu rezerve grafta da aynı biçimde birleşiyor. Sıra aynı.

### Adım 4. Sıkı yerleşimde 29-30 sahte bariyer (kesin, nedeni koddan)

- Sıkı (tam boyutlu) yerleşimde tensörler bitişik. 512 sütunlu yerleşimde ise yuvalar büyük olduğu için aralarında boşluk var.
- `ggml-metal-common.cpp:140`: `mr.p0 < cmp.p1 && mr.p1 >= cmp.p0`. Aralıklar yarı açık ([p0, p1)). `>=` birbirine değen aralığı çakışma sayıyor.
- Kanıt, `-ub` taraması (`kos3.sh`, birleştirmeler kapalı, 3 sütunlu doğrulama grafı): rezerve boyutu 3 iken 451 bariyer. 4, 5, 6, 8, 12, 16, 32, 64, 128, 256, 512 iken hep 421.
  Yani yalnız yerleşim tam boyutlu olduğunda bariyer artıyor. Birleştirme bayrağı gerekmiyor.
- Kanıt, günlük benzetimi (`parse_gd.py`, `gd_3.err`): bellek aralıklarını günlükten okuyup kuralı yeniden uyguladım.
  - `>=` ile 423-424. `>` ile 394. 512 ve 4 sütunlu yerleşimde her iki kuralla 394.
  - Benzetim gerçek sayıdan hep 28 düşük (422 ve 394). Bu sabit farkı, günlükte görünmeyen çekirdek içi bariyerlere (tahmin) bağlıyorum. Doğrulamadım.
  - Yani strict kuralla sıkı yerleşim, 512 sütunlu yerleşimle aynı bariyer sayısına iniyor.
- Senin daha önce denediğin sıkı kontrol de bu yüzden hızı düzeltmedi: bariyer sayısı düşse bile zaman kaybı bariyerlerden gelmiyordu (aşağıda).

### Adım 5. Asıl zaman kaybı: aktivasyon adresinin 1024'e göre kalanı (ölçüldü, nedeni bilinmiyor)

Ölçüm yöntemi: kopyada `GGML_METAL_GPU_TIME=1` düğmesi ekledim. Komut tamponlarının GPU başlangıç ve bitiş zaman damgalarından, 957 düğümlü 3 sütunlu grafın GPU süresini alıyor. Her koşuda 42 doğrulama grafı, tabloda koşu medyanlarının ortalaması.

Yerleşimin etkisi (birleştirmeler kapalı, tur sayısı 3-4, dönüşümlü sıra):

| yerleşim | GPU süresi (ms) | tur |
|---|---|---|
| 512 sütunlu eski yerleşim (taban) | 23.76 | 4 |
| rezerve 4 sütun (421 bariyer) | 25.70 | 4 |
| ekleme açık, yeni yerleşim (451 bariyer) | 26.33 | 4 |

Bu kayıplar bariyer sayısıyla açıklanamıyor: rezerve 4 sütunlu yerleşimde bariyer sayısı tabanla aynı (421) ama 1.9 ms yavaş.

Kaybın yeri: işlem atlama (`GGML_METAL_SKIP_OPS`, `GGML_METAL_SKIP_MM_NAME`; çıktı yanlış, yalnız süre), her karşılaştırma 2 tur, dönüşümlü. İki karşılaştırmada da aynı sonuç çıktı:
- Rezerve 4 sütun ile taban (fark +2.02 ms, `kos8.sh`, `kos9.sh`): tüm matris çarpımları atlanınca fark kayboluyor (-0.09). Flash attention atlanınca +1.99, küçük işlemler atlanınca +1.88 kalıyor. ffn_down atlanınca fark +0.58, ffn_gate ve ffn_up atlanınca +2.00, attn q/k/v/o atlanınca +1.06.
- Ekleme açık (hizalama 256) ile taban (fark +2.99 ms, `kos15.sh`): tüm matris çarpımları atlanınca -0.08. Flash attention atlanınca +2.98, küçük işlemler atlanınca +2.97. ffn_down atlanınca +1.32, ffn_gate ve ffn_up atlanınca +2.95, attn q/k/v/o atlanınca +1.72.
- Yani fark ffn_down ve dikkat matris çarpımlarında. gate ve up'ta yok. Bariyerler yerinde kaldığı halde fark kaybolduğu için bariyer maliyeti küçük.

Adres bağımlılığı (kaydırma taraması, `kos17.sh`, tüm tensörleri sabit bir bayt kadar kaydırıyorum, yerleşim ve bariyerler aynı, 1 koşu/değer):

| kaydırma (bayt) | 0 | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 | 3072 | 3840 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU ms | 26.78 | 29.36 | 28.96 | 30.72 | 26.77 | 29.24 | 28.85 | 30.70 | 26.77 | 26.66 | 30.69 |

Periyot 1024 bayt. Hiçbir şey değişmeden yalnız adres kayınca süre 26.7 ile 30.7 ms arasında oynuyor. 256 baytlık kayma 2.6 ms ekliyor.

Matris çarpımlarının `src1` adresinin 1024'e göre kalanı (GRAPH_DEBUG=3 günlüğünden, son 3 sütunlu graf, 28 katman; `analiz3.py`):

| yerleşim | Q/K/V, o-proj, ffn_down girdisi | ffn_gate, ffn_up girdisi | hız |
|---|---|---|---|
| 512 sütunlu eski yerleşim | 0 | 0 | hızlı |
| rezerve 3 sütun, hizalama 256 | 0 | 0 | hızlı |
| rezerve 3 sütun, hizalama 32 | 128 | 0 | yavaş |
| rezerve 4 sütun, hizalama 32 | 128 | 0 | yavaş |
| ekleme açık, hizalama 32 | 640 | 0 | yavaş |
| ekleme açık, hizalama 256 | 512 | 0 | yavaş |

Hızlı olan her yerleşimde tüm girdiler 1024'e hizalı. Yavaş olan her yerleşimde Q/K/V, o-proj ve ffn_down girdisi hizasız. ffn_gate ve ffn_up girdisi hep hizalı, ve bu iki sınıf atlama deneyinde yavaşlamadı. Örtüşme tam. Bu yine de bir ilinti. Doğrudan kanıt kaydırma ve hizalama denemeleri.

Hizalamayı zorlama (ayırıcı hizalaması `GGML_ALLOC_ALIGN`, kopyada; bu çözümün kendisi değil, nedeni doğrulayan deney; tur 3):

| yerleşim | hizalama 32 (mevcut) | 256 | 1024 | 4096 |
|---|---|---|---|---|
| ekleme açık (yeni yerleşim) | 26.43 | 26.87 | 24.12 | 23.84 |
| rezerve 3 sütun | 25.94 | 23.86 | 24.00 | 23.64 |
| rezerve 4 sütun | 25.97 | 23.59 | - | 23.64 |
| taban (512 sütun) | 23.79 | - | - | - |

- 4096 hizalamayla ekleme açık yerleşim tabanla aynı hıza iniyor (23.84 ve 23.79).
- 256 yetmiyor. Rezerve 3 ve 4 sütunda 256 yetti ama ekleme açık yerleşimde yetmedi. 256'ya hizalı ama 1024'e göre kalanı 512 olan adresler hâlâ yavaş. Rezerve 3 sütunda kalan şans eseri sıfır çıktı (günlükle doğruladım). Rezerve 4 sütunda günlüğe bakmadım.
- 1024 ile 4096 arasındaki 0.3 ms, bariyer farkıyla uyumlu (4096'da boşluklar bitişikliği kaldırıyor). Bariyer sayısını bu iki satırda ayrıca saymadım.
- Metin her deneyde aynı (md5 `ba139316`).

Çıkarım: kayıp bayrağın, birleştirmenin, bariyerin ya da CPU yan etkisinin kendisinden değil. Matris çarpımı çekirdeklerinin `src1` adresine duyarlılığından geliyor.

## Emin olduklarım ve olmadıklarım

Emin olduklarım (kodla ve ölçümle):
- Yeniden ayırma zinciri: kapalıda sıfır olay, açıkta iki olay.
- 30 bariyer sıkı yerleşimden ve `>=` kuralından (`-ub` taraması ve günlük benzetimi).
- Kaybın matris çarpımlarında olduğu (atlama deneyi) ve adres kaymasına duyarlı olduğu.
- 4096 hizalamanın bu 3 sütunlu grafta kaybı kapattığı.

Emin olmadıklarım:
- Donanım nedeni. 1 KiB periyodun neden olduğunu bilmiyorum. Apple belgelerinde (`apple-silicon-docs`) bunu açıklayan bir şey aramadım. Doğrusu hız, `src1` okuma biçimine ya da önbellek eşlemesine bağlı olabilir.
- Hangi çekirdek ve hangi okuma. Yalnız sınıf düzeyinde biliyorum.
- `dst` adresinin etkisi. Tek ipucu: ffn_gate ve ffn_up çıktıları kalan 512'de durdu ve yavaşlamadılar.
- Bariyer maliyeti. 0.3 ms, iki ölçümün farkından çıktı. Doğrudan ölçmedim.
- 1 sütunlu graf, 0.5B ve 1.5B modeller. Hiçbirinde ölçmedim.
- Ayırıcının sıra numarasıyla uygulama davranışının başka graflarda yanlış tensöre yerleşim verip vermediği. Koddan şüphem var (`ggml-alloc.c:1008-1049` tensör kimliğine bakmıyor), ölçmedim.

Geçersiz çıkan koşular:
- zsh'te `env $E ...` değişkeni bölmedi. Bir GRAPH_DEBUG koşusu yanlış ortamla çalıştı. `kos11.sh` ile bash'te yeniden yaptım. Yalnız yeniden yapılan sonuç kullanıldı.
- `GGML_ALLOC_PAD` düğmesi çöktü (boşaltmada boyut tutarsızlığı). Sonuçlarını kullanmadım.
- `kos4.sh` son turunda rezerve 4 sütun koşusu 1217 ms ile aykırı çıktı (diğer üçü 1123-1126). Sonuca katmadım.

## Önerilen çözüm (uygulamadım)

1. Metal tampon türünün hizalamasını 4096 yap. `ggml-metal.cpp:260` (shared) ve `ggml-metal.cpp:336` (private) şu an `return 32;`. Ayırıcı bu sayıyı `ggml_backend_buft_get_alignment` ile alıyor (`ggml-alloc.c:523`). Bu, ggml'de arka ucun çekirdeklerinin istediği hizalamayı ayırıcıya bildirdiği yer. Yerleşim ister eski ister yeni olsun, her aktivasyon aynı faza düşer.
   - Artı: tek yerde, ayırıcıya ve çizge düzenlemeye dokunmuyor. Yerleşim nereden gelirse gelsin hız aynı.
   - Eksi: her tensör en çok 4 KB boşluk bırakır. Canlı tensör sayısı kadar küçük bir artış, ölçmedim. Ağırlıklar mmap değilse onlara da +1.4 MB civarı (339 tensör × en çok 4 KB, hesap).
   - `mapped` türü (satır 412) dosya ofsetini kullanıyor, aktivasyon taşımıyor. Dokunmaya gerek yok.
2. Bu tek başına yeterli. Bariyer düzeltmesi isteğe bağlı: `ggml-metal-common.cpp:140` içinde `>=` yerine `>`. 4096 hizalamayla tensörlerin çoğu arasında zaten boşluk kalıyor. Yani sahte bariyerler de büyük ölçüde gidiyor (451 yerine 421 beklenir, hizalamayla bariyer sayısını ayrıca saymadım). Tek başına yapılırsa kazanç yalnız 0.3 ms ve bir çekirdeğin padding'e dokunup dokunmadığı ayrıca kontrol edilmeli.
3. Sağlamlık için (ayrı iş): `ggml_gallocr_needs_realloc`, önbellek eşleşmesini düğüm sırasına ve kimliğine göre doğrulamalı. Şimdi yalnız sayı ve boyuta bakıyor.

## Tek hedefli test

Soru: 4096 hizalamayla, tüm birleştirmeler açıkken tahminli üretme kapalıdan yavaş mı?

1. Çalışma kopyasının bir çatalında (worktree) yalnız madde 1'i uygula, derle.
2. Aynı koşu: 7B hedef + 0.5B taslak, k=2, 96 token, `--spec-draft-p-min 0`, tüm birleştirmeler kapalı ve tüm açık, 5 tur dönüşümlü.
3. Beklenti: açık, kapalıdan hızlı (+%3.6 civarı, taslakta açık iken görülen). Hedefin doğrulama süresi yaklaşık 1215 ms. Mevcut kayıp (1324 ms) kaybolmalı.
4. Kontroller: metin md5 aynı, `GGML_METAL_GPU_TIME` ile 3 sütunlu graf ~23.8 ms, 3 sütunlu bariyer sayısı kaydedilsin.

Teşhis düğmeleri için kopya: `neden-kopya/ggml/src/ggml-metal/ggml-metal-context.m` (GPU_TIME), `ggml-alloc.c` (`GGML_ALLOC_ALIGN`, `GGML_ALLOC_SHIFT`), `ggml-backend.cpp` (`GGML_SCHED_LOG_REALLOC`), `ggml-metal-ops.cpp` (`GGML_METAL_SKIP_MM_NAME`). Bunlar çözüm değil, ölçüm aracı.

## Bu bulgunun listedeki diğer maddelere etkisi (ölçülmedi)

- Madde 3'teki tek token'lık kazançlar da bu yüzden kirli olabilir. Birleştirmeler açıkken 1 sütunlu graf da yeni yerleşim alıyor (adım 3). Kapalıyken ise eski, şanslı yerleşimi kullanıyor. Doğru hizalamayla birleştirmenin kazancı daha büyük çıkabilir. Bunu `GGML_ALLOC_ALIGN=4096` ile `llama-bench -n 64` koşup açık ve kapalıyı yeniden karşılaştırarak ölçmek gerekir.
- Madde 1'deki "komut hazırlama tabanı" ve matris çarpımı payları birleştirmeler kapalıyken ölçüldü. Yani şanslı yerleşimle. Büyük olasılıkla etkilenmediler, ama bunu ölçmedim.
