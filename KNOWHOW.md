# LLM decode hızı ve bellek bant genişliği: know-how paketi

Amaç: bu çalışmayı hiç görmemiş bir modelin (ya da kişinin) kodu ve komutları optimizasyon ve darboğaz gözüyle yeniden incelemesi.
Bu belge anlatıdır; kod dosyalarında anlatı yoktur. Sayılar için `SONUCLAR.md`, ham çıktılar için `raw_results/`.

## 0. Önce şunu bil: bu paket yeniden yazıldı

- Makine bellek baskısıyla kernel panic ile yeniden başladı; `/tmp` altındaki tüm çalışma dizini silindi.
- Bu paketteki tüm dosyalar, oturum bağlamından **yeniden yazıldı**. Orijinal bayt bayt kopya değil.
- Yeniden yazılan hiçbir dosya, yazıldıktan sonra **derlenmedi ve çalıştırılmadı**. Yazım hatası olabilir.
- İlk çalıştırmada derleme hatası çıkarsa önce şuna bak: `.metal` dosyalarında örnekleme (instantiation) satırları ya dosyanın içinde ya da `gen_*_instances.py` üreticisinde.
- Ayrıntılı eksik listesi: bölüm 9.

## 1. Çalışmanın özeti (tek paragraf)

M4 Pro (16 çekirdek GPU, 24 GB, ~254 GB/s ölçülen akış tavanı) üzerinde llama.cpp Metal ile q4_0 decode bellek sınırlı (7B: ~54 tok/s, ~241 GB/s etkin).
Sorular: (1) bayt yerine hesapla ağırlık türetmek hız verir mi, (2) 2-8 tokenlık doğrulama grubu neden "token başına tam fiyat" ödüyor (spekülatif decode'u öldürüyor), donanım mı yazılım mı?
Bulgular: (a) 2-8 klifi implementasyondur: llama.cpp'nin `mul_mv_ext` çekirdeği x'i satırlar arasında paylaşmıyor; benzer işi bir kez okuyan çekirdekler B=2'de 1.0x, B=3'te 1.1-1.6x, B=4-8'de 1.7-2.4x veriyor (llama.cpp 1.9 ile 7.5x). (b) Yamayla 7B+0.5B spekülatif 1.12-1.16x (tabandan hızlı) oldu; yamasız 0.63-0.70x idi. (c) Türetme: q4'te okumayı geçemez (aynı bayt), q8/fp16'ya karşı kazanır; ama ilk q4 sonuçlarım yavaş uchar→float dönüşümünden bozuktu (bölüm 8).

## 2. Dizin haritası

```
KNOWHOW.md                       bu belge
SONUCLAR.md                      tüm sayılar, her tablonun başında kaynak/tekrar/doğrulama satırı
metal/
  util/probe.swift               .metal dosyasını çalışma anında derleyip fonksiyon adlarını yazar (derleme testi)
  util/diag.swift                pipeline başına maxTotalThreadsPerThreadgroup / tgMem yazar (yazmaç baskısı ipucu; hep 1024 çıktı, bilgilendirici değil)
  micro/fma.{metal,swift}        ilk FMA/half/int/cvt tepe testi (8 zincir; DÜŞÜK ölçtü)
  micro/fma2.{metal,swift}       32 zincirli FMA + simdgroup_matrix MMA tepe testi (6.15 / 6.4 TFLOPS)
  micro/dq.{metal,swift}         nibble→float dönüşüm varyantları mikro-ölçümü (cvt vs sihirli sayı vs LUT)
  era1_cvt/                      İLK DÖNEM (yavaş dönüşüm): kernels*.metal + bench*.swift; q4/q8/fp16 format ve karışık deneyleri
  era2_fast_dequant/             HIZLI dönüşüm (mv9 ailesi): kernels_mv9.metal + gen_mv9_instances.py + bench12/13.swift
  batch/                         toplu (B sütun) mat-mat: kernels6/7/14 (skaler), kernels11 (simdgroup_matrix), bench6/7/8r/15
scripts/
  mk_prompts.py, mk_prompts2.py  lookup istemleri (JSON, düz yazı, kod, esse)
  matrix.sh, spec_sweep.sh       yamasız lookup matrisi ve spekülatif tarama (ilk tur)
  llama_runs.sh                  yamalı llama.cpp ölçümlerinin komut kaydı (fonksiyon fonksiyon)
  analysis_snippets.py           ayrıştırma ve projeksiyon parçaları
  legacy/deney2-5.sh             ana oturumun önceki betikleri (ben yazmadım, bağlamda görüp kurtardım)
llama_patch/
  apply_patch.py                 YETKİLİ yama (commit e475fa2b5'e uygular, çapa doğrular)
  sb_q4_0.diff                   aynı yamanın okunabilir diff hâli (make_diff.py ile üretildi; git apply ile DENENMEDİ)
  make_diff.py                   diff üreticisi
raw_results/                     ham çıktılar (bağlamdan aktarıldı)
```

## 3. Ortam ve araçlar

- Donanım: MacBook Pro, Apple M4 Pro, 16 çekirdek GPU, 24 GB birleşik bellek. Metal 4. `recommendedMaxWorkingSetSize` ≈ 19 GB.
- Modeller: `~/llm-bench/models/` altında Qwen2.5 0.5B/1.5B/7B instruct (q4_0 ve 1.5B için q8_0, q3_K_M, q2_K, fp16), OLMoE-1B-7B q4_0. 7B parçalı: ilk parçayı vermek yeter.
- Metal çekirdekleri çalışma anında derlenir: Swift'ten `device.makeLibrary(source:)`. Çevrimdışı `metal` derleyicisi başta yoktu (Xcode sonradan kuruldu). Swift programları: `swiftc -O X.swift -o X`.
- llama.cpp: `ggml-org/llama.cpp`, commit `e475fa2b5` (build 9730, ggml 0.15.2), Homebrew'daki sürümle aynı. Kendi kopyam scratchpad'de derlendi (brew'e dokunulmadı). `cmake` sistemde yoktu: sanal ortama `pip install cmake`.
- Python: `python3 -I` kullandım (yerel modül yüklemesin diye).
- Bellek disiplini (panikten sonra): aynı anda tek ağır iş, derlemede `-j4`, xctrace kaydı sırasında model yükleme yok.

## 4. Ölçüm düzeninin mantığı (Metal çekirdekleri)

- **Süre:** `MTLCommandBuffer.gpuStartTime/gpuEndTime`. Bir komut tamponunda `iters` (3-4) dispatch; süre / iters. Bu GPU zamanıdır, CPU/başlatma masrafı içermez.
- **Isınma ve istatistik:** her çekirdek için 2 ısınma tamponu; sonra `reps` (5-15) tekrar; **medyan** raporlanır (en iyi de bazı tablolarda).
- **Karıştırma (interleave):** karşılaştırılan çekirdekler her tekrarda sırayla koşar (K1,K2,K1,K2...). Termal/sıklık sürüklenmesi iki tarafa eşit dağılsın diye.
- **Çalışma kümesi:** dispatch başına sabit **2^29 ağırlık** (`T = 1<<29`; bazı ilk deneylerde 2^31). Matris boyutuna göre `Mx = T/(rows*cols)` matris tek dispatch'te (grid.y = matris). Amaç: (1) önbelleği aşmak (q4'te 268 MB, fp16'da 1 GB), (2) dispatch masrafını amortize etmek, (3) boyutları kıyaslanabilir tutmak.
- **Sanal bayt:** formatın saklanma bayt sayısı: q4_0 benzeri 18 B / 32 ağırlık (16 kod + 2 ölçek), q8_0 benzeri 34 B, fp16 64 B. "Sanal GB/s" = ağırlık sayısı × (bayt/ağırlık) / süre. Üretilen ağırlıkların da belleği "okumuş gibi" sayılması bilinçli karar; "gerçek GB/s" ayrıca yazıldı (üretenin sadece ölçeği okuduğu).
- **Bant tavanı:** `stream_read` çekirdeği (uint4 yük, 4 bağımsız yük/yineleme, 1M thread) aynı tampon üzerinde: ~243-254 GB/s. Tüm "tavan" yorumları buna göre.
- **Doğrulama:** her ölçüm programı çıktıyı bir tampona yazar (derleyici işi silemez) ve **CPU'da çift hassasiyetle yeniden hesaplanan örnek satırlarla** karşılaştırır (10 satır: baş, orta, son; sütunlar dahil). Göreli hata (max|fark|/max|y|) ≤ 5e-7 olmayan hiçbir sonuç raporlanmadı. MMA çekirdeğinde x yarım hassasiyete yuvarlanıp referansta da aynısı yapıldı (`halfX`).
- **Okuyan ↔ üreten eşitliği:** aynı rastgele içerik (`gen_block`/`hashw` ile) hem bellekte doldurulup okunuyor hem çekirdekte üretiliyor; bu yüzden y1 = y2 (ilk dönemde bit düzeyinde, sonra göreli ≤ 2e-7) olmalı. Karışık çekirdeklerde satır türü kuralı: `(satır & 3) < GF` satırlar üretilir (satır grubu uzunluğu 4'ün katı olduğundan derleme zamanında bilinir, SIMD ıraksaması yok).
- **7B eşdeğeri:** 7.62e9 / 2^29 = 14.2. Mat-mat süresi × 14.2 ≈ 7B'de o parti için matmul süresi. B=1 için 1.17 ms × 14.2 = 16.6 ms; llama.cpp 18.2 ms (attention vs. dahil).
- **Veri yerleşimi (çift düzeni):** q4 sözcüğünde ağırlık 2s ← bit 4s (alt 16 bit), ağırlık 2s+1 ← bit 16+4s. Sihirli sayı half2 çıkarımı iki 16 bitlik alanı bir seferde yapsın diye. q8: bayt çiftleri aynı biçim. fp16: sözcük = iki half. (llama.cpp'nin gerçek q4_0 düzeni farklı: alt nibble'lar 0-15, üst nibble'lar 16-31; yama bunu ayrıca ele alır.)

## 5. Komutlar

Metal programları (kökte `swiftc -O X.swift -o X`):
```
swiftc -O probe.swift -o probe && ./probe kernels.metal             # derleme testi
swiftc -O fma2.swift -o fma2 && ./fma2 fma2.metal                    # tepe testi
swiftc -O dq.swift -o dq && ./dq dq.metal                            # dönüşüm mikro-ölçümü
# ilk dönem
./bench kernels.metal 15 ; ./bench2 kernels.metal 15 decomp|hybrid
./bench3 kernels3.metal 15 [log2T] [codes_u32/blok]   # T varsayılan 29
./bench4 kernels3.metal 15 ; ./bench5 kernels5.metal 15 A|B|C
# hızlı dönüşüm
python3 gen_mv9_instances.py v9 > inst.txt ; cat kernels_mv9.metal inst.txt > kernels12.metal ; ./bench12 kernels12.metal 7
python3 gen_mv9_instances.py w  > inst.txt ; cat kernels_mv9.metal inst.txt > kernels13.metal ; ./bench13 kernels13.metal 5
# toplu
python3 gen_batch_instances.py k14 > inst.txt ; cat kernels14.metal inst.txt > k14.metal
./bench15 k14.metal kernels11.metal 5 8192 8192         # B=1..16 karşılaştırması
./bench15 k14.metal kernels11.metal 3 8192 8192 verify  # 96 skaler çekirdeği CPU ile doğrular
python3 gen_batch_instances.py k8r > inst.txt ; cat kernels14.metal inst.txt > k8r.metal ; ./bench8r k8r.metal 7 m0|m1p3|m2|m3 [Bmax]
```
llama.cpp:
```
python3 -m venv venv && venv/bin/pip install cmake
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git src && (cd src && git checkout e475fa2b5)
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build build -j4 --target llama-bench llama-speculative llama-lookup llama-completion test-backend-ops
cp -R build/bin bin_base                       # yamadan önce
python3 -I llama_patch/apply_patch.py src      # yama; sonra yeniden derle
```
Ölçüm komutları (hepsi `scripts/llama_runs.sh` içinde):
- Parti eğrisi: `llama-bench -m MODEL -ngl 99 -p 1,2,...,16 -n 0 -r 3 -o csv` (parti süresi = n / avg_ts × 1000).
- Karar: `-n 128 -p 0` → tg128.
- Spekülatif: `llama-speculative -m 7B -md 0.5B -ngl 99 -ngld 99 --temp 0 -n 256 --spec-draft-n-max k -p "$ESSAY"`. **`--spec-draft-p-min` etkisiz** (n_drafted aynı). `--draft-max` "kaldırıldı" mesajına rağmen aynı sonucu verdi.
- Lookup: `llama-lookup -m 7B -ngl 99 --temp 0 -n 500 -c 4096 -f istem.txt --spec-draft-n-max k` (k=0 düz decode tabanı). İstem dosyaları chat şablonlu ham metin (`<|im_start|>...`).
- Doğruluk: `test-backend-ops test -o MUL_MAT -b MTL0 -p "type_a=q4_0"`; kernel bazlı: `test-backend-ops perf ...` (sonuç satırı, kendinden önceki MUL_MAT satırına ait; araya derleme logları giriyor).
- Yama ortam değişkenleri: `GGML_METAL_SB` (0 kapalı, 1/yok varsayılan, `s` yalnız skaler, `m` MMA B=2-8), `GGML_METAL_SB_TM` (1|2), `GGML_METAL_SB_R` (skaler satır/simdgroup), `GGML_METAL_MM_MIN` (mat-vec↔mat-mat eşiği, varsayılan 8).
- xctrace (başarısız, bkz. `raw_results/07_xctrace_notes.txt`): `xcrun xctrace record --instrument 'Metal GPU Counters' ... --launch -- <komut>`; `xctrace export --toc`.

## 6. Çekirdek tasarımı ve önemli ilkeler (ne öğrenildi)

1. **Yarım-bayt → float dönüşümü pahalı.** `float4(uchar4)` ağırlık başına ~4.84 FMA-yuvası. Sihirli sayı: `as_type<half2>(((w >> s) & 0x000F000F) | 0x64006400) - half2(1032)` iki nibble'ı ~2 yuvada tam olarak (n-8) yapar. Mikro-ölçüm 570 → 1374 G ağırlık/s. Bu, ilk dönemin "hesap sınırlı q4" yorumunu çürüttü.
2. **Okuma yolu bellek sınırlı** (q4_0 B=1: ~250 GB/s, tavan ~254). B=1'de dönüşümü hızlandırmak okumayı hızlandırmaz.
3. **Mat-mat'te darboğaz x trafiği:** her lane her sütun için x'i yeniden yüklüyorsa (llama.cpp `mul_mv_ext` gibi) ağırlık başına B×4 B önbellek trafiği doğar. Çözüm: lane başına 8 ağırlık (bir 32-bit sözcük), x lane'ler arası bitişik; R satır aynı x'i paylaşır.
4. **Yazmaç/derleyici duyarlılığı:** R×B birikeç + x yazmaçları büyüyünce B=4-6 ve B>=9'da ani sıçramalar/patlamalar var (B=6'da 10x). R ve B kombinasyonu taranmadan sonuç yorumlanmaz. Yazılım ön-yükleme (prefetch) iyileştirmedi.
5. **simdgroup_matrix:** lane→eleman eşlemesi `qid=lane/4; fm=(qid&4)+((lane/2)&3); fn=((qid&2)<<1)+((lane&1)<<1)`; lane (fm satırı, fn ve fn+1 sütunu) elemanlarına `thread_elements()[0/1]` ile yazar. Ağırlık karosu A (8 satır × 8 k) doğrudan kayıtta üretilir; B karosu (k × sütun) x'ten. Ölçek blok sonunda satır başına uygulanır (T2 birikeci). B'den bağımsız (8 sütun karosu): B≤8'de düz, B=9-16'da 2 karo.
6. **FMA tepesi 6.15 TFLOPS** (32 bağımsız zincir); 8 zincirle 5.5 çıkar. MMA ~6.2-6.4. Half FMA fp32'den yalnız %7 hızlı. 256 girişli LUT sihirli sayıdan yavaş.
7. **Türetme (tohum+katsayı):** 16 bit tohum + 4×4 bit katsayı / 8 ağırlık; tohumdan xorshift ile P taban (4 bit giriş); `Σ_p t_p·(u_p·x)` biçimi. Bellek q4 ile aynı; fark yalnız hesap. En ucuz üretici E: tek çarpımlı hash, ±1 giriş, 4 değerli seçim (`±t0±t1`); tohum gösteriminin bellek tavanına ulaşır.
8. **Karışık satır kuralı** ve **sanal bayt** tanımı: bölüm 4.

## 7. llama.cpp yaması (kısa özet)

Dosyalar: `ggml-metal.metal`, `ggml-metal-device.cpp/.h`, `ggml-metal-ops.cpp`. Tam metin: `llama_patch/apply_patch.py` (yetkili) ve `sb_q4_0.diff`.
- `ggml_metal_op_mul_mat` başına bir dal: **yalnız** `src0=Q4_0`, `src1=F32`, `ne00 % 256 == 0`, `2 ≤ ne11 ≤ 8`, batch boyutları 1, `nb10 == 4`, `dst` bitişik. B=2-3: `kernel_mul_mv_sb_q4_0_f32_n{2,3}_r{R}` (lane başına 8 ağırlık, R satır paylaşımlı; R varsayılan B=2 için 4, B=3 için 3). B=4-8: `kernel_mul_mm_sb_q4_0_f32_tm{1,2}` (simdgroup_matrix, varsayılan tm=2).
- ggml'nin gerçek q4_0 düzeni (18 B blok: half d + 16 B qs; ağırlık 0-15 alt nibble, 16-31 üst nibble) çekirdeklerde ele alınır: 2 baytlık hizalı `ushort` yükleri, iki ushort → uint birleştirme, sihirli sayı çıkarımı.
- Eşik ortam değişkeni (`GGML_METAL_MM_MIN`): kodda zaten var olan `mul_mv_ext` yolunun üst sınırı (sabit 8) ayarlanabilir; yamasız tek başına B=5-8'i 104-143 → ~92 ms'ye çeker.
- **Yamalanmayanlar:** q6_K lm_head (B=2-3'te eski yoldan 2-3 kez okunur), q4_0 dışı tipler, B>8 (mat-mat 5x taban), `ne00 % 256 != 0` (0.5B taslak modelin K=896'sı), MoE (`mul_mat_id`), batch/yayın boyutları, flash-attention.
- **Doğrulama:** ggml `test-backend-ops` (CPU referansı) n=2..7 q4_0 k=256/1024 m=16 geçti. Gerçek model boyutlarında sayısal doğruluk (greedy çıktı eşitliği, PPL) **kontrol edilmedi**.
- **Uyarı (AGENTS.md):** llama.cpp deposu tamamen/ağırlıklı yapay zekâ üretimi PR'ları kabul etmiyor ("Private forks are exempt"). Yama yerelde kaldı; upstream'e açılacaksa insan yazarlığı ve anlama gerekir.

## 8. Bilimsel hijyen: hata günlüğü (neden ihtiyat)

- **İlk q4 sonuçlarım bozuktu.** İlk dönemde "q4'te bayt yerine hesap yaramaz (türetme karışımı 243→70 GB/s)" dedim. Sebep, yavaş `uchar4→float4` dönüşümüydü; hızlı dönüşümle sonuç q4'te rastgele üretim 0.94x → 1.34x, türetme P=3 0.28x → 0.55x oldu. Tek kaynak ve doğrulanmamış baz çizgisiyle çıkarım yapmanın bedeli.
- İlk okuyan çekirdeğim tavanın altındaydı (175-214 GB/s, tavan 254): kod düzeni yüzünden. Baz çizgisini tavanla doğrulamadan "okuma bellek sınırlı" demek yanıltıcıydı.
- FMA tepesini 2.76 T FMA/s ölçtüm (8 zincir), gerçek ~3.07. "B=4-8'de 1.35x alt sınır, inemez" iddiası geri çekildi.
- Projeksiyon (k=2'de 1.30x) iyimserdi: gerçek 1.12x. Çekirdek oranını tüm token süresine uygulamak yanlış (attention, taslak, lm_head payı).
- `--draft-max` bayrağı "kaldırıldı" mesajına rağmen aynı sonuç verdi: sessiz alias; her bayrak sayıyı değiştirdi mi ayrıca doğrulanmalı (p-min değiştirmedi).
- "Okuma, ölçek yok" varyantı (mv_var MODE 4) okuma yolunda **yavaşladı** (8.70 vs 6.13 ms): derleyici sıralama etkisi; yorumlanmadı.

## 9. Kurtarma notu: yeniden üretilemeyenler ve belirsizlikler

- **Ana oturumun zip/entropi/CPU ölçümleri:** `compress-test.sh`, `lz4-test.sh`, `zstd-q8.sh`, `cpu-paralel.sh`, `byteplane.py` scratchpad'deydi; içeriklerini ve çıktılarını bu bağlamda **hiç görmedim**. Hiçbiri pakette yok.
- Orijinal `out*.txt` dosyaları ve `.trace` dosyaları yok; `raw_results/` bağlamdan aktarıldı (bazı uzun tabloların yalnızca özeti).
- Birebir kopyası olmayanlar: `bench8.swift` (bench7'nin `m7_b→m8_b` ve sayaç tabanlı rastgele referans farkı), `bench11.swift` (bench15'in atası: türev modu xorshift m1p3, doğrulama bloğu ve `m3` yok, `kernels9r`+`kernels11` ile koştu), `bench14.swift` (bench8r'nin B≤8 ve m3 kopyası; bench8r'ye birleştirildi), `kernels8/9r/10r` (kernels14 ve `mm8_prefetch_variant.metal.txt` ile ifade edildi; instantiation'lar `gen_batch_instances.py`'de), `bench8r` başlığı sadeleştirildi (kullanılmayan referans kodu atıldı).
- Örnekleme (instantiation) satırları üreticilerle yeniden üretilir; orijinaller tek tek Python parçalarıyla yazılmıştı. İsimler (`m8_b{B}_r{R}_{...}`, `w_f{F}_g{G}_...`, `v9_...`, `mma_tm{T}_n{N}_m{M}`) bench kodunun beklediği biçimde.
- `llama.cpp` klonu, derleme dizini ve `bin_base` yok. Yeniden klonlayıp `apply_patch.py` ile yamanabilir.
- `matrix.sh` içindeki yollar göreli yazıldı (orijinal mutlak scratchpad yoluydu).
- `sb_q4_0.diff` sahte ağaçta üretildi; bağlam satırları (ops.cpp 2055-2091, device.h 131-135) gerçek metinlerden; device.cpp ve .metal hunk'larının önceki bağlamı yok (git apply'da önce bağlamsız eşleşir). `git apply` ile **denenmedi**. Yetkili yöntem `apply_patch.py`.
- Doğrulanmamış hipotezler: (1) JSON lookup'ta B≥9'daki 1.4x tuhaflığı kısa taslaklı turlardan geliyor olabilir; (2) JSON k=0 yamasızdaki 41.0 tok/s'nin sebebi bilinmiyor; (3) K-quant yavaşlığı dönüşüm kaynaklı olabilir (ölçülmedi); (4) q4_0 B=3'ün hesap yuvası bütçesini aşması (hesap tahmini, sayaçla doğrulanmadı).

## 10. Cloud analizi için açık sorular (yönlendirme değil, malzeme)

- B=3 skaler çekirdek neden 1.09x (mikro) yerine 1.57x (gerçek)? Yük dağılımı (ushort yükleri, hizalama), threadgroup sayısı (m=4096'da 256 tg), lm_head q6_K'nın B=3'te 3 kez okunması, attention/diğer op payı.
- MMA B=4-8 per-op 2.4x vs mikro 1.66x: az threadgroup (m=4096, tm=2 → 64 tg), x yük deseni (`thread_elements` ile 2 skaler yük/lane/kstep), karo tm=1 daha mı iyi tek matriste?
- Dönüşüm/FMA bütçesi: q4 B=2-3 için ağırlık başına ~7 yuva "bedava" (bellek süresi 2.28 ps/ağırlık × 3.07 T/s). Dönüşüm + B FMA bunu aşıyor mu? Hangi işlem sayısı azaltılabilir (ofset ertelemesi işe yaramadı; LUT yaramadı; half birikim +%4)?
- x'i simdgroup başına threadgroup belleğinde paylaşmak, ağırlığı yeniden paketlemek (q4_0 → nibble çifti düzeni, yükleme sırasında bir kez), B=9-16 için daha ucuz MMA (mevcut 3.1x vs llama.cpp 5.1x).
- Spekülatif tarafında taslak maliyeti (token başına 3.4-4.2 ms, 0.5B bağımsız 3.16 ms) ve yield; kısa taslaklı lookup.
- Sayaçlar: xctrace yolu çalışmadı ("not supported on target device"); Xcode Metal Debugger ile .gputrace yolu çalışıyor, sonuç `main_session/TUR1-DOGRULAMA.md` bölüm 19.
