# M4 Pro LLM Decode Bandwidth Deneyleri

Apple M4 Pro (24 GB, 273 GB/s) üzerinde, llama.cpp/Metal ile LLM decode hızının bellek bandı darboğazı üzerine bir günlük deney serisinin tüm kodu, ölçümleri ve know-how'ı.

## Öne çıkan bulgular

- Decode bellek bandı sınırlı, prefill hesap sınırlı; quantize ağırlıklarda kayıpsız sıkıştırmanın tavanı %5-8 (byte entropisi ölçümü).
- llama.cpp Metal'de 2-8 tokenlık toplu decode, token başına tam fiyat ödüyor; bu bir donanım sınırı değil — ağırlığı bir kez okuyan özel kernel B=2'yi tek token fiyatına işliyor. Yama ile kopya-ağır işlerde (kod/JSON düzenleme) 7B modelde 2-2.8x uçtan uca hızlanma ölçüldü.
- "Ağırlığı okuma, tohumdan türet" (SeedLM benzeri) yaklaşımı bu GPU'da q4_0 okumasını geçemiyor; kazanç penceresi yalnız fp16/q8 ikamesi.
- Nibble→float dönüşümünde sihirli sayı hilesi açmayı 2.4x hızlandırıyor; K-quant'ların yavaşlığının kaynağı bellek değil dönüşüm maliyeti.

## Önemli uyarı

Ölçümlerin çoğu tek deneycinin tek koşuluk sonuçlarıdır ve bağımsız tekrar yapılmamıştır; SONUCLAR.md'de her tablonun başında kaynak/tekrar/doğrulama etiketi vardır. Doğrulanmamış iddia olarak okuyun.

## Yapı

- `KNOWHOW.md` — komutlar, ölçüm düzeni, kernel tasarım ilkeleri, yama özeti, hata günlüğü
- `SONUCLAR.md` — tüm ölçüm tabloları, kaynak etiketleriyle
- `metal/` — Metal kernel'leri ve Swift ölçüm programları (mikro-ölçümler, dönüşüm, toplu mat-mat)
- `llama_patch/` — llama.cpp Metal arka ucu küçük-batch yaması (.diff + uygulayıcı)
- `scripts/` — istem üreticileri, tarama betikleri, ölçüm zincirleri
- `main_session/` — zip/entropi/CPU açma ölçümleri (script + sonuç)
- `raw_results/` — ham çıktılar

Tüm Metal/Swift dosyaları derleme kontrolünden geçmiştir; ölçüm programları `swiftc` ile derlenip kernel dosyası argümanıyla çalıştırılır (ayrıntı KNOWHOW.md'de).
