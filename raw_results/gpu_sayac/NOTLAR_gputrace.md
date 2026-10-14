# GPU yakalama notları (11 Eki 2026)

Amaç: B=4-8 doğrulama grubunda yamalı çarpma yolunun (kernel_mul_mm_sb_q4_0_f32_tm2) neden ~2x yavaş olduğunu sayaçla görmek.

## Yöntem
- llama.cpp e475fa2b5 + yama. Kanca: GGML_METAL_CAPTURE_COMPUTE=5 METAL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_DIR=~/claude-scratch/gputrace
- Yerel değişiklik: ggml-metal-context.m çıktı dizinini GGML_METAL_CAPTURE_DIR'den alır (varsayılan /tmp).
- Yerel değişiklik: tests/test-backend-ops.cpp'ye Qwen2.5-7B boyutları (18944x3584, 3584x18944; n=1,2,3,4,8) eklendi.
- Komut: test-backend-ops perf -o MUL_MAT -b MTL0 -p "type_a=q4_0,type_b=f32,m=M,n=N,k=K,"
- Sayaç okuma: Xcode 26.6 > .gputrace > Performance > Counters > CSV dışa aktarma (elle).

## İşlem başına süre (us, test-backend-ops perf, tek işlem)
| boyut (m x k) | n=1 | n=2 | n=3 | n=4 | n=8 |
|---|---|---|---|---|---|
| 4096 x 14336 | 141 | 152 | 240 | 397 | 388 |
| 18944 x 3584 (up) | - | - | - | 301 | 309 |
| 3584 x 18944 (down) | - | - | - | 435 | 461 |

up ve down aynı bayt ve aynı işlem sayısı; down %45 yavaş (56 grup vs 296 grup).

## n4_mma sayaçları (4096x14336, B=4, MMA tm2; encoder düzeyi toplam)
- Bellek: cihaz bant ~75 GB/s (tavan ~258), LLC limiter %11, L1 limiter %14, MMU %7, başlatma %0.02
- ALU kullanımı %41.6; F32 limiter %67.9, F32 kullanımı %55.7; F16 kullanımı %6.4
- Instruction Throughput limiter %64.4-64.7; Int&Conditional limiter %20.6
- Kernel Occupancy %14.5. Kaynak: 64 grup x 128 parçacık = 8192 parçacık / (16 çekirdek x ~3072) = %16.7 (hesap, sayaç değil)
- Kernel ALU komut payı: float %66.9, half %7.7, int&cond %21.6, int&complex %3.9

## Açık
- B=8 ve B=1 sayaçları yok; "8 sütun hesaplanıyor" için B=4 ile B=8 komut sayısı karşılaştırması gerekiyor.
- Dosyalar: n1/n2/n3/n4_mma/n8_sb1 (4096x14336), up_n4/up_n8, down_n4/down_n8 (.gputrace)
