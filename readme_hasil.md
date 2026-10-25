## ringkasan pendekatan

- backbone: `convnext_tiny.fb_in22k_ft_in1k` (pretrained via timm, fine-tuning penuh)
- strategi latihan: adamw, layer-wise lr decay 0.85, warmup 1 epoch lalu cosine, 1 epoch, batch 32, label smoothing 0.05, ema 0.99 (`AveragedModel`), mixed precision
- augmentasi: random resized crop (skala 0.5-1,0), flip horizontal, randaug, random erasing
- data tambahan: tidak ada
- ukuran input dan preprocessing: 256 x 256, resize langsung bicubic, normalisasi per backbone di dalam model
- penanganan label noisy: 0 gambar latih dikeluarkan setelah audit manual
- pemilihan model: split hold-out stratified 15 persen (`my_val`, 104 gambar), model baru dipakai hanya jika menang lebih dari 0.005 macro-f1 dari kontrol
- keputusan: kandidat terbaik (cnv2_t) hanya -0.1031 dari kontrol, tidak melewati margin 0.005, jadi kontrol dipertahankan
- model akhir dilatih ulang pada seluruh data dengan resep yang sama

## hasil

| putaran | commit | macro-f1 (split validasi lokal) | macro-f1 (diumumkan asisten) |
|---|---|---|---|
| preview 1 (14 okt) | | | |
| preview 2 (21 okt) | | | |
| final (25 okt) | | 0.3057 (hold-out, sebelum pelatihan ulang penuh) | |

perbandingan run hold-out pada `my_val` (epoch terakhir, ema):

| run | backbone | param (juta) | macro-f1 | macro-f1 rata-rata 3 epoch terakhir | akurasi |
|---|---|---|---|---|---|
| cnx_tiny | `convnext_tiny.fb_in22k_ft_in1k` | 27.8 | 0.3057 | 0.3057 | 0.3077 |
| cnv2_t | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 27.9 | 0.2026 | 0.2026 | 0.2404 |
| effv2_s | `tf_efficientnetv2_s.in21k_ft_in1k` | 20.2 | 0.1879 | 0.1879 | 0.1827 |

pemindaian ensemble:

| kombinasi | macro-f1 | ukuran fp16 (mib) |
|---|---|---|
| cnx_tiny + cnv2_t | 0.3212 | 106.3 |
| cnx_tiny + cnv2_t + effv2_s | 0.1918 | 144.8 |
| cnx_tiny + effv2_s | 0.1889 | 91.6 |
| cnv2_t + effv2_s | 0.1889 | 91.7 |

catatan jujur: `my_val` hanya sekitar 104 gambar, jadi selisih di bawah 0.005 tidak dapat dibedakan dari noise. skor model akhir (setelah pelatihan ulang penuh) tidak punya estimasi hold-out sendiri.

## cara reproduksi

1. buka notebook ini di colab dengan gpu, jalankan sekali dengan `smoke = True` untuk memeriksa alur, lalu ubah ke `False`.
2. jalankan semua sel dari atas ke bawah. folder dataset dibaca dari google drive dan tidak disalin ke repo.
3. salin `submission/` (dan `evaluate.py`) ke repo, lalu jalankan `python evaluate.py --submission ./submission --data ./my_val`.
