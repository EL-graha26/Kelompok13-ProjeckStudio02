# Studio 2 — Klasifikasi Sampah · K13B - Synapse

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Najlatika | 123140078 | Kontributor |
| 2  | Hanifah Hasanah | 123140082 | Kontributor |
| 3  | Muhammad Piela Nugraha | 123140200 | Kontributor |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, pustaka `timm`, pretrained ImageNet-22k), head diganti 5 kelas, seluruh layer di-fine-tune. Satu model, tanpa TTA.
- **Strategi pelatihan**: AdamW (LR backbone 4e-4 dengan layer-wise LR decay 0,85, weight decay 0,05), warmup 1 epoch lalu cosine, 12 epoch, batch 32, label smoothing 0,05, drop path 0,1, gradient clipping 1,0, EMA bobot 0,99 (`torch.optim.swa_utils.AveragedModel`), mixed precision. Model final dilatih ulang pada seluruh data dengan resep yang sama.
- **Augmentasi**: RandomResizedCrop (skala 0,5–1,0), flip horizontal, RandAugment, RandomErasing.
- **Data tambahan**: tidak ada (hanya dataset kompetisi).
- **Ukuran input & preprocessing**: 256 × 256, Resize langsung (bicubic); normalisasi mean/std per backbone dikerjakan di dalam model.
- **Penanganan label noisy**: audit integritas (gambar rusak, duplikat isi, konflik label) pada 2.753 gambar; 0 gambar dikeluarkan.
- **Pemilihan model**: split hold-out stratified 15% (`my_val`, 413 gambar) dibuat sekali. Kandidat baru dipakai hanya jika menang lebih dari 0,005 Macro-F1 dari kontrol (ConvNeXt-Tiny) dan muat batas ukuran file 95 MiB.
- **Catatan**: kode notebook dibuat dengan bantuan asisten AI; bobot awal pretrained dari `timm`.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `747c274` | | 0.8941 |
| Preview 2 (21 Okt) | `0b03b3a` | | 0.8979 |
| Final (25 Okt)     | [ISI hash commit setelah push] | 0.9720 | |

Perbandingan run hold-out pada `my_val` (epoch terakhir, EMA):

| Run | Backbone | Param (juta) | Macro-F1 | Rata-rata 3 epoch terakhir | Akurasi |
|---|---|---|---|---|---|
| cnx_tiny | `convnext_tiny.fb_in22k_ft_in1k` | 27.8 | 0.9720 | 0.9720 | 0.9734 |
| cnv2_t | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 27.9 | 0.9736 | 0.9744 | 0.9734 |
| effv2_s | `tf_efficientnetv2_s.in21k_ft_in1k` | 20.2 | 0.9255 | 0.9247 | 0.9274 |

Pemindaian ensemble (dihitung dari probabilitas validasi):

| Kombinasi | Macro-F1 | Ukuran fp16 (MiB) |
|---|---|---|
| cnx_tiny + cnv2_t | 0.9802 | 106.3 |
| cnx_tiny + cnv2_t + effv2_s | 0.9771 | 144.8 |
| cnv2_t + effv2_s | 0.9756 | 91.7 |
| cnx_tiny + effv2_s | 0.9724 | 91.6 |

Keputusan: kandidat terbaik yang muat batas ukuran hanya +0,0036 dari kontrol (di bawah margin 0,005), jadi ConvNeXt-Tiny tunggal dipertahankan.

Catatan jujur: `my_val` hanya 413 gambar, jadi selisih di bawah ±0,005 tidak dapat dibedakan dari noise. Skor model akhir (setelah pelatihan ulang penuh) tidak punya estimasi hold-out sendiri, dan skor hold-out lokal terlihat lebih tinggi daripada skor Preview resmi.

## Cara Reproduksi

1. Buka `studio2_klasifikasi_sampah.ipynb` di Google Colab dengan GPU. Jalankan sekali dengan `smoke = True` untuk memeriksa alur, lalu ubah ke `False`.
2. Jalankan semua sel dari atas ke bawah. Dataset dibaca dari Google Drive dan tidak disalin ke repo.
3. Salin `submission/` hasil notebook ke repo, lalu: `python evaluate.py --submission ./submission --data ./my_val`.

`train_baseline.py` dan `export_submission.py` adalah skrip percobaan awal; bobot akhir dihasilkan oleh notebook di atas.

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model akhir (fp16)
  config.json       # catatan konfigurasi hasil ekspor
  requirements.txt  # paket tambahan (timm)
evaluate.py         # salinan skrip evaluasi resmi 
studio2_klasifikasi_sampah.ipynb  # notebook pelatihan 
train_baseline.py, export_submission.py   # skrip percobaan awal
hasil_akhir.json    # ringkasan keputusan pemilihan model
```
