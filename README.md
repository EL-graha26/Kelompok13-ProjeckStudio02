# Studio 2 — Klasifikasi Sampah · K13B - Synapse

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Najlatika    | 123140078   | Kontributor                 |
| 2  | Hanifah Hasanah | 123140082    | Kontributor                   |
| 3  | Muhammad Piela Nugraha | 123140200    | Kontributor                   |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: (mis. ResNet-50 pretrained ImageNet, CNN from scratch, ...)
- **Strategi pelatihan**: (freezing, learning rate, scheduler, epoch, augmentasi, ...)
- **Data tambahan**: (tidak ada / sebutkan sumber dan jumlahnya)
- **Ukuran input & preprocessing**:

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | | 0.8941 |
| Preview 2 (21 Okt) | | | 0.8979 |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
