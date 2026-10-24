"""Submission Studio 2 — Klasifikasi Sampah (IF25-40401) · Kelompok 13.

Kontrak evaluate.py:
    get_model()     -> nn.Module, output [batch, 5], hanya ARSITEKTUR (tanpa unduh bobot)
    get_transform() -> PIL.Image (RGB) -> Tensor [3, H, W]

Pengaturan (backbone, resolusi, TTA) dibaca dari config.json di folder yang sama. File itu
ditulis otomatis oleh export_submission.py, jadi model.py / config.json / model.pth selalu
cocok satu sama lain. Urutan kelas (= ImageFolder, alfabetis):
    0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam
"""
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 5
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path=CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan export_submission.py, lalu commit config.json "
            "bersama model.pth.")
    return json.loads(path.read_text())


class Member(nn.Module):
    """Satu backbone timm + normalisasi (mean/std) miliknya sendiri.

    Input: tensor 0..1 [B, 3, H, W]. Normalisasi sengaja ditaruh DI DALAM model supaya
    backbone yang beda mean/std (mis. ConvNeXt vs EfficientNetV2) bisa digabung dalam satu
    ensemble dengan satu get_transform().
    """

    def __init__(self, name, pretrained=False, drop_path=0.0, drop_rate=0.0):
        super().__init__()
        kw = {}
        if drop_path > 0:
            kw["drop_path_rate"] = drop_path
        if drop_rate > 0:
            kw["drop_rate"] = drop_rate
        try:
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES, **kw)
        except TypeError:  # backbone yang tidak mengenal drop_path / drop_rate
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES)
        cfg = timm.data.resolve_model_data_config(self.net)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class Ensemble(nn.Module):
    """Rata-rata probabilitas beberapa Member (+ opsional TTA flip). Output: log-probabilitas."""

    def __init__(self, names, tta_hflip=False):
        super().__init__()
        self.members = nn.ModuleList([Member(n) for n in names])
        self.tta_hflip = tta_hflip

    def forward(self, x):
        views = (x, torch.flip(x, dims=[-1])) if self.tta_hflip else (x,)
        total, n = 0.0, 0
        for m in self.members:
            for v in views:
                total = total + m(v).float().softmax(dim=-1)
                n += 1
        return (total / n).clamp_min(1e-8).log()  # argmax = prediksi


def build_eval_transform(size):
    """Resize langsung ke size x size (sama dengan cara data dilihat saat training)."""
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


def get_model():
    cfg = load_config()
    return Ensemble(cfg["members"], cfg.get("tta_hflip", False))


def get_transform():
    return build_eval_transform(load_config()["img_size"])
