# submission studio 2 · kelompok 13 (k13b - synapse)
# get_model() = arsitektur saja, bobot dimuat evaluate.py
# urutan kelas: 0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

n_kelas = 5
arsitektur = ['convnext_tiny']
ukuran = 256
rata = [(0.485, 0.456, 0.406)]
simpangan = [(0.229, 0.224, 0.225)]


class gabungan(nn.Module):
    def __init__(self, nama_arsitektur):
        super().__init__()
        self.anggota = nn.ModuleList(
            [timm.create_model(a, pretrained=False, num_classes=n_kelas) for a in nama_arsitektur]
        )

    def forward(self, x):
        # normalisasi per anggota (mean/std bisa beda)
        keluaran = []
        for m, mu, sd in zip(self.anggota, rata, simpangan):
            mu_t = torch.tensor(mu, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            sd_t = torch.tensor(sd, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            keluaran.append(m((x - mu_t) / sd_t).float())
        if len(keluaran) == 1:
            return keluaran[0]
        prob = torch.stack([k.softmax(dim=-1) for k in keluaran]).mean(dim=0)
        return prob.clamp_min(1e-8).log()


def get_model():
    return gabungan(arsitektur)


def get_transform():
    return transforms.Compose([
        transforms.Resize((ukuran, ukuran), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
