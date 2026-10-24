import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import evaluate                          # CLASSES, EXTS, metrics() — identik dengan penilai
from submission.model import Member

CLASSES = evaluate.CLASSES
NUM_CLASSES = len(CLASSES)


# ----------------------------------------------------------------------------- data
def ensure_val_split(src, val_dir, frac, seed):
    """Buat my_val/ (salinan 15% gambar per kelas) HANYA jika belum ada.
    Kalau my_val/ sudah ada (mis. dari baseline starter kit) split lama dipakai ulang,
    jadi angka lama dan baru bisa dibandingkan."""
    val_dir = Path(val_dir)
    if val_dir.exists():
        return
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in evaluate.EXTS)
        rng.shuffle(files)
        out = val_dir / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for f in files[:int(len(files) * frac)]:
            shutil.copy2(f, out / f.name)


def val_keys(val_dir):
    return {(p.parent.name, p.name) for p in Path(val_dir).glob("*/*")
            if p.suffix.lower() in evaluate.EXTS}


def train_transform(size):
    # Output tensor 0..1 (normalisasi dikerjakan di dalam Member)
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3),
                            interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), value=0),
    ])


def eval_transform(size):
    return T.Compose([T.Resize((size, size), interpolation=InterpolationMode.BICUBIC), T.ToTensor()])


# ----------------------------------------------------------------------------- model utils
class EMA:
    """Rata-rata bergerak eksponensial dari bobot: lebih stabil daripada bobot epoch terakhir."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1 - d)
            else:
                v.copy_(src[k])


def param_groups(model, lr, head_mult, wd):
    """4 grup: (backbone | head) x (pakai | tanpa weight decay). Bias & parameter 1-D (norm) tanpa decay."""
    head_ids = {id(p) for p in model.net.get_classifier().parameters()}
    buckets = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            key = ("head" if id(p) in head_ids else "body", p.ndim > 1 and not n.endswith(".bias"))
            buckets.setdefault(key, []).append(p)
    return [dict(params=ps, lr=lr * (head_mult if part == "head" else 1.0),
                 weight_decay=wd if decay else 0.0) for (part, decay), ps in buckets.items()]


@torch.no_grad()
def run_eval(model, dl, device, amp):
    model.eval()
    probs, ys = [], []
    for x, y in dl:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            p = model(x.to(device, non_blocking=True)).float().softmax(-1)
        probs.append(p.cpu())
        ys.append(y)
    probs, ys = torch.cat(probs), torch.cat(ys)
    return evaluate.metrics(ys.tolist(), probs.argmax(1).tolist()), probs


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder train/ dari dataset (ImageFolder)")
    ap.add_argument("--val-dir", default="my_val")
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32, help="harus genap (Mixup)")
    ap.add_argument("--lr", type=float, default=1e-4, help="LR backbone")
    ap.add_argument("--head-mult", type=float, default=10.0, help="LR head = lr x nilai ini")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--drop-rate", type=float, default=0.2, help="dropout sebelum head")
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--mix-prob", type=float, default=0.5, help="peluang Mixup/CutMix per batch (0 = mati)")
    ap.add_argument("--mix-off-last", type=int, default=3, help="Mixup/CutMix dimatikan di N epoch terakhir")
    ap.add_argument("--ema", type=float, default=0.995, help="decay EMA (0 = mati)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--full", action="store_true", help="latih di SEMUA data (my_val ikut) = model final")
    ap.add_argument("--no-pretrained", action="store_true", help="tanpa bobot ImageNet (hanya untuk uji kode)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    assert a.batch_size % 2 == 0, "batch-size harus genap"

    seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    amp = device == "cuda"
    name = a.name or f"{a.backbone.split('.')[0]}_{time.strftime('%m%d-%H%M')}"
    run = Path("runs") / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=2))

    # ---- data: train = semua gambar di --data KECUALI my_val (atau semua, bila --full)
    ensure_val_split(a.data, a.val_dir, 0.15, a.split_seed)
    ds_tr = datasets.ImageFolder(a.data, train_transform(a.img_size))
    ds_ev = datasets.ImageFolder(a.data, eval_transform(a.img_size))
    assert ds_tr.classes == CLASSES, f"urutan kelas salah / ada folder asing: {ds_tr.classes}"
    vk = val_keys(a.val_dir)
    keys = [(Path(p).parent.name, Path(p).name) for p, _ in ds_tr.samples]
    va_idx = [i for i, k in enumerate(keys) if k in vk]
    tr_idx = list(range(len(keys))) if a.full else [i for i, k in enumerate(keys) if k not in vk]
    assert len(va_idx) > 0, "my_val tidak cocok dengan --data (nama file beda?)"
    print(f"device={device} amp={amp} backbone={a.backbone} img={a.img_size} | "
          f"train={len(tr_idx)} val={len(va_idx)} mode={'FULL' if a.full else 'hold-out'}")

    kw = dict(num_workers=a.workers, pin_memory=amp, persistent_workers=a.workers > 0)
    dl_tr = DataLoader(Subset(ds_tr, tr_idx), a.batch_size, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(Subset(ds_ev, va_idx), 64, shuffle=False, **kw)

    # ---- model, optimizer, scheduler
    model = Member(a.backbone, pretrained=not a.no_pretrained,
                   drop_path=a.drop_path, drop_rate=a.drop_rate).to(device)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.head_mult, a.wd))
    total, warm = a.epochs * len(dl_tr), max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        max(0.01, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))
    mix = (Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=a.mix_prob, switch_prob=0.5,
                 label_smoothing=a.smooth, num_classes=NUM_CLASSES) if a.mix_prob > 0 else None)
    soft_ce = SoftTargetCrossEntropy()
    hard_ce = nn.CrossEntropyLoss(label_smoothing=a.smooth)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ---- training loop
    log_f = open(run / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "f1_raw", "f1_ema", "acc_ema", "mix", "sec"])
    f1_hist, last = [], None
    for ep in range(1, a.epochs + 1):
        t0, loss_sum, n = time.time(), 0.0, 0
        use_mix = mix is not None and ep <= a.epochs - a.mix_off_last
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_mix:
                x, y = mix(x, y)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = soft_ce(out, y) if use_mix else hard_ce(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            loss_sum += loss.item() * len(x)
            n += len(x)
        m_raw, _ = run_eval(model, dl_va, device, amp)
        m_ema, probs = run_eval(ema.module, dl_va, device, amp) if ema else (m_raw, None)
        last = (m_ema, probs)
        f1_hist.append(m_ema["macro_f1"])
        sec = time.time() - t0
        log.writerow([ep, f"{loss_sum / n:.4f}", f"{m_raw['macro_f1']:.4f}", f"{m_ema['macro_f1']:.4f}",
                      f"{m_ema['accuracy']:.4f}", int(use_mix), f"{sec:.0f}"])
        log_f.flush()
        print(f"epoch {ep:2d}/{a.epochs} loss={loss_sum / n:.4f} | F1 raw={m_raw['macro_f1']:.4f} "
              f"ema={m_ema['macro_f1']:.4f} acc={m_ema['accuracy']:.4f} | mix={'on' if use_mix else 'off'} "
              f"{sec:.0f}s", flush=True)
    log_f.close()

    # ---- simpan epoch TERAKHIR (bobot EMA kalau aktif), bukan "epoch terbaik"
    final = ema.module if ema else model
    torch.save({k: v.detach().cpu().clone() for k, v in final.state_dict().items()}, run / "member.pth")
    m, probs = last
    k = min(3, len(f1_hist))
    info = dict(backbone=a.backbone, img_size=a.img_size, full=a.full, epochs=a.epochs,
                n_train=len(tr_idx), n_val=len(va_idx), timm_version=timm.__version__,
                torch_version=torch.__version__, f1_last=round(m["macro_f1"], 4),
                f1_mean_last3=round(float(np.mean(f1_hist[-k:])), 4),
                per_class_f1={c: round(v, 4) for c, v in m["per_class_f1"].items()},
                confusion_matrix=m["confusion_matrix"])
    (run / "info.json").write_text(json.dumps(info, indent=2))

    if not a.full:  # daftar gambar paling "yakin salah" -> bahan audit label noise
        rows = []
        for (path, lab), p in zip((ds_ev.samples[i] for i in va_idx), probs):
            pr = int(p.argmax())
            rows.append([f"{Path(path).parent.name}/{Path(path).name}", CLASSES[lab], CLASSES[pr],
                         round(float(p[lab]), 4), round(float(p[pr]), 4), int(pr == lab)])
        with open(run / "val_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", "pred", "p_label", "p_pred", "benar"])
            w.writerows(sorted(rows, key=lambda r: r[3]))

    print("\n" + "=" * 70)
    print(f"run: {run}   Macro-F1 epoch terakhir = {m['macro_f1']:.4f}   rata-rata 3 epoch terakhir = {info['f1_mean_last3']:.4f}")
    print("F1 per kelas:", {c: round(v, 4) for c, v in m["per_class_f1"].items()})
    print("confusion matrix (baris=asli, kolom=prediksi):")
    for c, row in zip(CLASSES, m["confusion_matrix"]):
        print(f"  {c:<15} {row}")
    if a.full:
        print("CATATAN: --full -> my_val ikut dilatih, skor di atas BUKAN estimasi generalisasi.")
    print(f"\nLangkah berikut: python export_submission.py --runs {run}")
    print("=" * 70)


if __name__ == "__main__":
    main()
