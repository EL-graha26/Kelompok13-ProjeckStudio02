"""export_submission.py — ubah hasil training (runs/*) menjadi folder submission/ yang siap dinilai.

  # 1) (opsional) lihat pengaruh resolusi uji & TTA pada my_val, TANPA menulis apa pun:
  python export_submission.py --runs runs/cnx_tiny --scan
  # 2) tulis submission/ (config.json, model.pth, requirements.txt) lalu jalankan evaluate.py resmi:
  python export_submission.py --runs runs/cnx_tiny_full
  # ensemble: sebut beberapa run
  python export_submission.py --runs runs/cnx_tiny_full runs/effv2_s_full --img-size 256

submission/model.py TIDAK diubah (membaca config.json). Bobot disimpan fp16 kalau > 95 MB
(batas GitHub 100 MB); argmax praktis identik. Bila masih > 100 MB pakai Git LFS / GitHub Release.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import timm
import torch
from PIL import Image

import evaluate
from submission import model as sub


def load_runs(run_dirs):
    infos, states = [], []
    for r in map(Path, run_dirs):
        infos.append(json.loads((r / "info.json").read_text()))
        states.append(torch.load(r / "member.pth", map_location="cpu", weights_only=True))
    return infos, states


def merge(states):
    return {f"members.{i}.{k}": v for i, st in enumerate(states) for k, v in st.items()}


@torch.no_grad()
def macro_f1(model, tf, items, device, bs=32):
    model.eval()
    y_true, y_pred = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in chunk]).to(device)
        y_pred += model(x).argmax(1).cpu().tolist()
        y_true += [c for _, c in chunk]
    return evaluate.metrics(y_true, y_pred)["macro_f1"]


@torch.no_grad()
def cpu_budget(sub_dir, val_dir, n=32, n_val=740, limit_min=15):
    """Perkiraan waktu evaluasi n_val gambar di CPU 2 core (mesin asisten bisa tanpa GPU)."""
    model, tf = evaluate.load_submission(sub_dir, "cpu")
    items = evaluate.list_images(val_dir)
    items = random.Random(0).sample(items, min(n, len(items)))
    torch.set_num_threads(2)
    t0 = time.time()
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in items])
    t_pre = time.time() - t0
    model(x[:2])  # pemanasan
    t0 = time.time()
    model(x)
    t_fwd = time.time() - t0
    est = (t_pre + t_fwd) / len(items) * n_val / 60
    print(f"CPU 2 thread: praproses {t_pre / len(items) * 1000:.0f} ms + forward {t_fwd / len(items) * 1000:.0f} ms "
          f"per gambar -> perkiraan {est:.1f} menit untuk {n_val} gambar (batas {limit_min} menit)")
    if est > limit_min / 2:
        print("PERINGATAN: perkiraan > separuh batas. Mesin penilai bisa lebih lambat -> kecilkan resolusi, "
              "kurangi anggota ensemble, atau matikan TTA.")
    else:
        print("Anggaran waktu CPU aman (< separuh batas).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="folder runs/<name> (urutan = urutan anggota ensemble)")
    ap.add_argument("--img-size", type=int, default=None, help="resolusi uji (default: resolusi training run pertama)")
    ap.add_argument("--tta", action="store_true", help="TTA flip horizontal (menggandakan waktu inferensi)")
    ap.add_argument("--half", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--scan", action="store_true", help="hanya tampilkan Macro-F1 my_val per resolusi/TTA, lalu keluar")
    ap.add_argument("--val-dir", default="my_val")
    ap.add_argument("--sub-dir", default="submission")
    a = ap.parse_args()

    infos, states = load_runs(a.runs)
    names = [i["backbone"] for i in infos]
    size = a.img_size or infos[0]["img_size"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    merged = merge(states)

    ens = sub.Ensemble(names, a.tta)
    ens.load_state_dict(merged, strict=True)  # gagal di sini = arsitektur & bobot tidak cocok
    ens.to(device)

    if a.scan:
        items = evaluate.list_images(a.val_dir)
        print(f"my_val = {len(items)} gambar | 1 gambar salah ~ {100 / len(items):.2f}% akurasi: "
              "selisih kecil (< ~0.005) HANYA noise.")
        print(f"{'ukuran':>7} {'TTA':>5} {'Macro-F1':>9}")
        for s in sorted({size, size + 32, size + 64}):
            for tta in (False, True):
                ens.tta_hflip = tta
                print(f"{s:>7} {str(tta):>5} {macro_f1(ens, sub.build_eval_transform(s), items, device):>9.4f}", flush=True)
        return

    sub_dir = Path(a.sub_dir)
    sub_dir.mkdir(exist_ok=True)
    cfg = dict(members=names, img_size=size, tta_hflip=bool(a.tta),
               timm_version=timm.__version__, torch_version=torch.__version__)
    (sub_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    mb32 = sum(v.numel() * v.element_size() for v in merged.values()) / 2 ** 20
    half = a.half == "yes" or (a.half == "auto" and mb32 > 95)
    out = {k: (v.half() if (half and v.is_floating_point()) else v) for k, v in merged.items()}
    torch.save(out, sub_dir / "model.pth")
    (sub_dir / "requirements.txt").write_text(f"timm=={timm.__version__}\n")
    mb = (sub_dir / "model.pth").stat().st_size / 2 ** 20
    print(f"ditulis: {sub_dir}/config.json, model.pth ({mb:.1f} MB, {'fp16' if half else 'fp32'}), requirements.txt")
    print(f"anggota: {names} | resolusi {size} | TTA flip {a.tta}")
    if mb > 500:
        print("!! MELEBIHI batas 500 MB.")
    elif mb > 100:
        print("! > 100 MB: GitHub menolak push biasa -> pakai Git LFS atau GitHub Release, tulis tautannya di README.")

    print("\n=== evaluate.py resmi ===")
    r = subprocess.run([sys.executable, "evaluate.py", "--submission", str(sub_dir), "--data", a.val_dir])
    if r.returncode != 0:
        raise SystemExit("evaluate.py GAGAL — perbaiki sebelum commit.")
    print("\n=== anggaran waktu ===")
    cpu_budget(sub_dir, a.val_dir)
    if any(i.get("full") for i in infos):
        print("\nCATATAN: ada run --full -> my_val ikut dilatih, skor evaluate.py di atas hanya cek 'bisa jalan'.")


if __name__ == "__main__":
    main()
