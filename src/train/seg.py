#!/usr/bin/env python
"""Lesion segmentation: pretraining on simulated B-mode frames, zero-shot and fine-tuning on real data.

Model: U-Net with a ResNet34 encoder (segmentation_models_pytorch), 1-channel 256 x 256 input, per-image
z-score intensity. Recipe (identical for every init): Dice + BCE, AdamW (lr 5e-4, wd 1e-4), cosine
schedule with 200-iteration warm-up, batch 16, GPU augmentation (random scale 0.75-1.25; 0.8-2 in pretraining, rotation
+-15 deg, translation, horizontal flip, gamma / contrast / brightness, Gaussian noise and blur),
fixed iteration budget (no early stopping, so the test split is never used for selection).

    python src/train/seg.py cache-sim --sim $BIG/simdata/v1 --out $BIG/simdata/v1/sim256.h5
    python src/train/seg.py pretrain --data $BIG/simdata/v1/sim256.h5 --init imagenet --out $BIG/seg/sim_in
    python src/train/seg.py finetune --real $BIG/realseg/BUSI.h5 --init sim:$BIG/seg/sim_in/model.pt \
        --n-train 32 --seed 0 --out $BIG/seg/ft/BUSI_n32_simin_s0
    python src/train/seg.py evaluate --model $BIG/seg/sim_in/model.pt --real $BIG/realseg/*.h5

Runs checkpoint every 500 iterations and resume (preemptible Slurm partitions).
Metrics: per-image Dice and IoU on the test split at 256 x 256 (threshold 0.5), averaged over images.
"""
import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

S = 256


def amp(dev):
    """bf16 autocast on GPUs with native bf16 (compute capability >= 8: A100 / H100); fp32 otherwise (V100)."""
    on = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on)


# ------------------------------------------------------------------ data
def zscore(x):
    m = x.mean((-1, -2), keepdim=True)
    s = x.std((-1, -2), keepdim=True)
    return (x - m) / (s + 1e-6)


def load_real(path, split, n=None, seed=0):
    with h5py.File(path, "r") as f:
        sel = np.nonzero(f["split"][:] == split.encode())[0]
        if n is not None and n < len(sel):
            sel = np.sort(np.random.default_rng(seed).choice(sel, n, replace=False))
        return f["img"][sel], f["mask"][sel], sel


def load_sim(path, val_frac=0.1):
    """sim256.h5 -> train / val split by phantom (val phantoms are never trained on)."""
    with h5py.File(path, "r") as f:
        img, mask = f["img"][:], f["mask"][:]
        ph = np.array([json.loads(m)["phantom"] for m in f["meta"][:]])
    names = sorted(set(ph))
    rng = np.random.default_rng(0)
    val = set(rng.choice(names, max(1, int(round(val_frac * len(names)))), replace=False))
    v = np.isin(ph, list(val))
    return (img[~v], mask[~v]), (img[v], mask[v]), sorted(val)


def to_gpu(img, mask, dev):
    x = torch.from_numpy(img).to(dev).float()[:, None] / 255
    y = torch.from_numpy(mask).to(dev).float()[:, None].clamp(0, 1)
    return x, y


def augment(x, y, g, zoom=(0.75, 1.25)):
    n, dev = x.shape[0], x.device
    ang = (torch.rand(n, device=dev, generator=g) * 2 - 1) * math.radians(15)
    lz = torch.log(torch.tensor(zoom, device=dev))
    sc = torch.exp(lz[0] + torch.rand(n, device=dev, generator=g) * (lz[1] - lz[0]))   # > 1 zooms in
    flip = torch.where(torch.rand(n, device=dev, generator=g) < 0.5, -1.0, 1.0)
    t = (torch.rand(n, 2, device=dev, generator=g) * 2 - 1) * 0.1
    th = torch.zeros(n, 2, 3, device=dev)
    th[:, 0, 0], th[:, 0, 1] = torch.cos(ang) / sc * flip, -torch.sin(ang) / sc
    th[:, 1, 0], th[:, 1, 1] = torch.sin(ang) / sc * flip, torch.cos(ang) / sc
    th[:, :, 2] = t
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y = (F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5).float()
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, 1, 1, 1, device=dev, generator=g)
    x = x.clamp(0, 1) ** r(0.7, 1.5)
    x = (x - 0.5) * r(0.75, 1.25) + 0.5 + r(-0.1, 0.1)
    blur = torch.rand(n, device=dev, generator=g) < 0.2
    if blur.any():
        k = torch.exp(-0.5 * (torch.arange(-3, 4, device=dev) / 1.0) ** 2)
        k = (k / k.sum())
        xb = F.conv2d(F.conv2d(x[blur], k.view(1, 1, 1, 7), padding=(0, 3)), k.view(1, 1, 7, 1), padding=(3, 0))
        x[blur] = xb
    x = x + torch.randn(x.shape, device=dev, generator=g) * r(0, 0.05)
    return zscore(x), y


# ----------------------------------------------------------------- model
def make_model(init):
    import segmentation_models_pytorch as smp
    weights = "imagenet" if init in ("imagenet",) else None
    m = smp.Unet("resnet34", encoder_weights=weights, in_channels=1, classes=1)
    if init.startswith("sim:"):
        m.load_state_dict(torch.load(init[4:], map_location="cpu"))
    return m


def loss_fn(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + y.sum() + 1)
    return dice + F.binary_cross_entropy_with_logits(logit, y)


@torch.no_grad()
def predict(model, img, dev, bs=64):
    model.eval()
    out = []
    for i in range(0, len(img), bs):
        x = zscore(torch.from_numpy(img[i:i + bs]).to(dev).float()[:, None] / 255)
        with amp(dev):
            lo = model(x) + model(x.flip(-1)).flip(-1)        # flip test-time augmentation
        out.append((lo.float() > 0).cpu().numpy()[:, 0])
    model.train()
    return np.concatenate(out)


def metrics(pred, mask):
    m = mask.astype(bool)
    inter = (pred & m).sum((1, 2))
    ps, ms = pred.sum((1, 2)), m.sum((1, 2))
    union = ps + ms - inter
    dice = np.where(ps + ms > 0, 2 * inter / np.maximum(ps + ms, 1), 1.0)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return dict(dice=round(float(dice.mean()), 4), iou=round(float(iou.mean()), 4),
                dice_median=round(float(np.median(dice)), 4), empty_pred=round(float((ps == 0).mean()), 4), n=len(m))


def train(model, x, y, out, iters, dev, lr=5e-4, bs=16, seed=0, log_every=500, evalf=None, zoom=(0.75, 1.25)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 200) * 0.5 * (1 + math.cos(math.pi * i / iters)))
    g = torch.Generator(device=dev).manual_seed(seed)
    start, log = 0, []
    ck = out / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        g.set_state(st["gen"].cpu())
        start, log = st["it"], st["log"]
        print(f"resumed at {start}", flush=True)
    t0, run, nrun = time.time(), 0.0, 0
    for it in range(start, iters):
        idx = torch.randint(0, x.shape[0], (bs,), device=dev, generator=g)
        xb, yb = augment(x[idx], y[idx], g, zoom)
        with amp(dev):
            lo = model(xb)
        loss = loss_fn(lo.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run, nrun = run + loss.item(), nrun + 1
        if (it + 1) % log_every == 0 or it + 1 == iters:
            rec = dict(it=it + 1, loss=round(run / nrun, 4), s=round(time.time() - t0, 1))
            if evalf is not None:
                rec.update(evalf(model))
            log.append(rec)
            print(json.dumps(rec), flush=True)
            run, nrun = 0.0, 0
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=g.get_state(), it=it + 1, log=log), ck)
    torch.save(model.state_dict(), out / "model.pt")
    return log


# -------------------------------------------------------------- commands
def cmd_cache_sim(a):
    """Shards (native 0.078 mm frames) -> one 256 x 256 HDF5 (bilinear image, nearest mask)."""
    imgs, masks, metas = [], [], []
    for s in sorted(Path(a.sim).glob("shard_*.h5")):
        with h5py.File(s, "r") as f:
            for i in range(0, f["img"].shape[0], 256):
                x = torch.from_numpy(f["img"][i:i + 256]).float()[:, None]
                m = torch.from_numpy(f["mask"][i:i + 256]).float()[:, None]
                imgs.append(F.interpolate(x, (S, S), mode="bilinear", antialias=True, align_corners=False)
                            .round().clamp(0, 255).byte().numpy()[:, 0])
                masks.append(F.interpolate(m, (S, S), mode="nearest").byte().numpy()[:, 0])
            metas += list(f["meta"][:])
    with h5py.File(a.out, "w") as f:
        f.create_dataset("img", data=np.concatenate(imgs), compression="gzip")
        f.create_dataset("mask", data=np.concatenate(masks), compression="gzip")
        f.create_dataset("meta", data=np.array(metas, dtype=object), dtype=h5py.string_dtype())
    print(f"{len(metas)} frames -> {a.out}")


def real_evaluator(paths, dev):
    sets = {Path(p).stem: load_real(p, "test")[:2] for p in paths}
    return lambda model: {k: metrics(predict(model, i, dev), m)["dice"] for k, (i, m) in sets.items()}


def cmd_pretrain(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (ti, tm), (vi, vm), val_ph = load_sim(a.data)
    x, y = to_gpu(ti, tm, dev)
    print(f"sim train {len(ti)} / val {len(vi)} frames (val phantoms {val_ph})", flush=True)
    ev_real = real_evaluator(a.real, dev) if a.real else None

    def evalf(model):
        r = {"sim_val": metrics(predict(model, vi, dev), vm)["dice"]}
        return {**r, **(ev_real(model) if ev_real else {})}     # monitoring only, never used for selection
    model = make_model(a.init)
    log = train(model, x, y, a.out, a.iters, dev, seed=a.seed, log_every=1000, evalf=evalf, zoom=tuple(a.zoom))
    (Path(a.out) / "train.json").write_text(json.dumps(dict(args=vars(a), val_phantoms=val_ph, log=log)))


def cmd_finetune(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ti, tm, sel = load_real(a.real, "train", a.n_train, a.seed)
    x, y = to_gpu(ti, tm, dev)
    model = make_model(a.init)
    out = Path(a.out)
    if (out / "result.json").exists():
        print("done", out)
        return
    log = train(model, x, y, out, a.iters, dev, seed=a.seed, log_every=a.iters)
    test_i, test_m, _ = load_real(a.real, "test")
    res = dict(dataset=Path(a.real).stem, init=a.init_name or a.init, n_train=len(sel), seed=a.seed,
               iters=a.iters, **metrics(predict(model, test_i, dev), test_m), loss=log[-1]["loss"])
    (out / "result.json").write_text(json.dumps(res))
    (out / "ckpt.pt").unlink(missing_ok=True)          # keep only the final weights
    if not a.keep:
        (out / "model.pt").unlink(missing_ok=True)
    print(json.dumps(res), flush=True)


def cmd_evaluate(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model("none")
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.to(dev)
    res = {}
    for p in a.real:
        for split in a.splits.split(","):
            i, m, _ = load_real(p, split)
            res[f"{Path(p).stem}/{split}"] = metrics(predict(model, i, dev), m)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_summary(a):
    """Fine-tuning results (result.json under --ft) -> markdown table: mean +- sd over seeds of test Dice / IoU."""
    import collections
    rows = collections.defaultdict(list)
    for f in Path(a.ft).glob("*/result.json"):
        r = json.loads(f.read_text())
        rows[(r["dataset"], r["n_train"], r["init"])].append(r)
    inits = ["none", "imagenet", "sim0", "simin"]
    print("| dataset | n train | " + " | ".join(f"{i} Dice / IoU" for i in inits) + " |")
    print("|---|---|" + "---|" * len(inits))
    for ds, n in sorted({k[:2] for k in rows}):
        cells = []
        for i in inits:
            rs = rows.get((ds, n, i), [])
            d, u = np.array([r["dice"] for r in rs]), np.array([r["iou"] for r in rs])
            cells.append(f"{d.mean():.3f}+-{d.std():.3f} / {u.mean():.3f} (n={len(rs)})" if rs else "-")
        print(f"| {ds} | {n} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache-sim")
    c.add_argument("--sim", required=True)
    c.add_argument("--out", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--data", required=True)
    p.add_argument("--init", default="imagenet", help="imagenet | none")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--real", nargs="*", help="real HDF5s, test Dice logged for monitoring only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zoom", type=float, nargs=2, default=(0.8, 2.0),
                   help="scale range of the augmentation (log-uniform, > 1 zooms in): simulated lesions are smaller "
                        "relative to the frame than in the real sets")
    p.add_argument("--out", required=True)
    f = sub.add_parser("finetune")
    f.add_argument("--real", required=True)
    f.add_argument("--init", default="imagenet", help="imagenet | none | sim:<model.pt>")
    f.add_argument("--init-name")
    f.add_argument("--n-train", type=int)
    f.add_argument("--iters", type=int, default=3000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--keep", action="store_true", help="keep the fine-tuned weights")
    f.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--model", required=True)
    e.add_argument("--real", nargs="+", required=True)
    e.add_argument("--splits", default="test")
    e.add_argument("--out")
    m = sub.add_parser("summary")
    m.add_argument("--ft", required=True, help="directory of fine-tuning runs")
    a = ap.parse_args()
    {"cache-sim": cmd_cache_sim, "pretrain": cmd_pretrain, "finetune": cmd_finetune, "evaluate": cmd_evaluate,
     "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
